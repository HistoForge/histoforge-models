"""Fork of histoforge_models.train.train_seg() with real-time wandb logging.

Retains the upstream Cellpose data/flow primitives while adding the explicit
HistoForge task, optimizer, checkpoint-selection, and W&B contracts below.
When nclasses > 0, uses the requested CE, Tversky, or CE+Tversky
classification objective via MultiTaskLoss, with optional uncertainty
weighting and per-class F1 metrics.
"""

import sys
import time
import numpy as np
from histoforge_models import models, io
from histoforge_models.transforms import normalize_img, random_rotate_and_resize
from histoforge_models.pathology.training.eval import compute_seg_ap
from histoforge_models.pathology.decode_profile import HOVER_TILE_OVERLAP
from histoforge_models.pathology.optimization import build_param_groups
from histoforge_models.pathology.metric_aggregation import (
    DEFAULT_SEGMENTATION_AGGREGATION,
    aggregate_segmentation_counts,
    class_f1,
    mean_defined,
    validate_segmentation_aggregation,
)
from pathlib import Path
import torch
from torch import nn
import logging
from histoforge_models.train import (
    _loss_fn_seg,
    _loss_fn_class,
    _loss_fn_tversky,
    mL1_ACE_loss,
    _process_train_test,
    _get_batch,
    _split_flow_weight_channel,
)

train_logger = logging.getLogger(__name__)


def _log_host_peak_rss():
    """Log the main process peak RSS when the platform provides resource."""
    try:
        import resource
    except ModuleNotFoundError:
        return

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    train_logger.info(
        ">>> host RSS after data loading: %.1f GiB (peak, main process)",
        peak_rss / divisor,
    )


class MultiTaskLoss(nn.Module):
    """Learnable multi-task loss balancing via homoscedastic uncertainty.

    Implements Kendall, Gal & Cipolla (2018) "Multi-Task Learning Using
    Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR.

    Each task gets a learnable log-variance parameter s_k. The total loss
    is: sum_k [ exp(-s_k) * L_k + s_k ].
    """

    def __init__(self, num_tasks, learnable=True):
        super().__init__()
        self.num_tasks = num_tasks
        self.learnable = learnable
        self.log_sigma_sq = nn.Parameter(torch.zeros(num_tasks),
                                         requires_grad=learnable)

    def forward(self, *task_losses):
        assert len(task_losses) == self.num_tasks
        stacked = torch.stack(list(task_losses))
        inv_variance = torch.exp(-self.log_sigma_sq)
        weighted = inv_variance * stacked
        if self.learnable:
            weighted = weighted + self.log_sigma_sq
        return weighted.sum()

    def get_task_weights(self, names):
        """Return current per-task precision weights as a dict.

        Args:
            names: list of strings, one per task, used as dict keys.
        """
        with torch.no_grad():
            w = torch.exp(-self.log_sigma_sq)
            return {name: w[i].item() for i, name in enumerate(names)}


def _classification_task_names(cls_loss_fn, has_classes=True):
    """Return the ordered classification-loss contract for a run."""
    choices = {
        "ce": ("cls_ce",),
        "tversky": ("cls_tversky",),
        "ce+tversky": ("cls_ce", "cls_tversky"),
    }
    if cls_loss_fn not in choices:
        raise ValueError("cls_loss_fn must be 'ce', 'tversky', or 'ce+tversky', "
                         f"got {cls_loss_fn!r}")
    return choices[cls_loss_fn] if has_classes else ()


def train_seg_wandb(
        net, train_data=None, train_labels=None, train_files=None,
        train_labels_files=None, train_probs=None, test_data=None, test_labels=None,
        test_files=None, test_labels_files=None, test_probs=None, channel_axis=None,
        load_files=True, batch_size=1, learning_rate=1e-5, n_epochs=100,
        weight_decay=0.1, normalize=True, compute_flows=False, save_path=None,
        nimg_per_epoch=None, nimg_test_per_epoch=None, rescale=False, scale_range=None,
        bsize=256, min_train_masks=5, model_name=None, class_weights=None,
        wandb_run=None, freeze_backbone=False, nclasses=0, freeze_encoder_only=False,
        gradient_checkpointing=False, freeze_seg_head=False, cls_loss_fn="ce",
        tversky_alpha=0.3, tversky_gamma=1.33, cls_loss_weight=1.0, lr_scheduler="step",
        augmentation_strategy="none", use_uncertainty_weighting=False, num_workers=0,
        llrd_factor=None, patch_embed_lr=None, adapter_lr=None,
        ap_every=0, cellpose_model=None, test_names=None, lazy_train=False,
        log_priors=None, logit_adjustment_tau=0.0, calibration_loss=False,
        calibration_n_bins=15, calibration_variant="hard", head_type="flow",
        grad_clip=0.0, hv_edge_threshold=0.4, hv_object_size=10, hv_fg_threshold=0.5,
        hv_morph_kernel=5, class_weights_target="both", seed=42,
        tile_overlap=HOVER_TILE_OVERLAP, flow_loss_weighting="pixel",
        segmentation_metric_aggregation=DEFAULT_SEGMENTATION_AGGREGATION):
    """Train the network with images for segmentation (and optional classification).

    When nclasses > 0 (semantic mode), uses joint seg+cls training:
    - Computes the classification terms selected by ``cls_loss_fn``
    - Uses MultiTaskLoss for loss combination (optional uncertainty weighting)
    - No autocast (full float32 precision, matching joint training)
    - Per-batch normalization when augmentation is enabled
    - Per-class F1 metrics during validation

    Args:
        augmentation_strategy: "none" (no color aug), "color" (HED stain
            perturbation), or "all" (HED/NMF stain + blur/noise/jitter).
        use_uncertainty_weighting: Learn loss weights via Kendall et al. 2018.
        ap_every: if >0 and cellpose_model is given, compute val AP@0.5 every N
            epochs (full mask inference) and save a `_best_ap` checkpoint.
        cellpose_model: CellposeModel wrapping `net`, used for AP mask inference.
        test_names: val image filenames, for per-dataset AP grouping.
        lazy_train: if True, stream the train corpus from disk via
            LazyTrainingDataset (train_files/train_labels_files) rather than
            holding it in RAM.
        flow_loss_weighting: ``"pixel"`` preserves the established flow MSE;
            ``"equal_instance"`` applies inverse-full-instance-area weighting
            to the flow vectors only. It is never inferred from dataset identity.
        segmentation_metric_aggregation: ``"pooled"`` (default) pools instance
            counts before AP/mSA; ``"per_image"`` averages defined image scores.
            Both comparisons are logged. Pixel F1 is independently pooled.

    Returns:
        tuple: (model_path, train_losses, test_losses)
    """
    validate_segmentation_aggregation(segmentation_metric_aggregation)
    if flow_loss_weighting not in ("pixel", "equal_instance"):
        raise ValueError(
            "flow_loss_weighting must be 'pixel' or 'equal_instance', got "
            f"{flow_loss_weighting!r}")
    if head_type != "flow" and flow_loss_weighting != "pixel":
        raise ValueError(
            "equal-instance flow weighting applies only to the Cellpose flow head")
    if freeze_seg_head and flow_loss_weighting != "pixel":
        raise ValueError(
            "equal-instance flow weighting requires a trainable flow head")

    device = net.device

    original_net_dtype = net.dtype
    if net.dtype == torch.bfloat16:
        train_logger.info(">>> converting bfloat16 network to float32 for training")
        net.dtype = torch.float32

    if gradient_checkpointing:
        net.gradient_checkpointing_enable()
        train_logger.info(">>> gradient checkpointing enabled")

    scale_range = 0.5 if scale_range is None else scale_range

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default.copy()
        normalize_params["normalize"] = normalize

    has_classes = nclasses > 0
    augmenting = augmentation_strategy != "none"
    is_hovernet = head_type == "hovernet"
    classification_tasks = _classification_task_names(cls_loss_fn,
                                                      has_classes=has_classes)
    if cls_loss_weight <= 0:
        raise ValueError(f"cls_loss_weight must be positive, got {cls_loss_weight}")
    use_ce = "cls_ce" in classification_tasks
    use_tversky = "cls_tversky" in classification_tasks
    # Class-weight target — "both" (default, matches Classpose/Arm B legacy)
    # weights BOTH CE and Tversky by class_weights; "ce" weights only CE and
    # lets Tversky's built-in per-class averaging + α/β asymmetry handle
    # imbalance alone. `weight_tversky` is used at the tversky call sites
    # (train + val) so a single flip toggles both.
    if class_weights_target not in ("ce", "both"):
        raise ValueError(
            f"class_weights_target must be 'ce' or 'both', got {class_weights_target!r}"
        )
    weight_tversky_by_class = (class_weights_target == "both")
    # HoVer has four segmentation channels; flow has three.
    nclasses_slice = -4 if is_hovernet else -3
    # HoVer-Net seg loss (MSE + MSGE + CE + Dice) — replaces `_loss_fn_seg`
    # when head_type='hovernet'. Instantiated once so the module lives on
    # the training device without per-batch reallocation.
    if is_hovernet:
        from histoforge_models.pathology.managed_training._hover_losses import HVMapLoss
        hv_loss_fn = HVMapLoss().to(device)
    else:
        hv_loss_fn = None
    if augmenting:
        # Defer normalization — color augmentations need raw pixel values.
        # _reshape_norm will still enforce 3-channel CHW format.
        ptt_normalize = {**normalize_params, "normalize": False}
    else:
        ptt_normalize = normalize_params

    out = _process_train_test(
        train_data=train_data, train_labels=train_labels, train_files=train_files,
        train_labels_files=train_labels_files, train_probs=train_probs,
        test_data=test_data, test_labels=test_labels, test_files=test_files,
        test_labels_files=test_labels_files, test_probs=test_probs,
        load_files=load_files, min_train_masks=min_train_masks,
        compute_flows=compute_flows, channel_axis=channel_axis,
        normalize_params=ptt_normalize, device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
    _log_host_peak_rss()
    # GT instance masks + raw images for val AP. Eager val uses the in-RAM
    # arrays; file-based (lazy) val reads them from disk once — val is small.
    gt_val_masks, val_ap_images = None, None
    if ap_every and cellpose_model is not None:
        if test_labels is not None:
            gt_val_masks = [
                test_labels[k][0].astype(np.int32) for k in range(len(test_labels))
            ]
            val_ap_images = test_data
        elif test_labels_files is not None:
            gt_val_masks = [io.imread(f)[0].astype(np.int32) for f in test_labels_files]
            val_ap_images = [io.imread(f) for f in test_files]
    # already normalized, do not normalize during training
    kwargs = {} if normed else {"normalize_params": ptt_normalize}

    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    # Freeze backbone if requested (train only output heads).
    if freeze_backbone:
        for name, param in net.named_parameters():
            if not name.startswith("out"):
                param.requires_grad = False
        n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        train_logger.info(">>> backbone frozen, training %d params "
                          "(out head only)", n_trainable)

    # CellVTA-style freeze: pin ONLY the DINOv3 ViT encoder (`backbone.*`),
    # keep the CNN adapter (spm/cti_modules), neck, readout, and cls heads
    # trainable. Tests "is the Phase-1 encoder representation
    # good enough that we only adapt + read out", and slashes ~300M trainable
    # params to fight overfitting on small Phase-2 sets. Mutually exclusive
    # with freeze_backbone (which is the stronger decoder-only freeze).
    if freeze_encoder_only:
        for name, param in net.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
        n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        train_logger.info(
            ">>> ViT encoder frozen (CellVTA-style), training "
            "%d params (adapter + readout + heads)", n_trainable)

    if freeze_seg_head:
        for param in net.out.parameters():
            param.requires_grad = False
        net.W2.requires_grad = False
        n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        train_logger.info(">>> seg head frozen, training %d params", n_trainable)

    if class_weights is not None and isinstance(class_weights,
                                                (list, np.ndarray, tuple)):
        class_weights = torch.from_numpy(np.float32(class_weights)).to(device).float()
        train_logger.info(">>> class_weights: %s", class_weights)

    if log_priors is not None:
        if not isinstance(log_priors, torch.Tensor):
            log_priors = torch.from_numpy(np.float32(log_priors))
        log_priors = log_priors.to(device=device, dtype=torch.float32)
        train_logger.info(">>> log_priors (tau=%.2f): %s", logit_adjustment_tau,
                          log_priors)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    # learning rate schedule
    warmup = np.linspace(0, learning_rate, 10)
    if lr_scheduler == "cosine":
        n_cosine = max(0, n_epochs - 10)
        cosine = 0.5 * learning_rate * (1 +
                                        np.cos(np.pi * np.arange(n_cosine) / n_cosine))
        LR = np.concatenate([warmup, cosine])
    elif lr_scheduler == "step":
        LR = np.append(warmup, learning_rate * np.ones(max(0, n_epochs - 10)))
        if n_epochs > 300:
            LR = LR[:-100]
            for i in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(10))
        elif n_epochs > 99:
            LR = LR[:-50]
            for i in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")
    train_logger.info(
        f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}"
    )

    # Optimizer — LLRD param groups for DINOv3 paths, else single group.
    llrd_active = llrd_factor is not None
    if llrd_active:
        param_groups = build_param_groups(
            net, base_lr=learning_rate, llrd_factor=llrd_factor,
            patch_embed_lr=patch_embed_lr or learning_rate, weight_decay=weight_decay,
            adapter_lr=adapter_lr)
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate,
                                      weight_decay=weight_decay)
        train_logger.info(
            ">>> LLRD enabled: factor=%.3f, patch_embed_lr=%.2e, "
            "%d param groups", llrd_factor, patch_embed_lr or learning_rate,
            len(param_groups))
    else:
        trainable = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate,
                                      weight_decay=weight_decay)

    # MultiTaskLoss covers the active segmentation/classification losses.
    seg_trainable = not freeze_seg_head and (any(
        p.requires_grad for p in net.out.parameters()) or net.W2.requires_grad)
    if flow_loss_weighting == "equal_instance" and not seg_trainable:
        raise ValueError(
            "equal-instance flow weighting requires an active trainable "
            "segmentation objective")
    task_names = []
    if seg_trainable:
        task_names.append("seg")
    task_names.extend(classification_tasks)
    if calibration_loss:
        task_names.append("calibration")
    use_multitask = len(task_names) > 1
    task_loss = None
    if use_multitask:
        task_loss = MultiTaskLoss(num_tasks=len(task_names),
                                  learnable=use_uncertainty_weighting).to(device)
        if use_uncertainty_weighting:
            optimizer.add_param_group({
                "params": task_loss.parameters(),
                "lr": learning_rate * 0.1,
            })
            train_logger.info(">>> uncertainty weighting enabled (%d losses): %s",
                              len(task_names), task_names)
        else:
            train_logger.info(">>> equal loss weighting (%d losses): %s",
                              len(task_names), task_names)

    # LLRD: snapshot each group's base LR so the epoch schedule scales it
    # (rather than overwriting). Done after add_param_group so the MultiTaskLoss
    # log-variance group is included.
    if llrd_active:
        for g in optimizer.param_groups:
            g["initial_lr"] = g["lr"]

    use_dataset = num_workers > 0
    train_logger.info(f">>> num_workers={num_workers}, use_dataset={use_dataset}")
    if use_dataset:
        from histoforge_models.pathology.training.dataset import (
            TrainingDataset,
            LazyTrainingDataset,
            numpy_collate,
            seed_worker,
        )
        from torch.utils.data import DataLoader
        if lazy_train:
            train_dataset = LazyTrainingDataset(
                image_files=train_files, flow_files=train_labels_files,
                diameter_array=diam_train, augmentation_strategy=augmentation_strategy,
                diam_mean=net.diam_mean.item(), rescale=rescale,
                scale_range=scale_range, bsize=bsize,
                normalize_params=normalize_params,
                flow_loss_weighting=flow_loss_weighting)
        else:
            train_dataset = TrainingDataset(
                data_array=train_data, label_array=train_labels,
                diameter_array=diam_train, augmentation_strategy=augmentation_strategy,
                diam_mean=net.diam_mean.item(), rescale=rescale,
                scale_range=scale_range, bsize=bsize,
                normalize_params=normalize_params,
                flow_loss_weighting=flow_loss_weighting)
        train_logger.info(f">>> DataLoader with {num_workers} workers")
    train_logger.info(f">>> augmentation_strategy='{augmentation_strategy}'")

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    filename = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)

    train_logger.info(f">>> saving model to {filename}")

    best_val_loss = float("inf")
    filename_best = str(filename) + "_best"
    best_val_ap = -1.0
    filename_best_ap = str(filename) + "_best_ap"
    best_val_macro_f1 = -1.0
    filename_best_f1 = str(filename) + "_best_f1"

    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    for iepoch in range(n_epochs):
        epoch_seed = int(seed) + iepoch
        np.random.seed(epoch_seed)
        if augmenting:
            epoch_rng = np.random.default_rng(int(seed) * 1_000_003 + iepoch)
        if nimg != nimg_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            if train_probs is not None:
                rperm = np.random.choice(np.arange(0, nimg), size=(nimg,),
                                         p=train_probs)
            else:
                rperm = np.random.permutation(np.arange(0, nimg))
        if llrd_active:
            # scale every group's base LR by the schedule; preserves the
            # per-layer LLRD ratios instead of flattening them to one LR
            scale = LR[iepoch] / learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"] * scale
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = LR[iepoch]
        net.train()
        epoch_seg, epoch_ce, epoch_tversky = 0.0, 0.0, 0.0
        epoch_calibration = 0.0
        epoch_hv_mse = epoch_hv_msge = epoch_np_ce = epoch_np_dice = 0.0
        # Grad-norm accumulators (only used when grad_clip > 0).
        epoch_grad_norm_sum = 0.0
        epoch_grad_norm_max = 0.0
        num_batches = 0

        if use_dataset:
            nimg_epoch = nimg_per_epoch
            loader_generator = torch.Generator()
            loader_generator.manual_seed(epoch_seed)
            epoch_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      sampler=rperm.tolist(), num_workers=num_workers,
                                      collate_fn=numpy_collate,
                                      worker_init_fn=seed_worker,
                                      generator=loader_generator, drop_last=False)
            batch_iter = iter(epoch_loader)
        else:
            nimg_epoch = nimg_per_epoch
        epoch_samples = 0

        for k in range(0, nimg_epoch, batch_size):
            if use_dataset:
                imgi, lbl = next(batch_iter)
            else:
                kend = min(k + batch_size, nimg_per_epoch)
                inds = rperm[k:kend]
                imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels,
                                        files=train_files,
                                        labels_files=train_labels_files,
                                        flow_loss_weighting=flow_loss_weighting,
                                        **kwargs)
                diams = np.array([diam_train[i] for i in inds])
                rsc = diams / net.diam_mean.item() if rescale else np.ones(
                    len(diams), "float32")

                if augmenting:
                    from histoforge_models.pathology.training.augmentation import augment_batch
                    imgs = augment_batch(imgs, augmentation_strategy, rng=epoch_rng)

                imgi, lbl = random_rotate_and_resize(imgs, Y=lbls, rescale=rsc,
                                                     scale_range=scale_range,
                                                     xy=(bsize, bsize))[:2]

                if augmenting:
                    imgi = np.stack([
                        normalize_img(imgi[i], axis=0, **normalize_params)
                        for i in range(len(imgi))
                    ])

            X = torch.from_numpy(imgi).to(device)
            lbl = torch.from_numpy(lbl).to(device)
            lbl, flow_pixel_weights = _split_flow_weight_channel(
                lbl, flow_loss_weighting)

            # No autocast for semantic training (full float32 precision)
            if has_classes:
                y = net(X)[0]
            else:
                with torch.autocast(device_type=device.type, dtype=net.dtype):
                    y = net(X)[0]

            # --- Unified loss block: active losses in the SAME order as
            # task_names so MultiTaskLoss.get_task_weights() names align.
            active = []
            if seg_trainable:
                if is_hovernet:
                    # Layout: y = [cls_0..cls_{N-1}, V, H, np_bg, np_fg];
                    # lbl (post-strip) = [class, fg, V_gt, H_gt].
                    hv_res = hv_loss_fn(y[:, -4:-2], y[:, -2:], lbl[:, -2:],
                                        (lbl[:, -3] > 0.5).long())
                    loss_seg = hv_res["total"]
                    epoch_hv_mse += hv_res["hv_mse"].item()
                    epoch_hv_msge += hv_res["hv_msge"].item()
                    epoch_np_ce += hv_res["np_ce"].item()
                    epoch_np_dice += hv_res["np_dice"].item()
                else:
                    loss_seg = _loss_fn_seg(
                        lbl, y, device,
                        flow_pixel_weights=flow_pixel_weights)
                active.append(loss_seg)
                epoch_seg += loss_seg.item()
            if use_ce:
                loss_ce = _loss_fn_class(lbl, y, class_weights=class_weights,
                                         log_priors=log_priors,
                                         tau=logit_adjustment_tau,
                                         nclasses_slice=nclasses_slice)
                active.append(cls_loss_weight * loss_ce)
                epoch_ce += loss_ce.item()
            if use_tversky:
                loss_tv = _loss_fn_tversky(
                    lbl, y, n_classes=nclasses,
                    class_weights=(class_weights if weight_tversky_by_class else None),
                    log_priors=log_priors, tau=logit_adjustment_tau,
                    nclasses_slice=nclasses_slice, alpha=tversky_alpha,
                    gamma=tversky_gamma)
                active.append(cls_loss_weight * loss_tv)
                epoch_tversky += loss_tv.item()
            if calibration_loss:
                loss_cal = mL1_ACE_loss(y[:, :nclasses_slice], lbl[:, 0].long(),
                                        n_bins=calibration_n_bins,
                                        variant=calibration_variant)
                active.append(loss_cal)
                epoch_calibration += loss_cal.item()

            loss = task_loss(*active) if use_multitask else active[0]

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                # clip_grad_norm_ returns the PRE-clip total norm; log it so
                # `train/grad_norm` shows whether we're regularly hitting the
                # ceiling (defensive tune signal). Clipping only applies to
                # net params; task_loss (uncertainty-weighting log_sigma) is
                # small and naturally bounded.
                gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                       max_norm=grad_clip)
                gval = float(gnorm)
                epoch_grad_norm_sum += gval
                epoch_grad_norm_max = max(epoch_grad_norm_max, gval)
            optimizer.step()

            train_loss = loss.item() * len(imgi)
            lavg += train_loss
            nsum += len(imgi)
            train_losses[iepoch] += train_loss
            epoch_samples += len(imgi)
            num_batches += 1

        train_losses[iepoch] /= max(epoch_samples, 1)

        # --- wandb: build log_dict (every epoch) ---
        if wandb_run is not None:
            log_dict = {
                "train/loss": float(train_losses[iepoch]),
                "lr": float(LR[iepoch]),
                "train/effective_samples": epoch_samples,
                "train/optimizer_steps": num_batches,
            }
            if num_batches > 0:
                if seg_trainable:
                    log_dict["train/seg_loss"] = epoch_seg / num_batches
                    if is_hovernet:
                        log_dict["train/hv_mse"] = epoch_hv_mse / num_batches
                        log_dict["train/hv_msge"] = epoch_hv_msge / num_batches
                        log_dict["train/np_ce"] = epoch_np_ce / num_batches
                        log_dict["train/np_dice"] = epoch_np_dice / num_batches
                if use_ce:
                    log_dict["train/ce_loss"] = epoch_ce / num_batches
                if use_tversky:
                    log_dict["train/tversky_loss"] = epoch_tversky / num_batches
                if calibration_loss:
                    log_dict["train/calibration_loss"] = epoch_calibration / num_batches
                if grad_clip > 0:
                    log_dict["train/grad_norm"] = epoch_grad_norm_sum / num_batches
                    log_dict["train/grad_norm_max"] = epoch_grad_norm_max
                if use_uncertainty_weighting and task_loss is not None:
                    tw = task_loss.get_task_weights(task_names)
                    for name, weight in tw.items():
                        log_dict[f"train/weight_{name}"] = weight

        if True:
            lavgt = 0.
            val_seg_sum, val_ce_sum, val_tv_sum, val_batches = 0., 0., 0., 0
            val_calibration_sum = 0.
            val_hv_mse_sum = val_hv_msge_sum = 0.
            val_np_ce_sum = val_np_dice_sum = 0.
            if has_classes:
                tp_per_class = np.zeros(nclasses)
                fp_per_class = np.zeros(nclasses)
                fn_per_class = np.zeros(nclasses)
            if test_data is not None or test_files is not None:
                np.random.seed(int(seed) + 10_000_000 + iepoch)
                if nimg_test != nimg_test_per_epoch:
                    rperm = np.random.choice(np.arange(0, nimg_test),
                                             size=(nimg_test_per_epoch,), p=test_probs)
                else:
                    rperm = np.random.permutation(np.arange(0, nimg_test))
                for ibatch in range(0, len(rperm), batch_size):
                    with torch.no_grad():
                        net.eval()
                        inds = rperm[ibatch:ibatch + batch_size]
                        imgs, lbls = _get_batch(inds, data=test_data,
                                                labels=test_labels, files=test_files,
                                                labels_files=test_labels_files,
                                                flow_loss_weighting=flow_loss_weighting,
                                                **kwargs)
                        diams = np.array([diam_test[i] for i in inds])
                        rsc = diams / net.diam_mean.item() if rescale else np.ones(
                            len(diams), "float32")
                        imgi, lbl = random_rotate_and_resize(imgs, Y=lbls, rescale=rsc,
                                                             scale_range=scale_range,
                                                             xy=(bsize, bsize))[:2]
                        if augmenting:
                            imgi = np.stack([
                                normalize_img(imgi[i], axis=0, **normalize_params)
                                for i in range(len(imgi))
                            ])
                        X = torch.from_numpy(imgi).to(device)
                        lbl = torch.from_numpy(lbl).to(device)
                        lbl, flow_pixel_weights = _split_flow_weight_channel(
                            lbl, flow_loss_weighting)

                        if has_classes:
                            y = net(X)[0]
                        else:
                            with torch.autocast(device_type=device.type,
                                                dtype=net.dtype):
                                y = net(X)[0]

                        # Unified val loss block — mirrors training, same
                        # ordering as task_names for MultiTaskLoss weighting.
                        active = []
                        if seg_trainable:
                            if is_hovernet:
                                hv_res_v = hv_loss_fn(y[:, -4:-2], y[:, -2:],
                                                      lbl[:, -2:], (lbl[:, -3]
                                                                    > 0.5).long())
                                loss_seg_v = hv_res_v["total"]
                                val_hv_mse_sum += hv_res_v["hv_mse"].item()
                                val_hv_msge_sum += hv_res_v["hv_msge"].item()
                                val_np_ce_sum += hv_res_v["np_ce"].item()
                                val_np_dice_sum += hv_res_v["np_dice"].item()
                            else:
                                loss_seg_v = _loss_fn_seg(
                                    lbl, y, device,
                                    flow_pixel_weights=flow_pixel_weights)
                            active.append(loss_seg_v)
                            val_seg_sum += loss_seg_v.item()
                        if use_ce:
                            loss_ce_v = _loss_fn_class(lbl, y,
                                                       class_weights=class_weights,
                                                       log_priors=log_priors,
                                                       tau=logit_adjustment_tau,
                                                       nclasses_slice=nclasses_slice)
                            active.append(cls_loss_weight * loss_ce_v)
                            val_ce_sum += loss_ce_v.item()
                        if use_tversky:
                            loss_tv_v = _loss_fn_tversky(
                                lbl, y, n_classes=nclasses,
                                class_weights=(class_weights if weight_tversky_by_class
                                               else None), log_priors=log_priors,
                                tau=logit_adjustment_tau, nclasses_slice=nclasses_slice,
                                alpha=tversky_alpha, gamma=tversky_gamma)
                            active.append(cls_loss_weight * loss_tv_v)
                            val_tv_sum += loss_tv_v.item()
                        if calibration_loss:
                            loss_cal_v = mL1_ACE_loss(y[:, :nclasses_slice],
                                                      lbl[:, 0].long(),
                                                      n_bins=calibration_n_bins,
                                                      variant=calibration_variant)
                            active.append(loss_cal_v)
                            val_calibration_sum += loss_cal_v.item()
                        val_batches += 1

                        loss = task_loss(*active) if use_multitask else active[0]

                        test_loss = loss.item() * len(imgi)
                        lavgt += test_loss

                        # Per-class precision, recall, F1
                        if has_classes:
                            cls_pred = y[:, :nclasses_slice].argmax(dim=1)
                            cls_true = lbl[:, 0].long()
                            valid = cls_true != 255
                            cls_pred = cls_pred[valid]
                            cls_true = cls_true[valid]
                            for c in range(nclasses):
                                c_true = cls_true == c
                                c_pred = cls_pred == c
                                tp_per_class[c] += (c_true & c_pred).sum().item()
                                fp_per_class[c] += (~c_true & c_pred).sum().item()
                                fn_per_class[c] += (c_true & ~c_pred).sum().item()

                lavgt /= len(rperm)
                test_losses[iepoch] = lavgt
                # Save best model
                if lavgt < best_val_loss:
                    best_val_loss = lavgt
                    train_logger.info(
                        f"new best val_loss={lavgt:.4f} at epoch {iepoch}, saving to {filename_best}"
                    )
                    torch.save(net.state_dict(), filename_best)
                # --- wandb: add val metrics ---
                if wandb_run is not None:
                    log_dict["val/loss"] = float(test_losses[iepoch])
                    if val_batches > 0:
                        if seg_trainable:
                            log_dict["val/seg_loss"] = val_seg_sum / val_batches
                            if is_hovernet:
                                log_dict["val/hv_mse"] = val_hv_mse_sum / val_batches
                                log_dict["val/hv_msge"] = val_hv_msge_sum / val_batches
                                log_dict["val/np_ce"] = val_np_ce_sum / val_batches
                                log_dict["val/np_dice"] = val_np_dice_sum / val_batches
                        if use_ce:
                            log_dict["val/ce_loss"] = val_ce_sum / val_batches
                        if use_tversky:
                            log_dict["val/tversky_loss"] = val_tv_sum / val_batches
                        if calibration_loss:
                            log_dict[
                                "val/calibration_loss"] = val_calibration_sum / val_batches
                if has_classes:
                    # This is pooled PIXEL F1 on the sampled validation crops,
                    # not decoded-object F1. Include prediction-only classes;
                    # omit only classes with TP=FP=FN=0. Selection is independent
                    # of whether W&B logging is enabled.
                    f1_values = class_f1(tp_per_class[1:], fp_per_class[1:],
                                         fn_per_class[1:])
                    macro_f1 = float(mean_defined(f1_values))
                    if wandb_run is not None:
                        precisions, recalls = [], []
                        for c, f1 in enumerate(f1_values, start=1):
                            if np.isfinite(f1):
                                log_dict[f"val/pixel_f1_class_{c}"] = float(f1)
                                # Retain the existing auxiliary precision/recall
                                # zero-denominator convention. Only F1 determines
                                # class inclusion and the selected checkpoint.
                                precision = float(tp_per_class[c] / (
                                    tp_per_class[c] + fp_per_class[c] + 1e-8))
                                recall = float(tp_per_class[c] / (
                                    tp_per_class[c] + fn_per_class[c] + 1e-8))
                                log_dict[f"val/pixel_prec_class_{c}"] = precision
                                log_dict[f"val/pixel_rec_class_{c}"] = recall
                                precisions.append(precision)
                                recalls.append(recall)
                        if np.isfinite(macro_f1):
                            log_dict["val/pixel_macro_f1"] = macro_f1
                            log_dict["val/pixel_macro_prec"] = float(np.mean(precisions))
                            log_dict["val/pixel_macro_rec"] = float(np.mean(recalls))
                    if macro_f1 > best_val_macro_f1:
                        best_val_macro_f1 = macro_f1
                        train_logger.info(
                            f"new best val/pixel_macro_f1={macro_f1:.4f} "
                            f"at epoch {iepoch}, saving to {filename_best_f1}")
                        torch.save(net.state_dict(), filename_best_f1)
            lavg /= nsum
            train_logger.info(
                f"{iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
            )
            lavg, nsum = 0, 0

        # --- val AP@0.5 (own cadence; full mask inference on the val set) ---
        if (ap_every and cellpose_model is not None and gt_val_masks is not None and
            (iepoch % ap_every == 0 or iepoch == n_epochs - 1)):
            net.eval()
            with torch.no_grad():
                if is_hovernet:
                    from histoforge_models.pathology.training.hovernet_eval import (
                        compute_hovernet_seg_ap,
                    )
                    ap_res = compute_hovernet_seg_ap(
                        net, val_ap_images, gt_val_masks, bsize=bsize,
                        batch_size=batch_size, tile_overlap=tile_overlap,
                        edge_threshold=hv_edge_threshold, object_size=hv_object_size,
                        fg_threshold=hv_fg_threshold, morph_kernel=hv_morph_kernel,
                        segmentation_metric_aggregation=segmentation_metric_aggregation)
                else:
                    ap_res = compute_seg_ap(cellpose_model, val_ap_images, gt_val_masks,
                                            bsize=bsize, batch_size=batch_size,
                                            segmentation_metric_aggregation=segmentation_metric_aggregation)
            net.train()
            ap50 = ap_res["AP@0.5"]
            # Retain the declared primary (HoVer mSA, flow AP50); both now use
            # the same Cellpose matcher and explicitly selected aggregation.
            if is_hovernet:
                best_score = ap_res["mSA"]
                best_label = "mSA"
            else:
                best_score = ap50
                best_label = "AP@0.5"
            if best_score > best_val_ap:
                best_val_ap = best_score
                train_logger.info(
                    f"new best val {best_label}={best_score:.4f} at epoch "
                    f"{iepoch}, saving to {filename_best_ap}")
                torch.save(net.state_dict(), filename_best_ap)
            train_logger.info(
                f">>> val AP@0.5={ap50:.4f}, AP@0.75={ap_res['AP@0.75']:.4f}, "
                f"mSA={ap_res['mSA']:.4f} (epoch {iepoch})")
            if wandb_run is not None:
                log_dict.update({f"val/{key}": value for key, value in ap_res.items()
                                 if isinstance(value, float) and np.isfinite(value)})
                # per-dataset breakdown only when ALL val filenames are prefixed
                # `<dataset>__...` (the prepare_he_pretrain.py convention). For
                # single-dataset training (PUMA, Leydig) where filenames have no
                # `__` separator, skip — otherwise every image would become its
                # own "dataset" key in wandb.
                if test_names is not None and test_names and all(
                        "__" in nm for nm in test_names):
                    groups = {}
                    for i, nm in enumerate(test_names):
                        groups.setdefault(nm.split("__")[0], []).append(i)
                    per_ds = []
                    for ds, idxs in sorted(groups.items()):
                        scores = aggregate_segmentation_counts(
                            ap_res["tp"][idxs], ap_res["fp"][idxs], ap_res["fn"][idxs],
                            segmentation_metric_aggregation)
                        ds_ap = float(scores[0])
                        if np.isfinite(ds_ap):
                            log_dict[f"val/AP@0.5/{ds}"] = ds_ap
                            per_ds.append(ds_ap)
                    if per_ds:
                        log_dict["val/AP@0.5/macro"] = float(np.mean(per_ds))

        # --- wandb: flush once per epoch (outside the if block) ---
        if wandb_run is not None:
            wandb_run.log(log_dict, step=iepoch)

    torch.save(net.state_dict(), filename)
    train_logger.info(f"saved final model to {filename}")
    if original_net_dtype != torch.float32:
        train_logger.info(
            f">>> converting network back to {original_net_dtype} after training")
        net.dtype = original_net_dtype

    return filename, train_losses, test_losses
