"""HistoForge adapter training script for datasets with separate image/label directories.

Bypasses cellpose's io.load_train_test_data() which requires images and masks
in the same directory. Instead loads from separate directories and calls
train_seg_wandb() (fork of train.train_seg with real-time wandb logging).

Two orthogonal controls:
  --checkpoint    What weights to start from (defaults to official DINOv3 initialization)
  --freeze_backbone   What to train (full model vs output head only)

Classification requires per-instance JSON class labels in a separate
directory, matched by basename to the mask files. Each JSON has the format:
    {"instance_id": class_id, ...}
where keys are string instance IDs matching mask pixel values, and values
are 0-indexed foreground class IDs in the dataset's declared class order.
Background (class 0) is assigned automatically to unlabeled pixels.
Set nclasses = num_foreground_classes + 1.

Usage:
    # 1. Segmentation only (from the official DINOv3 backbone)
    python -m histoforge_models.pathology.training.train_seg \
        --train_images /path/to/train/images \
        --train_labels /path/to/train/labels \
        --save_path /path/to/output

    # 2. Segmentation from a checkpoint
    python -m histoforge_models.pathology.training.train_seg \
        --checkpoint /path/to/model \
        --train_images /path/to/train/images \
        --train_labels /path/to/train/labels \
        --save_path /path/to/output

    # 3. Joint seg + classification (full adapter model)
    python -m histoforge_models.pathology.training.train_seg \
        --train_images /path/to/train/images \
        --train_labels /path/to/train/labels \
        --class_labels_dir /path/to/train/class_labels \
        --nclasses 5 --class_weights_mode inverse_freq \
        --save_path /path/to/output

    # 4. Classification head only (frozen backbone, from checkpoint)
    python -m histoforge_models.pathology.training.train_seg \
        --checkpoint /path/to/finetuned/model \
        --train_images /path/to/train/images \
        --train_labels /path/to/train/labels \
        --class_labels_dir /path/to/train/class_labels \
        --nclasses 5 --freeze_backbone \
        --class_weights_mode inverse_freq \
        --save_path /path/to/output

    # 5. Dense semantic segmentation (per-pixel class masks, no JSON labels)
    python -m histoforge_models.pathology.training.train_seg \
        --checkpoint /path/to/seg_model \
        --train_images /path/to/images \
        --train_labels /path/to/dense_masks \
        --nclasses 2 --dense_semantic \
        --cls_loss_fn ce+tversky --class_weights_mode inverse_freq \
        --save_path /path/to/output
"""

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from histoforge_models import io
from histoforge_models.pathology.metric_aggregation import (
    DEFAULT_SEGMENTATION_AGGREGATION,
    SEGMENTATION_AGGREGATIONS,
)
from histoforge_models.pathology.training.data_utils import (
    get_paired_files,
    load_arrays,
    prepare_semantic_labels,
    prepare_dense_semantic_labels,
    prepare_hovernet_labels,
    compute_class_weights,
)
from histoforge_models.pathology.training.initialization import create_cellpose_model
from histoforge_models.pathology.training.train_seg_loop import train_seg_wandb
from histoforge_models.pathology.decode_profile import (
    HOVER_DECODE_PROFILE_VERSION,
    HOVER_EDGE_THRESHOLD,
    HOVER_FG_THRESHOLD,
    HOVER_MORPH_KERNEL,
    HOVER_OBJECT_SIZE,
    HOVER_TILE_OVERLAP,
)

logger = logging.getLogger(__name__)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_set_summary(paths):
    """Content-address the exact ordered-independent set of input files."""
    digest = hashlib.sha256()
    total_bytes = 0
    normalized = sorted(Path(path) for path in paths)
    for path in normalized:
        size = path.stat().st_size
        total_bytes += size
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {
        "count": len(normalized),
        "total_bytes": total_bytes,
        "content_sha256": digest.hexdigest(),
    }


def _git_state():
    repo = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                         text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo,
                                    text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _write_recipe(path, args, run_config, file_sets):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint = {
            "path": str(checkpoint_path.resolve()),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256_file(checkpoint_path),
        }
    recipe = {
        "format_version": 2,
        "command": sys.argv,
        "git": _git_state(),
        "checkpoint": checkpoint,
        "configuration": run_config,
        "file_sets": {
            name: _file_set_summary(files) for name, files in file_sets.items() if files
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    recipe_path = output / "recipe.json"
    with open(recipe_path, "w") as stream:
        json.dump(recipe, stream, indent=2, sort_keys=True)
        stream.write("\n")
    logger.info("Wrote run recipe to %s", recipe_path)


def _resolve_flow_files(image_files, images_dir, mask_files):
    """Map each training image to its precomputed {stem}_flows.tif.

    Flows live in a `flows/` dir sibling to the images dir (written by
    datasets/dataset_converters/precompute_flows.py). Raises if any are missing.
    """
    flows_dir = Path(images_dir).parent / "flows"
    if len(image_files) != len(mask_files):
        raise ValueError("image/mask counts differ while resolving flow targets")
    flow_files = [str(flows_dir / (Path(f).stem + "_flows.tif")) for f in image_files]
    missing = [f for f in flow_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(flow_files)} flow files missing under "
            f"{flows_dir} (e.g. {missing[0]}). Run precompute_flows.py first.")
    stale = []
    for mask_file, flow_file in zip(mask_files, flow_files):
        sidecar = Path(flow_file).with_name(Path(flow_file).stem + ".mask.sha256")
        if not sidecar.is_file():
            stale.append(f"{flow_file}: missing mask-hash sidecar")
            continue
        if sidecar.read_text().strip() != _sha256_file(mask_file):
            stale.append(f"{flow_file}: mask hash mismatch")
    if stale:
        raise ValueError(f"{len(stale)} precomputed flows are stale or unverifiable "
                         f"(e.g. {stale[0]}). Re-run precompute_flows.py.")
    return flow_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the fixed HistoForge adapter from separate image/label directories")

    # Data paths
    parser.add_argument("--train_images", required=True,
                        help="Directory containing training images")
    parser.add_argument("--train_labels", required=True,
                        help="Directory containing training label masks")
    parser.add_argument("--val_images", help="Directory containing validation images")
    parser.add_argument("--val_labels",
                        help="Directory containing validation label masks")
    parser.add_argument("--test_images",
                        help="Directory containing test images (post-training eval)")
    parser.add_argument(
        "--test_labels",
        help="Directory containing test label masks (post-training eval)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Training hyperparameters
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--lr_scheduler", default="step", choices=["cosine", "step"],
        help="LR schedule: 'cosine' (annealing after warmup) "
        "or 'step' (constant plateau + halving decay)")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of patches per gradient step")
    parser.add_argument(
        "--ap_every", type=int, default=0,
        help="Compute val AP@0.5 (full mask inference) every "
        "N epochs; 0 = off. Adds a few minutes per pass.")
    parser.add_argument(
        "--segmentation_metric_aggregation", choices=SEGMENTATION_AGGREGATIONS,
        default=DEFAULT_SEGMENTATION_AGGREGATION,
        help="AP/mSA reporting and checkpoint selection: pooled counts (default) "
        "or equal per-image scores. Both summaries are also reported separately.")
    parser.add_argument(
        "--lazy_train", action="store_true",
        help="Stream the training corpus from disk per item "
        "instead of loading it all into RAM. Requires "
        "nclasses=0, explicit --val_images/--val_labels, "
        "and precomputed flows (precompute_flows.py).")

    # Architecture
    parser.add_argument(
        "--checkpoint", help="Path to model checkpoint to initialize from. "
        "Training uses it as a warm start: matching "
        "encoder/adapter/head tensors load, while a "
        "deliberately changed task or class head starts "
        "fresh. Defaults to the backbone base if omitted.")
    parser.add_argument("--bsize", type=int, default=256,
                        help="Crop/tile size (default 256)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute activations during backward to save GPU memory")
    parser.add_argument(
        "--llrd_factor", type=float, default=None,
        help="Layer-wise LR decay for the dinov3 backbone: "
        "block N gets learning_rate*llrd_factor**(23-N). "
        "None = single LR.")
    parser.add_argument(
        "--patch_embed_lr", type=float, default=None,
        help="Separate LR for the dinov3 patch_embed conv "
        "(used with --llrd_factor). None = learning_rate.")
    parser.add_argument(
        "--adapter_lr", type=float, default=None,
        help="LR for adapter params (SPM, CTI) in "
        "dinov3_adapter backbone. None = 1e-3.")
    # Semantic classification
    parser.add_argument(
        "--nclasses", type=int, default=0, help="Number of semantic classes including "
        "background (0 = segmentation only)")
    parser.add_argument(
        "--class_labels_dir", help="Dir with per-instance JSON class labels "
        "for training (matched by basename)")
    parser.add_argument(
        "--val_class_labels_dir", help="Dir with per-instance JSON class labels "
        "for validation (matched by basename)")
    parser.add_argument(
        "--dense_semantic", action="store_true",
        help="Dense per-pixel semantic mode: labels are "
        "(H,W) class masks instead of instance masks. "
        "Implies freeze_seg_head=True, min_train_masks=0, "
        "no --class_labels_dir needed.")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze encoder, train only output head")
    parser.add_argument(
        "--freeze_encoder_only", action="store_true",
        help="CellVTA-style: freeze ONLY the ViT encoder "
        "(backbone.*); keep the CNN adapter (spm/cti), "
        "neck, readout and cls heads trainable. "
        "Distinct from --freeze_backbone (which also "
        "freezes the adapter = decoder-only fine-tune).")
    parser.add_argument("--freeze_seg_head", action="store_true",
                        help="Freeze seg output head (out/W2), train only cls head")
    parser.add_argument("--cls_loss_fn", choices=["ce", "tversky", "ce+tversky"],
                        default="ce", help="Classification loss function")
    parser.add_argument("--tversky_alpha", type=float, default=0.3,
                        help="Tversky FP weight (lower = penalize FN more)")
    parser.add_argument("--tversky_gamma", type=float, default=1.33,
                        help="Tversky focal exponent")
    parser.add_argument(
        "--cls_loss_weight", type=float, default=1.0,
        help="Scalar weight for classification loss (loss = seg + w * cls)")
    parser.add_argument("--class_weights_mode", choices=["none", "inverse_freq"],
                        default="none", help="Class weight computation strategy")
    parser.add_argument(
        "--class_weights_target", choices=["ce", "both"], default="both",
        help="Where to apply the per-class weights: 'both' "
        "(default) weights CE and Tversky; 'ce' weights only CE "
        "and leaves Tversky to its built-in per-class "
        "averaging + α/β asymmetry.")
    parser.add_argument(
        "--grad_clip", type=float, default=0.0,
        help="Global gradient-norm clip on net.parameters() "
        "before the optimizer step. 0 disables clipping. "
        "Logs train/grad_norm and train/grad_norm_max "
        "to W&B when active.")
    parser.add_argument(
        "--use_uncertainty_weighting", action="store_true",
        help="Learn loss weights via Kendall et al. 2018 "
        "uncertainty-based multi-task weighting")
    parser.add_argument(
        "--logit_adjustment_tau", type=float, default=0.0,
        help="Logit adjustment temperature (Menon ICLR 2021): "
        "softmax(z_y + tau * log pi_y). 0 disables, "
        "1.0 is Bayes-optimal. Requires --nclasses > 0.")
    parser.add_argument(
        "--head_type", choices=["flow", "hovernet"], default="flow",
        help="Segmentation output head: 'flow' (cellpose "
        "3-ch [flow_y, flow_x, cellprob], default) or "
        "'hovernet' (4-ch [V, H, np_bg, np_fg] for "
        "HoVer-Net-style watershed post-processing on "
        "structure/tubule tasks).")
    parser.add_argument(
        "--flow_loss_weighting", choices=["pixel", "equal_instance"],
        default="pixel",
        help="Flow-vector MSE weighting. 'pixel' preserves the historical "
        "pixel mean; 'equal_instance' gives every complete source instance "
        "equal total flow influence. Foreground BCE is unchanged.")
    parser.add_argument(
        "--hv_edge_threshold", type=float, default=HOVER_EDGE_THRESHOLD,
        help="HoVer watershed val-AP eval: edge-map threshold "
        "for marker splitting. Use the evaluated settings "
        "for the target task. Only affects head_type=hovernet.")
    parser.add_argument(
        "--hv_object_size", type=int, default=HOVER_OBJECT_SIZE,
        help="HoVer watershed val-AP eval: minimum marker size "
        "in pixels. Smaller marker regions are discarded; "
        "use the evaluated settings for the target task.")
    parser.add_argument(
        "--hv_fg_threshold", type=float, default=HOVER_FG_THRESHOLD,
        help="HoVer watershed val-AP eval: fg-probability "
        "threshold for the watershed mask. Use the evaluated "
        "settings for the target task.")
    parser.add_argument(
        "--hv_morph_kernel", type=int, default=HOVER_MORPH_KERNEL,
        help="HoVer watershed val-AP eval: marker morphological-"
        "opening kernel in pixels (odd; 1 disables opening). "
        "Use the evaluated settings for the target task.")
    parser.add_argument(
        "--tile_overlap", type=float, default=HOVER_TILE_OVERLAP,
        help="HoVer overlap for validation AP and post-training evaluation "
        "(default 0.25). Flow evaluation retains Cellpose's 0.1 overlap.")
    parser.add_argument(
        "--calibration_loss", action="store_true",
        help="mL1-ACE differentiable calibration loss "
        "(Barfoot 2026). Reduces overconfidence on rare "
        "classes. Requires --nclasses > 0.")
    parser.add_argument("--calibration_n_bins", type=int, default=15,
                        help="Number of confidence bins for mL1-ACE.")
    parser.add_argument(
        "--calibration_variant", choices=["hard", "soft"], default="hard",
        help="hard = bucketize, soft = Gaussian kernels "
        "(differentiable everywhere).")
    parser.add_argument(
        "--oversample", action="store_true", help="Oversample images with rare classes "
        "(instance-weighted class balancing)")
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of data augmentation workers "
        "(0 = single-process)")
    parser.add_argument(
        "--augmentation_strategy", choices=["none", "color", "all"], default="none",
        help="Color augmentation: 'none' (disabled), "
        "'color' (HED stain perturbation only), "
        "'all' (HED/NMF stain + blur/noise/color jitter)")
    parser.add_argument(
        "--train_probs_mode", choices=["uniform", "area"], default="uniform",
        help="Image sampling: 'uniform' (every image equally "
        "likely) or 'area' (P(image) proportional to H*W, "
        "so a dataset's epoch share equals its share of "
        "total tissue area, not its image count)")

    # Output
    parser.add_argument("--save_path", required=True,
                        help="Directory for saving models and outputs")
    parser.add_argument("--model_name", help="Name for saved model file")
    parser.add_argument("--min_train_masks", type=int, default=5,
                        help="Minimum masks per image to include in training")
    parser.add_argument(
        "--val_patches_per_image", type=int, default=1,
        help="Number of random patches per val image per epoch. "
        "Higher = smoother val loss, slower eval.")

    # Wandb
    parser.add_argument("--wandb_project", default="cellpose-finetuning",
                        help="Wandb project name")
    parser.add_argument(
        "--run_name", help="Wandb run name (falls back to RUN_NAME env var "
        "or model_name)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Remove file handlers; existing stream handlers still receive log messages.
    for lgr in (logging.getLogger("histoforge_models"), logging.getLogger()):
        for h in lgr.handlers[:]:
            if isinstance(h, logging.FileHandler):
                lgr.removeHandler(h)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Allow TF32 for eligible CUDA matrix multiplies and cuDNN operations.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Dense semantic mode overrides
    if args.dense_semantic:
        args.freeze_seg_head = True
        args.min_train_masks = 0
        if args.nclasses == 0:
            raise ValueError("--dense_semantic requires --nclasses > 0")

    # Validate classification args
    if args.freeze_backbone and args.freeze_encoder_only:
        raise ValueError(
            "--freeze_backbone and --freeze_encoder_only are mutually "
            "exclusive: the former freezes the adapter too (decoder-only), "
            "the latter keeps the adapter trainable (CellVTA-style).")
    if args.freeze_seg_head and args.nclasses == 0:
        raise ValueError("--freeze_seg_head requires --nclasses > 0")
    if args.nclasses > 0 and not args.dense_semantic:
        assert args.class_labels_dir, (
            "--class_labels_dir required when --nclasses > 0 "
            "(unless --dense_semantic)")
    if args.logit_adjustment_tau > 0 and args.nclasses == 0:
        raise ValueError("--logit_adjustment_tau > 0 requires --nclasses > 0")
    if args.logit_adjustment_tau > 0 and args.class_weights_mode == "inverse_freq":
        raise ValueError(
            "--logit_adjustment_tau > 0 is incompatible with "
            "--class_weights_mode inverse_freq. Both inflate the rare-class "
            "gradient; stacking them double-counts the imbalance correction. "
            "Per Menon ICLR 2021, logit-adj is a PRINCIPLED REPLACEMENT for "
            "inverse-frequency class weights; use --class_weights_mode none.")
    if args.calibration_loss and args.nclasses == 0:
        raise ValueError("--calibration_loss requires --nclasses > 0")
    # Set up cellpose logging early so output goes to SLURM logs
    cp_logger, log_file = io.logger_setup()
    logger.info("Cellpose log file: %s", log_file)

    if args.head_type == "hovernet":
        if args.dense_semantic:
            raise ValueError(
                "--head_type hovernet is incompatible with --dense_semantic "
                "(HoVer targets need instance masks, not dense class masks)")
    if args.flow_loss_weighting == "equal_instance":
        if args.head_type != "flow":
            raise ValueError(
                "--flow_loss_weighting equal_instance requires --head_type flow")
        if args.dense_semantic:
            raise ValueError(
                "--flow_loss_weighting equal_instance requires true instance masks")
        if args.freeze_seg_head:
            raise ValueError(
                "--flow_loss_weighting equal_instance requires a trainable flow head")

    if args.lazy_train:
        if args.nclasses > 0:
            raise ValueError("--lazy_train supports only nclasses=0 (seg-only)")
        if args.head_type == "hovernet":
            raise ValueError(
                "--lazy_train streams precomputed Cellpose flow targets and "
                "does not support --head_type hovernet. Use eager HoVer "
                "training or add a dedicated precomputed-HV dataset path.")
        if not (args.val_images and args.val_labels):
            raise ValueError(
                "--lazy_train requires explicit --val_images / --val_labels")

    # Load training data
    logger.info("Loading training data...")
    train_img_files, train_lbl_files = get_paired_files(args.train_images,
                                                        args.train_labels)
    train_flow_files = None
    if args.lazy_train:
        # Lazy: keep file paths only; flows are streamed from disk per item.
        train_flow_files = _resolve_flow_files(train_img_files, args.train_images,
                                               train_lbl_files)
        train_data, train_labels = None, None
        logger.info("Lazy train: %d image/flow pairs (corpus not loaded)",
                    len(train_img_files))
    else:
        train_data = load_arrays(train_img_files)
        train_labels = load_arrays(train_lbl_files)
        logger.info("Loaded %d training image-label pairs", len(train_data))

    # Prepare semantic labels if classification mode
    class_weights = None
    if args.nclasses > 0 or args.head_type == "hovernet":
        logger.info(
            "Preparing semantic labels (nclasses=%d, dense=%s, "
            "head_type=%s)...", args.nclasses, args.dense_semantic, args.head_type)
        if args.dense_semantic:
            train_labels = prepare_dense_semantic_labels(train_labels)
        elif args.head_type == "hovernet":
            # nclasses=0 -> seg-only HV (no class json); nclasses>0 -> joint w/ cls
            train_labels = prepare_hovernet_labels(
                train_labels, train_lbl_files,
                args.class_labels_dir if args.nclasses > 0 else None)
        else:
            train_labels = prepare_semantic_labels(train_labels, train_lbl_files,
                                                   args.class_labels_dir)

    # Load or auto-split validation data
    val_data, val_labels, val_names = None, None, None
    val_img_files, val_lbl_files, val_flow_files = None, None, None
    if args.val_images or args.val_labels:
        # Both must be provided together
        if not (args.val_images and args.val_labels):
            raise ValueError(
                "Both --val_images and --val_labels must be provided together")
        logger.info("Loading validation data...")
        val_img_files, val_lbl_files = get_paired_files(args.val_images,
                                                        args.val_labels)
        if len(val_img_files) == 0:
            raise ValueError(f"Val directories are empty: {args.val_images}, "
                             f"{args.val_labels}")
        if args.lazy_train:
            val_flow_files = _resolve_flow_files(val_img_files, args.val_images,
                                                 val_lbl_files)
            logger.info("Lazy val: %d image/flow pairs (not loaded)",
                        len(val_img_files))
        else:
            val_data = load_arrays(val_img_files)
            val_labels = load_arrays(val_lbl_files)
            if args.nclasses > 0 or args.head_type == "hovernet":
                if args.dense_semantic:
                    val_labels = prepare_dense_semantic_labels(val_labels)
                elif args.head_type == "hovernet":
                    # nclasses=0 -> seg-only HV; nclasses>0 needs the val class json
                    if args.nclasses > 0 and not args.val_class_labels_dir:
                        raise ValueError("--val_class_labels_dir required for hovernet "
                                         "with --nclasses > 0")
                    val_labels = prepare_hovernet_labels(
                        val_labels, val_lbl_files,
                        args.val_class_labels_dir if args.nclasses > 0 else None)
                else:
                    if not args.val_class_labels_dir:
                        raise ValueError("--val_class_labels_dir required when "
                                         "--nclasses > 0 with val data")
                    val_labels = prepare_semantic_labels(val_labels, val_lbl_files,
                                                         args.val_class_labels_dir)
            logger.info("Loaded %d validation image-label pairs", len(val_data))
        val_names = [os.path.basename(f) for f in val_img_files]
    else:
        # Auto-split training data 85/15
        n_total = len(train_data)
        n_val = max(1, int(n_total * 0.15))
        rng = np.random.RandomState(42)
        perm = rng.permutation(n_total)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        val_data = [train_data[i] for i in val_idx]
        val_labels = [train_labels[i] for i in val_idx]
        train_data = [train_data[i] for i in train_idx]
        train_labels = [train_labels[i] for i in train_idx]
        val_img_files = [train_img_files[i] for i in val_idx]
        val_lbl_files = [train_lbl_files[i] for i in val_idx]
        train_img_files = [train_img_files[i] for i in train_idx]
        train_lbl_files = [train_lbl_files[i] for i in train_idx]
        logger.info("No val set provided, auto-split: %d train, %d val",
                    len(train_data), len(val_data))
        val_names = [os.path.basename(path) for path in val_img_files]

    # Compute class weights
    if args.nclasses > 0 and args.class_weights_mode == "inverse_freq":
        class_weights = compute_class_weights(train_labels, args.nclasses)
        logger.info("Class weights: %s", class_weights)

    # Compute log priors for logit adjustment (Menon ICLR 2021).
    # Pattern mirrors compute_class_weights: channel 1 of the semantic label
    # is the per-pixel class map; ignore index 255 is excluded.
    log_priors = None
    if args.nclasses > 0 and args.logit_adjustment_tau > 0:
        all_classes = np.concatenate([lbl[1].ravel() for lbl in train_labels])
        valid = (all_classes >= 0) & (all_classes < args.nclasses)
        counts = np.maximum(
            np.bincount(all_classes[valid].astype(int),
                        minlength=args.nclasses).astype("float64"), 1.0)
        log_priors_np = np.log(counts / counts.sum()).astype("float32")
        logger.info("Logit adjustment: tau=%.2f, log_priors=%s",
                    args.logit_adjustment_tau,
                    log_priors_np.round(4).tolist())
        log_priors = torch.from_numpy(log_priors_np)  # moved to device inside loop

    # Compute oversampling probabilities
    train_probs = None
    if args.oversample and args.nclasses > 0:
        from histoforge_models.pathology.training.data_utils import compute_sampling_weights
        train_probs = compute_sampling_weights(train_labels, args.nclasses)
        logger.info("Oversampling enabled")

    # Area-weighted sampling: P(image) proportional to its pixel area, so a
    # dataset's epoch share equals its share of total tissue, not image count.
    if args.train_probs_mode == "area" and train_probs is None:
        if args.lazy_train:
            # Read image shapes one at a time when the corpus is not in RAM.
            areas = np.zeros(len(train_img_files), dtype=np.float64)
            for i, f in enumerate(train_img_files):
                s = io.imread(f).shape
                areas[i] = float(s[0]) * float(s[1])
        else:
            areas = np.array(
                [float(im.shape[0]) * float(im.shape[1]) for im in train_data],
                dtype=np.float64)
        train_probs = areas / areas.sum()
        logger.info("train_probs: area-weighted over %d images "
                    "(min=%.2e, max=%.2e)", len(train_probs), float(train_probs.min()),
                    float(train_probs.max()))

    run_name = args.run_name or os.environ.get("RUN_NAME", args.model_name)
    samples_per_epoch = max(2, len(train_img_files))
    run_config = {
        "bsize":
            args.bsize,
        "n_epochs":
            args.n_epochs,
        "learning_rate":
            args.learning_rate,
        "lr_scheduler":
            args.lr_scheduler,
        "weight_decay":
            args.weight_decay,
        "batch_size":
            args.batch_size,
        "n_train":
            len(train_img_files),
        "n_val":
            len(val_img_files) if val_img_files else (len(val_data) if val_data else 0),
        "save_path":
            args.save_path,
        "model_name":
            args.model_name,
        "nclasses":
            args.nclasses,
        "freeze_backbone":
            args.freeze_backbone,
        "freeze_encoder_only":
            args.freeze_encoder_only,
        "checkpoint":
            args.checkpoint,
        "seed":
            args.seed,
        "class_weights_mode":
            args.class_weights_mode,
        "gradient_checkpointing":
            args.gradient_checkpointing,
        "freeze_seg_head":
            args.freeze_seg_head,
        "dense_semantic":
            args.dense_semantic,
        "cls_loss_fn":
            args.cls_loss_fn,
        "tversky_alpha":
            args.tversky_alpha,
        "tversky_gamma":
            args.tversky_gamma,
        "cls_loss_weight":
            args.cls_loss_weight,
        "use_uncertainty_weighting":
            args.use_uncertainty_weighting,
        "oversample":
            args.oversample,
        "num_workers":
            args.num_workers,
        "augmentation_strategy":
            args.augmentation_strategy,
        "backbone":
            "dinov3_adapter",
        "adapter_dim":
            256,
        "adapter_schema":
            "ln-spm-cti-w2",
        "llrd_factor":
            args.llrd_factor,
        "patch_embed_lr":
            args.patch_embed_lr,
        "adapter_lr":
            args.adapter_lr,
        "train_probs_mode":
            args.train_probs_mode,
        "ap_every":
            args.ap_every,
        "segmentation_metric_aggregation":
            args.segmentation_metric_aggregation,
        "logit_adjustment_tau":
            args.logit_adjustment_tau,
        "calibration_loss":
            args.calibration_loss,
        "calibration_n_bins":
            args.calibration_n_bins,
        "calibration_variant":
            args.calibration_variant,
        "head_type":
            args.head_type,
        "flow_loss_weighting":
            args.flow_loss_weighting,
        "grad_clip":
            args.grad_clip,
        "class_weights_target":
            args.class_weights_target,
        "lazy_train":
            args.lazy_train,
        "min_train_masks":
            args.min_train_masks,
        "val_patches_per_image":
            args.val_patches_per_image,
        "tile_overlap":
            args.tile_overlap,
        "hv_edge_threshold":
            args.hv_edge_threshold,
        "hv_object_size":
            args.hv_object_size,
        "hv_fg_threshold":
            args.hv_fg_threshold,
        "hv_morph_kernel":
            args.hv_morph_kernel,
        "hover_decode_profile_version":
            HOVER_DECODE_PROFILE_VERSION,
        "steps_per_epoch":
            int(np.ceil(samples_per_epoch / args.batch_size)),
        "samples_per_epoch":
            samples_per_epoch,
    }

    def matching_class_files(label_files, directory):
        if not directory:
            return []
        return [Path(directory) / f"{Path(path).stem}.json" for path in label_files]

    class_train_files = matching_class_files(train_lbl_files, args.class_labels_dir)
    class_val_files = matching_class_files(
        val_lbl_files or [], args.val_class_labels_dir or args.class_labels_dir)
    _write_recipe(
        args.save_path, args, run_config, {
            "train_images": train_img_files,
            "train_labels": train_lbl_files,
            "train_flows": train_flow_files or [],
            "train_class_labels": class_train_files,
            "val_images": val_img_files or [],
            "val_labels": val_lbl_files or [],
            "val_flows": val_flow_files or [],
            "val_class_labels": class_val_files,
        })

    # Init wandb only after the immutable local recipe has been written.
    import wandb
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=run_config,
    )

    # Create model
    logger.info("Creating fixed adapter (bsize=%d, nclasses=%d)",
                args.bsize, args.nclasses)
    model = create_cellpose_model(
        gpu=True, pretrained_model=args.checkpoint, nclasses=args.nclasses,
        head_type=args.head_type, warm_start=True)

    logger.info("Model on device: %s", model.device)

    # Train (real-time wandb logging via forked train_seg)
    model_path, train_losses, test_losses = train_seg_wandb(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        train_probs=train_probs,
        test_data=val_data,
        test_labels=val_labels,
        test_files=val_img_files if args.lazy_train else None,
        test_labels_files=val_flow_files,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        nimg_per_epoch=samples_per_epoch,
        nimg_test_per_epoch=(len(val_data) *
                             args.val_patches_per_image if val_data else None),
        save_path=args.save_path,
        model_name=args.model_name,
        min_train_masks=args.min_train_masks,
        bsize=args.bsize,
        wandb_run=run,
        class_weights=class_weights,
        freeze_backbone=args.freeze_backbone,
        freeze_encoder_only=args.freeze_encoder_only,
        nclasses=args.nclasses,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_seg_head=args.freeze_seg_head,
        cls_loss_fn=args.cls_loss_fn,
        tversky_alpha=args.tversky_alpha,
        tversky_gamma=args.tversky_gamma,
        cls_loss_weight=args.cls_loss_weight,
        lr_scheduler=args.lr_scheduler,
        use_uncertainty_weighting=args.use_uncertainty_weighting,
        num_workers=args.num_workers,
        augmentation_strategy=args.augmentation_strategy,
        llrd_factor=args.llrd_factor,
        patch_embed_lr=args.patch_embed_lr,
        adapter_lr=args.adapter_lr,
        cellpose_model=model,
        test_names=val_names,
        ap_every=args.ap_every,
        segmentation_metric_aggregation=args.segmentation_metric_aggregation,
        train_files=train_img_files if args.lazy_train else None,
        train_labels_files=train_flow_files,
        load_files=not args.lazy_train,
        lazy_train=args.lazy_train,
        log_priors=log_priors,
        logit_adjustment_tau=args.logit_adjustment_tau,
        calibration_loss=args.calibration_loss,
        calibration_n_bins=args.calibration_n_bins,
        calibration_variant=args.calibration_variant,
        head_type=args.head_type,
        grad_clip=args.grad_clip,
        hv_edge_threshold=args.hv_edge_threshold,
        hv_object_size=args.hv_object_size,
        hv_fg_threshold=args.hv_fg_threshold,
        hv_morph_kernel=args.hv_morph_kernel,
        class_weights_target=args.class_weights_target,
        seed=args.seed,
        tile_overlap=args.tile_overlap,
        flow_loss_weighting=args.flow_loss_weighting,
    )
    logger.info("Training complete. Model saved to: %s", model_path)

    # Save loss curves
    save_dir = Path(args.save_path)
    np.save(save_dir / "train_losses.npy", train_losses)
    np.save(save_dir / "test_losses.npy", test_losses)
    logger.info("Loss curves saved to %s", save_dir)

    # Post-training evaluation on held-out test set (optional)
    if args.test_images and args.test_labels:
        from histoforge_models.pathology.training.eval import evaluate_model
        logger.info("Running post-training evaluation on test set...")
        test_results = evaluate_model(
            model_path=str(model_path),
            eval_images_dir=args.test_images,
            eval_labels_dir=args.test_labels,
            bsize=args.bsize,
            save_dir=args.save_path,
            dataset_name="test",
            batch_size=args.batch_size,
            tile_overlap=args.tile_overlap,
            hv_edge_threshold=args.hv_edge_threshold,
            hv_object_size=args.hv_object_size,
            hv_fg_threshold=args.hv_fg_threshold,
            hv_morph_kernel=args.hv_morph_kernel,
            segmentation_metric_aggregation=args.segmentation_metric_aggregation,
        )
        run.log({
            f"test/{k}": v for k, v in test_results.items()
            if isinstance(v, float) and np.isfinite(v)
        })

    run.finish()


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
