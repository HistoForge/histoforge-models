import time
import os
import numpy as np
from histoforge_models import io, utils, models, dynamics
from histoforge_models.transforms import normalize_img, random_rotate_and_resize
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange

import logging

train_logger = logging.getLogger(__name__)


def _loss_fn_class(lbl, y, class_weights=None, log_priors=None, tau=0.0,
                   nclasses_slice=-3):
    """Cross-entropy classification loss with optional logit adjustment.

    Args:
        lbl: (B, C_lbl, H, W); channel 0 is the per-pixel class map.
        y:   (B, C_y, H, W); class logits in y[:, :nclasses_slice].
        class_weights: optional per-class weights for CE.
        log_priors: 1-D tensor of log P(y=k); shifts logits when tau>0
            (Menon ICLR 2021, "softmax(z_y + tau * log pi_y)").
        tau: logit-adjustment temperature. 0 disables.
        nclasses_slice: -3 for Cellpose flow or -4 for HoVer NP/HV. Selects
            the class-logit channels off the end of y.
    """
    logits = y[:, :nclasses_slice]
    if log_priors is not None and tau > 0:
        logits = logits + tau * log_priors.view(1, -1, 1, 1)
    criterion3 = nn.CrossEntropyLoss(reduction="mean", weight=class_weights,
                                     ignore_index=255)
    return criterion3(logits, lbl[:, 0].long())


def _loss_fn_tversky(lbl, y, n_classes=None, class_weights=None, alpha=0.3, gamma=1.33,
                     eps=1e-6, log_priors=None, tau=0.0, nclasses_slice=-3):
    """Focal Tversky loss for semantic classification.

    Based on Abraham & Khan (ISBI 2019, arxiv 1810.07842):
    TI_c = TP_c / (TP_c + alpha * FP_c + beta * FN_c)
    Loss = clip(1 - TI_c, eps, 1-eps) ^ (1/gamma)

    Class weights are applied per-class after the focal power, before
    averaging. Logit adjustment (log_priors, tau) shifts logits before
    softmax, mirroring _loss_fn_class.
    """
    beta = 1 - alpha
    logits = y[:, :nclasses_slice]  # (B, C, H, W)
    if log_priors is not None and tau > 0:
        logits = logits + tau * log_priors.view(1, -1, 1, 1)
    target = lbl[:, 0].long()  # (B, H, W)
    nclasses = n_classes if n_classes is not None else logits.shape[1]
    probs = torch.softmax(logits, dim=1)

    valid = (target != 255)  # (B, H, W)
    safe_target = target.clone()
    safe_target[~valid] = 0  # safe for one_hot
    target_oh = F.one_hot(safe_target, nclasses).permute(0, 3, 1, 2).float()
    target_oh *= valid.unsqueeze(1).float()  # zero ignore in target
    masked_probs = probs * valid.unsqueeze(1).float()  # zero ignore in probs

    tp = (masked_probs * target_oh).sum(dim=(2, 3))
    fp = (masked_probs * (1 - target_oh)).sum(dim=(2, 3))
    fn = ((1 - masked_probs) * target_oh).sum(dim=(2, 3))

    loss = 1.0 - tp / (tp + alpha * fp + beta * fn)
    loss = torch.clip(loss, eps, 1 - eps)
    loss = loss.pow(1.0 / gamma)
    if class_weights is not None:
        loss = loss * class_weights

    return loss.mean()


def mL1_ACE_loss(logits, target, n_bins=15, variant="hard", ignore_index=255):
    """Differentiable Average Calibration Error (Barfoot et al. 2026,
    arXiv 2506.03942) for segmentation/classification.

    For each pixel, take the predicted class and its softmax confidence.
    Bin pixels by confidence; compute |mean_confidence - mean_accuracy|
    per bin; return the mean over n_bins.

    Args:
        logits: (B, C, H, W) class logits.
        target: (B, H, W) long target class map. Pixels equal to
            ignore_index are excluded.
        n_bins: Number of confidence bins.
        variant: "hard" (bucketize + per-bin reduction) or "soft"
            (Gaussian kernel at each bin center — differentiable
            everywhere, slightly more expensive).
        ignore_index: Target value to mask out.

    Returns:
        Scalar tensor (dtype matching logits).
    """
    probs = F.softmax(logits, dim=1)
    confidence, prediction = probs.max(dim=1)  # (B, H, W)
    valid = (target != ignore_index)
    confidence = confidence[valid]
    prediction = prediction[valid]
    target_v = target[valid]
    if confidence.numel() == 0:
        return logits.new_zeros(())
    correct = (prediction == target_v).to(logits.dtype)

    if variant == "hard":
        bin_idx = torch.clamp((confidence * n_bins).long(), 0, n_bins - 1)
        ace = logits.new_zeros(())
        for b in range(n_bins):
            sel = (bin_idx == b)
            if sel.any():
                ace = ace + (confidence[sel].mean() - correct[sel].mean()).abs()
        return ace / n_bins

    # soft variant: weight each pixel by a Gaussian centered at the bin
    # midpoint — fully differentiable, no bucketize/STE needed.
    centers = torch.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins,
                             device=logits.device, dtype=logits.dtype)
    sigma = 1.0 / n_bins
    w = torch.exp(-((confidence[None, :] - centers[:, None])**2) /
                  (2 * sigma**2))  # (n_bins, N)
    w_sum = w.sum(dim=1) + 1e-8
    conf_bin = (w * confidence[None, :]).sum(dim=1) / w_sum
    acc_bin = (w * correct[None, :]).sum(dim=1) / w_sum
    return (conf_bin - acc_bin).abs().mean()


def _normalized_instance_flow_weights(instance_weight_map, foreground):
    """Normalize inverse-area weights without changing total flow-loss mass.

    ``instance_weight_map`` is created from complete instances before spatial
    augmentation or cropping. Each source instance has unit mass. A visible
    crop-edge fragment therefore retains only its visible fraction of that
    mass. Within each sample, visible foreground weights are rescaled to sum to
    the number of foreground pixels; background receives weight one. The total
    weight consequently remains equal to the number of pixels, matching the
    scale of the historical pixel-mean flow loss.
    """
    if instance_weight_map.ndim != 3 or foreground.ndim != 3:
        raise ValueError("instance flow weights and foreground must be (B,H,W)")
    if instance_weight_map.shape != foreground.shape:
        raise ValueError(
            "instance flow weights and foreground must have identical shapes")
    if not instance_weight_map.dtype.is_floating_point:
        raise ValueError("instance flow weights must use a floating dtype")
    if not torch.isfinite(instance_weight_map).all():
        raise ValueError("instance flow weights must be finite")
    if torch.any(instance_weight_map < 0):
        raise ValueError("instance flow weights must be non-negative")

    normalized = torch.ones_like(instance_weight_map)
    foreground = foreground.bool()
    for batch_index in range(instance_weight_map.shape[0]):
        sample_foreground = foreground[batch_index]
        if not torch.any(sample_foreground):
            continue
        visible_weights = instance_weight_map[batch_index][sample_foreground]
        if torch.any(visible_weights <= 0):
            raise ValueError(
                "every foreground pixel must have a positive instance-flow weight")
        source_mass = visible_weights.sum()
        target_mass = sample_foreground.sum().to(instance_weight_map.dtype)
        normalized[batch_index][sample_foreground] = (
            visible_weights * target_mass / source_mass)
    return normalized


def _loss_fn_seg(lbl, y, device, flow_pixel_weights=None):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        device (torch.device): Device on which the tensors are located.
        flow_pixel_weights (torch.Tensor or None): Optional pre-augmentation
            inverse-instance-area map with shape ``[B,H,W]``. When present,
            it weights only the two flow-vector MSE channels.

    Returns:
        torch.Tensor: Loss value.

    """
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    veci = 5. * lbl[:, -2:]
    if flow_pixel_weights is None:
        # Preserve the historical arithmetic exactly in the default path.
        criterion = nn.MSELoss(reduction="mean")
        loss = criterion(y[:, -3:-1], veci)
        loss /= 2.
    else:
        normalized = _normalized_instance_flow_weights(
            flow_pixel_weights.to(device=y.device, dtype=y.dtype),
            lbl[:, -3] > 0.5)
        squared_error = (y[:, -3:-1] - veci).square()
        loss = (squared_error * normalized[:, None]).mean() / 2.
    loss2 = criterion2(y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype))
    loss = loss + loss2
    return loss


def _inverse_instance_area_map(instance_mask):
    """Return ``1 / full_instance_area`` on instances and zero elsewhere."""
    mask = np.asarray(instance_mask)
    if mask.ndim != 2:
        raise ValueError(f"instance mask must be 2-D, got shape={mask.shape}")
    if not np.issubdtype(mask.dtype, np.number) or not np.isfinite(mask).all():
        raise ValueError("instance mask must contain finite numeric IDs")
    if np.any(mask < 0) or not np.equal(mask, np.floor(mask)).all():
        raise ValueError("instance mask IDs must be non-negative integers")

    instance_ids, inverse, counts = np.unique(
        mask, return_inverse=True, return_counts=True)
    per_instance = np.zeros(instance_ids.shape, dtype=np.float32)
    foreground = instance_ids != 0
    per_instance[foreground] = 1.0 / counts[foreground].astype(np.float32)
    return per_instance[inverse].reshape(mask.shape)


def _prepare_flow_training_label(raw_label, flow_loss_weighting="pixel"):
    """Strip instance IDs and optionally retain a warpable weight map.

    The final two channels remain the flow-vector pair because
    :func:`random_rotate_and_resize` rotates and linearly interpolates those
    channels. The inserted inverse-area map sits immediately before the pair,
    so it receives the same nearest-neighbour transform as categorical targets.
    """
    raw_label = np.asarray(raw_label)
    if flow_loss_weighting == "pixel":
        return raw_label[1:]
    if flow_loss_weighting != "equal_instance":
        raise ValueError(
            "flow_loss_weighting must be 'pixel' or 'equal_instance', got "
            f"{flow_loss_weighting!r}")
    if raw_label.ndim != 3 or raw_label.shape[0] < 4:
        raise ValueError(
            "equal-instance flow weighting requires [instance, ..., "
            f"cellprob, flowY, flowX], got shape={raw_label.shape}")

    instance_mask = raw_label[0]
    foreground = raw_label[-3] > 0.5
    if np.any(foreground != (instance_mask > 0)):
        raise ValueError(
            "equal-instance flow weighting requires instance IDs and "
            "foreground targets to agree")
    task_label = raw_label[1:]
    instance_weights = _inverse_instance_area_map(instance_mask)
    return np.concatenate(
        [task_label[:-2], instance_weights[None], task_label[-2:]], axis=0)


def _split_flow_weight_channel(label, flow_loss_weighting="pixel"):
    """Remove the temporary weight map after spatial augmentation."""
    if flow_loss_weighting == "pixel":
        return label, None
    if flow_loss_weighting != "equal_instance":
        raise ValueError(
            "flow_loss_weighting must be 'pixel' or 'equal_instance', got "
            f"{flow_loss_weighting!r}")
    if label.ndim != 4 or label.shape[1] < 4:
        raise ValueError(
            "equal-instance augmented labels must have shape [B,C>=4,H,W]")
    flow_pixel_weights = label[:, -3]
    task_label = torch.cat([label[:, :-3], label[:, -2:]], dim=1)
    return task_label, flow_pixel_weights


def _reshape_norm(data, channel_axis=None, normalize_params={"normalize": False}):
    """
    Reshapes and normalizes the input data.

    Args:
        data (list): List of input data, with channels axis first or last.
        normalize_params (dict, optional): Dictionary of normalization parameters. Defaults to {"normalize": False}.

    Returns:
        list: List of reshaped and normalized data.
    """
    if (np.array([td.ndim != 3 for td in data]).sum() > 0 or
            np.array([td.shape[0] != 3 for td in data]).sum() > 0):
        data_new = []
        for td in data:
            if td.ndim == 3:
                channel_axis0 = channel_axis if channel_axis is not None else np.array(
                    td.shape).argmin()
                # put channel axis first
                td = np.moveaxis(td, channel_axis0, 0)
                td = td[:3]  # keep at most 3 channels
            if td.ndim == 2 or (td.ndim == 3 and td.shape[0] == 1):
                td = np.stack((td, 0 * td, 0 * td), axis=0)
            elif td.ndim == 3 and td.shape[0] < 3:
                td = np.concatenate((td, 0 * td[:1]), axis=0)
            data_new.append(td)
        data = data_new
    if normalize_params["normalize"]:
        data = [normalize_img(td, normalize=normalize_params, axis=0) for td in data]
    return data


def _get_batch(inds, data=None, labels=None, files=None, labels_files=None,
               normalize_params={"normalize": False},
               flow_loss_weighting="pixel"):
    """Get a batch of images and labels.

    Args:
        inds (list): List of indices indicating which images and labels to retrieve.
        data (list or None): List of image data. If None, images will be loaded from files.
        labels (list or None): List of label data. If None, labels will be loaded from files.
        files (list or None): List of file paths for images.
        labels_files (list or None): List of file paths for labels.
        normalize_params (dict): Dictionary of parameters for image normalization (will be faster, if loading from files to pre-normalize).
    Returns:
        tuple: A tuple containing two lists: the batch of images and the batch of labels.
    """
    if data is None:
        lbls = None
        imgs = [io.imread(files[i]) for i in inds]
        imgs = _reshape_norm(imgs, normalize_params=normalize_params)
        if labels_files is not None:
            lbls = [
                _prepare_flow_training_label(
                    io.imread(labels_files[i]), flow_loss_weighting)
                for i in inds
            ]
    else:
        imgs = [data[i] for i in inds]
        lbls = [
            _prepare_flow_training_label(labels[i], flow_loss_weighting)
            for i in inds
        ]
    return imgs, lbls


def _process_train_test(train_data=None, train_labels=None, train_files=None,
                        train_labels_files=None, train_probs=None, test_data=None,
                        test_labels=None, test_files=None, test_labels_files=None,
                        test_probs=None, load_files=True, min_train_masks=5,
                        compute_flows=False, normalize_params={"normalize": False},
                        channel_axis=None, device=None):
    """
    Process train and test data.

    Args:
        train_data (list or None): List of training data arrays.
        train_labels (list or None): List of training label arrays.
        train_files (list or None): List of training file paths.
        train_labels_files (list or None): List of training label file paths.
        train_probs (ndarray or None): Array of training probabilities.
        test_data (list or None): List of test data arrays.
        test_labels (list or None): List of test label arrays.
        test_files (list or None): List of test file paths.
        test_labels_files (list or None): List of test label file paths.
        test_probs (ndarray or None): Array of test probabilities.
        load_files (bool): Whether to load data from files.
        min_train_masks (int): Minimum number of masks required for training images.
        compute_flows (bool): Whether to compute flows.
        channels (list or None): List of channel indices to use.
        channel_axis (int or None): Axis of channel dimension.
        rgb (bool): Convert training/testing images to RGB.
        normalize_params (dict): Dictionary of normalization parameters.
        device (torch.device): Device to use for computation.

    Returns:
        tuple: A tuple containing the processed train and test data and sampling probabilities and diameters.
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
            'mps') if torch.backends.mps.is_available() else None

    if train_data is not None and train_labels is not None:
        # if data is loaded
        nimg = len(train_data)
        nimg_test = len(test_data) if test_data is not None else None
    else:
        # otherwise use files
        nimg = len(train_files)
        if train_labels_files is None:
            train_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in train_files
            ]
            train_labels_files = [tf for tf in train_labels_files if os.path.exists(tf)]
        if (test_data is not None or
                test_files is not None) and test_labels_files is None:
            test_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in test_files
            ]
            test_labels_files = [tf for tf in test_labels_files if os.path.exists(tf)]
        if not load_files:
            train_logger.info(">>> using files instead of loading dataset")
        else:
            # load all images
            train_logger.info(">>> loading images and labels")
            train_data = [io.imread(train_files[i]) for i in trange(nimg)]
            train_labels = [io.imread(train_labels_files[i]) for i in trange(nimg)]
        nimg_test = len(test_files) if test_files is not None else None
        if load_files and nimg_test:
            test_data = [io.imread(test_files[i]) for i in trange(nimg_test)]
            test_labels = [io.imread(test_labels_files[i]) for i in trange(nimg_test)]

    ### check that arrays are correct size
    if ((train_labels is not None and nimg != len(train_labels)) or
        (train_labels_files is not None and nimg != len(train_labels_files))):
        error_message = "train data and labels not same length"
        train_logger.critical(error_message)
        raise ValueError(error_message)
    if ((test_labels is not None and nimg_test != len(test_labels)) or
        (test_labels_files is not None and nimg_test != len(test_labels_files))):
        train_logger.warning("test data and labels not same length, not using")
        test_data, test_files = None, None
    if train_labels is not None:
        if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
            error_message = "training data or labels are not at least two-dimensional"
            train_logger.critical(error_message)
            raise ValueError(error_message)
        if train_data[0].ndim > 3:
            error_message = "training data is more than three-dimensional (should be 2D or 3D array)"
            train_logger.critical(error_message)
            raise ValueError(error_message)

    ### check that flows are computed
    if train_labels is not None:
        train_labels = dynamics.labels_to_flows(train_labels, files=train_files,
                                                device=device)
        if test_labels is not None:
            test_labels = dynamics.labels_to_flows(test_labels, files=test_files,
                                                   device=device)
    elif compute_flows:
        for k in trange(nimg):
            tl = dynamics.labels_to_flows(io.imread(train_labels_files),
                                          files=train_files, device=device)
        if test_files is not None:
            for k in trange(nimg_test):
                tl = dynamics.labels_to_flows(io.imread(test_labels_files),
                                              files=test_files, device=device)

    ### compute diameters
    nmasks = np.zeros(nimg)
    diam_train = np.zeros(nimg)
    train_logger.info(">>> computing diameters")
    for k in trange(nimg):
        tl = (train_labels[k][0]
              if train_labels is not None else io.imread(train_labels_files[k])[0])
        diam_train[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam_train[diam_train < 5] = 5.
    if test_data is not None:
        diam_test = np.array(
            [utils.diameters(test_labels[k][0])[0] for k in trange(len(test_labels))])
        diam_test[diam_test < 5] = 5.
    elif test_labels_files is not None:
        diam_test = np.array([
            utils.diameters(io.imread(test_labels_files[k])[0])[0]
            for k in trange(len(test_labels_files))
        ])
        diam_test[diam_test < 5] = 5.
    else:
        diam_test = None

    ### check to remove training images with too few masks
    if min_train_masks > 0:
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            train_logger.warning(
                f"{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set"
            )
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            if train_data is not None:
                train_data = [train_data[i] for i in ikeep]
                train_labels = [train_labels[i] for i in ikeep]
            if train_files is not None:
                train_files = [train_files[i] for i in ikeep]
            if train_labels_files is not None:
                train_labels_files = [train_labels_files[i] for i in ikeep]
            if train_probs is not None:
                train_probs = train_probs[ikeep]
            diam_train = diam_train[ikeep]
            nimg = len(ikeep)

    ### normalize probabilities
    train_probs = 1. / nimg * np.ones(nimg,
                                      "float64") if train_probs is None else train_probs
    train_probs /= train_probs.sum()
    if test_files is not None or test_data is not None:
        test_probs = 1. / nimg_test * np.ones(
            nimg_test, "float64") if test_probs is None else test_probs
        test_probs /= test_probs.sum()

    ### reshape and normalize train / test data
    normed = False
    if normalize_params["normalize"]:
        train_logger.info(f">>> normalizing {normalize_params}")
    if train_data is not None:
        train_data = _reshape_norm(train_data, channel_axis=channel_axis,
                                   normalize_params=normalize_params)
        normed = True
    if test_data is not None:
        test_data = _reshape_norm(test_data, channel_axis=channel_axis,
                                  normalize_params=normalize_params)

    return (train_data, train_labels, train_files, train_labels_files, train_probs,
            diam_train, test_data, test_labels, test_files, test_labels_files,
            test_probs, diam_test, normed)


def train_seg(net, train_data=None, train_labels=None, train_files=None,
              train_labels_files=None, train_probs=None, test_data=None,
              test_labels=None, test_files=None, test_labels_files=None,
              test_probs=None, channel_axis=None, load_files=True, batch_size=1,
              learning_rate=1e-5, SGD=False, n_epochs=100, weight_decay=0.1,
              normalize=True, compute_flows=False, save_path=None, save_every=100,
              save_each=False, nimg_per_epoch=None, nimg_test_per_epoch=None,
              rescale=False, scale_range=None, bsize=256, min_train_masks=5,
              model_name=None, class_weights=None, flow_loss_weighting="pixel"):
    """
    Train the network with images for segmentation.

    Args:
        net (object): The network model to train. If `net` is a bfloat16 model it will be converted to float32 for training. The saved models will be in float32, but the original model will be returned as the original dtype for consistency. 
        train_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for training. Defaults to None.
        train_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for train_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        train_files (List[str], optional): List of strings - file names for images in train_data (to save flows for future runs). Defaults to None.
        train_labels_files (list or None): List of training label file paths. Defaults to None.
        train_probs (List[float], optional): List of floats - probabilities for each image to be selected during training. Defaults to None.
        test_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for testing. Defaults to None.
        test_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for test_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        test_files (List[str], optional): List of strings - file names for images in test_data (to save flows for future runs). Defaults to None.
        test_labels_files (list or None): List of test label file paths. Defaults to None.
        test_probs (List[float], optional): List of floats - probabilities for each image to be selected during testing. Defaults to None.
        load_files (bool, optional): Boolean - whether to load images and labels from files. Defaults to True.
        batch_size (int, optional): Integer - number of patches to run simultaneously on the GPU. Defaults to 1.
        learning_rate (float or List[float], optional): Float or list/np.ndarray - learning rate for training. Defaults to 1e-5.
        n_epochs (int, optional): Integer - number of times to go through the whole training set during training. Defaults to 100.
        weight_decay (float, optional): Float - weight decay for the optimizer. Defaults to 0.1.
        SGD (bool, optional): Deprecated in v4.0.1+ - AdamW always used.
        normalize (bool or dict, optional): Boolean or dictionary - whether to normalize the data. Defaults to True.
        compute_flows (bool, optional): Boolean - whether to compute flows during training. Defaults to False.
        save_path (str, optional): String - where to save the trained model. Defaults to None.
        save_every (int, optional): Integer - save the network every [save_every] epochs. Defaults to 100.
        save_each (bool, optional): Boolean - save the network to a new filename at every [save_each] epoch. Defaults to False.
        nimg_per_epoch (int, optional): Integer - minimum number of images to train on per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): Integer - minimum number of images to test on per epoch. Defaults to None.
        rescale (bool, optional): Boolean - whether or not to rescale images during training. Defaults to False.
        min_train_masks (int, optional): Integer - minimum number of masks an image must have to use in the training set. Defaults to 5.
        model_name (str, optional): String - name of the network. Defaults to None.
        flow_loss_weighting (str, optional): ``"pixel"`` preserves the
            historical flow-vector pixel mean. ``"equal_instance"`` gives
            each complete source instance equal total flow influence. The
            foreground BCE remains pixel-weighted. Defaults to ``"pixel"``.

    Returns:
        tuple: A tuple containing the path to the saved model weights, training losses, and test losses.
       
    """
    if SGD:
        train_logger.warning("SGD is deprecated, using AdamW instead")
    if flow_loss_weighting not in ("pixel", "equal_instance"):
        raise ValueError(
            "flow_loss_weighting must be 'pixel' or 'equal_instance', got "
            f"{flow_loss_weighting!r}")

    device = net.device

    original_net_dtype = net.dtype
    if net.dtype == torch.bfloat16:
        # NOTE: this produces a side effect of returning a network that is not of a guaranteed dtype \
        train_logger.info(">>> converting bfloat16 network to float32 for training")
        net.dtype = torch.float32

    scale_range = 0.5 if scale_range is None else scale_range

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    out = _process_train_test(
        train_data=train_data, train_labels=train_labels, train_files=train_files,
        train_labels_files=train_labels_files, train_probs=train_probs,
        test_data=test_data, test_labels=test_labels, test_files=test_files,
        test_labels_files=test_labels_files, test_probs=test_probs,
        load_files=load_files, min_train_masks=min_train_masks,
        compute_flows=compute_flows, channel_axis=channel_axis,
        normalize_params=normalize_params, device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {"normalize_params": normalize_params, "channel_axis": channel_axis}

    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    if class_weights is not None and isinstance(class_weights,
                                                (list, np.ndarray, tuple)):
        class_weights = torch.from_numpy(class_weights).to(device).float()
        print(class_weights)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    # learning rate schedule
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
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
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    filename = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)

    train_logger.info(f">>> saving model to {filename}")

    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            # choose random images for epoch with probability train_probs
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            # otherwise use all images
            rperm = np.random.permutation(np.arange(0, nimg))
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]  # set learning rate
        net.train()
        for k in range(0, nimg_per_epoch, batch_size):
            kend = min(k + batch_size, nimg_per_epoch)
            inds = rperm[k:kend]
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels,
                                    files=train_files, labels_files=train_labels_files,
                                    flow_loss_weighting=flow_loss_weighting,
                                    **kwargs)
            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / net.diam_mean.item() if rescale else np.ones(
                len(diams), "float32")
            # augmentations
            imgi, lbl = random_rotate_and_resize(imgs, Y=lbls, rescale=rsc,
                                                 scale_range=scale_range,
                                                 xy=(bsize, bsize))[:2]
            # network and loss optimization
            X = torch.from_numpy(imgi).to(device)
            lbl = torch.from_numpy(lbl).to(device)
            lbl, flow_pixel_weights = _split_flow_weight_channel(
                lbl, flow_loss_weighting)

            with torch.autocast(device_type=device.type, dtype=net.dtype):
                y = net(X)[0]
            loss = _loss_fn_seg(
                lbl, y, device, flow_pixel_weights=flow_pixel_weights)
            if y.shape[1] > 3:
                loss3 = _loss_fn_class(lbl, y, class_weights=class_weights)
                loss += loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            train_loss *= len(imgi)

            # keep track of average training loss across epochs
            lavg += train_loss
            nsum += len(imgi)
            # per epoch training loss
            train_losses[iepoch] += train_loss
        train_losses[iepoch] /= nimg_per_epoch

        if iepoch == 5 or iepoch % 10 == 0:
            lavgt = 0.
            if test_data is not None or test_files is not None:
                np.random.seed(42)
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
                        X = torch.from_numpy(imgi).to(device)
                        lbl = torch.from_numpy(lbl).to(device)
                        lbl, flow_pixel_weights = _split_flow_weight_channel(
                            lbl, flow_loss_weighting)

                        with torch.autocast(device_type=device.type, dtype=net.dtype):
                            y = net(X)[0]
                        loss = _loss_fn_seg(
                            lbl, y, device,
                            flow_pixel_weights=flow_pixel_weights)
                        if y.shape[1] > 3:
                            loss3 = _loss_fn_class(lbl, y, class_weights=class_weights)
                            loss += loss3
                        test_loss = loss.item()
                        test_loss *= len(imgi)
                        lavgt += test_loss
                lavgt /= len(rperm)
                test_losses[iepoch] = lavgt
            lavg /= nsum
            train_logger.info(
                f"{iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
            )
            lavg, nsum = 0, 0

        if iepoch == n_epochs - 1 or (iepoch % save_every == 0 and iepoch != 0):
            if save_each and iepoch != n_epochs - 1:  #separate files as model progresses
                filename0 = str(filename) + f"_epoch_{iepoch:04d}"
            else:
                filename0 = filename
            train_logger.info(f"saving network parameters to {filename0}")
            net.save_model(filename0)

    net.save_model(filename)
    if original_net_dtype != torch.float32:
        train_logger.info(
            f">>> converting network back to {original_net_dtype} after training")
        net.dtype = original_net_dtype

    return filename, train_losses, test_losses
