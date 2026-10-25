"""Utilities for data loading, file pairing, and training label preparation."""

import json
import logging
from pathlib import Path

import numpy as np
from natsort import natsorted

from histoforge_models import dynamics, io

logger = logging.getLogger(__name__)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def get_paired_files(img_dir, lbl_dir):
    """Load and validate paired image/label file paths from separate directories.

    Files are matched by basename (stem) after natural sorting.

    Returns:
        Tuple of (image_paths, label_paths) as lists of strings.
    """
    img_files = natsorted(
        [f for f in Path(img_dir).iterdir() if f.suffix.lower() in IMG_EXTS])
    lbl_files = natsorted(
        [f for f in Path(lbl_dir).iterdir() if f.suffix.lower() in IMG_EXTS])
    assert len(img_files) == len(lbl_files), (
        f"Count mismatch: {len(img_files)} images vs {len(lbl_files)} labels")
    for img, lbl in zip(img_files, lbl_files):
        assert img.stem == lbl.stem, (
            f"Basename mismatch: {img.stem} vs {lbl.stem}")
    return [str(f) for f in img_files], [str(f) for f in lbl_files]


def load_arrays(file_paths):
    """Read a list of file paths into numpy arrays via histoforge_models.io.imread."""
    return [io.imread(f) for f in file_paths]


def list_images(img_dir):
    """List image file paths in a directory, naturally sorted."""
    return natsorted([
        str(f) for f in Path(img_dir).iterdir()
        if f.suffix.lower() in IMG_EXTS
    ])


def prepare_semantic_labels(masks_list, mask_files, class_labels_dir,
                            device=None):
    """Convert instance masks + per-instance JSON class labels to
    5-channel semantic flow labels.

    For each image:
        1. Load JSON: {"instance_id": class_id, ...}
        2. Build per-pixel class map from instance mask + JSON
           (class IDs shifted +1 to reserve 0 for background)
        3. Compute flows via dynamics.labels_to_flows()
        4. Assemble 5-channel label:
           [masks, class_map, cellprob, flowY, flowX]

    Class ID convention:
        JSON files use 0-indexed foreground classes (e.g., 0=leydig,
        1=intra_tubule, ...). These are shifted +1 internally so that
        0=background in the class map. Set nclasses = num_foreground + 1.

    Args:
        masks_list: List of 2D instance mask arrays (H, W).
        mask_files: List of mask file paths (for basename matching to JSONs).
        class_labels_dir: Directory containing JSON files matched by
            basename to mask files. Each JSON: {"instance_id": class_id, ...}
        device: Torch device for flow computation.

    Returns:
        List of [5, H, W] float32 arrays.
    """
    class_dir = Path(class_labels_dir)
    semantic_labels = []
    for mask, mask_path in zip(masks_list, mask_files):
        stem = Path(mask_path).stem
        json_path = class_dir / f"{stem}.json"
        with open(json_path) as f:
            inst_classes = json.load(f)

        # Build per-pixel class map (0 = background for unlabeled pixels,
        # foreground classes shifted +1 so class 0 in JSON becomes 1)
        class_map = np.zeros(mask.shape, "float32")
        for inst_id_str, cls_id in inst_classes.items():
            class_map[mask == int(inst_id_str)] = cls_id + 1

        # Compute flows from instance masks
        # Returns [4, H, W]: [masks, cellprob, flowY, flowX]
        lbl_all = dynamics.labels_to_flows(
            [mask[np.newaxis]], device=device)[0]

        # Assemble: [masks, class_map, cellprob, flowY, flowX]
        semantic_lbl = np.concatenate(
            (lbl_all[:1], class_map[np.newaxis], lbl_all[1:]),
            axis=0)
        semantic_labels.append(semantic_lbl)

    logger.info("Prepared %d semantic labels (shape=%s)",
                len(semantic_labels), semantic_labels[0].shape)
    return semantic_labels


def prepare_hovernet_labels(masks_list, mask_files, class_labels_dir):
    """Convert instance masks + per-instance JSON class labels to 5-channel
    HoVer-Net training labels.

    Produces the same 5-channel envelope as `prepare_semantic_labels` so it
    is a drop-in swap in the pipeline, but the last-2 channels are the HoVer
    HV maps `[V_gt, H_gt]` (from `hv_transforms.gen_instance_hv_map`) instead
    of cellpose flows. Channels:

        0: instance mask (int-like, kept as float32)
        1: per-pixel class map (0=bg, shifted foreground classes)
        2: foreground binary (analog to cellprob)
        3: V_gt (y-axis gradient), in [-1, 1], bg=0
        4: H_gt (x-axis gradient), in [-1, 1], bg=0

    Skips `dynamics.labels_to_flows` — HoVer targets are cheap to precompute
    on CPU and don't need a GPU pass. `random_rotate_and_resize` rotates the
    last-2 channels correctly for the (V, H) rotational pair (see cellpose
    flow convention: last-2 = [flow_y, flow_x]).
    """
    from histoforge_models.pathology.managed_training._hover_targets import gen_instance_hv_map

    # class_labels_dir=None -> seg-only HoVer (no classes): class_map stays
    # all-background, so the 5-ch envelope is preserved (loss reads only the
    # HV/NP channels via negative indexing; the cls branch is off at nclasses=0).
    class_dir = Path(class_labels_dir) if class_labels_dir is not None else None
    labels = []
    for mask, mask_path in zip(masks_list, mask_files):
        class_map = np.zeros(mask.shape, "float32")
        if class_dir is not None:
            with open(class_dir / f"{Path(mask_path).stem}.json") as f:
                inst_classes = json.load(f)
            for inst_id_str, cls_id in inst_classes.items():
                class_map[mask == int(inst_id_str)] = cls_id + 1

        fg = (mask > 0).astype("float32")
        hv = gen_instance_hv_map(mask.astype("int32"))  # (2, H, W) [V, H]

        lbl = np.stack([
            mask.astype("float32"),
            class_map,
            fg,
            hv[0],   # V
            hv[1],   # H
        ], axis=0)
        labels.append(lbl)

    logger.info("Prepared %d HoVer labels (shape=%s)",
                len(labels), labels[0].shape)
    return labels


def prepare_dense_semantic_labels(masks_list):
    """Convert dense per-pixel class masks to training label format.

    Args:
        masks_list: List of 2D arrays (H, W), pixel values = class indices.

    Returns:
        List of [5, H, W] float32 arrays (class map at channel 1).
    """
    labels = []
    for mask in masks_list:
        h, w = mask.shape[:2]
        lbl = np.zeros((5, h, w), dtype="float32")
        lbl[1] = mask
        labels.append(lbl)
    logger.info("Prepared %d dense semantic labels (shape=%s)",
                len(labels), labels[0].shape)
    return labels


def compute_class_weights(labels_list, nclasses):
    """Compute median-normalized sqrt class weights from semantic labels.

    Formula: weights = (median(counts) / counts)^0.5
    This reduces extreme weights for very rare classes compared to raw
    inverse frequency.

    Args:
        labels_list: List of 5-channel label arrays [5, H, W].
            Channel 1 is the per-pixel class map.
        nclasses: Number of classes including background.

    Returns:
        numpy array of shape (nclasses,), float32.
    """
    all_classes = np.concatenate([lbl[1].ravel() for lbl in labels_list])
    valid = (all_classes >= 0) & (all_classes < nclasses)
    counts = np.bincount(all_classes[valid].astype(int),
                         minlength=nclasses).astype("float64")
    counts = np.maximum(counts, 1)  # avoid div by zero
    weights = (np.median(counts) / counts) ** 0.5
    weights = weights.astype("float32")
    logger.info("Class weights (sqrt-median): %s", weights.round(4).tolist())
    return weights


def compute_sampling_weights(labels_list, nclasses):
    """Compute per-image sampling probabilities for class-balanced training.

    Images with more instances of rare classes get higher sampling
    probability. Background (class 0) is excluded from weighting.

    Args:
        labels_list: List of 5-channel label arrays [5, H, W].
            Channel 0 is instance mask, channel 1 is class map.
        nclasses: Number of classes including background.

    Returns:
        numpy array of shape (n_images,), float64 — normalized probs.
    """
    # Per-class pixel counts across dataset → inverse weights
    all_classes = np.concatenate([lbl[1].ravel() for lbl in labels_list])
    valid = (all_classes >= 0) & (all_classes < nclasses)
    counts = np.bincount(all_classes[valid].astype(int),
                         minlength=nclasses).astype("float64")
    counts = np.maximum(counts, 1)
    inv_freq = 1.0 / counts
    inv_freq[0] = 0  # exclude background

    # Per-image: count unique instances per class, weight by inv_freq
    weights = np.zeros(len(labels_list), "float64")
    for i, lbl in enumerate(labels_list):
        masks = lbl[0]
        class_map = lbl[1]
        for c in range(1, nclasses):
            n_instances = len(np.unique(masks[class_map == c]))
            weights[i] += n_instances * inv_freq[c]

    weights = np.maximum(weights, 1e-10)
    weights /= weights.sum()
    logger.info("Oversampling probs range: [%.6f, %.6f]",
                weights.min(), weights.max())
    return weights
