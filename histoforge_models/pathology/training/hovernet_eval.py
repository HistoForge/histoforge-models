"""Standalone HoVer inference followed by the shared Cellpose mask evaluator.

This research evaluator normalizes a complete image before tiling, as do the
product wrappers. Only targets, losses and watershed decoding are head-specific;
mask matching and metric aggregation are shared with the flow evaluator.
"""

import logging
from typing import List

import cv2
import numpy as np

from ..decode_profile import (
    HOVER_EDGE_THRESHOLD, HOVER_FG_THRESHOLD, HOVER_MORPH_KERNEL,
    HOVER_OBJECT_SIZE, HOVER_TILE_OVERLAP,
)
from ..dinov3 import DinoV3AdapterBackbone
from ..hv_decode import watershed_from_hv_maps
from ..metric_aggregation import (
    DEFAULT_SEGMENTATION_AGGREGATION, validate_segmentation_aggregation,
)
from .segmentation_metrics import evaluate_instance_masks

logger = logging.getLogger(__name__)


def compute_hovernet_seg_ap(
    net: DinoV3AdapterBackbone,
    images: List[np.ndarray],
    gt_masks: List[np.ndarray],
    bsize: int = 256,
    batch_size: int = 8,
    tile_overlap: float = HOVER_TILE_OVERLAP,
    edge_threshold: float = HOVER_EDGE_THRESHOLD,
    object_size: int = HOVER_OBJECT_SIZE,
    fg_threshold: float = HOVER_FG_THRESHOLD,
    morph_kernel: int = HOVER_MORPH_KERNEL,
    segmentation_metric_aggregation: str = DEFAULT_SEGMENTATION_AGGREGATION,
) -> dict:
    """Val AP for HoVer-mode training. Mirrors `eval.compute_seg_ap` for flow.

    Runs cellpose's tiled forward (`core.run_net`) on each val image, slices
    HV + NP from the stitched output, decodes to instances via
    `watershed_from_hv_maps`, then uses the same Cellpose matcher and pooled or
    per-image aggregation as the flow head. No optional evaluator or fallback.

    Args:
        net: HoVer-mode backbone with weights loaded (its `.eval()` state is
            managed by the caller).
        images: list of raw (H, W) or (H, W, 3) val images.
        gt_masks: list of int GT instance masks, same length as `images`.
        bsize: model tile size (default 256; caller should match training).
        batch_size: tiles per forward pass.
        tile_overlap: cellpose tile-overlap fraction. Bump to 0.5 if
            watershed leaks at tile boundaries during eval — HV is
            centroid-relative so context matters.

    Returns:
        Selected AP/mSA, named pooled/per_image comparisons, and per-image counts.
    """
    from ...core import run_net
    from ...transforms import normalize_img

    validate_segmentation_aggregation(segmentation_metric_aggregation)
    pred_masks: List[np.ndarray] = []
    for i, img in enumerate(images):
        # Coerce to (H, W, 3) float, normalize like cellpose.
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] != 3:
            arr = arr[..., :3]
        arr = normalize_img(arr, axis=-1)  # (H, W, 3)

        # run_net expects (Lz, Ly, Lx, C) — inject a Z=1 dim.
        stack = arr[np.newaxis]  # (1, H, W, 3)
        y_full, _ = run_net(net, stack, batch_size=batch_size, augment=False,
                            tile_overlap=tile_overlap, bsize=bsize)
        # y_full: (Lz, H, W, nchan) — take z=0.
        y_full = y_full[0]  # (H, W, nchan)

        hv = y_full[..., -4:-2]  # (H, W, 2) [V, H]
        np_logits = y_full[..., -2:]  # (H, W, 2)
        # Softmax across the NP axis to get fg probability.
        e = np.exp(np_logits - np_logits.max(axis=-1, keepdims=True))
        np_prob = e[..., 1] / e.sum(axis=-1)  # (H, W)

        pred_stack = np.stack(  # (3, H, W)
            [np_prob, hv[..., 0], hv[..., 1]], axis=0)
        pred = watershed_from_hv_maps(pred_stack, edge_threshold=edge_threshold,
                                      object_size=object_size,
                                      fg_threshold=fg_threshold,
                                      morph_kernel=morph_kernel)
        # run_net stitches back to the input size, but guard against any drift
        # vs the GT mask (tiling/overlap edges, odd dims) so the AP IoU aligns
        # — fixes "masks_true.size != masks_pred.size".
        gt_hw = np.asarray(gt_masks[i]).shape[:2]
        if pred.shape[:2] != gt_hw:
            pred = cv2.resize(pred.astype(np.int32), (gt_hw[1], gt_hw[0]),
                              interpolation=cv2.INTER_NEAREST)
        pred_masks.append(pred)

        # Diagnostic (first 2 val images): pinpoints an AP=0 failure mode —
        # np collapse (fg_frac~0 -> 0 pred), np over-fire (fg_frac~1 -> 1 giant
        # blob), degenerate HV, or a pred/GT instance-count/size mismatch.
        if i < 2:
            gt_i = np.asarray(gt_masks[i])
            logger.info(
                ">>> HoVer AP diag [img %d]: y_full%s nout=%d | "
                "np_prob(min/mean/max)=%.3f/%.3f/%.3f fg_frac=%.4f | "
                "V(min/max)=%.2f/%.2f H=%.2f/%.2f | "
                "pred_inst=%d gt_inst=%d | pred_hw=%s gt_hw=%s", i,
                tuple(y_full.shape), y_full.shape[-1], float(np_prob.min()),
                float(np_prob.mean()), float(np_prob.max()),
                float((np_prob > 0.5).mean()), float(hv[..., 0].min()),
                float(hv[..., 0].max()), float(hv[..., 1].min()), float(hv[...,
                                                                           1].max()),
                int(pred.max()), int(len(np.unique(gt_i)) - 1), tuple(pred.shape),
                tuple(gt_i.shape[:2]))

    return evaluate_instance_masks(gt_masks, pred_masks, segmentation_metric_aggregation)
