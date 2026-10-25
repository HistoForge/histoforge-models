"""HoVer-Net watershed post-processing.

Adapted from micro_sam/micro_sam/instance_segmentation.py:119-244
(Apache-2.0, Constantin Pape et al.). Underlying methodology: HoVer-Net
(Graham et al., Medical Image Analysis 2019, MIT license). Original:
github.com/vqdang/hover_net.

Input convention: pred = [fg_prob, V, H] (channels-first or channels-last),
matching this repo's cellpose-flow-analog HV ordering (V=y-axis, H=x-axis).
Micro_sam uses [fg_prob, H, V]; the axis-derivative assignments inside the
watershed are identical — cv2.Sobel is per-axis by construction.

Pure numpy/cv2/scipy/skimage — no torch dependency; safe to import from a
vendored inference module.
"""

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as scipy_label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from .decode_profile import (
    HOVER_EDGE_THRESHOLD,
    HOVER_FG_THRESHOLD,
    HOVER_MORPH_KERNEL,
    HOVER_OBJECT_SIZE,
    HOVER_SOBEL_KERNEL,
)


def watershed_from_hv_maps(
    pred: np.ndarray,
    object_size: int = HOVER_OBJECT_SIZE,
    ksize: int = HOVER_SOBEL_KERNEL,
    edge_threshold: float = HOVER_EDGE_THRESHOLD,
    morph_kernel: int = HOVER_MORPH_KERNEL,
    fg_threshold: float = HOVER_FG_THRESHOLD,
) -> np.ndarray:
    """Marker-controlled watershed on stitched HoVer maps.

    Args:
        pred: (3, H, W) or (H, W, 3), channels = [fg_prob, V, H].
            fg_prob in [0, 1]; V/H in [-1, 1] (background = 0).
        object_size: minimum pixel count for a surviving marker.
        ksize: cv2.Sobel kernel size for the gradient magnitude
            (default 21 — HoVer-Net's original setting).
        edge_threshold: threshold on the combined edge map for marker cores.
        morph_kernel: elliptical opening kernel size for marker cleanup
            (odd int; 1 disables). Smaller preserves thin structures.
        fg_threshold: probability threshold for the fg binary mask (also
            acts as the watershed's spatial ceiling).

    Returns:
        (H, W) int32 instance mask. 0 = background.
    """
    if pred.shape[0] == 3:
        pred = np.transpose(pred, (1, 2, 0))
    pred = np.asarray(pred, dtype=np.float32)
    blb_raw = pred[..., 0]
    v_dir_raw = pred[..., 1]
    h_dir_raw = pred[..., 2]

    blb = (blb_raw >= fg_threshold).astype(np.int32)
    blb = scipy_label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1

    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_32F)

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)
    sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F)

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall_thresh = (overall >= edge_threshold).astype(np.int32)
    marker = blb - overall_thresh
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype(np.uint8)

    if morph_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, k)
    marker = scipy_label(marker)[0]
    marker = remove_small_objects(marker, min_size=object_size)

    return watershed(dist, markers=marker, mask=blb).astype(np.int32)
