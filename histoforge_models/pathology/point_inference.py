"""Scored object centers decoded once after neural-field blending."""

import math

import numpy as np
import torch
from scipy import ndimage
from scipy.special import expit
from skimage.morphology import local_maxima

from .field_inference import run_model_fields


class PointDetectionLimitExceeded(ValueError):
    """The complete decoded population exceeds the caller's count limit."""


def decode_point_logits(
    logits: np.ndarray,
    *,
    score_threshold: float,
    peak_radius_model_px: int,
    max_detections: int,
    support_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return xy, class_indices and scores from finite HWC center logits.

    Native float16/32/64 arrays are accepted. Float16 is promoted losslessly to
    float32 for the morphology operations; float64 peak comparisons stay exact.

    A class-specific peak is maximal within a square neighborhood of radius
    peak_radius_model_px. An 8-connected equal-valued plateau emits its first
    pixel in row-major order. Coordinates refer to pixel centers, x + 0.5 and
    y + 0.5, in the supplied model-resolution image. Classes are zero-based
    emitted channels; scores are independent sigmoid values, not calibrated
    probabilities. Results sort by descending score, class, y, then x.

    Peak comparisons use logits before sigmoid to avoid manufacturing plateaus
    by float32 saturation. Unsupported pixels cannot produce or suppress peaks.
    Exceeding max_detections fails instead of silently dropping predictions.
    This function neither clips polygon geometry nor merges manual objects.
    """
    raw = np.asarray(logits)
    if (
        raw.ndim != 3
        or any(side == 0 for side in raw.shape)
        or raw.dtype
        not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64"))
        or not np.isfinite(raw).all()
    ):
        raise ValueError("logits must be finite float16/32/64 HWC")
    if raw.dtype == np.float16:
        raw = raw.astype(np.float32)
    _validate_decode_options(score_threshold, peak_radius_model_px, max_detections)
    support = (
        np.ones(raw.shape[:2], dtype=bool)
        if support_mask is None
        else np.asarray(support_mask)
    )
    if support.dtype != np.bool_ or support.shape != raw.shape[:2]:
        raise ValueError("support_mask must be boolean with logits' HW shape")

    coordinates, classes, scores = [], [], []
    # A larger radius cannot see anything beyond the image's existing pixels.
    radius = min(peak_radius_model_px, max(raw.shape[:2]))
    for channel in range(raw.shape[2]):
        values = np.where(support, raw[..., channel], -np.inf)
        maxima = ndimage.maximum_filter(
            values, size=2 * radius + 1, mode="constant", cval=-np.inf
        )
        # Regional maxima reject an entire flat shoulder connected to a higher
        # value. A neighborhood-only maximum filter would retain its far end.
        # The low border also defines maxima for a single-pixel/constant image.
        regional = local_maxima(
            np.pad(values, 1, constant_values=-np.inf),
            connectivity=2,
            allow_borders=True,
        )[1:-1, 1:-1]
        peaks = support & regional
        labels, count = ndimage.label(peaks, structure=np.ones((3, 3), dtype=bool))
        if count:
            # One evaluation precision keeps identical logits from crossing the
            # score threshold merely because an offline caller upcast the array.
            confidence_map = expit(values.astype(np.float64)).astype(np.float32)
            rejected = ndimage.maximum(
                (values != maxima)
                | (confidence_map.astype(np.float64) < score_threshold),
                labels,
                index=np.arange(1, count + 1),
            )
            permitted = np.concatenate(([False], ~rejected.astype(bool)))
            peaks &= permitted[labels]
            labels, count = ndimage.label(peaks, structure=np.ones((3, 3), dtype=bool))
        if len(coordinates) + count > max_detections:
            raise PointDetectionLimitExceeded("point result exceeds max_detections")
        if count == 0:
            continue
        ys, xs = np.nonzero(peaks)
        _, first_indices = np.unique(labels[ys, xs], return_index=True)
        for index in first_indices:
            y, x = ys[index], xs[index]
            coordinates.append((x + 0.5, y + 0.5))
            classes.append(channel)
            scores.append(confidence_map[y, x])

    xy = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    class_indices = np.asarray(classes, dtype=np.int64)
    confidence = np.asarray(scores, dtype=np.float32)
    order = np.lexsort((xy[:, 0], xy[:, 1], class_indices, -confidence))
    return {
        "xy": xy[order],
        "class_indices": class_indices[order],
        "scores": confidence[order],
    }


def _validate_decode_options(score_threshold, peak_radius_model_px, max_detections):
    if (
        isinstance(score_threshold, bool)
        or not isinstance(score_threshold, (int, float))
        or not math.isfinite(score_threshold)
        or not 0 <= score_threshold <= 1
    ):
        raise ValueError("score_threshold must be finite and in [0, 1]")
    for value, name in (
        (peak_radius_model_px, "peak_radius_model_px"),
        (max_detections, "max_detections"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


@torch.no_grad()
def run_point_inference(
    net: torch.nn.Module,
    image: np.ndarray,
    *,
    field_profile: dict,
    score_threshold: float,
    peak_radius_model_px: int,
    max_detections: int,
    batch_size: int = 8,
    normalize: bool = True,
    real_pixel_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Predict one finite image in FP32 and decode its complete blended map.

    The caller supplies an explicit existing variable-field profile. Shared
    inference normalizes real image pixels once and blends raw logits before
    peak extraction. Returned coordinates remain on this image's model grid;
    the application owns physical/slide transforms and exact ROI admission.
    No large-region or lower-precision capability is implied by this API.
    """
    if getattr(net, "head_type", None) != "point_detection":
        raise ValueError("point inference requires a point-detection head")
    if net.dtype != torch.float32 or any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in net.parameters()
    ):
        raise ValueError("point inference requires FP32 network weights")
    if not isinstance(field_profile, dict):
        raise ValueError("point inference requires an explicit field_profile")
    _validate_decode_options(score_threshold, peak_radius_model_px, max_detections)
    # Validation can be called inside a mixed-precision training context.
    # Disable it locally without changing the caller's context or other heads.
    with torch.autocast(device_type=torch.device(net.device).type, enabled=False):
        output = run_model_fields(
            net,
            image,
            fixed_field_defaults=(
                field_profile["preferredTileSideModelPx"],
                field_profile["tileOverlap"],
            ),
            field_profile=field_profile,
            batch_size=batch_size,
            normalize=normalize,
            real_pixel_mask=real_pixel_mask,
        )
    if output.shape != (*image.shape[:2], net.nclasses):
        raise ValueError("point network output must match image HW and emitted classes")
    return decode_point_logits(
        output,
        score_threshold=score_threshold,
        peak_radius_model_px=peak_radius_model_px,
        max_detections=max_detections,
        support_mask=real_pixel_mask,
    )
