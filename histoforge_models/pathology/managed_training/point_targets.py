"""Center heatmaps for point labels, without inventing object contours."""

import math

import numpy as np


def build_point_targets(
    points_xy: np.ndarray,
    class_indices: np.ndarray,
    *,
    class_count: int,
    sigma_model_px: float,
    support_mask: np.ndarray,
) -> np.ndarray:
    """Return float32 CHW heatmaps for one supervised model-resolution field.

    Coordinates use image pixel edges: pixel (row, column) has its center at
    (column + 0.5, row + 0.5). Each mark supervises the pixel containing it.
    Each emitted class has its own sigmoid channel; there is no background
    channel. Gaussian shoulders reduce the negative loss near a positive,
    rather than label an object's area. Their spread is explicit, not inferred
    from MIDOG's evaluation tolerance or a marker's displayed size.

    The caller transforms RGB and points together, then intersects reviewed
    supervision with real source support. Context and artificial padding must
    be false in support_mask. Mark identities remain with the caller. Distinct
    same-class marks in one output pixel fail instead of silently merging.
    """
    points = np.asarray(points_xy)
    classes = np.asarray(class_indices)
    support = np.asarray(support_mask)
    if support.ndim != 2 or support.dtype != np.bool_ or not support.any():
        raise ValueError(
            "support_mask must contain supervised real pixels in boolean HW"
        )
    if (
        isinstance(class_count, bool)
        or not isinstance(class_count, int)
        or class_count < 1
    ):
        raise ValueError("class_count must be a positive integer")
    if (
        isinstance(sigma_model_px, bool)
        or not isinstance(sigma_model_px, (int, float))
        or not math.isfinite(sigma_model_px)
        or sigma_model_px <= 0
    ):
        raise ValueError("sigma_model_px must be finite and positive")
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not (
            np.issubdtype(points.dtype, np.floating)
            or np.issubdtype(points.dtype, np.integer)
        )
        or not np.isfinite(points).all()
    ):
        raise ValueError("points_xy must be a finite real Nx2 array")
    if (
        classes.shape != (len(points),)
        or not np.issubdtype(classes.dtype, np.integer)
        or np.any(classes < 0)
        or np.any(classes >= class_count)
    ):
        raise ValueError(
            "class_indices must contain one valid integer channel per point"
        )
    height, width = support.shape
    if np.any(points < 0) or np.any(points >= [width, height]):
        raise ValueError("point coordinates must lie inside the field's pixel edges")
    cells = np.floor(points).astype(np.int64)
    if len(points) and not support[cells[:, 1], cells[:, 0]].all():
        raise ValueError("point center lies outside supervised real support")
    keys = np.column_stack((classes, cells))
    if len(np.unique(keys, axis=0)) != len(points):
        raise ValueError("same-class point centers collide in one output pixel")

    targets = np.zeros((class_count, height, width), dtype=np.float32)
    # The image bounds cap allocation even when the requested spread is broad.
    radius = math.ceil(min(3 * sigma_model_px, max(height, width)))
    for (x, y), channel in zip(cells, classes):
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        yy, xx = np.ogrid[y0 - y : y1 - y, x0 - x : x1 - x]
        with np.errstate(over="ignore", under="ignore"):
            shoulder = np.exp(
                -0.5 * ((xx / sigma_model_px) ** 2 + (yy / sigma_model_px) ** 2)
            )
        # Only a true center is exactly 1, including under float32 rounding.
        shoulder = np.minimum(shoulder, np.nextafter(np.float32(1), np.float32(0)))
        shoulder[y - y0, x - x0] = 1
        patch = targets[channel, y0:y1, x0:x1]
        np.maximum(patch, shoulder, out=patch)
    targets[:, ~support] = 0
    return targets
