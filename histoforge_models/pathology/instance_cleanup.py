"""Head-specific instance cleanup shared by full-field and window inference."""

from __future__ import annotations

from collections.abc import Collection

import fastremap
import numpy as np


def cleanup_cellpose_instances(
    instance_mask: np.ndarray,
    min_instance_pixels: int,
    *,
    protected_instance_ids: Collection[int] = (),
) -> np.ndarray:
    """Apply Cellpose cleanup while protecting provisional window fragments.

    The ordinary path calls Cellpose's helper once inside a temporary
    zero-background frame. The protected path composes the same filter, fill,
    filter sequence from the upstream hole filler and HistoForge's ID-aware
    area filter. A boolean snapshot of the protected pixels carries protection
    across Cellpose's renumbering.
    """
    from ..utils import fill_holes_and_remove_small_masks

    mask = _validated_instance_mask(instance_mask, min_instance_pixels)
    protected = {int(value) for value in protected_instance_ids if int(value) > 0}
    if not protected:
        # Upstream treats uniq[0] as background without checking that its value
        # is zero. Keep an explicit border through both size passes because
        # hole filling can consume the last background pixel in the mask.
        working = np.pad(mask, 1, mode="constant", constant_values=0)
        result = fill_holes_and_remove_small_masks(
            working,
            min_size=max(1, min_instance_pixels),
        )
        return result[1:-1, 1:-1].copy()

    protected_pixels = np.isin(mask, list(protected))
    result = remove_instances_below_area(
        mask,
        min_instance_pixels,
        protected_instance_ids=protected,
    )
    result = fill_holes_and_remove_small_masks(result, min_size=1)
    protected_after_fill = {
        int(value) for value in np.unique(result[protected_pixels]) if value > 0
    }
    return remove_instances_below_area(
        result,
        min_instance_pixels,
        protected_instance_ids=protected_after_fill,
    )


def remove_instances_below_area(
    instance_mask: np.ndarray,
    min_instance_pixels: int,
    *,
    protected_instance_ids: Collection[int] = (),
) -> np.ndarray:
    """Remove decoded instances smaller than a model-grid pixel threshold.

    Zero disables this final area filter and preserves instance IDs. Positive
    instance IDs need not be consecutive, and a mask need not contain
    background. Protected IDs remain available for a later completeness check,
    even when their visible fragment is below the threshold. When the filter is
    enabled, retained positive IDs are renumbered consecutively for downstream
    class voting and feature conversion.
    """

    mask = _validated_instance_mask(instance_mask, min_instance_pixels)
    result = mask.copy()
    if min_instance_pixels == 0 or not np.any(result):
        return result

    labels, counts = fastremap.unique(result, return_counts=True)
    protected = np.fromiter(protected_instance_ids, dtype=labels.dtype)
    remove = labels[
        (labels > 0) & (counts < min_instance_pixels) & ~np.isin(labels, protected)
    ]
    if remove.size:
        result = fastremap.mask(result, remove)
    fastremap.renumber(result, in_place=True)
    return result


def _validated_instance_mask(instance_mask, min_instance_pixels):
    require_min_instance_pixels(min_instance_pixels)

    mask = np.asarray(instance_mask)
    if mask.ndim != 2:
        raise ValueError("instance_mask must be a two-dimensional array")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("instance_mask must use an integer dtype")
    if np.any(mask < 0):
        raise ValueError("instance_mask must be nonnegative")

    return mask


def require_min_instance_pixels(min_instance_pixels: int) -> int:
    """Validate and return the recipe's nonnegative Cellpose area threshold."""
    if (
        isinstance(min_instance_pixels, bool)
        or not isinstance(min_instance_pixels, int)
        or min_instance_pixels < 0
    ):
        raise ValueError("min_instance_pixels must be a nonnegative integer")
    return min_instance_pixels
