"""Instance-balanced Cellpose flow weighting for the vendorable trainer."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def inverse_instance_area_map(instance_mask: np.ndarray) -> np.ndarray:
    """Return ``1 / full_instance_area`` on instances and zero elsewhere."""

    mask = np.asarray(instance_mask)
    if mask.ndim != 2:
        raise ValueError("instance mask must be two-dimensional")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("instance mask must use an integer dtype")
    if np.any(mask < 0):
        raise ValueError("instance mask IDs must be nonnegative")

    instance_ids, inverse, counts = np.unique(
        mask, return_inverse=True, return_counts=True
    )
    per_instance = np.zeros(instance_ids.shape, dtype=np.float32)
    foreground = instance_ids != 0
    per_instance[foreground] = 1.0 / counts[foreground].astype(np.float32)
    return per_instance[inverse].reshape(mask.shape)


def normalized_instance_flow_weights(
    instance_weight_map: Tensor, foreground: Tensor
) -> Tensor:
    """Normalize visible instance weights while preserving total pixel mass."""

    if instance_weight_map.ndim != 3 or foreground.ndim != 3:
        raise ValueError("instance flow weights and foreground must be [B,H,W]")
    if instance_weight_map.shape != foreground.shape:
        raise ValueError(
            "instance flow weights and foreground must have identical shapes"
        )
    if not instance_weight_map.dtype.is_floating_point:
        raise ValueError("instance flow weights must use a floating dtype")
    if not torch.isfinite(instance_weight_map).all():
        raise ValueError("instance flow weights must be finite")
    if torch.any(instance_weight_map < 0):
        raise ValueError("instance flow weights must be nonnegative")

    normalized = torch.ones_like(instance_weight_map)
    foreground = foreground.bool()
    for batch_index in range(instance_weight_map.shape[0]):
        sample_foreground = foreground[batch_index]
        if not torch.any(sample_foreground):
            continue
        visible_weights = instance_weight_map[batch_index][sample_foreground]
        if torch.any(visible_weights <= 0):
            raise ValueError(
                "every foreground pixel must have a positive instance-flow weight"
            )
        source_mass = visible_weights.sum()
        target_mass = sample_foreground.sum().to(instance_weight_map.dtype)
        normalized[batch_index][sample_foreground] = (
            visible_weights * target_mass / source_mass
        )
    return normalized
