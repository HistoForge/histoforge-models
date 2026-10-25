"""Cellpose flow integration in the complete logical target's coordinates.

Raw map execution lives in pathology.window_maps. A caller supplies one
processing image's predicted maps. Its trajectory
coordinates remain anchored to the complete target; ordinary window-local flow
integration is not numerically equivalent. Output ownership and resource limits
remain with large-region execution.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_cellpose_masks_in_target_coordinates(
    d_p: Any,
    cell_probability: Any,
    *,
    target_height: int,
    target_width: int,
    window_x: int,
    window_y: int,
    niter: int = 200,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    min_instance_pixels: int = 15,
    max_size_fraction: float = 0.4,
    device: Any = None,
) -> np.ndarray:
    """Decode one Cellpose window while retaining target-global trajectories.

    Cellpose's ordinary dynamics normalize pixel coordinates by the array being
    decoded.  Decoding the same raw field in differently sized CoreHalo windows
    can therefore change a trajectory by a pixel.  This decoder keeps
    trajectory integration and clamping in complete-target coordinates while
    sampling only the local raw field. Histogram labeling remains local and is
    valid only for objects contained by the supplied halo. Labels touching an
    artificial window edge remain provisional until the caller's ownership and
    context gate validates them.
    """

    import torch

    from ..dynamics import (
        get_masks_torch,
        remove_bad_flow_masks,
    )
    from .instance_cleanup import (
        cleanup_cellpose_instances,
    )

    flows = np.asarray(d_p, dtype=np.float32)
    probability = np.asarray(cell_probability, dtype=np.float32)
    if flows.ndim != 3 or flows.shape[0] != 2:
        raise ValueError("d_p must have shape (2, height, width)")
    if probability.shape != flows.shape[1:]:
        raise ValueError("cell_probability shape must match d_p")
    target_height = _positive_int(target_height, "target_height")
    target_width = _positive_int(target_width, "target_width")
    if target_height <= 1 or target_width <= 1:
        raise ValueError("target dimensions must exceed one pixel")
    window_x = _nonnegative_int(window_x, "window_x")
    window_y = _nonnegative_int(window_y, "window_y")
    if window_x + probability.shape[1] > target_width:
        raise ValueError("Cellpose window exceeds target_width")
    if window_y + probability.shape[0] > target_height:
        raise ValueError("Cellpose window exceeds target_height")
    device = torch.device("cpu") if device is None else torch.device(device)

    foreground = probability > float(cellprob_threshold)
    if not foreground.any():
        return np.zeros(probability.shape, dtype=np.uint16)
    indices = np.nonzero(foreground)
    final_global = _target_coordinate_flow_steps(
        flows * foreground / 5.0,
        indices,
        niter,
        target_height,
        target_width,
        window_y,
        window_x,
        device,
    ).int()
    origin = torch.tensor(
        [[window_y], [window_x]],
        dtype=torch.int32,
        device=final_global.device,
    )
    final_local = final_global - origin
    if device.type == "mps":
        final_local = final_local.to(torch.device("cpu"))
    mask = get_masks_torch(
        final_local,
        indices,
        np.array(probability.shape, dtype=np.int32),
        max_size_fraction=float(max_size_fraction),
    )
    if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
        mask = remove_bad_flow_masks(
            mask,
            flows,
            threshold=float(flow_threshold),
            device=device,
        )
    protected_ids = _artificial_edge_instance_ids(
        mask,
        target_height=target_height,
        target_width=target_width,
        window_x=window_x,
        window_y=window_y,
    )
    mask = cleanup_cellpose_instances(
        mask,
        min_instance_pixels,
        protected_instance_ids=protected_ids,
    )
    if mask.max() < 2**16 and mask.dtype != np.uint16:
        mask = mask.astype(np.uint16)
    return mask


def _artificial_edge_instance_ids(
    mask: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    window_x: int,
    window_y: int,
) -> set[int]:
    """Keep truncated candidates visible to the later CoreHalo context gate."""

    artificial_edges = []
    if window_y > 0:
        artificial_edges.append(mask[0, :])
    if window_y + mask.shape[0] < target_height:
        artificial_edges.append(mask[-1, :])
    if window_x > 0:
        artificial_edges.append(mask[:, 0])
    if window_x + mask.shape[1] < target_width:
        artificial_edges.append(mask[:, -1])
    return {
        int(instance_id)
        for edge in artificial_edges
        for instance_id in np.unique(edge)
        if instance_id > 0
    }


def _target_coordinate_flow_steps(
    d_p: np.ndarray,
    indices: tuple[np.ndarray, np.ndarray],
    niter: int,
    target_height: int,
    target_width: int,
    window_y: int,
    window_x: int,
    device: Any,
) -> Any:
    """Integrate local Cellpose flows in complete-target coordinates."""

    import torch

    if isinstance(niter, bool) or not isinstance(niter, int) or niter < 0:
        raise ValueError("niter must be a nonnegative integer")
    device = torch.device(device)
    local_height, local_width = d_p.shape[1:]
    point_count = len(indices[0])
    points = torch.zeros(
        (1, 1, point_count, 2),
        dtype=torch.float32,
        device=device,
    )
    field = torch.zeros(
        (1, 2, local_height, local_width),
        dtype=torch.float32,
        device=device,
    )
    points[0, 0, :, 1] = torch.from_numpy(indices[0]).to(
        device, dtype=torch.float32
    ) + float(window_y)
    points[0, 0, :, 0] = torch.from_numpy(indices[1]).to(
        device, dtype=torch.float32
    ) + float(window_x)
    field[0, 1] = torch.from_numpy(d_p[0]).to(device, dtype=torch.float32)
    field[0, 0] = torch.from_numpy(d_p[1]).to(device, dtype=torch.float32)

    target_extent = np.array([target_width, target_height], dtype=float) - 1.0
    for axis in range(2):
        field[:, axis] *= 2.0 / target_extent[axis]
        points[..., axis] /= target_extent[axis]
    points *= 2.0
    points -= 1.0

    complete_window = (
        window_x == 0
        and window_y == 0
        and local_width == target_width
        and local_height == target_height
    )
    target_shape = (target_width, target_height)
    local_shape = (local_width, local_height)
    origin = (window_x, window_y)
    for _ in range(niter):
        if complete_window:
            sampling_points = points
        else:
            sampling_points = torch.empty_like(points)
            for axis in range(2):
                sampling_points[..., axis] = (
                    (points[..., axis] + 1.0) * target_shape[axis] - 2.0 * origin[axis]
                ) / local_shape[axis] - 1.0
        delta = torch.nn.functional.grid_sample(
            field,
            sampling_points,
            align_corners=False,
        )
        for axis in range(2):
            points[..., axis] = torch.clamp(
                points[..., axis] + delta[:, axis],
                -1.0,
                1.0,
            )

    points += 1.0
    points *= 0.5
    for axis in range(2):
        points[..., axis] *= target_extent[axis]
    return points[0, 0][:, [1, 0]].T.contiguous()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value
