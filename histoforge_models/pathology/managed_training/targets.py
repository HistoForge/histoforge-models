"""Target construction shared by managed flow and HoVer training.

For each field-crossing instance, callers rasterize its complete captured
geometry on a sufficient bounded object-local grid, call
:func:`build_managed_targets`, and only then crop and assemble the sampled model
field. Cropping the instance first would move Cellpose flow centers and HoVer
centroids to an artificial field boundary. A slide-sized or large-ROI target
tensor is neither required nor intended.
"""

from __future__ import annotations

from dataclasses import dataclass
import fastremap
import numpy as np
import torch

from ._flow_targets import masks_to_flows_gpu
from ._flow_weighting import inverse_instance_area_map
from ._hover_targets import gen_instance_hv_map
from .contracts import HeadType


@dataclass(frozen=True)
class ManagedSegmentationTargets:
    """Complete object-local targets from which bounded fields may be cropped.

    Flow targets also include ``instance_flow_weights`` with unit mass per
    complete instance. Callers must crop and spatially transform this map with
    the other scalar targets; recomputing it from a cropped instance would give
    boundary fragments the weight of complete objects. HoVer targets leave it
    unset because their objective does not consume Cellpose flow weights.
    """

    directional: np.ndarray
    foreground: np.ndarray
    classes: np.ndarray | None
    instance_flow_weights: np.ndarray | None = None


def build_managed_targets(
    instance_labels: np.ndarray,
    *,
    head_type: HeadType,
    class_labels: np.ndarray | None = None,
    flow_device: torch.device | None = None,
) -> ManagedSegmentationTargets:
    """Derive targets from complete instances on a sufficient local grid.

    Instance IDs are normalized only in the private raster passed to target
    generators because Cellpose flow construction requires dense positive IDs.
    The caller's IDs and provenance are not changed. Callers crop the returned
    arrays afterward so field boundaries cannot move flow centers or HoVer
    centroids. HistoForge owns object rasterization and field assembly.
    """

    instances = np.asarray(instance_labels)
    if instances.ndim != 2:
        raise ValueError("instance_labels must be a two-dimensional array")
    if not np.issubdtype(instances.dtype, np.integer):
        raise ValueError("instance_labels must use an integer dtype")
    if np.any(instances < 0):
        raise ValueError("instance_labels must be nonnegative")

    classes = None
    if class_labels is not None:
        classes = np.asarray(class_labels)
        if classes.shape != instances.shape:
            raise ValueError("class_labels must match instance_labels")
        if not np.issubdtype(classes.dtype, np.integer):
            raise ValueError("class_labels must use an integer dtype")
        if np.any(classes < 0):
            raise ValueError("class_labels must be nonnegative")
        if np.any((instances == 0) & (classes != 0)):
            raise ValueError("background pixels cannot carry a foreground class")
        if np.any((instances != 0) & (classes == 0)):
            raise ValueError("foreground pixels must carry a foreground class")
        classes = classes.astype(np.int64, copy=False)

    normalized, _ = fastremap.renumber(instances, in_place=False)
    normalized = normalized.astype(np.int32, copy=False)
    if head_type == "flow":
        device = flow_device if flow_device is not None else torch.device("cpu")
        directional, _ = masks_to_flows_gpu(normalized, device=device)
        labels, counts = np.unique(normalized[normalized > 0], return_counts=True)
        singleton_labels = labels[counts == 1]
        if singleton_labels.size:
            directional[:, np.isin(normalized, singleton_labels)] = 0
        instance_flow_weights = inverse_instance_area_map(normalized)
    elif head_type == "hovernet":
        directional = gen_instance_hv_map(normalized)
        instance_flow_weights = None
    else:
        raise ValueError("head_type must be 'flow' or 'hovernet'")

    return ManagedSegmentationTargets(
        directional=np.asarray(directional, dtype=np.float32),
        foreground=(normalized > 0).astype(np.float32),
        classes=classes,
        instance_flow_weights=instance_flow_weights,
    )
