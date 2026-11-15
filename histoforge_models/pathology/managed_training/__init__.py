"""Scientific training primitives for instance and dense models.

The package owns target construction, masked losses, task weighting and explicit
real-pixel appearance operators. The application owns optimizer sequencing,
dataset preparation, field sampling, augmentation selection, parameters, order
and seeds, validation, checkpointing and publication.

All package-internal imports stay within this directory. This makes the exact
directory suitable for provenance-bound vendoring without copying unrelated
Cellpose training, inference, command-line, or research code.
"""

from .contracts import (
    ClassificationObjective,
    FlowLossWeighting,
    HeadType,
    ManagedLossResult,
)

# Standalone embedding must load targets (and fastremap) before the Torch losses.
# A host that imports Torch first can still encounter the macOS runtime conflict.
from .targets import ManagedSegmentationTargets, build_managed_targets
from .losses import (
    ManagedSegmentationBatch,
    managed_segmentation_loss,
    validate_managed_loss_configuration,
)
from .objective import ManagedObjective
from .dense_losses import ManagedDenseBatch, managed_dense_loss
from .augmentation import (
    augment_blur,
    augment_brightness_contrast,
    augment_hed,
    augment_noise,
)

__all__ = [
    "ClassificationObjective",
    "FlowLossWeighting",
    "HeadType",
    "ManagedLossResult",
    "ManagedDenseBatch",
    "ManagedObjective",
    "ManagedSegmentationBatch",
    "ManagedSegmentationTargets",
    "augment_blur",
    "augment_brightness_contrast",
    "augment_hed",
    "augment_noise",
    "build_managed_targets",
    "managed_segmentation_loss",
    "managed_dense_loss",
    "validate_managed_loss_configuration",
]
