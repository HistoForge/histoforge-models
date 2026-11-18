"""Masked pixel-classification objectives with equal weight per model field."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from .classification_losses import (
    classification_field_loss,
    validate_classification_loss_configuration,
)
from .contracts import ClassificationObjective, ManagedLossResult
from .objective import ManagedObjective


@dataclass(frozen=True)
class ManagedDenseBatch:
    """RGB and neural-channel targets; no instances or directional targets.

    Background is channel zero. Context and artificial padding can be visible
    to the network, but only supervised real pixels contribute to losses.
    """

    images: Tensor
    class_targets: Tensor
    supervision_mask: Tensor
    real_pixel_mask: Tensor

    @property
    def effective_support(self) -> Tensor:
        """Return the intersection of supervision and real image support."""

        return self.supervision_mask & self.real_pixel_mask

    def validate(self, class_count: int) -> None:
        """Validate the dense scientific boundary before model execution."""

        if class_count < 1:
            raise ValueError("dense batches require class channels")
        if self.images.ndim != 4 or self.images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        batch, _, height, width = self.images.shape
        if batch < 1:
            raise ValueError("dense batches must contain at least one field")
        expected = (batch, height, width)
        for name, value in (
            ("class_targets", self.class_targets),
            ("supervision_mask", self.supervision_mask),
            ("real_pixel_mask", self.real_pixel_mask),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape [B, H, W]")
        if (
            self.supervision_mask.dtype != torch.bool
            or self.real_pixel_mask.dtype != torch.bool
        ):
            raise ValueError("support masks must be boolean")
        if self.class_targets.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("class_targets must use an integer dtype")
        support = self.effective_support
        if torch.any(support.flatten(1).sum(dim=1) == 0):
            raise ValueError("every field must contain supervised real pixels")
        valid = (self.class_targets >= 0) & (self.class_targets < class_count)
        if torch.any(~valid & (self.class_targets != 255)):
            raise ValueError("class_targets must be valid class-channel indices")
        if torch.any(support & ~valid):
            raise ValueError("class 255 is invalid inside effective support")


def managed_dense_loss(
    logits: Tensor,
    batch: ManagedDenseBatch,
    *,
    class_count: int,
    classification_objective: ClassificationObjective = "cross_entropy+tversky",
    classification_loss_weight: float = 1.0,
    class_weights: Tensor | None = None,
    weight_tversky_by_class: bool = True,
    tversky_alpha: float = 0.3,
    tversky_gamma: float = 1.33,
    objective: ManagedObjective | None = None,
) -> ManagedLossResult:
    """Reduce each field independently, then return sums for accumulation.

    ``logits`` contains only class channels. Model-specific unused output heads
    are outside this contract. A learned objective has one parameter per active
    classification loss, never a dummy segmentation task.
    """

    validate_classification_loss_configuration(
        class_count,
        classification_objective,
        classification_loss_weight,
        class_weights,
        weight_tversky_by_class,
        tversky_alpha,
        tversky_gamma,
    )
    batch.validate(class_count)
    expected = (batch.images.shape[0], class_count, *batch.images.shape[2:])
    if logits.shape != expected:
        raise ValueError(
            f"logits must have shape {expected}, got {tuple(logits.shape)}"
        )
    if logits.dtype != torch.float32:
        raise ValueError("managed loss requires float32 predictions")
    if objective is None:
        objective = ManagedObjective(
            class_count=class_count,
            classification_objective=classification_objective,
            classification_loss_weight=classification_loss_weight,
            include_segmentation=False,
        ).to(logits.device)
    objective.validate_loss_configuration(
        class_count=class_count,
        classification_objective=classification_objective,
        classification_loss_weight=classification_loss_weight,
        include_segmentation=False,
    )
    components: dict[str, list[Tensor]] = {}
    for index in range(batch.images.shape[0]):
        values = classification_field_loss(
            logits[index : index + 1],
            batch.class_targets[index : index + 1],
            batch.effective_support[index],
            objective=classification_objective,
            class_weights=class_weights,
            weight_tversky_by_class=weight_tversky_by_class,
            alpha=tversky_alpha,
            gamma=tversky_gamma,
        )
        for name, value in values.items():
            components.setdefault(name, []).append(value)
    sums = {name: torch.stack(values).sum() for name, values in components.items()}
    return ManagedLossResult(
        total_sum=objective.combine(sums, field_count=batch.images.shape[0]),
        task_sums=sums,
        component_sums=sums,
        field_count=batch.images.shape[0],
    )
