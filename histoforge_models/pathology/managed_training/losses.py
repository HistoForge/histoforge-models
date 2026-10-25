"""Managed flow and HoVer loss contracts for bounded model fields.

The model output layouts are:

* flow: ``[optional class logits..., dY, dX, foreground_logit]``
* HoVer: ``[optional class logits..., V, H, np_background, np_foreground]``

Every loss is reduced independently per model field and then fields are weighted
equally. Effective support is the intersection of the supervision and real-pixel
masks. Padding and context outside the annotated region can therefore remain
visible to the network without becoming training targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor
from torch.nn import functional as F

from ._hover_losses import HVMapLoss
from ._flow_weighting import normalized_instance_flow_weights
from .contracts import (
    ClassificationObjective,
    FlowLossWeighting,
    HeadType,
    ManagedLossResult,
)
from .objective import ManagedObjective
from .classification_losses import (
    classification_field_loss,
    validate_classification_loss_configuration,
)


@dataclass(frozen=True)
class ManagedSegmentationBatch:
    """One already-sampled and already-augmented model batch.

    ``directional_targets`` contains Cellpose ``[dY, dX]`` flows or HoVer
    ``[V, H]`` maps. ``class_targets`` uses background zero and positive
    foreground-class indices. ``instance_flow_weights``, when supplied, is the
    inverse-full-instance-area map created before field cropping. Unsupervised
    pixels are represented only by the support masks. All tensors may include
    batch-collation padding; that padding must be false in ``real_pixel_mask``.
    """

    images: Tensor
    directional_targets: Tensor
    foreground_targets: Tensor
    supervision_mask: Tensor
    real_pixel_mask: Tensor
    class_targets: Tensor | None = None
    instance_flow_weights: Tensor | None = None

    def validate(
        self,
        class_count: int,
        *,
        flow_loss_weighting: FlowLossWeighting = "pixel",
    ) -> None:
        """Reject malformed or scientifically contradictory batch tensors.

        Validation covers tensor layout, effective supervision support, binary
        foreground labels, and agreement between instance foreground and class
        labels. Class value 255 is permitted only outside effective support,
        where it is ignored by every objective.
        """

        if self.images.ndim != 4 or self.images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        batch, _, height, width = self.images.shape
        if self.directional_targets.shape != (batch, 2, height, width):
            raise ValueError("directional_targets must have shape [B, 2, H, W]")
        expected_scalar = (batch, height, width)
        for name, value in (
            ("foreground_targets", self.foreground_targets),
            ("supervision_mask", self.supervision_mask),
            ("real_pixel_mask", self.real_pixel_mask),
        ):
            if value.shape != expected_scalar:
                raise ValueError(f"{name} must have shape [B, H, W]")
        if class_count:
            if self.class_targets is None:
                raise ValueError(
                    "class_targets are required when class_count is nonzero"
                )
            if self.class_targets.shape != expected_scalar:
                raise ValueError("class_targets must have shape [B, H, W]")
        elif self.class_targets is not None:
            raise ValueError("class_targets require a nonzero class_count")
        if self.instance_flow_weights is not None:
            if self.instance_flow_weights.shape != expected_scalar:
                raise ValueError(
                    "instance_flow_weights must have shape [B, H, W]"
                )
            if not self.instance_flow_weights.dtype.is_floating_point:
                raise ValueError("instance_flow_weights must use a floating dtype")
            if not torch.isfinite(self.instance_flow_weights).all():
                raise ValueError("instance_flow_weights must be finite")
            if torch.any(self.instance_flow_weights < 0):
                raise ValueError("instance_flow_weights must be nonnegative")
        if flow_loss_weighting == "equal_instance":
            if self.instance_flow_weights is None:
                raise ValueError(
                    "equal-instance flow weighting requires "
                    "instance_flow_weights derived from complete instances"
                )
        effective = self.effective_support
        if effective.dtype != torch.bool:
            raise ValueError("support masks must be boolean")
        if torch.any(effective.flatten(1).sum(dim=1) == 0):
            raise ValueError("every field must contain supervised real pixels")
        if not torch.all(
            (self.foreground_targets == 0) | (self.foreground_targets == 1)
        ):
            raise ValueError("foreground_targets must contain only zero or one")
        if flow_loss_weighting == "equal_instance":
            assert self.instance_flow_weights is not None
            foreground = self.foreground_targets != 0
            if torch.any(
                effective & foreground & (self.instance_flow_weights <= 0)
            ):
                raise ValueError(
                    "every supervised foreground pixel requires a positive "
                    "instance-flow weight"
                )
            if torch.any(
                effective & ~foreground & (self.instance_flow_weights != 0)
            ):
                raise ValueError(
                    "supervised background pixels must have zero "
                    "instance-flow weight"
                )
        if class_count:
            assert self.class_targets is not None
            if (
                self.class_targets.dtype == torch.bool
                or self.class_targets.dtype.is_floating_point
            ):
                raise ValueError("class_targets must use an integer dtype")
            valid_class = (self.class_targets >= 0) & (self.class_targets < class_count)
            if torch.any(~valid_class & (self.class_targets != 255)):
                raise ValueError("class_targets must be valid class-channel indices")
            if torch.any(effective & ~valid_class):
                raise ValueError("class 255 is invalid inside effective support")
            foreground = self.foreground_targets != 0
            classified = self.class_targets != 0
            if torch.any(effective & (foreground != classified)):
                raise ValueError(
                    "foreground and class targets must agree on background"
                )

    @property
    def effective_support(self) -> Tensor:
        """Return pixels that are both real image data and supervised."""

        return self.supervision_mask & self.real_pixel_mask


def managed_segmentation_loss(
    predictions: Tensor,
    batch: ManagedSegmentationBatch,
    *,
    head_type: HeadType,
    class_count: int = 0,
    classification_objective: ClassificationObjective = "cross_entropy",
    classification_loss_weight: float = 1.0,
    class_weights: Tensor | None = None,
    weight_tversky_by_class: bool = True,
    tversky_alpha: float = 0.3,
    tversky_gamma: float = 1.33,
    objective: ManagedObjective | None = None,
    flow_loss_weighting: FlowLossWeighting = "pixel",
) -> ManagedLossResult:
    """Compute masked objectives and give every actual field equal weight.

    Each field is reduced independently over its effective support. ``pixel``
    preserves the established flow-vector mean. ``equal_instance`` changes only
    that flow MSE; foreground and optional classification objectives keep their
    established pixel reductions. The result contains sums rather than means so
    the optimizer loop can combine uneven microbatches and divide once by the
    actual number of accumulated fields.
    """

    validate_managed_loss_configuration(
        head_type,
        class_count,
        classification_objective,
        classification_loss_weight,
        class_weights,
        weight_tversky_by_class,
        tversky_alpha,
        tversky_gamma,
        flow_loss_weighting,
    )
    batch.validate(class_count, flow_loss_weighting=flow_loss_weighting)
    if predictions.dtype != torch.float32:
        raise ValueError("managed loss requires float32 predictions")
    if objective is not None:
        objective.validate_loss_configuration(
            class_count=class_count,
            classification_objective=classification_objective,
            classification_loss_weight=classification_loss_weight,
        )
    segmentation_channels = 3 if head_type == "flow" else 4
    expected_shape = (
        batch.images.shape[0],
        class_count + segmentation_channels,
        batch.images.shape[2],
        batch.images.shape[3],
    )
    if predictions.shape != expected_shape:
        raise ValueError(
            f"predictions must have shape {expected_shape}, got "
            f"{tuple(predictions.shape)}"
        )

    component_lists: dict[str, list[Tensor]] = {}
    task_lists: dict[str, list[Tensor]] = {}
    hover_loss = HVMapLoss().to(predictions.device) if head_type == "hovernet" else None

    for index in range(batch.images.shape[0]):
        support = batch.effective_support[index]
        segmentation = predictions[index : index + 1, class_count:]
        direction = batch.directional_targets[index : index + 1]
        foreground = batch.foreground_targets[index : index + 1]
        if head_type == "flow":
            flow_weights = (
                None
                if batch.instance_flow_weights is None
                else batch.instance_flow_weights[index : index + 1]
            )
            components = _flow_field_loss(
                segmentation,
                direction,
                foreground,
                support,
                flow_loss_weighting=flow_loss_weighting,
                instance_flow_weights=flow_weights,
            )
        else:
            assert hover_loss is not None
            components = _hover_field_loss(
                hover_loss, segmentation, direction, foreground, support
            )

        tasks = {"segmentation": components["segmentation"]}
        if class_count:
            assert batch.class_targets is not None
            classification = classification_field_loss(
                predictions[index : index + 1, :class_count],
                batch.class_targets[index : index + 1],
                support,
                objective=classification_objective,
                class_weights=class_weights,
                weight_tversky_by_class=weight_tversky_by_class,
                alpha=tversky_alpha,
                gamma=tversky_gamma,
            )
            components.update(classification)
            if "classification_cross_entropy" in classification:
                tasks["classification_cross_entropy"] = classification[
                    "classification_cross_entropy"
                ]
            if "classification_tversky" in classification:
                tasks["classification_tversky"] = classification[
                    "classification_tversky"
                ]
        for name, value in tasks.items():
            task_lists.setdefault(name, []).append(value)
        for name, value in components.items():
            component_lists.setdefault(name, []).append(value)

    task_sums = {name: torch.stack(values).sum() for name, values in task_lists.items()}
    component_sums = {
        name: torch.stack(values).sum() for name, values in component_lists.items()
    }
    if objective is None:
        objective = ManagedObjective(
            class_count=class_count,
            classification_objective=classification_objective,
            classification_loss_weight=classification_loss_weight,
        ).to(predictions.device)
    return ManagedLossResult(
        total_sum=objective.combine(task_sums, field_count=batch.images.shape[0]),
        task_sums=task_sums,
        component_sums=component_sums,
        field_count=batch.images.shape[0],
    )


def _flow_field_loss(
    prediction: Tensor,
    directional_target: Tensor,
    foreground_target: Tensor,
    support: Tensor,
    *,
    flow_loss_weighting: FlowLossWeighting,
    instance_flow_weights: Tensor | None,
) -> dict[str, Tensor]:
    """Compute established Cellpose flow and foreground loss for one field.

    Unsupported pixels contribute neither values nor gradients. The directional
    scale and reduction match the existing Cellpose training objective.
    """

    support_float = support.to(prediction.dtype)
    support_count = support_float.sum()
    flow_squared = (prediction[:, :2] - 5.0 * directional_target) ** 2
    if flow_loss_weighting == "pixel":
        flow_mse = (
            flow_squared * support_float.unsqueeze(0).unsqueeze(0)
        ).sum() / (4.0 * support_count)
    else:
        assert instance_flow_weights is not None
        supervised_foreground = (
            (foreground_target > 0.5) & support.unsqueeze(0)
        )
        normalized = normalized_instance_flow_weights(
            instance_flow_weights.to(
                device=prediction.device, dtype=prediction.dtype
            ),
            supervised_foreground,
        )
        effective_weights = normalized[0] * support_float
        flow_mse = (
            flow_squared * effective_weights.unsqueeze(0).unsqueeze(0)
        ).sum() / (4.0 * support_count)
    foreground = (foreground_target > 0.5).to(prediction.dtype)
    foreground_bce = (
        F.binary_cross_entropy_with_logits(
            prediction[:, 2], foreground, reduction="none"
        )
        * support_float.unsqueeze(0)
    ).sum() / support_count
    return {
        "segmentation": flow_mse + foreground_bce,
        "flow_mse": flow_mse,
        "foreground_bce": foreground_bce,
    }


def _hover_field_loss(
    loss: HVMapLoss,
    prediction: Tensor,
    directional_target: Tensor,
    foreground_target: Tensor,
    support: Tensor,
) -> dict[str, Tensor]:
    """Compute HoVer directional and nuclear-pixel losses for one field.

    The support mask also invalidates derivative neighborhoods that touch a
    masked pixel or the tensor exterior, preventing padding and crop edges from
    influencing the MSGE term.
    """

    result = loss(
        prediction[:, :2],
        prediction[:, 2:],
        directional_target,
        foreground_target.long(),
        valid_mask=support.unsqueeze(0),
        tensor_exterior_is_invalid=True,
    )
    return {
        "segmentation": result["total"],
        "hover_mse": result["hv_mse"],
        "hover_msge": result["hv_msge"],
        "foreground_cross_entropy": result["np_ce"],
        "foreground_dice": result["np_dice"],
    }


def validate_managed_loss_configuration(
    head_type: str,
    class_count: int,
    classification_objective: str,
    classification_loss_weight: float,
    class_weights: Tensor | None,
    weight_tversky_by_class: bool,
    alpha: float,
    gamma: float,
    flow_loss_weighting: FlowLossWeighting = "pixel",
) -> None:
    """Validate the complete loss configuration without running a model."""

    if head_type not in {"flow", "hovernet"}:
        raise ValueError("head_type must be 'flow' or 'hovernet'")
    if flow_loss_weighting not in {"pixel", "equal_instance"}:
        raise ValueError(
            "flow_loss_weighting must be 'pixel' or 'equal_instance'"
        )
    if head_type != "flow" and flow_loss_weighting != "pixel":
        raise ValueError(
            "equal-instance flow weighting applies only to the Cellpose flow head"
        )
    validate_classification_loss_configuration(
        class_count,
        classification_objective,
        classification_loss_weight,
        class_weights,
        weight_tversky_by_class,
        alpha,
        gamma,
    )
