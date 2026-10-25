"""Run-level combination of the active managed training losses."""

from __future__ import annotations

from collections.abc import Mapping
import math
import torch
from torch import Tensor, nn

from .contracts import ClassificationObjective


class ManagedObjective(nn.Module):
    """Combine the active tasks for one instance or dense training run.

    Fixed mode sums the resolved task losses. Uncertainty mode reproduces the
    established homoscedastic weighting contract, with one learned log-variance
    per active task. Classification scaling is applied before either mode.
    """

    def __init__(
        self,
        *,
        class_count: int,
        classification_objective: ClassificationObjective = "cross_entropy",
        classification_loss_weight: float = 1.0,
        use_uncertainty_weighting: bool = False,
        include_segmentation: bool = True,
    ) -> None:
        """Create the exact active-task set and its combination policy."""

        super().__init__()
        _validate_objective_options(
            class_count,
            classification_objective,
            classification_loss_weight,
            use_uncertainty_weighting,
        )
        self.class_count = class_count
        self.classification_objective = classification_objective
        self.classification_loss_weight = float(classification_loss_weight)
        if not isinstance(include_segmentation, bool):
            raise ValueError("include_segmentation must be boolean")
        if not include_segmentation and class_count == 0:
            raise ValueError("a dense objective requires class channels")
        self.include_segmentation = include_segmentation
        task_names = ["segmentation"] if include_segmentation else []
        task_scales = [1.0] if include_segmentation else []
        if class_count:
            if classification_objective in {
                "cross_entropy",
                "cross_entropy+tversky",
            }:
                task_names.append("classification_cross_entropy")
                task_scales.append(self.classification_loss_weight)
            if classification_objective in {
                "tversky",
                "cross_entropy+tversky",
            }:
                task_names.append("classification_tversky")
                task_scales.append(self.classification_loss_weight)
        self.task_names = tuple(task_names)
        self.register_buffer(
            "task_scales", torch.tensor(task_scales, dtype=torch.float32)
        )
        if use_uncertainty_weighting and len(task_names) > 1:
            self.log_variances = nn.Parameter(
                torch.zeros(len(task_names), dtype=torch.float32)
            )
        else:
            self.register_parameter("log_variances", None)

    @property
    def uses_uncertainty_weighting(self) -> bool:
        """Return whether this objective has learned relative task weights."""

        return self.log_variances is not None

    def validate_loss_configuration(
        self,
        *,
        class_count: int,
        classification_objective: ClassificationObjective,
        classification_loss_weight: float,
        include_segmentation: bool = True,
    ) -> None:
        """Reject a loss call that disagrees with this run-level objective."""

        if class_count != self.class_count:
            raise ValueError("objective class_count does not match the loss")
        if include_segmentation != self.include_segmentation:
            raise ValueError("objective segmentation task does not match the loss")
        if classification_objective != self.classification_objective:
            raise ValueError(
                "objective classification_objective does not match the loss"
            )
        if float(classification_loss_weight) != self.classification_loss_weight:
            raise ValueError(
                "objective classification_loss_weight does not match the loss"
            )

    def combine(self, task_sums: Mapping[str, Tensor], *, field_count: int) -> Tensor:
        """Combine task sums while preserving equal weight per actual field."""

        if field_count < 1:
            raise ValueError("field_count must be positive")
        if set(task_sums) != set(self.task_names):
            raise ValueError("task sums do not match the objective's active tasks")
        values = torch.stack([task_sums[name] for name in self.task_names])
        scales = self.task_scales.to(device=values.device, dtype=values.dtype)
        scaled = values * scales
        if self.log_variances is None:
            return scaled.sum()
        if self.log_variances.device != values.device:
            raise ValueError("objective and task losses must use the same device")
        log_variances = self.log_variances.to(dtype=values.dtype)
        return (torch.exp(-log_variances) * scaled + field_count * log_variances).sum()


def _validate_objective_options(
    class_count: int,
    classification_objective: str,
    classification_loss_weight: float,
    use_uncertainty_weighting: bool,
) -> None:
    """Validate the run-level task-combination configuration."""

    if class_count < 0:
        raise ValueError("class_count must be nonnegative")
    if classification_objective not in {
        "cross_entropy",
        "tversky",
        "cross_entropy+tversky",
    }:
        raise ValueError("unsupported classification_objective")
    if not math.isfinite(classification_loss_weight) or classification_loss_weight <= 0:
        raise ValueError("classification_loss_weight must be finite and positive")
    if not isinstance(use_uncertainty_weighting, bool):
        raise ValueError("use_uncertainty_weighting must be boolean")
