"""Pixel-classification arithmetic shared by instance and dense supervision."""

from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import ClassificationObjective


def classification_field_loss(
    logits: Tensor,
    target: Tensor,
    support: Tensor,
    *,
    objective: ClassificationObjective,
    class_weights: Tensor | None,
    weight_tversky_by_class: bool,
    alpha: float,
    gamma: float,
) -> dict[str, Tensor]:
    """Reduce the selected objectives over one field's supervised real pixels.

    CE retains weighted-mean semantics. Focal Tversky averages over all neural
    channels, including background; class weights apply after the focal power.
    The caller validates tensor shapes, channel indices and nonempty support.
    """

    valid = support.unsqueeze(0)
    safe_target = torch.where(valid, target, torch.zeros_like(target)).long()
    result: dict[str, Tensor] = {}
    if objective in {"cross_entropy", "cross_entropy+tversky"}:
        valid_logits = logits.permute(0, 2, 3, 1)[valid]
        valid_targets = safe_target[valid]
        cross_entropy = F.cross_entropy(
            valid_logits,
            valid_targets,
            weight=class_weights,
            reduction="mean",
        )
        result["classification_cross_entropy"] = cross_entropy
    if objective in {"tversky", "cross_entropy+tversky"}:
        tversky = _masked_tversky(
            logits,
            safe_target,
            valid,
            class_weights=class_weights if weight_tversky_by_class else None,
            alpha=alpha,
            gamma=gamma,
        )
        result["classification_tversky"] = tversky
    return result


def _masked_tversky(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    class_weights: Tensor | None,
    alpha: float,
    gamma: float,
) -> Tensor:
    """Compute focal Tversky over supported pixels, independently per class."""

    beta = 1.0 - alpha
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
    mask = valid.unsqueeze(1).to(logits.dtype)
    probabilities = probabilities * mask
    one_hot = one_hot * mask
    true_positive = (probabilities * one_hot).sum(dim=(2, 3))
    false_positive = (probabilities * (1.0 - one_hot) * mask).sum(dim=(2, 3))
    false_negative = ((1.0 - probabilities) * one_hot).sum(dim=(2, 3))
    denominator = true_positive + alpha * false_positive + beta * false_negative
    score = true_positive / denominator.clamp_min(torch.finfo(logits.dtype).eps)
    losses = (1.0 - score).clamp(1.0e-6, 1.0 - 1.0e-6).pow(1.0 / gamma)
    if class_weights is not None:
        losses = losses * class_weights
    return losses.mean()


def validate_classification_loss_configuration(
    class_count: int,
    classification_objective: str,
    classification_loss_weight: float,
    class_weights: Tensor | None,
    weight_tversky_by_class: bool,
    alpha: float,
    gamma: float,
) -> None:
    """Validate shared class-loss settings before model execution."""

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
    if not isinstance(weight_tversky_by_class, bool):
        raise ValueError("weight_tversky_by_class must be boolean")
    if class_count == 0 and class_weights is not None:
        raise ValueError("class_weights require a nonzero class_count")
    if class_weights is not None and tuple(class_weights.shape) != (class_count,):
        raise ValueError("class_weights must contain one value per class channel")
    if class_weights is not None and (
        not torch.all(torch.isfinite(class_weights)) or torch.any(class_weights <= 0)
    ):
        raise ValueError("class_weights must contain finite positive values")
    if not 0.0 < alpha < 1.0:
        raise ValueError("tversky_alpha must be between zero and one")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("tversky_gamma must be finite and positive")
