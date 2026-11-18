"""CenterNet-style modified focal loss over supervised real pixels."""

import torch
from torch import Tensor
from torch.nn import functional as F


def point_heatmap_loss(logits: Tensor, targets: Tensor, support_mask: Tensor) -> Tensor:
    """Return one loss per field, leaving batch/accumulation reduction to callers.

    logits and targets are BCHW with one independent channel per emitted class.
    support_mask is boolean BHW, intersecting reviewed and real-pixel support.
    Positive cells have target 1. Gaussian shoulders below 1 downweight nearby
    negatives by (1 - target)^4. Focal exponent is 2. A field's summed loss is
    divided by its number of centers, or 1 for a valid all-negative field.
    No padded/context pixel participates. Compute in float32, including when
    the preceding network forward used lower precision.
    """
    if (
        logits.ndim != 4
        or any(side == 0 for side in logits.shape)
        or not logits.is_floating_point()
        or targets.shape != logits.shape
        or targets.dtype != torch.float32
    ):
        raise ValueError(
            "logits must be floating-point BCHW with matching float32 targets"
        )
    if (
        support_mask.shape != (logits.shape[0], *logits.shape[2:])
        or support_mask.dtype != torch.bool
    ):
        raise ValueError("support_mask must be boolean BHW matching logits")
    if torch.any(~support_mask.flatten(1).any(dim=1)):
        raise ValueError("every field requires supervised real support")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("point losses require finite logits and targets")
    if torch.any((targets < 0) | (targets > 1)):
        raise ValueError("center targets must be in [0, 1]")

    prediction = logits.float()
    truth = targets.float()
    support = support_mask[:, None]
    # Zero excluded context before arithmetic: it must neither affect the loss
    # nor overflow an intermediate when it contains extreme finite values.
    prediction = torch.where(support, prediction, 0)
    truth = torch.where(support, truth, 0)
    positive = (truth == 1) & support
    negative = (truth < 1) & support
    probability = prediction.sigmoid()
    positive_loss = -F.logsigmoid(prediction) * (1 - probability).square()
    negative_loss = (
        -F.logsigmoid(-prediction) * probability.square() * (1 - truth).pow(4)
    )
    total = (positive_loss * positive + negative_loss * negative).sum(dim=(1, 2, 3))
    denominator = positive.sum(dim=(1, 2, 3)).clamp_min(1)
    result = total / denominator
    if not torch.isfinite(result).all():
        raise ValueError("point heatmap loss is nonfinite")
    return result
