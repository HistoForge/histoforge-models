"""One Cellpose mask evaluator for flow and HoVer standalone training."""

import numpy as np

from histoforge_models import metrics
from histoforge_models.pathology.metric_aggregation import (
    DEFAULT_SEGMENTATION_AGGREGATION,
    SEGMENTATION_AGGREGATIONS,
    aggregate_segmentation_counts,
    mean_defined,
    validate_segmentation_aggregation,
)

# Preserve the standalone Cellpose evaluation threshold grid.
THRESHOLDS = list(np.arange(0.5, 1.0, 0.05))


def evaluate_instance_masks(ground_truth, prediction,
                            aggregation=DEFAULT_SEGMENTATION_AGGREGATION):
    """Match once per threshold; report both reductions from those same counts.

    Unqualified AP/mSA keys follow the selected mode. The upstream per-image AP
    array is retained unchanged, and count arrays support dataset-group summaries
    without repeating inference or matching. Undefined scalar scores are NaN.
    """
    validate_segmentation_aggregation(aggregation)
    if len(ground_truth) != len(prediction) or not len(ground_truth):
        raise ValueError("evaluation requires equally sized nonempty mask lists")
    # Cellpose's empty-empty AP is intentionally 0/0. Suppress that warning only;
    # count validation below still rejects unexpected nonfinite counts.
    with np.errstate(invalid="ignore"):
        ap, tp, fp, fn = metrics.average_precision(
            ground_truth, prediction, threshold=THRESHOLDS)
    result = {"ap_array": ap, "tp": tp, "fp": fp, "fn": fn,
              "segmentation_metric_aggregation": aggregation}
    for mode in SEGMENTATION_AGGREGATIONS:
        scores = aggregate_segmentation_counts(tp, fp, fn, mode)
        summary = {
            "AP@0.5": float(scores[0]),
            "AP@0.75": float(scores[5]),
            "AP@0.9": float(scores[8]),
            "mSA": float(mean_defined(scores)),
            "SA50": float(scores[0]),
            "SA75": float(scores[5]),
        }
        result.update({f"{key}_{mode}": value for key, value in summary.items()})
        if mode == aggregation:
            result.update(summary)
    return result
