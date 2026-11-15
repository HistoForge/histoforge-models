"""Count reductions shared by standalone and managed validation.

Callers own matching, supervised support, and the population represented by each
axis. No masks, model inference, class taxonomy, or checkpoint policy lives here.
NaN denotes only a zero denominator; invalid counts are errors.
"""

import numpy as np

SEGMENTATION_AGGREGATIONS = ("pooled", "per_image")
DEFAULT_SEGMENTATION_AGGREGATION = "pooled"


def validate_segmentation_aggregation(value):
    if value not in SEGMENTATION_AGGREGATIONS:
        raise ValueError("segmentation_metric_aggregation must be 'pooled' or 'per_image'")
    return value


def _counts(tp, fp, fn):
    counts = tuple(np.asarray(value, dtype=np.float64) for value in (tp, fp, fn))
    if not (counts[0].shape == counts[1].shape == counts[2].shape):
        raise ValueError("TP, FP and FN must have the same shape")
    for value in counts:
        if not np.all(np.isfinite(value) & (value >= 0) & (value == np.floor(value))):
            raise ValueError("TP, FP and FN must be finite nonnegative integer counts")
    return counts


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator,
                     out=np.full(denominator.shape, np.nan, dtype=np.float64),
                     where=denominator != 0)


def mean_defined(values, axis=None):
    """Mean of defined scores; all-undefined populations stay NaN, without warnings."""
    values = np.asarray(values, dtype=np.float64)
    if np.any(np.isinf(values)):
        raise ValueError("metric scores must not be infinite")
    defined = ~np.isnan(values)
    return _ratio(np.where(defined, values, 0).sum(axis=axis), defined.sum(axis=axis))


def segmentation_accuracy(tp, fp, fn):
    """Cellpose instance score TP/(TP+FP+FN), elementwise on count arrays."""
    tp, fp, fn = _counts(tp, fp, fn)
    return _ratio(tp, tp + fp + fn)


def aggregate_segmentation_counts(tp, fp, fn, aggregation=DEFAULT_SEGMENTATION_AGGREGATION):
    """Reduce [image, threshold] counts to one score per threshold.

    Pooled sums counts before division. Per-image gives each defined image equal
    weight; empty-empty images are excluded. Neither mode rematches objects.
    """
    validate_segmentation_aggregation(aggregation)
    tp, fp, fn = _counts(tp, fp, fn)
    if tp.ndim != 2:
        raise ValueError("segmentation counts must have image and threshold axes")
    if aggregation == "pooled":
        return segmentation_accuracy(tp.sum(axis=0), fp.sum(axis=0), fn.sum(axis=0))
    return mean_defined(segmentation_accuracy(tp, fp, fn), axis=0)


def class_f1(tp, fp, fn):
    """Per-class 2TP/(2TP+FP+FN); prediction-only or target-only support is zero."""
    tp, fp, fn = _counts(tp, fp, fn)
    return _ratio(2 * tp, 2 * tp + fp + fn)
