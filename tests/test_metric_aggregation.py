"""Independent count oracles; no model, dataset, or optional evaluator required."""

import numpy as np
import pytest

from histoforge_models.pathology.metric_aggregation import (
    aggregate_segmentation_counts,
    class_f1,
    mean_defined,
    segmentation_accuracy,
)


def test_unequal_populations_distinguish_pooling_from_image_average():
    tp, fp, fn = [[1], [0]], [[0], [0]], [[0], [9]]
    np.testing.assert_allclose(aggregate_segmentation_counts(tp, fp, fn), [0.1])
    np.testing.assert_allclose(
        aggregate_segmentation_counts(tp, fp, fn, "per_image"), [0.5])


def test_pooling_is_invariant_to_order_and_count_partition():
    tp = np.array([[1, 0], [4, 3], [0, 0]])
    fp = np.array([[2, 3], [1, 2], [5, 5]])
    fn = np.array([[0, 1], [3, 4], [0, 0]])
    expected = [5 / 16, 3 / 18]
    np.testing.assert_allclose(aggregate_segmentation_counts(tp, fp, fn), expected)
    np.testing.assert_allclose(
        aggregate_segmentation_counts(tp[::-1], fp[::-1], fn[::-1]), expected)
    np.testing.assert_allclose(aggregate_segmentation_counts(
        tp.sum(0, keepdims=True), fp.sum(0, keepdims=True), fn.sum(0, keepdims=True)),
        expected)


def test_undefined_is_zero_denominator_not_absent_ground_truth():
    # matched class, target-only class, prediction-only class, absent class
    result = class_f1([3, 0, 0, 0], [1, 0, 4, 0], [2, 4, 0, 0])
    np.testing.assert_allclose(result, [2 / 3, 0, 0, np.nan], equal_nan=True)
    assert float(mean_defined(result)) == pytest.approx(2 / 9)
    np.testing.assert_allclose(segmentation_accuracy([0, 0], [4, 0], [0, 0]),
                               [0, np.nan], equal_nan=True)


def test_all_empty_stays_undefined_without_runtime_warnings():
    with np.errstate(all="raise"):
        assert np.isnan(mean_defined(class_f1([0], [0], [0])))
        for mode in ("pooled", "per_image"):
            assert np.isnan(aggregate_segmentation_counts([[0]], [[0]], [[0]], mode)[0])
            np.testing.assert_array_equal(
                aggregate_segmentation_counts(np.zeros((0, 2)), np.zeros((0, 2)),
                                               np.zeros((0, 2)), mode), [np.nan, np.nan])


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, 0.5])
@pytest.mark.parametrize("reduce", [segmentation_accuracy, class_f1])
def test_invalid_counts_fail_instead_of_becoming_unavailable(bad, reduce):
    with pytest.raises(ValueError, match="integer counts"):
        reduce([bad], [0], [0])


def test_count_shape_and_aggregation_are_explicit():
    with pytest.raises(ValueError, match="same shape"):
        class_f1([1, 2], [0], [0])
    with pytest.raises(ValueError, match="image and threshold"):
        aggregate_segmentation_counts([1], [0], [0])
    with pytest.raises(ValueError, match="aggregation"):
        aggregate_segmentation_counts([[1]], [[0]], [[0]], "automatic")
    with pytest.raises(ValueError, match="infinite"):
        mean_defined([np.inf])
