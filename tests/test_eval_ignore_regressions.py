"""Synthetic regressions for empty matching and ignored classification labels.

  1. evaluate_semantic: empty float64 index array when an image has GT and predictions
     but zero IoU>0.5 pairs.
  2. evaluate_semantic: a prediction matched to an ignore-labelled GT still counted as FP.
  3. evaluate_classification: fn[254] indexed into a num_classes-length array.

Per-instance class 254 becomes 255 in the background-offset pixel class map.
"""

import numpy as np


from histoforge_models.pathology.training.eval import evaluate_classification, evaluate_semantic

IGNORE = 254


def _blob(arr, r0, c0, val, size=6):
    arr[r0:r0 + size, c0:c0 + size] = val


# ===========================================================================
# evaluate_semantic
# ===========================================================================

class TestEvaluateSemanticRegressions:

    def test_zero_iou_matches_does_not_crash(self):
        """GT top-left, prediction bottom-right: no pair clears IoU>0.5."""
        mt = np.zeros((40, 40), np.int32)
        mp = np.zeros((40, 40), np.int32)
        ct = np.zeros((40, 40), np.uint8)
        cp = np.zeros((40, 40), np.uint8)
        _blob(mt, 2, 2, 1)
        _blob(ct, 2, 2, 1)
        _blob(mp, 30, 30, 1)
        _blob(cp, 30, 30, 1)

        res = evaluate_semantic([mp], [cp], [mt], [ct], nclasses=3)
        assert res["tp"].sum() == 0
        assert res["fn"][0, 0] == 1          # the GT was missed
        assert res["fp"][0, 0] == 1          # the prediction is spurious

    def test_prediction_on_ignore_gt_is_not_a_false_positive(self):
        """An ignore GT (254 -> class map 255) plus a prediction squarely on it.

        The prediction must be absorbed, not charged: tp = fp = fn = 0.
        """
        mt = np.zeros((40, 40), np.int32)
        mp = np.zeros((40, 40), np.int32)
        ct = np.zeros((40, 40), np.uint8)
        cp = np.zeros((40, 40), np.uint8)
        _blob(mt, 10, 10, 1)
        _blob(ct, 10, 10, IGNORE + 1)        # build_gt_class_map stores cls+1 -> 255
        _blob(mp, 10, 10, 1)
        _blob(cp, 10, 10, 1)                 # predicted as foreground class 0

        res = evaluate_semantic([mp], [cp], [mt], [ct], nclasses=3)
        assert res["tp"].sum() == 0
        assert res["fn"].sum() == 0
        assert res["fp"].sum() == 0, "prediction on an ignore nucleus must not be an FP"

    def test_real_nucleus_still_scored_alongside_an_ignore_one(self):
        """Censoring ignore must not suppress scoring of genuine nuclei."""
        mt = np.zeros((60, 60), np.int32)
        mp = np.zeros((60, 60), np.int32)
        ct = np.zeros((60, 60), np.uint8)
        cp = np.zeros((60, 60), np.uint8)
        _blob(mt, 5, 5, 1)
        _blob(ct, 5, 5, 1)                   # real, class 0
        _blob(mt, 40, 40, 2)
        _blob(ct, 40, 40, IGNORE + 1)        # ignore
        _blob(mp, 5, 5, 1)
        _blob(cp, 5, 5, 1)
        _blob(mp, 40, 40, 2)
        _blob(cp, 40, 40, 1)

        res = evaluate_semantic([mp], [cp], [mt], [ct], nclasses=3)
        assert res["tp"][0, 0] == 1
        assert res["fp"].sum() == 0
        assert res["fn"].sum() == 0


# ===========================================================================
# evaluate_classification
# ===========================================================================

class TestEvaluateClassificationRegressions:

    def test_ignore_class_does_not_raise(self):
        """Previously: IndexError, index 254 out of bounds for axis 0 with size 2."""
        mt = np.zeros((40, 40), np.int32)
        mp = np.zeros((40, 40), np.int32)
        _blob(mt, 10, 10, 1)
        _blob(mp, 10, 10, 1)

        res = evaluate_classification([mp], [{"1": 0}], [mt], [{"1": IGNORE}],
                                      num_classes=2)
        assert res["per_class_tp"].sum() == 0
        assert res["per_class_fn"].sum() == 0
        assert res["per_class_fp"].sum() == 0, "ignore-matched pred must not be an FP"

    def test_ignore_alongside_a_real_instance(self):
        mt = np.zeros((60, 60), np.int32)
        mp = np.zeros((60, 60), np.int32)
        _blob(mt, 5, 5, 1)
        _blob(mt, 40, 40, 2)
        _blob(mp, 5, 5, 1)
        _blob(mp, 40, 40, 2)

        res = evaluate_classification([mp], [{"1": 0, "2": 0}], [mt],
                                      [{"1": 0, "2": IGNORE}], num_classes=2)
        assert res["per_class_tp"][0] == 1
        assert res["per_class_fp"].sum() == 0
        assert res["per_class_fn"].sum() == 0

    def test_ignore_in_the_empty_prediction_branch(self):
        """n_pred == 0 path must also skip ignore rather than indexing fn[254]."""
        mt = np.zeros((40, 40), np.int32)
        _blob(mt, 10, 10, 1)
        res = evaluate_classification([np.zeros((40, 40), np.int32)], [{}], [mt],
                                      [{"1": IGNORE}], num_classes=2)
        assert res["per_class_fn"].sum() == 0

    def test_normal_case_unchanged(self):
        """Ignore handling must not change scoring when every class is valid."""
        mt = np.zeros((60, 60), np.int32)
        mp = np.zeros((60, 60), np.int32)
        _blob(mt, 5, 5, 1)
        _blob(mt, 40, 40, 2)
        _blob(mp, 5, 5, 1)
        _blob(mp, 40, 40, 2)

        res = evaluate_classification([mp], [{"1": 0, "2": 1}], [mt],
                                      [{"1": 0, "2": 1}], num_classes=2)
        assert res["per_class_tp"].tolist() == [1, 1]
        assert res["per_class_fp"].sum() == 0
        assert res["per_class_fn"].sum() == 0
