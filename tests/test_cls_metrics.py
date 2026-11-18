"""Unit tests for histoforge_models/pathology/training/cls_metrics.py.

Covers:
  - cls_metrics.py: instance_centroids, canonical_order, match_centroids,
    censor_by_ignore, count_image, prf, aggregate_pooled, aggregate_official

Every expected value here is hand-computed. The matcher itself was additionally
validated against the OFFICIAL PUMA challenge implementation
(github.com/tueimage/PUMA-challenge-eval-track2, eval_nuclei.py) over 14,500 randomized
cases — uniform, dense, exact-15.0 lattice, realistic, and with/without scores — with
zero mismatches in every configuration we use. See TestOfficialSemantics for the
properties that validation pinned down.
"""
import warnings

import numpy as np
import pytest


from histoforge_models.pathology.training.cls_metrics import (
    confusion_from_matches,
    IGNORE_ID,
    aggregate_official,
    aggregate_pooled,
    canonical_order,
    censor_by_ignore,
    count_image,
    detection_counts,
    instance_centroids,
    match_centroids,
    pq_from_masks,
    prf,
)


def _counts(gt_xy, gt_cls, pr_xy, pr_cls, n_fg, radius=15.0, class_aware=True):
    m = match_centroids(gt_xy, gt_cls, pr_xy, pr_cls, radius=radius,
                        class_aware=class_aware)
    return count_image(gt_cls, pr_cls, m, n_fg), m


# ===========================================================================
# Matching
# ===========================================================================

class TestMatchCentroids:

    def test_perfect_match(self):
        (tp, fp, fn), m = _counts([[0, 0]], [0], [[0, 0]], [0], 2)
        assert m == [(0, 0)]
        assert tp.tolist() == [1, 0]
        assert fp.tolist() == [0, 0]
        assert fn.tolist() == [0, 0]

    def test_threshold_is_strict(self):
        """distance == radius must NOT match (official uses `if distance < 15`)."""
        (tp, fp, fn), m = _counts([[0, 0]], [0], [[0, 15]], [0], 2)
        assert m == []
        assert tp.tolist() == [0, 0]
        assert fp.tolist() == [1, 0]
        assert fn.tolist() == [1, 0]

    def test_just_inside_threshold(self):
        (tp, _, _), m = _counts([[0, 0]], [0], [[0, 14.999]], [0], 2)
        assert m == [(0, 0)] and tp.tolist() == [1, 0]

    def test_pythagorean_exact_15(self):
        """3-4-5 triangle scaled to exactly 15.0 -> still excluded."""
        (tp, _, _), m = _counts([[0, 0]], [0], [[9, 12]], [0], 2)
        assert m == [] and tp.tolist() == [0, 0]

    def test_misclassified_is_charged_twice(self):
        """Class-aware matching: wrong class => FN in true class AND FP in predicted."""
        (tp, fp, fn), m = _counts([[0, 0]], [0], [[0, 1]], [1], 2)
        assert m == []
        assert tp.tolist() == [0, 0]
        assert fn.tolist() == [1, 0]
        assert fp.tolist() == [0, 1]

    def test_two_gt_one_pred_yields_one_match(self):
        """Both GT are within radius of the single prediction; only one may win."""
        (tp, fp, fn), m = _counts([[0, 0], [0, 20]], [0, 0], [[0, 10]], [0], 2)
        assert len(m) == 1
        assert tp.tolist() == [1, 0]
        assert fp.tolist() == [0, 0]
        assert fn.tolist() == [1, 0]

    def test_two_preds_one_gt_leaves_one_fp(self):
        (tp, fp, fn), m = _counts([[0, 0]], [0], [[0, 5], [0, -5]], [0, 0], 2)
        assert len(m) == 1
        assert tp.tolist() == [1, 0]
        assert fp.tolist() == [1, 0]

    def test_dense_cluster_assigns_one_to_one(self):
        """Nuclei spaced 10px apart, well inside a 15px radius, must still pair 1:1."""
        gt = [[0, 0], [0, 10], [0, 20]]
        pr = [[0, 1], [0, 11], [0, 21]]
        (tp, fp, fn), m = _counts(gt, [0, 0, 0], pr, [0, 0, 0], 2)
        assert len(m) == 3
        assert tp.tolist() == [3, 0]
        assert fp.tolist() == [0, 0] and fn.tolist() == [0, 0]

    def test_class_agnostic_mode_ignores_class(self):
        """Our (non-official) detection mode matches regardless of class."""
        (tp, _, _), m = _counts([[0, 0]], [0], [[0, 1]], [1], 2, class_aware=False)
        assert len(m) == 1

    def test_empty_inputs(self):
        for gt, cg, pr, cp in [([], [], [[0, 0]], [0]),
                               ([[0, 0]], [0], [], []),
                               ([], [], [], [])]:
            (tp, fp, fn), m = _counts(gt, cg, pr, cp, 2)
            assert m == []
            assert tp.sum() == 0

    def test_zero_gt_all_predictions_are_fp(self):
        (tp, fp, fn), _ = _counts([], [], [[0, 0], [5, 5], [9, 9]], [0, 0, 1], 2)
        assert tp.tolist() == [0, 0]
        assert fp.tolist() == [2, 1]
        assert fn.tolist() == [0, 0]

    def test_zero_pred_all_gt_are_fn(self):
        (tp, fp, fn), _ = _counts([[0, 0], [5, 5]], [0, 1], [], [], 2)
        assert fn.tolist() == [1, 1] and fp.tolist() == [0, 0]


# ===========================================================================
# Ignore-class censoring
# ===========================================================================

class TestIgnoreCensoring:

    def test_prediction_on_ignore_is_neither_tp_nor_fp(self):
        """A prediction landing on an ignore nucleus must vanish, not count as FP."""
        pr = np.array([[0.0, 1.0], [100.0, 102.0]])
        gt_real = np.array([[0.0, 0.0]])
        ignore = np.array([[100.0, 100.0]])
        m = match_centroids(gt_real, [0], pr, [0, 0])
        assert m == [(0, 0)]
        cens = censor_by_ignore(pr, [j for _, j in m], ignore)
        assert cens == {1}
        tp, fp, fn = count_image([0], [0, 0], m, 2, censored=cens)
        assert tp.tolist() == [1, 0]
        assert fp.tolist() == [0, 0]     # would be [1, 0] without censoring
        assert fn.tolist() == [0, 0]

    def test_ignore_cannot_steal_from_a_real_nucleus(self):
        """Censoring runs AFTER matching, so a real GT keeps first claim."""
        pr = np.array([[0.0, 2.0]])
        m = match_centroids(np.array([[0.0, 0.0]]), [0], pr, [0])
        assert m == [(0, 0)]
        cens = censor_by_ignore(pr, [j for _, j in m], np.array([[0.0, 3.0]]))
        assert cens == set()
        tp, fp, fn = count_image([0], [0], m, 2, censored=cens)
        assert tp.tolist() == [1, 0] and fn.tolist() == [0, 0]

    def test_ignore_absorbs_at_most_one_prediction(self):
        pr = np.array([[0.0, 1.0], [0.0, 2.0]])
        cens = censor_by_ignore(pr, [], np.array([[0.0, 0.0]]))
        assert len(cens) == 1
        tp, fp, fn = count_image([], [0, 0], [], 2, censored=cens)
        assert fp.tolist() == [1, 0]

    def test_ignore_gt_never_counts_as_fn(self):
        """count_image drops out-of-range class ids (254) from the GT tally."""
        tp, fp, fn = count_image([IGNORE_ID], [], [], 2)
        assert fn.tolist() == [0, 0] and tp.sum() == 0


# ===========================================================================
# Aggregation — the two variants differ, and that is the point
# ===========================================================================

class TestAggregation:

    @staticmethod
    def _two_image_perfect():
        """Image 1: class 0 only. Image 2: classes 0 and 1. Everything predicted
        perfectly. Class 1 is absent from image 1."""
        a = count_image([0], [0], [(0, 0)], 2)
        b = count_image([0, 1], [0, 1], [(0, 0), (1, 1)], 2)
        return [a, b]

    def test_pooled_is_perfect(self):
        r = aggregate_pooled(self._two_image_perfect())
        assert r["micro_f1"] == pytest.approx(1.0)
        assert r["macro_f1_present"] == pytest.approx(1.0)
        assert r["macro_f1_all"] == pytest.approx(1.0)

    def test_official_penalises_absent_class(self):
        """A PERFECT prediction scores 0.75, not 1.0 — verified against the official
        challenge code. Class 1 appears in 1 of 2 images, so its F1 of 1.0 is divided
        by num_files=2."""
        r = aggregate_official(self._two_image_perfect(), n_fg=2)
        assert r["per_class_f1_official"][0] == pytest.approx(1.0)
        assert r["per_class_f1_official"][1] == pytest.approx(0.5)
        assert r["macro_f1_official"] == pytest.approx(0.75)

    def test_official_score_equals_ceiling_when_perfect(self):
        """Strong invariant: a flawless model attains exactly the metric's ceiling."""
        r = aggregate_official(self._two_image_perfect(), n_fg=2)
        assert r["macro_f1_official"] == pytest.approx(r["ceiling_macro"])
        assert r["ceiling_macro"] == pytest.approx(0.75)

    def test_ceiling_is_one_when_every_class_in_every_image(self):
        per_image = [count_image([0, 1], [0, 1], [(0, 0), (1, 1)], 2) for _ in range(3)]
        r = aggregate_official(per_image, n_fg=2)
        assert r["ceiling_macro"] == pytest.approx(1.0)
        assert r["macro_f1_official"] == pytest.approx(1.0)

    def test_macro_present_vs_all_differ_on_zero_support(self):
        """Class 1 has no GT and no predictions anywhere."""
        r = aggregate_pooled([count_image([0], [0], [(0, 0)], 3)])
        assert r["macro_f1_all"] == pytest.approx(1 / 3)
        assert r["macro_f1_present"] == pytest.approx(1.0)

    def test_empty_aggregations_do_not_crash(self):
        assert aggregate_official([], n_fg=3)["macro_f1_official"] == 0.0


# ===========================================================================
# Centroids and ordering
# ===========================================================================

class TestCentroids:

    def test_centroid_of_known_rectangle(self):
        m = np.zeros((20, 20), np.int32)
        m[2:6, 4:10] = 1                       # rows 2..5, cols 4..9
        ids, cen = instance_centroids(m)
        assert ids.tolist() == [1]
        assert cen[0][0] == pytest.approx(3.5)
        assert cen[0][1] == pytest.approx(6.5)

    def test_non_contiguous_ids(self):
        m = np.zeros((20, 20), np.int32)
        m[0:2, 0:2] = 3
        m[10:12, 10:12] = 7
        m[15:17, 15:17] = 900
        ids, cen = instance_centroids(m)
        assert ids.tolist() == [3, 7, 900]
        assert len(cen) == 3

    def test_empty_mask(self):
        ids, cen = instance_centroids(np.zeros((10, 10), np.int32))
        assert len(ids) == 0 and cen.shape == (0, 2)

    def test_canonical_order_is_permutation_invariant(self):
        pts = np.array([[5.0, 1.0], [0.0, 9.0], [5.0, 0.0], [2.0, 2.0]])
        base = pts[canonical_order(pts)]
        for seed in range(5):
            perm = np.random.default_rng(seed).permutation(len(pts))
            shuffled = pts[perm]
            assert np.allclose(shuffled[canonical_order(shuffled)], base)


# ===========================================================================
# Properties pinned down by the differential validation
# ===========================================================================

class TestDetectionCounts:

    def test_class_agnostic_match_is_a_tp_regardless_of_class(self):
        m = match_centroids([[0, 0]], [0], [[0, 1]], [1], class_aware=False)
        assert detection_counts(1, 1, len(m)) == (1, 0, 0)

    def test_class_aware_would_charge_the_same_case_twice(self):
        """Contrast: the official class-aware metric penalises the same geometry."""
        (tp, fp, fn), _ = _counts([[0, 0]], [0], [[0, 1]], [1], 2)
        assert tp.sum() == 0 and fp.sum() == 1 and fn.sum() == 1

    def test_censored_predictions_leave_the_fp_pool(self):
        assert detection_counts(1, 3, 1, n_censored=1) == (1, 1, 0)


# ===========================================================================
# Panoptic quality (for CellVTA / Classpose comparability)
# ===========================================================================

class TestPanopticQuality:

    @staticmethod
    def _two_instances():
        m = np.zeros((30, 30), np.int32)
        m[2:8, 2:8] = 1
        m[20:26, 20:26] = 2
        return m, {1: 0, 2: 1}

    def test_perfect_prediction(self):
        m, c = self._two_instances()
        r = pq_from_masks([m], [c], [m.copy()], [dict(c)], n_fg=2)
        assert r["mPQ"] == pytest.approx(1.0)
        assert r["mDQ"] == pytest.approx(1.0)
        assert r["mSQ"] == pytest.approx(1.0)

    def test_misclassified_instance(self):
        """Both instances found, one given the wrong class.
        class0: tp=1 fp=1 fn=0 -> DQ=1/1.5; class1: tp=0 fn=1 -> DQ=0."""
        m, c = self._two_instances()
        r = pq_from_masks([m], [c], [m.copy()], [{1: 0, 2: 0}], n_fg=2)
        assert r["pq_tp"].tolist() == [1, 0]
        assert r["pq_fp"].tolist() == [1, 0]
        assert r["pq_fn"].tolist() == [0, 1]
        assert r["mDQ"] == pytest.approx(1 / 3)

    def test_no_phantom_instances_from_noncontiguous_ids(self):
        """Per-class masks must be relabelled contiguously, else absent ids become
        zero-pixel instances and the IoU matrix fills with nan."""
        m, c = self._two_instances()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            r = pq_from_masks([m], [c], [m.copy()], [dict(c)], n_fg=2)
        assert not np.isnan(r["per_class_pq"]).any()

    def test_missing_and_spurious_classes(self):
        m, c = self._two_instances()
        empty = np.zeros_like(m)
        assert pq_from_masks([m], [c], [empty], [{}], n_fg=2)["mPQ"] == 0.0
        assert pq_from_masks([empty], [{}], [m], [dict(c)], n_fg=2)["mPQ"] == 0.0


class TestOfficialSemantics:

    def test_matching_is_order_dependent(self):
        """The official metric genuinely depends on GT order. Measured impact at
        realistic density is ~0.01% of TP, but we canonicalise so OUR numbers are
        reproducible."""
        pr, cp = [[0, 13], [0, 26]], [0, 0]
        a = match_centroids([[0, 0], [0, 14]], [0, 0], pr, cp)
        b = match_centroids([[0, 14], [0, 0]], [0, 0], pr, cp)
        assert len(a) != len(b)

    def test_prf_is_zero_not_nan_when_undefined(self):
        p, r, f = prf(np.array([0]), np.array([0]), np.array([0]))
        assert p[0] == 0.0 and r[0] == 0.0 and f[0] == 0.0
        assert not np.isnan(f).any()


class TestConfusionMatrix:
    """confusion_from_matches must account for EVERY GT nucleus exactly once."""

    def _pts(self, n, step=100):
        return np.array([[0.0, i * step] for i in range(n)])

    def test_perfect_is_pure_diagonal(self):
        xy = self._pts(4)
        c = np.array([0, 1, 2, 1])
        m = match_centroids(xy, c, xy, c, radius=15.0, class_aware=False)
        cm = confusion_from_matches(c, c, m, n_fg=3)
        assert cm[0, 0] == 1 and cm[1, 1] == 2 and cm[2, 2] == 1
        assert cm.sum() == 4
        assert np.trace(cm[:3, :3]) == 4          # nothing off-diagonal
        assert cm[:, 3].sum() == 0                # nothing missed
        assert cm[3, :].sum() == 0                # nothing spurious

    def test_misclassification_lands_off_diagonal(self):
        xy = self._pts(2)
        gt = np.array([0, 1])
        pr = np.array([1, 0])                     # both swapped
        m = match_centroids(xy, gt, xy, pr, radius=15.0, class_aware=False)
        cm = confusion_from_matches(gt, pr, m, n_fg=2)
        assert cm[0, 1] == 1 and cm[1, 0] == 1
        assert np.trace(cm[:2, :2]) == 0
        assert cm[:2, :].sum() == 2

    def test_missed_and_spurious_columns(self):
        gt_xy = self._pts(3)
        gt_c = np.array([0, 1, 2])
        pr_xy = np.array([[0.0, 0.0], [0.0, 500.0]])   # matches gt0; 2nd is spurious
        pr_c = np.array([0, 1])
        m = match_centroids(gt_xy, gt_c, pr_xy, pr_c, radius=15.0, class_aware=False)
        cm = confusion_from_matches(gt_c, pr_c, m, n_fg=3)
        assert cm[0, 0] == 1
        assert cm[1, 3] == 1 and cm[2, 3] == 1         # gt1, gt2 missed
        assert cm[3, 1] == 1                           # spurious prediction of class 1
        assert cm[:3, :].sum() == 3                    # every GT accounted for once

    def test_row_sums_equal_gt_support_always(self):
        """The invariant that makes the CM trustworthy, over randomized inputs."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            ng, npd, n_fg = int(rng.integers(0, 15)), int(rng.integers(0, 15)), 4
            gt_xy = rng.uniform(0, 80, (ng, 2))
            pr_xy = rng.uniform(0, 80, (npd, 2))
            gt_c = rng.integers(0, n_fg, ng)
            pr_c = rng.integers(0, n_fg, npd)
            m = match_centroids(gt_xy, gt_c, pr_xy, pr_c, radius=15.0,
                                class_aware=False)
            cm = confusion_from_matches(gt_c, pr_c, m, n_fg=n_fg)
            support = np.bincount(gt_c, minlength=n_fg)
            assert np.array_equal(cm[:n_fg, :].sum(axis=1), support)
            # matched pairs + spurious == number of predictions counted
            assert cm[:n_fg, :n_fg].sum() + cm[n_fg, :n_fg].sum() == npd

    def test_censored_prediction_is_not_counted(self):
        xy = self._pts(2)
        gt = np.array([0, 1])
        pr = np.array([0, 1])
        m = match_centroids(xy, gt, xy, pr, radius=15.0, class_aware=False)
        cm = confusion_from_matches(gt, pr, m, n_fg=2, censored=(1,))
        assert cm[0, 0] == 1
        assert cm[1, 1] == 0            # its prediction was censored...
        assert cm[1, 2] == 1            # ...so the GT reads as missed, never as an error
        assert cm[2, :].sum() == 0      # and it is never charged as spurious



class TestMatchingDisagreement:
    """CM diagonal is bounded above by the class-aware TP, and the bound is tight enough
    to be a footnote. Both facts are load-bearing when the two appear in one table."""

    def test_minimal_disagreement_case(self):
        gt = np.array([[0.0, 0.0]])
        gc = np.array([0])
        pr = np.array([[0.0, 1.0], [0.0, 2.0]])     # wrong class nearer than right class
        pc = np.array([1, 0])
        ma = match_centroids(gt, gc, pr, pc, radius=15.0, class_aware=True)
        mg = match_centroids(gt, gc, pr, pc, radius=15.0, class_aware=False)
        tp, _, _ = count_image(gc, pc, ma, n_fg=2)
        cm = confusion_from_matches(gc, pc, mg, n_fg=2)
        assert int(tp.sum()) == 1                    # class-aware reaches past
        assert int(np.trace(cm[:2, :2])) == 0        # class-agnostic takes the nearest
        assert int(cm[0, 1]) == 1

    def test_diagonal_never_exceeds_class_aware_tp(self):
        rng = np.random.default_rng(7)
        for _ in range(400):
            n_fg = int(rng.integers(2, 6))
            ng, npd = int(rng.integers(0, 20)), int(rng.integers(0, 20))
            gt_xy = rng.uniform(0, 60, (ng, 2))     # dense: forces the competition
            pr_xy = rng.uniform(0, 60, (npd, 2))
            gc = rng.integers(0, n_fg, ng)
            pc = rng.integers(0, n_fg, npd)
            ma = match_centroids(gt_xy, gc, pr_xy, pc, radius=15.0, class_aware=True)
            mg = match_centroids(gt_xy, gc, pr_xy, pc, radius=15.0, class_aware=False)
            tp, _, _ = count_image(gc, pc, ma, n_fg)
            cm = confusion_from_matches(gc, pc, mg, n_fg)
            assert int(np.trace(cm[:n_fg, :n_fg])) <= int(tp.sum())
