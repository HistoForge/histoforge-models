"""End-to-end integration tests for scripts/score_cls.py.

Unit tests cover the matcher; the parity suite covers agreement with the official PUMA
code. This file covers the layer between them: reading canonical directories, mapping
instance ids to centroids and classes, censoring the ignore class, and aggregating —
against analytically computed expectations.

Builds synthetic datasets on disk (no model, no GPU, no download) so it is portable. Two
properties are checked that self-scoring alone cannot show:
  * a DEGRADED prediction produces exactly the intended TP/FP/FN
  * ignore instances are censored rather than charged, at a realistic 30%+ rate

Nuclei are placed far apart (>2x the match radius) so every prediction can only match its
own ground-truth nucleus. That isolates directory/id/aggregation correctness from the
metric's genuine neighbour-rescue behaviour, where a misclassified prediction can be
matched by a same-class nucleus within the radius. That behaviour is inherited from the
official implementation and is covered by tests/test_puma_official_parity.py.
"""
import json
import os

import numpy as np
import pytest
import tifffile


import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from score_cls import score  # noqa: E402

IGNORE = 254
STEP = 60          # >> 2 * 15px radius, so nuclei never compete
SIZE = 8


def build_pair(tmp, n_images, per_image, n_fg, rng, p_drop=0.2, p_mis=0.3, p_ignore=0.3):
    """Create GT and a degraded prediction; return (gt_dir, pred_dir, expected counts)."""
    gt_dir, pr_dir = f"{tmp}/gt", f"{tmp}/pred"
    for d in (gt_dir, pr_dir):
        os.makedirs(f"{d}/instance_masks", exist_ok=True)
        os.makedirs(f"{d}/class_labels", exist_ok=True)

    tp = np.zeros(n_fg, np.int64)
    fp = np.zeros(n_fg, np.int64)
    fn = np.zeros(n_fg, np.int64)
    n_ignore = n_real = 0

    side = int(np.ceil(np.sqrt(per_image)))
    dim = (side + 1) * STEP

    for im in range(n_images):
        gt_mask = np.zeros((dim, dim), np.int32)
        pr_mask = np.zeros((dim, dim), np.int32)
        gt_cls, pr_cls = {}, {}
        for k in range(per_image):
            r = (k // side + 1) * STEP
            c = (k % side + 1) * STEP
            inst = k + 1
            gt_mask[r:r + SIZE, c:c + SIZE] = inst

            if rng.random() < p_ignore:
                n_ignore += 1
                gt_cls[str(inst)] = IGNORE
                # the model still detects it and assigns some real class
                pr_mask[r:r + SIZE, c:c + SIZE] = inst
                pr_cls[str(inst)] = int(rng.integers(0, n_fg))
                continue

            n_real += 1
            cls = int(rng.integers(0, n_fg))
            gt_cls[str(inst)] = cls
            u = rng.random()
            if u < p_drop:
                fn[cls] += 1                                  # not predicted at all
            elif u < p_drop + p_mis:
                wrong = int((cls + 1 + rng.integers(0, max(n_fg - 1, 1))) % n_fg)
                pr_mask[r:r + SIZE, c:c + SIZE] = inst
                pr_cls[str(inst)] = wrong
                if wrong == cls:
                    tp[cls] += 1
                else:
                    fn[cls] += 1
                    fp[wrong] += 1
            else:
                pr_mask[r:r + SIZE, c:c + SIZE] = inst
                pr_cls[str(inst)] = cls
                tp[cls] += 1

        stem = f"img{im:03d}"
        tifffile.imwrite(f"{gt_dir}/instance_masks/{stem}.tif", gt_mask.astype(np.uint16))
        tifffile.imwrite(f"{pr_dir}/instance_masks/{stem}.tif", pr_mask.astype(np.uint16))
        json.dump(gt_cls, open(f"{gt_dir}/class_labels/{stem}.json", "w"))
        json.dump(pr_cls, open(f"{pr_dir}/class_labels/{stem}.json", "w"))

    return gt_dir, pr_dir, tp, fp, fn, n_ignore, n_real


def _expected(tp, fp, fn):
    p = np.divide(tp, tp + fp, out=np.zeros(len(tp), float), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros(len(tp), float), where=(tp + fn) > 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros(len(tp), float), where=(p + r) > 0)
    sup = tp + fn
    macro = f1[sup > 0].mean() if (sup > 0).any() else 0.0
    T, F, N = tp.sum(), fp.sum(), fn.sum()
    mp_ = T / (T + F) if (T + F) else 0.0
    mr_ = T / (T + N) if (T + N) else 0.0
    micro = 2 * mp_ * mr_ / (mp_ + mr_) if (mp_ + mr_) else 0.0
    return f1, macro, micro, sup


class TestScoreClsIntegration:

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_degraded_prediction_matches_analytic_expectation(self, tmp_path, seed):
        n_fg = 5
        rng = np.random.default_rng(seed)
        gt, pr, tp, fp, fn, n_ign, n_real = build_pair(
            str(tmp_path), n_images=6, per_image=16, n_fg=n_fg, rng=rng)

        res, pc = score(gt, pr, n_fg, 15.0, verbose=False)
        f1, macro, micro, sup = _expected(tp, fp, fn)

        assert res["n_censored_pred"] == n_ign, "every ignore prediction must be censored"
        assert pc["support"] == sup.tolist()
        assert np.allclose(pc["f1_pooled"], f1)
        assert res["centroid/pooled_macro_f1_present"] == pytest.approx(macro)
        assert res["centroid/pooled_micro_f1"] == pytest.approx(micro)

        # detection is class-agnostic: a misclassified nucleus is still detected, and
        # every surviving prediction sits on its own GT, so precision is exactly 1
        det_tp = int(tp.sum() + fp.sum())
        assert res["det/recall"] == pytest.approx(det_tp / n_real)
        assert res["det/precision"] == pytest.approx(1.0)

    def test_perfect_prediction_is_perfect_and_hits_the_ceiling(self, tmp_path):
        n_fg = 4
        rng = np.random.default_rng(7)
        gt, pr, tp, fp, fn, n_ign, _ = build_pair(
            str(tmp_path), n_images=5, per_image=12, n_fg=n_fg, rng=rng,
            p_drop=0.0, p_mis=0.0, p_ignore=0.3)

        res, _ = score(gt, pr, n_fg, 15.0, verbose=False)
        assert res["centroid/pooled_micro_f1"] == pytest.approx(1.0)
        assert res["centroid/pooled_macro_f1_present"] == pytest.approx(1.0)
        assert res["det/f1"] == pytest.approx(1.0)
        assert res["pq/mPQ"] == pytest.approx(1.0)
        assert res["seg/mSA"] == pytest.approx(1.0)
        assert res["n_censored_pred"] == n_ign
        # the invariant: a flawless model attains exactly the official metric's ceiling
        assert res["centroid/official_macro_f1"] == pytest.approx(
            res["centroid/official_ceiling"])

    def test_ignore_predictions_are_not_charged_as_false_positives(self, tmp_path):
        """Same data scored twice: once with the ignore GT present, once with those
        nuclei removed from BOTH sides. Scores must be identical -- i.e. ignore truly
        contributes nothing."""
        n_fg = 4
        rng = np.random.default_rng(11)
        gt, pr, *_ = build_pair(str(tmp_path), n_images=4, per_image=16, n_fg=n_fg,
                                rng=rng, p_ignore=0.4)
        with_ignore, _ = score(gt, pr, n_fg, 15.0, verbose=False)

        # strip ignore instances from GT and their predictions from the prediction
        stripped = f"{tmp_path}/gt2", f"{tmp_path}/pred2"
        for d in stripped:
            os.makedirs(f"{d}/instance_masks", exist_ok=True)
            os.makedirs(f"{d}/class_labels", exist_ok=True)
        for stem_file in sorted(os.listdir(f"{gt}/instance_masks")):
            stem = os.path.splitext(stem_file)[0]
            gm = tifffile.imread(f"{gt}/instance_masks/{stem}.tif")
            pm = tifffile.imread(f"{pr}/instance_masks/{stem}.tif")
            gc = json.load(open(f"{gt}/class_labels/{stem}.json"))
            pc_ = json.load(open(f"{pr}/class_labels/{stem}.json"))
            ign = [int(k) for k, v in gc.items() if int(v) == IGNORE]
            tifffile.imwrite(f"{stripped[0]}/instance_masks/{stem}.tif",
                             np.where(np.isin(gm, ign), 0, gm).astype(np.uint16))
            tifffile.imwrite(f"{stripped[1]}/instance_masks/{stem}.tif",
                             np.where(np.isin(pm, ign), 0, pm).astype(np.uint16))
            json.dump({k: v for k, v in gc.items() if int(v) != IGNORE},
                      open(f"{stripped[0]}/class_labels/{stem}.json", "w"))
            json.dump({k: v for k, v in pc_.items() if int(k) not in ign},
                      open(f"{stripped[1]}/class_labels/{stem}.json", "w"))

        without, _ = score(stripped[0], stripped[1], n_fg, 15.0, verbose=False)
        for key in ("centroid/pooled_macro_f1_present", "centroid/pooled_micro_f1",
                    "det/f1", "pq/mPQ", "seg/mSA"):
            assert with_ignore[key] == pytest.approx(without[key]), key
