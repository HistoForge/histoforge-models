"""Differential parity: our centroid matcher vs the OFFICIAL PUMA challenge code.

This is the gate that licenses publishing a PUMA classification number. It drives BOTH
implementations from identical point sets, bypassing the polygon/JSON layer (where format
conventions, not metric semantics, live) so that any disagreement is a real algorithmic
difference.

Requires a local clone:

    git clone https://github.com/tueimage/PUMA-challenge-eval-track2.git \\
        ~/papers/repos/PUMA-challenge-eval-track2

Skipped automatically if absent. Marked slow; run with:

    pytest tests/test_puma_official_parity.py --runslow -q

Known upstream bug, deliberately NOT worked around: the official code deletes a matched
prediction by centroid VALUE (`np.array_equal`), so with two predictions at exactly the
same centroid but different scores it can delete the wrong one. We delete by index. This
is the only configuration in which the two implementations disagree, it cannot arise from
rasterised instance masks, and it is impossible in our distance-only configuration (equal
scores make duplicates interchangeable) — `test_duplicate_centroids_distance_only` pins
that immunity.
"""
import os
import sys

import numpy as np
import pytest


from histoforge_models.pathology.training.cls_metrics import count_image, match_centroids

OFFICIAL = os.path.expanduser("~/papers/repos/PUMA-challenge-eval-track2")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.isdir(OFFICIAL),
                       reason=f"official PUMA eval not cloned at {OFFICIAL}"),
]

NAMES = ["stroma", "tumor", "plasma", "histiocyte", "lymphocyte",
         "melanophage", "neutrophil", "endothelium", "epithelium", "apoptosis"]


def _official_mod():
    if OFFICIAL not in sys.path:
        sys.path.insert(0, OFFICIAL)
    import eval_nuclei
    return eval_nuclei


def _to_official(xy, cls, score=None):
    return [{"filename": "f.json", "category": NAMES[int(c)],
             "centroid": [float(p[0]), float(p[1])],
             "score": 1 if score is None else float(score[k])}
            for k, (p, c) in enumerate(zip(xy, cls))]


def _official_counts(gt_xy, gt_cls, pr_xy, pr_cls, score, n_fg=10):
    ev = _official_mod()
    g, p = _to_official(gt_xy, gt_cls, None), _to_official(pr_xy, pr_cls, score)
    m = ev.calculate_classification_metrics(ev.calculate_centroid_distance(g, p), g, p)
    tp = np.zeros(n_fg, np.int64)
    fp = np.zeros(n_fg, np.int64)
    fn = np.zeros(n_fg, np.int64)
    for c, nm in enumerate(NAMES):
        if nm in m:
            tp[c], fp[c], fn[c] = m[nm]["TP"], m[nm]["FP"], m[nm]["FN"]
    return tp, fp, fn


def _our_counts(gt_xy, gt_cls, pr_xy, pr_cls, score, n_fg=10):
    m = match_centroids(gt_xy, gt_cls, pr_xy, pr_cls, radius=15.0,
                        pred_score=score, class_aware=True)
    return count_image(gt_cls, pr_cls, m, n_fg)


def _assert_parity(gen, n_cases, seed, use_scores=False):
    rng = np.random.default_rng(seed)
    for t in range(n_cases):
        gt_xy, gt_cls, pr_xy, pr_cls = gen(rng)
        sc = rng.random(len(pr_xy)) if use_scores else None
        a = _official_counts(gt_xy, gt_cls, pr_xy, pr_cls, sc)
        b = _our_counts(gt_xy, gt_cls, pr_xy, pr_cls, sc)
        for name, x, y in zip(("tp", "fp", "fn"), a, b):
            assert np.array_equal(x, y), (
                f"case {t}: {name} differs\n official={x.tolist()}\n ours    ={y.tolist()}\n"
                f" gt={np.asarray(gt_xy).tolist()} {np.asarray(gt_cls).tolist()}\n"
                f" pr={np.asarray(pr_xy).tolist()} {np.asarray(pr_cls).tolist()}")


def _gen_uniform(rng):
    ng, npd = int(rng.integers(0, 40)), int(rng.integers(0, 40))
    return (rng.uniform(0, 300, (ng, 2)), rng.integers(0, 10, ng),
            rng.uniform(0, 300, (npd, 2)), rng.integers(0, 10, npd))


def _gen_dense(rng):
    """Tight packing so the radius creates constant ambiguity."""
    ng, npd = int(rng.integers(5, 60)), int(rng.integers(5, 60))
    return (rng.uniform(0, 60, (ng, 2)), rng.integers(0, 3, ng),
            rng.uniform(0, 60, (npd, 2)), rng.integers(0, 3, npd))


def _gen_lattice(rng):
    """Integer lattice so exact-15.0 distances occur often (probes the strict `<`)."""
    step = int(rng.choice([3, 4, 5, 9, 12, 15]))
    n = int(rng.integers(3, 8))
    gx = np.array([[i * step, j * step] for i in range(n) for j in range(n)], float)
    px = gx + rng.choice([0, step], size=gx.shape)
    return gx, rng.integers(0, 3, len(gx)), px, rng.integers(0, 3, len(px))


def _gen_realistic(rng):
    """Model-like: jittered GT, some missed, some misclassified, some spurious."""
    n = int(rng.integers(50, 200))
    gt = rng.uniform(0, 512, (n, 2))
    cg = rng.integers(0, 10, n)
    keep = rng.random(n) > 0.2
    pr = gt[keep] + rng.normal(0, 5, (keep.sum(), 2))
    cp = cg[keep].copy()
    flip = rng.random(len(cp)) < 0.3
    cp[flip] = rng.integers(0, 10, flip.sum())
    extra = rng.uniform(0, 512, (int(rng.integers(0, 40)), 2))
    if len(extra):
        pr = np.vstack([pr, extra])
        cp = np.concatenate([cp, rng.integers(0, 10, len(extra))])
    return gt, cg, pr, cp


def _gen_duplicate_centroids(rng):
    pts = rng.uniform(0, 40, (int(rng.integers(1, 4)), 2))
    pr = np.repeat(pts, 2, axis=0)
    gt = rng.uniform(0, 40, (int(rng.integers(1, 6)), 2))
    return gt, rng.integers(0, 2, len(gt)), pr, rng.integers(0, 2, len(pr))


class TestOfficialParity:

    def test_uniform_sparse(self):
        _assert_parity(_gen_uniform, 500, seed=0)

    def test_dense_constant_ambiguity(self):
        _assert_parity(_gen_dense, 500, seed=1)

    def test_integer_lattice_exact_threshold(self):
        _assert_parity(_gen_lattice, 300, seed=2)

    def test_realistic_model_like(self):
        _assert_parity(_gen_realistic, 200, seed=3)

    def test_with_real_scores(self):
        _assert_parity(_gen_uniform, 500, seed=4, use_scores=True)

    def test_duplicate_centroids_distance_only(self):
        """Our operating configuration is immune to the upstream delete-by-value bug:
        with equal scores, coincident predictions are interchangeable."""
        _assert_parity(_gen_duplicate_centroids, 500, seed=5, use_scores=False)
