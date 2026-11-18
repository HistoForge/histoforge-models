"""Model-independent centroid, classification and panoptic metrics.

Centroid matching follows the PUMA challenge's class-aware greedy rule: visit
ground-truth objects in order, choose an eligible prediction by descending score
then distance, and remove it from the pool. The default radius is 15 pixels;
the boundary is strict. Omitted scores all equal one. Raster-mask centroids use
centres of mass, not polygon-vertex averages.

The aggregations are distinct. ``official`` averages each class over all images,
including images without that class. ``pooled`` sums TP/FP/FN before computing
F1. Report them separately, not as interchangeable scores.

Per-instance class 254 means ignore. It contributes no false negative; a
remaining prediction may be censored by an ignored object only after ordinary
matching. Ignored objects therefore cannot take matches from scored objects.
"""

import fastremap
import numpy as np
from scipy.ndimage import center_of_mass

from histoforge_models import metrics as cp_metrics

IGNORE_ID = 254
DEFAULT_RADIUS_PX = 15.0


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def instance_centroids(mask, ids=None):
    """Centre-of-mass centroid per instance.

    Args:
        mask: (H, W) integer instance mask, 0 = background.
        ids: optional explicit instance ids; defaults to every nonzero label.

    Returns:
        (ids (N,) int64, centroids (N, 2) float64 as (y, x)).
    """
    if ids is None:
        ids = np.unique(mask)
        ids = ids[ids != 0]
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return ids, np.zeros((0, 2), dtype=np.float64)
    cen = center_of_mass(np.ones_like(mask, dtype=np.uint8), mask, ids)
    return ids, np.asarray(cen, dtype=np.float64).reshape(-1, 2)


def canonical_order(centroids):
    """Deterministic input order: lexicographic by (y, x).

    The official metric is order-dependent (feeding GT in a different order can change
    TP). Prediction instance ids come out of cellpose dynamics and are not a stable
    function of the point set, so we canonicalise before matching and apply the same rule
    to every method. Use `order=None` to preserve caller order for official parity tests.
    """
    if len(centroids) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.lexsort((centroids[:, 1], centroids[:, 0]))


# ---------------------------------------------------------------------------
# centroid matching — the official PUMA algorithm
# ---------------------------------------------------------------------------

def match_centroids(gt_xy, gt_cls, pred_xy, pred_cls, radius=DEFAULT_RADIUS_PX,
                    pred_score=None, class_aware=True):
    """Greedy centroid matching, transcribed from the official PUMA implementation.

    Iterates GT in the order given (canonicalise beforehand for reproducibility); for each
    GT collects eligible predictions with `distance < radius` (strict, matching the
    official `if distance < 15`), sorts by `(-score, distance)`, takes the best, and
    removes it from the pool so matches are unique.

    Args:
        gt_xy: (Ng, 2) GT centroids.
        gt_cls: (Ng,) GT class ids.
        pred_xy: (Np, 2) predicted centroids.
        pred_cls: (Np,) predicted class ids.
        radius: match radius in pixels (strict less-than).
        pred_score: optional (Np,) scores. The official code defaults these to 1, which
            makes the sort degenerate to nearest-first; pass None to reproduce that.
        class_aware: if True (the official behaviour) a GT only matches a prediction of
            the same class. If False, class is ignored — used for our class-agnostic
            detection-F1, which is NOT part of the official metric.

    Returns:
        List of (gt_index, pred_index) pairs, in GT iteration order.
    """
    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    gt_cls = np.asarray(gt_cls).reshape(-1)
    pred_cls = np.asarray(pred_cls).reshape(-1)
    if pred_score is None:
        pred_score = np.ones(len(pred_xy), dtype=np.float64)
    else:
        pred_score = np.asarray(pred_score, dtype=np.float64).reshape(-1)

    # bucket prediction indices by class (or one shared bucket when class-agnostic)
    pool = {}
    for j in range(len(pred_xy)):
        key = pred_cls[j] if class_aware else 0
        pool.setdefault(key, []).append(j)

    matches = []
    for i in range(len(gt_xy)):
        key = gt_cls[i] if class_aware else 0
        cand = pool.get(key)
        if not cand:
            continue
        eligible = []
        for j in cand:
            d = float(np.linalg.norm(gt_xy[i] - pred_xy[j]))
            if d < radius:                       # strict, per the official code
                eligible.append((-pred_score[j], d, j))
        if not eligible:
            continue
        eligible.sort(key=lambda t: (t[0], t[1]))
        j = eligible[0][2]
        matches.append((i, j))
        cand.remove(j)
    return matches


def censor_by_ignore(pred_xy, taken, ignore_xy, radius=DEFAULT_RADIUS_PX):
    """Absorb leftover predictions that landed on an ignore-labelled GT nucleus.

    Run AFTER the main matching pass so a real GT always gets first claim: otherwise an
    ignore nucleus sitting next to a real one could steal its prediction and turn a true
    positive into a false negative. Each ignore nucleus absorbs at most one prediction
    (1:1, greedy by distance), mirroring the matching rule.

    Args:
        pred_xy: (Np, 2) predicted centroids.
        taken: iterable of prediction indices already matched to real GT.
        ignore_xy: (Ni, 2) centroids of ignore-labelled GT.
        radius: same radius as matching.

    Returns:
        set of prediction indices to drop from the FP pool.
    """
    taken = set(taken)
    censored = set()
    if len(ignore_xy) == 0:
        return censored
    ignore_xy = np.asarray(ignore_xy, dtype=np.float64).reshape(-1, 2)
    pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    for k in range(len(ignore_xy)):
        best, best_d = None, radius
        for j in range(len(pred_xy)):
            if j in taken or j in censored:
                continue
            d = float(np.linalg.norm(ignore_xy[k] - pred_xy[j]))
            if d < best_d:
                best, best_d = j, d
        if best is not None:
            censored.add(best)
    return censored


# ---------------------------------------------------------------------------
# per-image counting
# ---------------------------------------------------------------------------

def count_image(gt_cls, pred_cls, matches, n_fg, censored=()):
    """Per-class TP/FP/FN for one image, following the official counting rule.

    The official code computes `FP_c = n_pred_c - TP_c` and `FN_c = n_gt_c - TP_c`, which
    is why a misclassified nucleus is charged twice. Reproduced exactly, with censored
    predictions removed from the prediction pool first.

    Returns:
        (tp, fp, fn) each (n_fg,) int64.
    """
    gt_cls = np.asarray(gt_cls).reshape(-1)
    pred_cls = np.asarray(pred_cls).reshape(-1)
    censored = set(censored)

    tp = np.zeros(n_fg, dtype=np.int64)
    for _, j in matches:
        c = int(pred_cls[j])
        if 0 <= c < n_fg:
            tp[c] += 1

    n_gt = np.zeros(n_fg, dtype=np.int64)
    for c in gt_cls:
        c = int(c)
        if 0 <= c < n_fg:                        # ignore ids fall outside and are dropped
            n_gt[c] += 1

    n_pred = np.zeros(n_fg, dtype=np.int64)
    for j, c in enumerate(pred_cls):
        if j in censored:
            continue
        c = int(c)
        if 0 <= c < n_fg:
            n_pred[c] += 1

    return tp, n_pred - tp, n_gt - tp


def detection_counts(n_gt, n_pred, n_matched, n_censored=0):
    """Class-agnostic detection TP/FP/FN.

    `count_image` buckets by predicted class, which is meaningless once matching ignored
    class, so class-agnostic detection needs its own tally. NOTE this metric is OUR
    addition — the official PUMA metric has no class-agnostic stage — and exists so we can
    report a det-F1 alongside a cls-F1 the way TAND does.
    """
    tp = int(n_matched)
    return tp, int(n_pred) - int(n_censored) - tp, int(n_gt) - tp


def prf(tp, fp, fn):
    """Precision, recall, F1 from counts. 0 where undefined (matches the official code)."""
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return p, r, f


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate_pooled(per_image):
    """Pooled aggregation: sum TP/FP/FN over images, then compute F1.

    ``macro_f1_present`` includes classes with ground-truth support;
    ``macro_f1_all`` includes every configured class.
    """
    tp = np.sum([x[0] for x in per_image], axis=0)
    fp = np.sum([x[1] for x in per_image], axis=0)
    fn = np.sum([x[2] for x in per_image], axis=0)
    p, r, f1 = prf(tp, fp, fn)
    T, F, N = tp.sum(), fp.sum(), fn.sum()
    mp = T / (T + F) if (T + F) > 0 else 0.0
    mr = T / (T + N) if (T + N) > 0 else 0.0
    micro = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
    support = tp + fn
    present = support > 0
    return {
        "per_class_tp": tp, "per_class_fp": fp, "per_class_fn": fn,
        "per_class_precision": p, "per_class_recall": r, "per_class_f1": f1,
        "support": support,
        "micro_f1": float(micro),
        "macro_f1_present": float(f1[present].mean()) if present.any() else 0.0,
        "macro_f1_all": float(f1.mean()),
    }


def aggregate_official(per_image, n_fg):
    """The challenge container's aggregation (`evaluate.py` main()).

    Per-image F1 per class, summed over images, divided by the TOTAL number of images, then
    averaged over classes. A class absent from an image contributes 0 to its numerator but
    still 1 to its denominator, so this is bounded above by
    mean_c(n_images_containing_c / n_images) — strictly below 1 whenever any class is
    missing from any image. That ceiling is returned too, because without it a score of
    ~0.5 reads as "half wrong" when much of the gap is structural.

    Note the official code computes a per-image micro F1 and then explicitly skips it, so
    there is no official aggregated micro. We do not invent one here.
    """
    n_images = len(per_image)
    if n_images == 0:
        return {"macro_f1_official": 0.0, "per_class_f1_official": np.zeros(n_fg),
                "ceiling_per_class": np.zeros(n_fg), "ceiling_macro": 0.0}
    acc = np.zeros(n_fg, dtype=np.float64)
    present_count = np.zeros(n_fg, dtype=np.int64)
    seen_count = np.zeros(n_fg, dtype=np.int64)
    for tp, fp, fn in per_image:
        _, _, f1 = prf(tp, fp, fn)
        # the official per-image class set is the union of GT and predicted classes
        in_image = (tp + fp + fn) > 0
        acc += np.where(in_image, f1, 0.0)
        present_count += ((tp + fn) > 0)
        seen_count += in_image
    per_class = acc / n_images
    ceiling = present_count / n_images

    # Same per-image averaging, but dividing by the images where the class actually
    # OCCURS instead of by all of them. This is what a correctly-written per-image macro
    # does, and its ceiling is 1.0. Reporting it beside the official number decomposes
    # the official score into (real model error) vs (the divide-by-all-images artefact):
    # the gap between the two is purely the artefact.
    per_class_present = np.divide(acc, seen_count, out=np.zeros(n_fg),
                                  where=seen_count > 0)
    any_seen = seen_count > 0
    return {
        "per_class_f1_official": per_class,
        "macro_f1_official": float(per_class.mean()),
        "per_class_f1_perimage_present": per_class_present,
        "macro_f1_perimage_present": (float(per_class_present[any_seen].mean())
                                      if any_seen.any() else 0.0),
        "ceiling_per_class": ceiling,
        "ceiling_macro": float(ceiling.mean()),
    }


# ---------------------------------------------------------------------------
# panoptic quality (IoU>0.5) — for comparability with CellVTA / Classpose / HoVer-Net
# ---------------------------------------------------------------------------

def pq_from_masks(masks_true, class_true, masks_pred, class_pred, n_fg,
                  match_iou=0.5):
    """Per-class PQ = DQ x SQ, pooled over images, matching HoVer-Net's definition.

    DQ = tp / (tp + 0.5 fp + 0.5 fn); SQ = mean IoU of matched pairs. This is the metric
    CellVTA and Classpose report, so it is what makes us legible to that audience.
    Only classes 0..n_fg-1 are scored. The caller must remove ignored instances
    and their censored predictions first; this function does not perform that step.
    """
    tp = np.zeros(n_fg, dtype=np.int64)
    fp = np.zeros(n_fg, dtype=np.int64)
    fn = np.zeros(n_fg, dtype=np.int64)
    iou_sum = np.zeros(n_fg, dtype=np.float64)

    for mt, ct, mp, cp in zip(masks_true, class_true, masks_pred, class_pred):
        for c in range(n_fg):
            gt_ids = [i for i, k in ct.items() if int(k) == c]
            pr_ids = [i for i, k in cp.items() if int(k) == c]
            if not gt_ids and not pr_ids:
                continue
            n_t, n_p = len(gt_ids), len(pr_ids)
            if n_t == 0:
                fp[c] += n_p
                continue
            if n_p == 0:
                fn[c] += n_t
                continue
            # Relabel to contiguous 1..N. Without this the IoU matrix is sized by max
            # label and every absent id becomes a zero-pixel phantom instance -> 0/0 nan.
            gt_c = fastremap.renumber(np.where(np.isin(mt, gt_ids), mt, 0))[0]
            pr_c = fastremap.renumber(np.where(np.isin(mp, pr_ids), mp, 0))[0]
            iou = cp_metrics._intersection_over_union(gt_c, pr_c)[1:, 1:]
            paired = iou > match_iou
            n_paired = int(paired.sum())
            tp[c] += n_paired
            fp[c] += n_p - n_paired
            fn[c] += n_t - n_paired
            iou_sum[c] += float(iou[paired].sum())

    dq = np.divide(tp, tp + 0.5 * fp + 0.5 * fn,
                   out=np.zeros(n_fg), where=(tp + 0.5 * fp + 0.5 * fn) > 0)
    sq = np.divide(iou_sum, tp, out=np.zeros(n_fg), where=tp > 0)
    return {"per_class_dq": dq, "per_class_sq": sq, "per_class_pq": dq * sq,
            "mDQ": float(dq.mean()), "mSQ": float(sq.mean()), "mPQ": float((dq * sq).mean()),
            "pq_tp": tp, "pq_fp": fp, "pq_fn": fn}


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
# A confusion matrix needs class-agnostic matching so wrong-class predictions
# remain paired. Its diagonal is not the class-aware true-positive count.
# For example, a class-0 object at x=0 and predictions (class 1, x=1) and
# (class 0, x=2) produce a 0->1 confusion here but a class-aware true positive.


def confusion_from_matches(gt_cls, pred_cls, matches, n_fg, censored=()):
    """Class-agnostic confusion matrix for one image.

    `matches` must come from match_centroids(..., class_aware=False) — with class-aware
    matching every off-diagonal cell is structurally empty.

    Returns (n_fg+1, n_fg+1) int64. Index n_fg is the MISSED row / SPURIOUS column:
        cm[g, p]      GT class g paired with a prediction of class p (g == p is correct)
        cm[g, n_fg]   GT class g with no prediction within the radius  -> missed
        cm[n_fg, p]   prediction of class p paired with no GT          -> spurious
        cm[n_fg,n_fg] always 0
    """
    gt_cls = np.asarray(gt_cls).reshape(-1)
    pred_cls = np.asarray(pred_cls).reshape(-1)
    censored = set(censored)
    cm = np.zeros((n_fg + 1, n_fg + 1), dtype=np.int64)

    matched_g, matched_p = set(), set()
    for gi, pj in matches:
        if pj in censored:
            continue            # paired to an ignore nucleus: censored, costs nothing
        g, p = int(gt_cls[gi]), int(pred_cls[pj])
        if 0 <= g < n_fg and 0 <= p < n_fg:
            cm[g, p] += 1
            matched_g.add(gi)
            matched_p.add(pj)

    for gi in range(len(gt_cls)):
        g = int(gt_cls[gi])
        if gi not in matched_g and 0 <= g < n_fg:
            cm[g, n_fg] += 1
    for pj in range(len(pred_cls)):
        p = int(pred_cls[pj])
        if pj not in matched_p and pj not in censored and 0 <= p < n_fg:
            cm[n_fg, p] += 1
    return cm
