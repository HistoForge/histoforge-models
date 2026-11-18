"""Score saved nucleus masks and classes without loading a model.

Stage 2 of the two-stage design. Reads a GT directory and a PREDICTION directory, both in
the input layout described in prod_docs/training.md:

    <root>/
      instance_masks/<stem>.tif     uint16, 0 = background, ids 1..N
      class_labels/<stem>.json      {"<instance_id>": <0-indexed class>}
      classes.json                  optional {"0": "name", ..., "254": "Ignore"}

Use the same scorer and conversion rules when comparing different methods.

Emits, per dataset:
  centroid/official_macro_f1   the PUMA leaderboard metric, plus its CEILING
  centroid/pooled_*            micro + macro over pooled TP/FP/FN
  det/*                        class-agnostic detection F1 (OURS, not official)
  pq/*                         mPQ/mDQ/mSQ for CellVTA / Classpose comparability
  seg/*                        AP@0.5, AP@0.75, mSA

Self-check without a model or GPU:

    python scripts/score_cls.py --gt <root>/val --pred <root>/val --nclasses 6

Usage:
    python scripts/score_cls.py --gt data/validation --pred predictions \\
        --nclasses 3 --label candidate
"""
import argparse
import glob
import json
import os
import sys

import fastremap
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from histoforge_models import metrics as cp_metrics
from histoforge_models.pathology.training.cls_metrics import (
    confusion_from_matches,
    DEFAULT_RADIUS_PX,
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

# PUMA's match radius is defined in pixels, not derived from calibration.
# Other datasets supply their own class count and radius through the CLI.
DATASETS = {
    "puma":   {"nclasses": 11, "radius_px": 15.0, "splits": ("train", "val", "test"),
               "headline": "official"},
}

# Public PUMA class order is a fallback when classes.json is absent.
FALLBACK_NAMES = {
    "puma": ["stroma", "tumor", "plasma", "histiocyte", "lymphocyte",
             "melanophage", "neutrophil", "endothelium", "epithelium", "apoptosis"],
}


def class_names(gt_root, n_fg, dataset=None):
    """classes.json next to class_labels/ -> registry fallback -> class_N."""
    cj = os.path.join(gt_root, "classes.json")
    if os.path.exists(cj):
        raw = json.load(open(cj))
        names = [raw.get(str(i), f"class_{i}") for i in range(n_fg)]
        return names
    if dataset in FALLBACK_NAMES and len(FALLBACK_NAMES[dataset]) == n_fg:
        return FALLBACK_NAMES[dataset]
    return [f"class_{i}" for i in range(n_fg)]


def load_dir(root):
    """{stem: (instance_mask, {inst_id: class})} for a canonical directory."""
    import tifffile
    out = {}
    for m in sorted(glob.glob(os.path.join(root, "instance_masks", "*"))):
        stem = os.path.splitext(os.path.basename(m))[0]
        j = os.path.join(root, "class_labels", f"{stem}.json")
        if not os.path.exists(j):
            raise FileNotFoundError(f"missing class labels for {stem}: {j}")
        mask = tifffile.imread(m).astype(np.int32)
        cls = {int(k): int(v) for k, v in json.load(open(j)).items()}
        out[stem] = (mask, cls)
    if not out:
        raise SystemExit(f"no instance_masks found under {root}")
    return out


def score(gt_root, pred_root, n_fg, radius, dataset=None, verbose=True,
          headline=None):
    gt = load_dir(gt_root)
    pr = load_dir(pred_root)
    missing = sorted(set(gt) - set(pr))
    if missing:
        raise SystemExit(f"{len(missing)} GT images have no prediction, e.g. {missing[:3]}")

    per_image = []
    det_tp = det_fp = det_fn = 0
    cm_total = np.zeros((n_fg + 1, n_fg + 1), dtype=np.int64)
    n_ignore_total = n_censored_total = 0
    masks_true, masks_pred, ct_all, cp_all = [], [], [], []
    dup_centroids = 0

    for stem in sorted(gt):
        mt, ct = gt[stem]
        mp, cp = pr[stem]
        if mt.shape != mp.shape:
            raise SystemExit(f"{stem}: shape mismatch GT {mt.shape} vs pred {mp.shape}")

        gt_ids, gt_xy = instance_centroids(mt)
        pr_ids, pr_xy = instance_centroids(mp)
        gt_cls = np.array([ct.get(int(i), IGNORE_ID) for i in gt_ids])
        pr_cls = np.array([cp.get(int(i), IGNORE_ID) for i in pr_ids])

        # coincident centroids are the one input that makes the official implementation
        # order-sensitive; count them so the report can assert they never occur
        if len(pr_xy):
            dup_centroids += len(pr_xy) - len(np.unique(pr_xy, axis=0))

        keep_gt = gt_cls != IGNORE_ID
        ign_xy = gt_xy[~keep_gt]
        n_ignore_total += int((~keep_gt).sum())

        # canonical order so results do not depend on instance-id assignment
        o_g = canonical_order(gt_xy[keep_gt])
        o_p = canonical_order(pr_xy)
        g_xy, g_cls = gt_xy[keep_gt][o_g], gt_cls[keep_gt][o_g]
        p_xy, p_cls = pr_xy[o_p], pr_cls[o_p]

        # --- class-aware (official) ---
        m = match_centroids(g_xy, g_cls, p_xy, p_cls, radius=radius, class_aware=True)
        cens = censor_by_ignore(p_xy, [j for _, j in m], ign_xy, radius=radius)
        n_censored_total += len(cens)
        per_image.append(count_image(g_cls, p_cls, m, n_fg, censored=cens))

        # --- class-agnostic detection (ours, not official) ---
        md = match_centroids(g_xy, g_cls, p_xy, p_cls, radius=radius, class_aware=False)
        cd = censor_by_ignore(p_xy, [j for _, j in md], ign_xy, radius=radius)
        a, b, c = detection_counts(len(g_xy), len(p_xy), len(md), len(cd))
        det_tp, det_fp, det_fn = det_tp + a, det_fp + b, det_fn + c

        # Confusion matrix rides on the class-agnostic pairing above: it is the only
        # matching that pairs a MISCLASSIFIED nucleus at all, so it is the only one that
        # can say what a class was confused WITH. Free — no extra matching pass.
        cm_total += confusion_from_matches(g_cls, p_cls, md, n_fg, censored=cd)

        # --- seg / PQ inputs ---
        # Ignore is don't-care for EVERY metric, so drop ignore GT instances *and* the
        # predictions censored onto them. Without the second half a prediction that
        # correctly finds an ignore nucleus would be excused by the centroid metrics but
        # still charged as a false positive by AP/mSA and PQ.
        # Kept with ORIGINAL ids so the class dicts below stay valid; pq_from_masks
        # relabels per class internally, and the AP path relabels at the call site.
        drop_gt = [int(i) for i, k in zip(gt_ids, gt_cls) if k == IGNORE_ID]
        drop_pred = [int(pr_ids[o_p[j]]) for j in cd]
        masks_true.append(np.where(np.isin(mt, drop_gt), 0, mt))
        masks_pred.append(np.where(np.isin(mp, drop_pred), 0, mp))
        ct_all.append({int(i): int(k) for i, k in zip(gt_ids, gt_cls) if k != IGNORE_ID})
        cp_all.append({int(i): int(k) for i, k in zip(pr_ids, pr_cls)
                       if k != IGNORE_ID and int(i) not in drop_pred})

    pooled = aggregate_pooled(per_image)
    official = aggregate_official(per_image, n_fg)
    dp, dr, df = prf(np.array([det_tp]), np.array([det_fp]), np.array([det_fn]))
    pq = pq_from_masks(masks_true, ct_all, masks_pred, cp_all, n_fg)
    # AP is class-agnostic, so relabel to contiguous ids here: removing ignore instances
    # leaves gaps, and cellpose sizes its IoU matrix by max label, turning each gap into a
    # zero-pixel phantom instance and a 0/0 nan.
    ap = cp_metrics.average_precision(
        [fastremap.renumber(m)[0] for m in masks_true],
        [fastremap.renumber(m)[0] for m in masks_pred],
        threshold=list(np.arange(0.5, 1.0, 0.05)))[0]

    res = {
        "n_images": len(gt),
        "n_ignore_gt": n_ignore_total,
        "n_censored_pred": n_censored_total,
        "n_duplicate_pred_centroids": dup_centroids,
        "centroid/official_macro_f1": official["macro_f1_official"],
        "centroid/official_ceiling": official["ceiling_macro"],
        "centroid/perimage_present_macro_f1": official["macro_f1_perimage_present"],
        "centroid/pooled_micro_f1": pooled["micro_f1"],
        "centroid/pooled_macro_f1_present": pooled["macro_f1_present"],
        "centroid/pooled_macro_f1_all": pooled["macro_f1_all"],
        "det/precision": float(dp[0]), "det/recall": float(dr[0]), "det/f1": float(df[0]),
        "pq/mPQ": pq["mPQ"], "pq/mDQ": pq["mDQ"], "pq/mSQ": pq["mSQ"],
        "seg/AP@0.5": float(np.nanmean(ap[:, 0])),
        "seg/AP@0.75": float(np.nanmean(ap[:, 5])),
        "seg/mSA": float(np.nanmean(ap)),
    }
    names = class_names(gt_root, n_fg, dataset)
    per_class = {
        "name": names,
        "confusion_matrix": cm_total.tolist(),
        "support": pooled["support"].tolist(),
        "f1_pooled": pooled["per_class_f1"].tolist(),
        "f1_official": official["per_class_f1_official"].tolist(),
        "ceiling": official["ceiling_per_class"].tolist(),
        "pq": pq["per_class_pq"].tolist(),
    }
    if verbose:
        _print(res, per_class, radius, headline, cm=cm_total.tolist(),
               names=names)
    return res, per_class


def _print(res, pc, radius, headline=None, cm=None, names=None):
    print(f"\n{'=' * 78}")
    print(f"images={res['n_images']}  radius={radius:g}px  "
          f"ignore_gt={res['n_ignore_gt']}  censored_pred={res['n_censored_pred']}  "
          f"dup_centroids={res['n_duplicate_pred_centroids']}")
    print("=" * 78)
    print(f"{'class':16s} {'support':>8s} {'F1 pooled':>10s} {'F1 offic':>9s} "
          f"{'ceiling':>8s} {'PQ':>7s}")
    for i, nm in enumerate(pc["name"]):
        print(f"{nm:16s} {pc['support'][i]:8d} {pc['f1_pooled'][i]:10.4f} "
              f"{pc['f1_official'][i]:9.4f} {pc['ceiling'][i]:8.3f} {pc['pq'][i]:7.4f}")
    print("-" * 78)
    star = lambda k: "  <== HEADLINE" if headline == k else ""
    print(f"  CENTROID official  macro-F1 = {res['centroid/official_macro_f1']:.4f}"
          f"  (ceiling {res['centroid/official_ceiling']:.4f}; per-image avg over ALL "
          f"images){star('official')}")
    print(f"  CENTROID per-image macro-F1 = "
          f"{res['centroid/perimage_present_macro_f1']:.4f}"
          f"  (same averaging, divided by images CONTAINING the class; ceiling 1.0)")
    print(f"  CENTROID pooled    micro/macro = {res['centroid/pooled_micro_f1']:.4f} / "
          f"{res['centroid/pooled_macro_f1_present']:.4f}"
          f"  (TP/FP/FN summed over all images; ceiling 1.0){star('pooled')}")
    if cm is not None and names is not None:
        n = len(names)
        w = max(len(x) for x in names) + 1
        print()
        print("  CONFUSION MATRIX (class-agnostic 15px pairing; rows=GT, cols=pred)")
        print(" " * (w + 2) + "".join(f"{i:>7d}" for i in range(n)) + f"{'miss':>8s}")
        for g in range(n):
            row = "".join(f"{cm[g][p]:>7d}" for p in range(n))
            print(f"  {names[g]:<{w}s}{row}{cm[g][n]:>8d}")
        print(f"  {'spurious':<{w}s}" + "".join(f"{cm[n][p]:>7d}" for p in range(n)))
    print(f"  DETECTION (class-agnostic)   P={res['det/precision']:.4f} "
          f"R={res['det/recall']:.4f} F1={res['det/f1']:.4f}")
    print(f"  PANOPTIC  mPQ={res['pq/mPQ']:.4f}  mDQ={res['pq/mDQ']:.4f}  "
          f"mSQ={res['pq/mSQ']:.4f}")
    print(f"  SEG       AP@0.5={res['seg/AP@0.5']:.4f}  mSA={res['seg/mSA']:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True, help="GT dir holding instance_masks/ + class_labels/")
    p.add_argument("--pred", required=True, help="prediction dir, same layout")
    p.add_argument("--dataset", choices=sorted(DATASETS), help="registry defaults")
    p.add_argument("--nclasses", type=int, help="incl. background; from --dataset if omitted")
    p.add_argument("--radius", type=float, help=f"match radius px (default {DEFAULT_RADIUS_PX:g})")
    p.add_argument("--label", default="model", help="row label for wandb/stdout")
    p.add_argument("--wandb_project", default="")
    p.add_argument("--json_out", default="", help="also write metrics to this path")
    a = p.parse_args()

    cfg = DATASETS.get(a.dataset, {})
    nclasses = a.nclasses or cfg.get("nclasses")
    if not nclasses:
        raise SystemExit("--nclasses required (or pass a known --dataset)")
    radius = a.radius or cfg.get("radius_px", DEFAULT_RADIUS_PX)

    res, per_class = score(a.gt, a.pred, nclasses - 1, radius, a.dataset,
                           headline=cfg.get("headline"))

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"label": a.label, "gt": a.gt, "pred": a.pred,
                       "nclasses": nclasses, "radius_px": radius,
                       "metrics": res, "per_class": per_class}, f, indent=2)
        print(f"\nwrote {a.json_out}")

    if a.wandb_project:
        import wandb
        run = wandb.init(project=a.wandb_project, name=a.label,
                         config={"gt": a.gt, "pred": a.pred, "dataset": a.dataset,
                                 "nclasses": nclasses, "radius_px": radius})
        log = dict(res)
        for i, nm in enumerate(per_class["name"]):
            log[f"cls/f1_pooled/{nm}"] = per_class["f1_pooled"][i]
            log[f"cls/f1_official/{nm}"] = per_class["f1_official"][i]
            log[f"cls/pq/{nm}"] = per_class["pq"][i]
            log[f"cls/support/{nm}"] = per_class["support"][i]

        # Log both a readable table and individually queryable confusion counts.
        cm = per_class["confusion_matrix"]
        names = list(per_class["name"])
        n = len(names)
        tbl = wandb.Table(columns=["gt"] + names + ["missed"])
        for g in range(n):
            tbl.add_data(names[g], *[cm[g][p] for p in range(n)], cm[g][n])
        tbl.add_data("spurious", *[cm[n][p] for p in range(n)], 0)
        log["cm/table"] = tbl
        for g in range(n):
            for p in range(n):
                if g != p and cm[g][p]:
                    log[f"cm/{names[g]}__as__{names[p]}"] = cm[g][p]
            log[f"cm/{names[g]}__missed"] = cm[g][n]
            log[f"cm/spurious__{names[g]}"] = cm[n][g]

        run.log(log)
        run.finish()


if __name__ == "__main__":
    main()
