"""Instance-segmentation evaluation for supported training checkpoints.

The CLI reports AP@0.5/0.75/0.9 and mSA. Flow checkpoints use Cellpose
integration; HoVer checkpoints use stitched NP/HV maps followed by watershed.
Classification scoring lives in ``scripts/eval_instance_cls.py`` and the
canonical prediction/scoring harness.

Usage (segmentation only):
    python -m histoforge_models.pathology.training.eval \
        --model_path /path/to/model \
        --eval_images /path/to/images \
        --eval_labels /path/to/labels

"""

import argparse
import json
import logging
from pathlib import Path

import fastremap
import numpy as np

from histoforge_models import metrics
from histoforge_models.pathology.metric_aggregation import (
    DEFAULT_SEGMENTATION_AGGREGATION,
    SEGMENTATION_AGGREGATIONS,
    validate_segmentation_aggregation,
)
from histoforge_models.pathology.training.segmentation_metrics import evaluate_instance_masks
from histoforge_models.pathology.training.data_utils import get_paired_files, load_arrays
from histoforge_models.pathology.decode_profile import (
    HOVER_EDGE_THRESHOLD,
    HOVER_FG_THRESHOLD,
    HOVER_MORPH_KERNEL,
    HOVER_OBJECT_SIZE,
    HOVER_TILE_OVERLAP,
)
from histoforge_models.pathology.training.initialization import create_cellpose_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Segmentation evaluation
# ---------------------------------------------------------------------------


def compute_seg_ap(cellpose_model, eval_data, eval_labels, bsize=256, batch_size=8,
                   segmentation_metric_aggregation=DEFAULT_SEGMENTATION_AGGREGATION):
    """Run mask inference with a ready CellposeModel and compute AP.

    Shared by evaluate_model and the training loop, so the val AP logged
    during training is produced by the exact same path as the benchmark.

    Args:
        cellpose_model: a CellposeModel (its .net may be mid-training).
        eval_data: list of raw images.
        eval_labels: list of int ground-truth instance masks.
        bsize: inference tile size.
        batch_size: tiles per forward pass.

    Returns:
        Selected AP/mSA, named pooled/per_image comparisons, and per-image counts.
    """
    validate_segmentation_aggregation(segmentation_metric_aggregation)
    masks = cellpose_model.eval(eval_data, batch_size=batch_size, bsize=bsize)[0]
    return evaluate_instance_masks(eval_labels, masks, segmentation_metric_aggregation)


def evaluate_model(model_path, eval_images_dir, eval_labels_dir, bsize=256,
                   batch_size=8, save_dir=None, dataset_name="eval",
                   tile_overlap=HOVER_TILE_OVERLAP,
                   hv_edge_threshold=HOVER_EDGE_THRESHOLD,
                   hv_object_size=HOVER_OBJECT_SIZE, hv_fg_threshold=HOVER_FG_THRESHOLD,
                   hv_morph_kernel=HOVER_MORPH_KERNEL,
                   segmentation_metric_aggregation=DEFAULT_SEGMENTATION_AGGREGATION):
    """Evaluate a fixed HistoForge adapter on a dataset.

    Args:
        model_path: Path to model weights file.
        eval_images_dir: Directory with evaluation images.
        eval_labels_dir: Directory with ground-truth label masks.
        bsize: Tile size for inference (must match training).
        batch_size: Batch size for model.eval().
        save_dir: If provided, saves per-image AP array and summary CSV.
        dataset_name: Name used in log messages and output filenames.

    Returns:
        Dict with keys: AP@0.5, AP@0.75, AP@0.9, mSA, SA50, SA75,
        and the raw ap array under "ap_array".
    """
    validate_segmentation_aggregation(segmentation_metric_aggregation)
    # Load data
    img_files, lbl_files = get_paired_files(eval_images_dir, eval_labels_dir)
    logger.info("Loading %d images for %s evaluation...", len(img_files), dataset_name)
    eval_data = load_arrays(img_files)
    eval_labels = [lbl.astype(np.int32) for lbl in load_arrays(lbl_files)]

    # Create model and run inference
    logger.info("Loading fixed adapter checkpoint %s", model_path)
    model = create_cellpose_model(gpu=True, pretrained_model=str(model_path))
    logger.info("Running inference on %d images...", len(eval_data))
    if getattr(model.net, "head_type", "flow") == "hovernet":
        from histoforge_models.pathology.training.hovernet_eval import (
            compute_hovernet_seg_ap,
        )
        results = compute_hovernet_seg_ap(
            model.net, eval_data, eval_labels, bsize=bsize, batch_size=batch_size,
            tile_overlap=tile_overlap, edge_threshold=hv_edge_threshold,
            object_size=hv_object_size, fg_threshold=hv_fg_threshold,
            morph_kernel=hv_morph_kernel,
            segmentation_metric_aggregation=segmentation_metric_aggregation)
    else:
        results = compute_seg_ap(model, eval_data, eval_labels, bsize=bsize,
                                 batch_size=batch_size,
                                 segmentation_metric_aggregation=segmentation_metric_aggregation)

    for key, val in results.items():
        if isinstance(val, float):
            logger.info("%s %s = %.4f", dataset_name, key, val)

    # Save outputs
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"{dataset_name}_ap.npy", results["ap_array"])

        csv_path = save_dir / f"{dataset_name}_metrics.csv"
        with open(csv_path, "w") as f:
            f.write("metric,value\n")
            for key, val in results.items():
                if isinstance(val, float):
                    value = f"{val:.6f}" if np.isfinite(val) else ""
                    f.write(f"{key},{value}\n")
                elif key == "segmentation_metric_aggregation":
                    f.write(f"{key},{val}\n")
        logger.info("Saved results to %s", save_dir)

    return results


# ---------------------------------------------------------------------------
# Semantic (pixel-level) classification evaluation
# ---------------------------------------------------------------------------


def build_gt_class_map(mask, json_path):
    """Build per-pixel class map from instance mask and JSON labels.

    Uses same convention as data_utils.prepare_semantic_labels:
    JSON class IDs are 0-indexed foreground, shifted +1 so 0=background.
    """
    with open(json_path) as f:
        inst_classes = json.load(f)
    class_map = np.zeros(mask.shape, dtype=np.uint8)
    for inst_id_str, cls_id in inst_classes.items():
        class_map[mask == int(inst_id_str)] = cls_id + 1
    return class_map


def evaluate_semantic(masks_pred, class_maps_pred, masks_true, class_maps_true,
                      nclasses):
    """Evaluate semantic segmentation with per-class AP and error rates.

    Matches predicted and true masks via IoU > 0.5, then checks
    per-instance class agreement for TP/FP/FN per foreground class.

    Background-classified predicted pixels are removed before matching.

    Args:
        masks_pred: List of predicted instance mask arrays.
        class_maps_pred: List of predicted per-pixel class maps (uint8).
        masks_true: List of ground-truth instance mask arrays.
        class_maps_true: List of ground-truth per-pixel class maps (uint8).
        nclasses: Number of classes including background.

    Returns:
        Dict with per_class_ap, mean_class_ap, per_class_error,
        mean_error, and raw tp/fp/fn arrays.
    """
    nimg = len(masks_true)
    n_fg = nclasses - 1
    tp_all = np.zeros((nimg, n_fg), "int")
    fp_all = np.zeros((nimg, n_fg), "int")
    fn_all = np.zeros((nimg, n_fg), "int")

    for i in range(nimg):
        mp = masks_pred[i].copy()
        mt = masks_true[i].copy()
        cp = class_maps_pred[i]
        ct = class_maps_true[i]

        # Remove background-classified predicted pixels before matching
        mp[cp == 0] = 0
        mp = fastremap.renumber(mp)[0]
        mt = fastremap.renumber(mt)[0]

        n_pred = int(mp.max())
        n_true = int(mt.max())

        if n_pred == 0 and n_true == 0:
            continue

        # Per-instance class (mode via bincount)
        cid = np.array([
            np.bincount(cp[mp == j]).argmax() for j in range(1, n_pred + 1)
        ]) if n_pred > 0 else np.array([], "int")

        tid = np.array([
            np.bincount(ct[mt == j]).argmax() for j in range(1, n_true + 1)
        ]) if n_true > 0 else np.array([], "int")

        if n_pred == 0 or n_true == 0:
            for c in range(n_fg):
                fn_all[i, c] = (int((tid == c + 1).sum()) if n_true > 0 else 0)
                fp_all[i, c] = (int((cid == c + 1).sum()) if n_pred > 0 else 0)
            continue

        # Match masks via IoU
        iout, inds = metrics.mask_ious(mt, mp)
        inds[iout < 0.5] = 0

        # dtype=int matters: with no IoU>0.5 pairs this list is empty and an untyped
        # np.array([]) is float64, which raises "arrays used as indices must be of
        # integer type". That fires on any image the model misses completely.
        cmatch = cid[np.array([ind - 1 for ind in inds if ind != 0], dtype=np.int64)]
        tmatch = tid[inds != 0]
        inds_match = inds[inds != 0]

        # Predictions matched to an ignore-labelled GT are don't-care. GT class ids are
        # stored +1 (0 = background), so real classes occupy 1..n_fg and the 254 ignore
        # sentinel lands at 255. Drop those predictions from the FP pool for every class;
        # otherwise detecting an untyped nucleus would count as a false positive.
        ignore_matched = inds_match[tmatch > n_fg] - 1

        for c in range(n_fg):
            tps = (cmatch == c + 1) & (tmatch == c + 1)
            tp_all[i, c] = int(tps.sum())
            fn_all[i, c] = int((tid == c + 1).sum()) - int(tps.sum())
            not_tp = np.ones(n_pred, "bool")
            not_tp[ignore_matched] = False
            not_tp[inds_match[tps] - 1] = False
            fp_all[i, c] = int((cid[not_tp] == c + 1).sum())

    denom_ap = np.maximum(tp_all + fp_all + fn_all, 1)
    denom_err = np.maximum(tp_all + fn_all, 1)
    aps = tp_all / denom_ap
    errors = (fp_all + fn_all) / denom_err
    errors[np.isinf(errors)] = np.nan

    return {
        "per_class_ap": aps.mean(axis=0).tolist(),
        "mean_class_ap": float(np.nanmean(aps)),
        "per_class_error": errors.mean(axis=0).tolist(),
        "mean_error": float(np.nanmean(errors)),
        "tp": tp_all,
        "fp": fp_all,
        "fn": fn_all,
    }


# ---------------------------------------------------------------------------
# Shared classification evaluation (pixel-cls and instance-cls)
# ---------------------------------------------------------------------------


def evaluate_classification(masks_pred, pred_classes, masks_true, true_classes,
                            num_classes):
    """Evaluate classification by matching predicted and true instances.

    Works for both pixel-level and instance-level classification.
    Both approaches normalize to {str(instance_id): class_id} dicts.

    Args:
        masks_pred: List of predicted instance mask arrays.
        pred_classes: List of {str(inst_id): class_id} dicts.
        masks_true: List of ground-truth instance mask arrays.
        true_classes: List of {str(inst_id): class_id} dicts.
        num_classes: Number of foreground classes (0-indexed).

    Returns:
        Dict with per_class_tp, per_class_fp, per_class_fn arrays,
        per_class_precision, per_class_recall, per_class_f1, overall
        accuracy, and macro_precision / macro_recall / macro_f1.
    """
    tp = np.zeros(num_classes, "int")
    fp = np.zeros(num_classes, "int")
    fn = np.zeros(num_classes, "int")

    for mp, pc, mt, tc in zip(masks_pred, pred_classes, masks_true, true_classes):
        n_pred = int(mp.max())
        n_true = int(mt.max())

        if n_pred == 0 and n_true == 0:
            continue

        if n_pred == 0 or n_true == 0:
            # All GT are FN, all pred are FP
            for inst_id, cls_id in tc.items():
                if 0 <= cls_id < num_classes:  # skip ignore (254), as below
                    fn[cls_id] += 1
            for inst_id, cls_id in pc.items():
                if 0 <= cls_id < num_classes:
                    fp[cls_id] += 1
            continue

        # Match masks via IoU > 0.5
        iout, inds = metrics.mask_ious(mt, mp)
        inds[iout < 0.5] = 0

        matched_pred = set()
        for true_idx in range(len(inds)):
            true_inst = true_idx + 1
            pred_inst = inds[true_idx]
            true_cls = tc.get(str(true_inst))
            if true_cls is None:
                continue

            if not (0 <= true_cls < num_classes):
                # Ignored GT is never a false negative. Consume its matched
                # prediction so the unmatched-prediction loop does not count
                # it as a false positive or index the ignore ID into fn.
                if pred_inst != 0:
                    matched_pred.add(pred_inst)
                continue

            if pred_inst == 0:
                # Unmatched GT → FN
                fn[true_cls] += 1
                continue

            matched_pred.add(pred_inst)
            pred_cls = pc.get(str(pred_inst))
            if pred_cls is not None and pred_cls == true_cls:
                tp[true_cls] += 1
            else:
                fn[true_cls] += 1
                if pred_cls is not None:
                    fp[pred_cls] += 1

        # Unmatched predictions → FP
        for inst_id, cls_id in pc.items():
            if int(inst_id) not in matched_pred:
                fp[cls_id] += 1

    # Compute metrics
    precision = np.zeros(num_classes, "float64")
    recall = np.zeros(num_classes, "float64")
    f1 = np.zeros(num_classes, "float64")

    for c in range(num_classes):
        if tp[c] + fp[c] > 0:
            precision[c] = tp[c] / (tp[c] + fp[c])
        if tp[c] + fn[c] > 0:
            recall[c] = tp[c] / (tp[c] + fn[c])
        if precision[c] + recall[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])

    total_correct = tp.sum()
    total = tp.sum() + fp.sum() + fn.sum()
    accuracy = float(total_correct / max(total, 1))

    return {
        "per_class_tp": tp,
        "per_class_fp": fp,
        "per_class_fn": fn,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fixed HistoForge adapter")

    parser.add_argument("--model_path", required=True, help="Path to model weights")
    parser.add_argument("--eval_images", required=True,
                        help="Directory containing evaluation images")
    parser.add_argument("--eval_labels", required=True,
                        help="Directory containing ground-truth labels")

    parser.add_argument("--bsize", type=int, default=256,
                        help="Tile size (must match training)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Tiles per forward pass")
    parser.add_argument(
        "--segmentation_metric_aggregation", choices=SEGMENTATION_AGGREGATIONS,
        default=DEFAULT_SEGMENTATION_AGGREGATION,
        help="AP/mSA reduction: pooled object counts (default) or equal per-image scores")
    parser.add_argument("--tile_overlap", type=float, default=HOVER_TILE_OVERLAP,
                        help="HoVer tile overlap")
    parser.add_argument("--hv_edge_threshold", type=float, default=HOVER_EDGE_THRESHOLD)
    parser.add_argument("--hv_object_size", type=int, default=HOVER_OBJECT_SIZE)
    parser.add_argument("--hv_fg_threshold", type=float, default=HOVER_FG_THRESHOLD)
    parser.add_argument("--hv_morph_kernel", type=int, default=HOVER_MORPH_KERNEL)

    parser.add_argument("--save_dir", help="Directory for saving metrics output")
    parser.add_argument("--dataset_name", default="eval",
                        help="Name for this evaluation run")

    parser.add_argument("--wandb_project", help="Wandb project (omit to skip wandb)")
    parser.add_argument("--run_name", help="Wandb run name")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    seg_results = evaluate_model(
        model_path=args.model_path,
        eval_images_dir=args.eval_images,
        eval_labels_dir=args.eval_labels,
        bsize=args.bsize,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        dataset_name=args.dataset_name,
        tile_overlap=args.tile_overlap,
        hv_edge_threshold=args.hv_edge_threshold,
        hv_object_size=args.hv_object_size,
        hv_fg_threshold=args.hv_fg_threshold,
        hv_morph_kernel=args.hv_morph_kernel,
        segmentation_metric_aggregation=args.segmentation_metric_aggregation,
    )

    # Optional wandb logging
    if args.wandb_project:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"eval-{args.dataset_name}",
            config={
                "model_path": args.model_path,
                "dataset_name": args.dataset_name,
                "bsize": args.bsize,
                "segmentation_metric_aggregation": args.segmentation_metric_aggregation,
                "backbone": "dinov3_adapter",
            },
        )
        log_dict = {
            f"{args.dataset_name}/{k}": v
            for k, v in seg_results.items()
            if isinstance(v, float) and np.isfinite(v)
        }
        run.log(log_dict)
        run.finish()


if __name__ == "__main__":
    main()
