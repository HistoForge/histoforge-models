"""Multi-dataset H&E nucleus-segmentation benchmark.

Evaluates one explicitly selected checkpoint on the held-out test splits of the
14 `nuclei_v2` H&E datasets and logs per-dataset AP@0.5, AP@0.75, and mSA to W&B.
This fixed Cellpose-comparison protocol retains per-image aggregation; the new
general training/evaluation CLI default must not silently change this benchmark.

The test data must already exist on disk. Every dataset has a
`{ds}/test/{images,instance_masks}/` folder (the nuclei_v2 tree). This script reads
those folders; it creates/converts nothing. GT is the uint16 instance masks (no
flow precompute, which is only needed for training).

Usage (explicit supported checkpoint):
    python scripts/he_seg_benchmark.py \
        --model_path checkpoints/model.safetensors \
        --model_label candidate \
        --nuclei_root data/nuclei

Set WANDB_MODE=disabled to disable remote logging. Inference still runs and
results print to stdout. Use only an evaluation population approved for the run.
"""

import argparse
import gc
import logging
import os
import sys

import numpy as np

# repo root on sys.path so `import histoforge_models...` works when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from histoforge_models.pathology.training.data_utils import load_arrays
from histoforge_models.pathology.training.eval import compute_seg_ap
from histoforge_models.pathology.training.initialization import create_cellpose_model

logger = logging.getLogger("he_seg_benchmark")

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
BENCHMARK_SEGMENTATION_AGGREGATION = "per_image"

# Registry: dataset -> (test_images subdir, test_labels subdir, mask suffix).
# Paths are relative to --nuclei_root (the nuclei_v2 tree). Every dataset stores
# its held-out split under <ds>/test/{images,instance_masks}; masks are uint16
# .tif whose stems match the images (no suffix). val/ is a SEPARATE carved split
# and is never merged into the test set.
HE_TEST_DATASETS = {
    "conic": ("conic/test/images", "conic/test/instance_masks", ""),
    "consep": ("consep/test/images", "consep/test/instance_masks", ""),
    "cpm15": ("cpm15/test/images", "cpm15/test/instance_masks", ""),
    "cpm17": ("cpm17/test/images", "cpm17/test/instance_masks", ""),
    "cryonuseg": ("cryonuseg/test/images", "cryonuseg/test/instance_masks", ""),
    "glysac": ("glysac/test/images", "glysac/test/instance_masks", ""),
    "lizard": ("lizard/test/images", "lizard/test/instance_masks", ""),
    "lynsec": ("lynsec/test/images", "lynsec/test/instance_masks", ""),
    "monusac": ("monusac/test/images", "monusac/test/instance_masks", ""),
    "monuseg": ("monuseg/test/images", "monuseg/test/instance_masks", ""),
    "nuinsseg": ("nuinsseg/test/images", "nuinsseg/test/instance_masks", ""),
    "pannuke": ("pannuke/test/images", "pannuke/test/instance_masks", ""),
    "puma": ("puma/test/images", "puma/test/instance_masks", ""),
    "tnbc": ("tnbc/test/images", "tnbc/test/instance_masks", ""),
}
DEFAULT_DATASETS = list(HE_TEST_DATASETS.keys())


def _pair_one_dir(images_dir, labels_dir, label_suffix):
    """Pair image/label files from one directory, matched by stem."""
    masks = {}
    for f in os.listdir(labels_dir):
        if f.lower().endswith(IMG_EXTS):
            masks[os.path.splitext(f)[0]] = os.path.join(labels_dir, f)
    img_paths, lbl_paths = [], []
    for f in sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)):
        stem = os.path.splitext(f)[0]
        mkey = stem + label_suffix
        if mkey not in masks:
            raise FileNotFoundError(
                f"{images_dir}/{f}: no matching mask '{mkey}' in {labels_dir}")
        img_paths.append(os.path.join(images_dir, f))
        lbl_paths.append(masks[mkey])
    return img_paths, lbl_paths


def pair_test_files(images_dir, labels_dir, label_suffix=""):
    """Held-out test set for one dataset: <ds>/test/{images,instance_masks}.

    nuclei_v2 stores the COMPLETE official test split under test/ (e.g. monuseg's
    full 14 images); val/ is a separately carved split and is NOT merged in.
    """
    return _pair_one_dir(images_dir, labels_dir, label_suffix)


def to_uint8_rgb(arr):
    """Coerce an image array to (H, W, 3) uint8 (from prepare_he_pretrain).

    Makes pixels byte-identical across source dtypes (pannuke float64,
    lynsec int32, ...) so every backbone sees the same input.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        if a.max() <= 1.0 + 1e-6:  # 0-1 float images
            a = a * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    return arr


def evaluate_one_dataset(model, images_dir, labels_dir, label_suffix, bsize,
                         batch_size):
    """Load one dataset's test split, run compute_seg_ap, return the metrics dict."""
    img_paths, lbl_paths = pair_test_files(images_dir, labels_dir, label_suffix)
    images = [to_uint8_rgb(im) for im in load_arrays(img_paths)]
    labels = [lb.astype(np.int32) for lb in load_arrays(lbl_paths)]
    res = compute_seg_ap(model, images, labels, bsize=bsize, batch_size=batch_size,
                         segmentation_metric_aggregation=BENCHMARK_SEGMENTATION_AGGREGATION)
    del images, labels
    gc.collect()
    return res


def run_benchmark(model_path, model_label, bsize, batch_size,
                  nuclei_root, datasets, wandb_project,
                  allow_partial=False):
    """Build one model, evaluate it on every dataset, log to wandb + stdout."""
    if not model_path:
        raise ValueError("model_path must name an explicit supported checkpoint")
    try:
        import wandb
        run = wandb.init(
            project=wandb_project, name=model_label, config={
                "model_path": model_path,
                "backbone": "dinov3_adapter",
                "bsize": bsize,
                "segmentation_metric_aggregation": BENCHMARK_SEGMENTATION_AGGREGATION,
            })
    except Exception as e:  # wandb missing or WANDB_MODE=disabled
        logger.warning("wandb unavailable (%s) — printing to stdout only", e)
        run = None

    logger.info("building fixed adapter: label=%s bsize=%d checkpoint=%s",
                model_label, bsize, model_path)
    model = create_cellpose_model(
        gpu=True, pretrained_model=model_path, head_type="flow")

    log, rows, failures = {}, [], {}
    for ds in datasets:
        img_sub, lbl_sub, suffix = HE_TEST_DATASETS[ds]
        try:
            res = evaluate_one_dataset(model, os.path.join(nuclei_root, img_sub),
                                       os.path.join(nuclei_root, lbl_sub), suffix,
                                       bsize, batch_size)
        except Exception as exc:
            logger.exception("dataset %s failed", ds)
            failures[ds] = f"{type(exc).__name__}: {exc}"
            continue
        ap = res["ap_array"]
        n = ap.shape[0]
        log[f"{ds}/AP@0.5"] = res["AP@0.5"]
        log[f"{ds}/AP@0.75"] = res["AP@0.75"]
        log[f"{ds}/mSA"] = res["mSA"]
        log[f"{ds}/n_images"] = n
        if run is not None:
            ap50 = ap[:, 0]
            ap50 = ap50[~np.isnan(ap50)]
            if ap50.size:
                log[f"{ds}/AP@0.5_hist"] = wandb.Histogram(ap50)
        rows.append([ds, n, res["AP@0.5"], res["AP@0.75"], res["mSA"]])
        logger.info("%-18s n=%-5d AP@0.5=%.4f AP@0.75=%.4f mSA=%.4f", ds, n,
                    res["AP@0.5"], res["AP@0.75"], res["mSA"])

    if failures and not allow_partial:
        if run is not None:
            run.summary["benchmark_complete"] = False
            run.summary["failed_datasets"] = failures
            run.finish(exit_code=1)
        raise RuntimeError(
            f"benchmark incomplete: {len(failures)} of {len(datasets)} datasets "
            f"failed: {failures}")

    log["benchmark/complete"] = not failures
    log["benchmark/requested_datasets"] = len(datasets)
    log["benchmark/completed_datasets"] = len(rows)
    if failures:
        log["benchmark/partial"] = True
        logger.warning("PARTIAL BENCHMARK explicitly allowed; failures=%s", failures)

    if rows:
        ns = np.array([r[1] for r in rows], dtype=float)
        ap50 = np.array([r[2] for r in rows])
        ap75 = np.array([r[3] for r in rows])
        msa = np.array([r[4] for r in rows])
        log["mean/AP@0.5"] = float(ap50.mean())
        log["mean/AP@0.75"] = float(ap75.mean())
        log["mean/mSA"] = float(msa.mean())
        log["mean_weighted/AP@0.5"] = float(np.average(ap50, weights=ns))
        log["mean_weighted/mSA"] = float(np.average(msa, weights=ns))
        if run is not None:
            log["per_dataset"] = wandb.Table(
                columns=["dataset", "n_images", "AP@0.5", "AP@0.75", "mSA"], data=rows)

    if run is not None:
        run.log(log)
        run.finish()

    # always print to stdout
    print(f"\n=== benchmark: {model_label} ===")
    print(f"{'dataset':18s} {'n':>5s} {'AP@0.5':>8s} {'AP@0.75':>8s} {'mSA':>8s}")
    for ds, n, a50, a75, m in rows:
        print(f"{ds:18s} {n:5d} {a50:8.4f} {a75:8.4f} {m:8.4f}")
    if rows:
        print(f"{'mean':18s} {'':5s} {log['mean/AP@0.5']:8.4f} "
              f"{log['mean/AP@0.75']:8.4f} {log['mean/mSA']:8.4f}")
        print(f"{'mean_weighted':18s} {'':5s} {log['mean_weighted/AP@0.5']:8.4f} "
              f"{'':8s} {log['mean_weighted/mSA']:8.4f}")
    return rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark a model on the 14 nuclei_v2 H&E test splits → wandb.")
    p.add_argument("--model_path", required=True,
                   help="Complete current HistoForge checkpoint.")
    p.add_argument("--model_label", required=True,
                   help="W&B run name for this checkpoint.")
    p.add_argument("--bsize", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--nuclei_root", required=True,
                   help="nuclei_v2 dir holding the <ds>/test/ folders.")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   choices=list(HE_TEST_DATASETS),
                   help="Subset to evaluate (default: every dataset in the registry).")
    p.add_argument("--wandb_project", default="cellpose-he-benchmark")
    p.add_argument(
        "--allow_partial", action="store_true",
        help="Exit successfully with visibly partial metrics when a requested "
        "dataset fails. Default is fail-closed.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    run_benchmark(model_path=args.model_path, model_label=args.model_label,
                  bsize=args.bsize,
                  batch_size=args.batch_size,
                  nuclei_root=os.path.expanduser(args.nuclei_root),
                  datasets=args.datasets, wandb_project=args.wandb_project,
                  allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
