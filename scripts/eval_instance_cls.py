"""Object-level classification F1 for a supported flow or HoVer checkpoint.

The checkpoint selects the strict inference contract. Point --checkpoint and
--val_* at one frozen
split and set --nclasses. Class names auto-load from <val>/classes.json.

The class head is dense: channel 0 is background and channels 1..N-1 are
foreground classes. Object classes are majority votes inside decoded instances,
then ``evaluate_semantic`` matches predictions to GT at IoU > 0.5. This is the
module-validation metric, not PUMA's separate centroid-distance protocol.

Macro-F1 includes foreground classes with ground-truth support, using
evaluate_semantic's raw TP/FP/FN counts.

Runs LOCALLY, one ViT-L resident at a time. Usage:

  python scripts/eval_instance_cls.py \
      --checkpoint checkpoints/model.safetensors \
      --val_images data/validation/images \
      --val_masks data/validation/instance_masks \
      --val_class_labels data/validation/class_labels \
      --nclasses 3 --bsize 1024
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from histoforge_models import io, metrics
from histoforge_models.pathology.common import (
    build_and_load,
    detect_head_type_from_state,
    detect_num_classes_from_state,
)
from histoforge_models.pathology.flow_inference import run_flow_inference
from histoforge_models.pathology.hovernet_inference import run_hovernet_inference
from histoforge_models.pathology.training.eval import build_gt_class_map, evaluate_semantic
from histoforge_models.pathology.training.data_utils import get_paired_files
from histoforge_models.pathology.decode_profile import (
    HOVER_EDGE_THRESHOLD,
    HOVER_FG_THRESHOLD,
    HOVER_MORPH_KERNEL,
    HOVER_OBJECT_SIZE,
    HOVER_TILE_OVERLAP,
)


def _load_checkpoint(path):
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val_images", required=True)
    ap.add_argument("--val_masks", required=True)
    ap.add_argument("--val_class_labels", required=True)
    ap.add_argument("--nclasses", type=int, default=6, help="incl. background")
    ap.add_argument("--bsize", type=int, default=1024)
    ap.add_argument("--tile_overlap", type=float, default=HOVER_TILE_OVERLAP)
    ap.add_argument("--hv_edge_threshold", type=float, default=HOVER_EDGE_THRESHOLD)
    ap.add_argument("--hv_object_size", type=int, default=HOVER_OBJECT_SIZE)
    ap.add_argument("--hv_fg_threshold", type=float, default=HOVER_FG_THRESHOLD)
    ap.add_argument("--hv_morph_kernel", type=int, default=HOVER_MORPH_KERNEL)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    # Display names for the fg classes (1..nclasses-1). Auto-load from the
    # module's classes.json (sibling of class_labels) so this works for ANY
    # pixel-classification module. classes.json is 0-indexed foreground:
    # {"0": name, ...}; model fg class c corresponds to json class c-1.
    cj = Path(args.val_class_labels).parent / "classes.json"
    if cj.exists():
        raw = json.load(open(cj))
        CLASS_NAMES = {
            int(k) + 1: v
            for k, v in raw.items()
            if k.isdigit() and 0 <= int(k) < args.nclasses - 1
        }
    else:
        CLASS_NAMES = {c: f"class_{c}" for c in range(1, args.nclasses)}
    print(f"classes (fg): {CLASS_NAMES}")

    # Load the checkpoint strictly and verify its class-channel count.
    sd = _load_checkpoint(args.checkpoint)
    nc_detected = detect_num_classes_from_state(sd)
    print(f"detected num_classes (incl bg) from ckpt = {nc_detected}")
    if nc_detected != args.nclasses:
        raise ValueError(f"--nclasses={args.nclasses} but checkpoint contains "
                         f"{nc_detected} class channels")
    head_type = detect_head_type_from_state(sd)
    net = build_and_load({"head_type": head_type, "num_classes": args.nclasses}, sd)
    net.eval()

    nc = args.nclasses
    cls_dir = Path(args.val_class_labels)
    img_files, lbl_files = get_paired_files(args.val_images, args.val_masks)
    print(f"{head_type} validation: {len(img_files)} image/mask pairs\n")

    masks_pred, class_maps_pred, masks_true, class_maps_true = [], [], [], []
    for i, (imgf, lblf) in enumerate(zip(img_files, lbl_files)):
        img = np.asarray(io.imread(imgf))
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[..., :3]
        gt = np.asarray(io.imread(lblf)).astype(np.int32)

        if head_type == "hovernet":
            out = run_hovernet_inference(net, img, num_classes=nc, bsize=args.bsize,
                                         tile_overlap=args.tile_overlap, batch_size=1,
                                         edge_threshold=args.hv_edge_threshold,
                                         object_size=args.hv_object_size,
                                         fg_threshold=args.hv_fg_threshold,
                                         morph_kernel=args.hv_morph_kernel)
        else:
            out = run_flow_inference(net, img, num_classes=nc, bsize=args.bsize,
                                     tile_overlap=args.tile_overlap, batch_size=1)
        pm, pc = out["instance_mask"], out["class_map"]
        # size-align guard (run_net drift), nearest for label maps
        if pm.shape[:2] != gt.shape[:2]:
            import cv2
            hw = (gt.shape[1], gt.shape[0])
            pm = cv2.resize(pm.astype(np.int32), hw, interpolation=cv2.INTER_NEAREST)
            pc = cv2.resize(pc.astype(np.int32), hw, interpolation=cv2.INTER_NEAREST)

        ct = build_gt_class_map(gt, cls_dir / f"{Path(lblf).stem}.json")
        masks_pred.append(pm)
        class_maps_pred.append(pc.astype(np.uint8))
        masks_true.append(gt)
        class_maps_true.append(ct)

        print(f"  [{i+1}/{len(img_files)}] {Path(imgf).stem[:40]:40s} "
              f"pred_inst={pm.max():4d} gt_inst={gt.max():4d} "
              f"pred_cls={sorted(set(np.unique(pc).tolist()))}")

    # ---- object-level cls (IoU>0.5) ----
    res = evaluate_semantic(masks_pred, class_maps_pred, masks_true, class_maps_true,
                            nclasses=nc)
    tp = res["tp"].sum(0).astype(np.int64)  # per fg class (index 0 -> class 1)
    fp = res["fp"].sum(0).astype(np.int64)
    fn = res["fn"].sum(0).astype(np.int64)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    support = tp + fn  # GT objects per class

    # seg AP@0.5 sanity
    ap50 = float(
        np.nanmean(
            metrics.average_precision(masks_true, masks_pred, threshold=[0.5])[0][:,
                                                                                  0]))

    print("\n" + "=" * 62)
    print("INSTANCE-LEVEL (object, IoU>0.5) classification")
    print("=" * 62)
    print(f"{'class':16s} {'support':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'F1':>7s}")
    present = support > 0
    for j in range(nc - 1):  # fg classes 1..nc-1
        c = j + 1
        print(f"{CLASS_NAMES.get(c, f'cls{c}'):16s} {support[j]:8d} "
              f"{tp[j]:5d} {fp[j]:5d} {fn[j]:5d} {f1[j]:7.4f}")
    macro = float(f1[present].mean()) if present.any() else float("nan")
    micro_tp, micro_fp, micro_fn = tp.sum(), fp.sum(), fn.sum()
    micro = 2 * micro_tp / max(2 * micro_tp + micro_fp + micro_fn, 1)
    print("-" * 62)
    print(f"macro-F1 (fg classes w/ GT) = {macro:.4f}")
    print(f"micro-F1                    = {micro:.4f}")
    print(f"seg AP@0.5 (class-agnostic) = {ap50:.4f}")


if __name__ == "__main__":
    main()
