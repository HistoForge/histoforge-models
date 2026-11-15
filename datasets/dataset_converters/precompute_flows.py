"""Precompute cellpose flow targets for a training corpus.

For every instance mask in ``--masks_dir``, computes the cellpose flow
representation and writes a 4-channel float32 ``{stem}_flows.tif`` into
``--flows_dir``. Run once on a GPU node before lazy training
(``train_seg.py --lazy_train``), which reads these precomputed flows from disk
per item instead of holding the whole corpus + flow targets in RAM.

Masks are streamed in chunks so this script's own memory stays flat.

Usage:
    python datasets/dataset_converters/precompute_flows.py \
        --masks_dir /path/he_pretrain/train/instance_masks \
        --flows_dir /path/he_pretrain/train/flows
"""

import argparse
import hashlib
import os
import tempfile

import torch

from histoforge_models import dynamics, io
from histoforge_models.pathology.training.data_utils import list_images


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flow_path(flows_dir, mask_file):
    stem = os.path.splitext(os.path.basename(mask_file))[0]
    return os.path.join(flows_dir, stem + "_flows.tif")


def flow_hash_path(flows_dir, mask_file):
    stem = os.path.splitext(os.path.basename(mask_file))[0]
    return os.path.join(flows_dir, stem + "_flows.mask.sha256")


def flow_is_current(flows_dir, mask_file):
    output = flow_path(flows_dir, mask_file)
    sidecar = flow_hash_path(flows_dir, mask_file)
    if not (os.path.isfile(output) and os.path.isfile(sidecar)):
        return False
    with open(sidecar) as stream:
        recorded = stream.read().strip()
    return recorded == sha256_file(mask_file)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks_dir", required=True,
                    help="Directory of instance-mask image files")
    ap.add_argument("--flows_dir", required=True,
                    help="Output directory for {stem}_flows.tif")
    ap.add_argument("--chunk", type=int, default=200,
                    help="Masks held in RAM at once (keeps memory flat)")
    ap.add_argument("--redo", action="store_true",
                    help="Recompute even if a _flows.tif already exists")
    args = ap.parse_args()

    if os.path.abspath(args.masks_dir) == os.path.abspath(args.flows_dir):
        raise ValueError("--masks_dir and --flows_dir must differ")
    os.makedirs(args.flows_dir, exist_ok=True)

    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    print(f"device: {device}")

    mask_files = list_images(args.masks_dir)
    print(f"{len(mask_files)} masks in {args.masks_dir}")

    todo = (mask_files if args.redo else
            [mask for mask in mask_files if not flow_is_current(args.flows_dir, mask)])
    print(f"{len(todo)} to compute, "
          f"{len(mask_files) - len(todo)} already present")

    for i in range(0, len(todo), args.chunk):
        chunk = todo[i:i + args.chunk]
        masks = [io.imread(mask_file) for mask_file in chunk]
        # Stage the TIFFs and their mask hashes in the destination filesystem,
        # then publish each pair with os.replace. An interrupted chunk never
        # leaves a partially written final flow that a later run can accept.
        with tempfile.TemporaryDirectory(prefix=".flow-stage-",
                                         dir=args.flows_dir) as stage:
            name_files = [
                os.path.join(stage, os.path.basename(mask_file)) for mask_file in chunk
            ]
            dynamics.labels_to_flows(masks, files=name_files, device=device,
                                     return_flows=False)
            for mask_file in chunk:
                staged_flow = flow_path(stage, mask_file)
                final_flow = flow_path(args.flows_dir, mask_file)
                staged_hash = flow_hash_path(stage, mask_file)
                final_hash = flow_hash_path(args.flows_dir, mask_file)
                with open(staged_hash, "w") as stream:
                    stream.write(sha256_file(mask_file) + "\n")
                os.replace(staged_flow, final_flow)
                os.replace(staged_hash, final_hash)
        print(f"  {min(i + args.chunk, len(todo))}/{len(todo)}")

    print(f"done -> {args.flows_dir}")


if __name__ == "__main__":
    main()
