# Standalone training and evaluation

This guide covers the source training CLI. Training started in QuPath uses the
HistoForge backend and its separate dataset, execution and checkpoint lifecycle.

## Environment and commands

Install the source environment with `conda env create -f environment.yml`, or
install `.[dinov3,training,dev]` in a Python 3.10+ environment. Choose a PyTorch build
appropriate for the execution host. Do not replace a working GPU installation
just to install this package.

```bash
python -m histoforge_models.pathology.training.train_seg --help
python -m histoforge_models.pathology.training.eval --help
```

Training without an explicit initializer uses gated DINOv3 pretrained weights.
Accept the terms for
[`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
and authenticate with Hugging Face first. Checkpoint loading for inference is
strict and does not download missing weights implicitly.

## Inputs and labels

Use separate image and label directories with matching file stems:

```text
data/train/
  images/sample.tif
  instance_masks/sample.tif
  class_labels/sample.json
```

| Task | Supervision |
| --- | --- |
| Cellpose-flow instances | Integer instance mask, with zero as background |
| HoVer instances | Integer instance mask; HV/NP targets are generated during preparation |
| Instance classification | Instance mask and JSON mapping instance ID to zero-indexed foreground class |
| Dense semantics | Integer per-pixel class labels |

For classification, `--nclasses` includes background. Per-instance class 254
means ignore; after the foreground-to-pixel offset it becomes pixel class 255.
Keep ignored supervision separate from background. The
[architecture contract](architecture.md) defines output channel order.

Provide explicit validation directories when subjects, slides or source groups
must stay together. The fallback split is an image-level, seed-42 85/15 split;
it does not establish patient- or slide-level independence. Keep evaluation data
out of fitting, normalization statistics, tuning and checkpoint selection.

## Lazy flow training

`--lazy_train` streams segmentation-only Cellpose inputs and precomputed flows.
It is not a general low-memory mode for classification or HoVer targets.

```bash
python datasets/dataset_converters/precompute_flows.py \
  --masks_dir data/train/instance_masks \
  --flows_dir data/train/flows
```

Each `sample_flows.tif` needs a `sample_flows.mask.sha256` sidecar binding it to
the exact mask. Recompute flows when the mask changes; do not create a sidecar
for an unverified old flow. Supply explicit validation directories for lazy mode.

## Sampling and checkpoints

With N files in the training split and batch B, the CLI requests `S = max(2, N)`
samples per epoch and `ceil(S/B)` optimizer steps. Mask-count filtering can reduce
the retained file population without reducing S. Uniform sampling shuffles the
files when S equals the retained count; otherwise it samples with replacement.
Weighted sampling always uses replacement. The final incomplete batch is retained.
Changing batch size changes optimizer-step exposure, not just memory use.

Declare initialization, seed, resolution, field size, augmentation, loss, schedule
and selection rule before training. H&E stain perturbations are not automatically
appropriate for other stains.

The standalone loop writes the final checkpoint and eligible validation-selected
checkpoints. `_best` uses validation loss; `_best_ap` uses the configured
segmentation metric; `_best_f1` uses foreground pixel macro-F1, not object-level
classification F1. Undefined observations do not replace defined best scores;
ties retain the earlier checkpoint. A final checkpoint is not automatically the
best deployment choice.

## Evaluation

The segmentation CLI reports AP and mean segmentation accuracy. Its
`--segmentation_metric_aggregation` is explicit:

- `pooled` sums matched-object counts across images before division.
- `per_image` gives each defined image score equal weight. Empty-empty images
  are omitted rather than scored as perfect.

Both reductions are reported from the same counts. The dedicated H&E benchmark
keeps its separate `per_image` protocol; it does not inherit the CLI default.

`scripts/eval_instance_cls.py` evaluates object classes from a supported
checkpoint. `scripts/score_cls.py` scores saved instance masks and class JSON
without loading a model. Its PUMA-style centroid matching and pooled scores are
different metrics and must be named separately. The optional `classes.json`
maps zero-indexed class IDs to display names.

Record the exact source, inputs, split, initializer, checkpoint and decode
settings with every reported result. Training writes `recipe.json` before W&B
initialization. Model quality, dataset rights and checkpoint redistribution need
their own evidence; passing source tests does not establish them.
