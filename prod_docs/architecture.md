# Supported architecture contract

## Scope

HistoForge modules use one fixed encoder/adapter implementation:

1. DINOv3 ViT-L/16 at its native stride 16, with one CLS token and four register
   tokens.
2. A LayerNorm spatial-prior module (SPM) producing 256-channel features at
   strides 4, 8, and 16. The stride-2 stem feeds these levels. The independent
   stride-1 branch remains registered for checkpoint and optimizer compatibility,
   in its original construction order, but is not executed because no head uses it.
3. Four bidirectional CNN/transformer interaction stages at ViT block boundaries
   `[0, 6, 12, 18, 24]`. The interaction operates over the stride-4/8/16 CNN
   features using multi-scale deformable attention and reinjects stride-16
   adapter features into the ViT.
4. A stride-4 W2 pixel-shuffle readout that restores full-resolution logits.

The code contract is `histoforge_models.pathology.dinov3.DinoV3AdapterBackbone`; construction
for training is through `histoforge_models.pathology.training.initialization.create_cellpose_model`,
and release inference is through `histoforge_models.pathology.common.build_and_load`.

## Output contracts

Classification channels, when present, are prepended to the segmentation tail.
`nclasses` includes background.

| mode | output tail | decode | use |
|---|---|---|---|
| Cellpose flow | `[dY, dX, cellprob]` | Cellpose dynamics | nuclei, cells, SPG |
| HoVer | `[V, H, np_bg, np_fg]` | stitched NP/HV maps, then watershed | large structures such as tubules and lumen |
| instance classification | `nclasses` logits + either tail above | majority class vote inside each decoded instance | joint IS + IC |
| dense semantic | `nclasses` logits + frozen flow tail | per-pixel softmax/argmax; no instance decode | pixel taxonomy maps |

HoVer targets are centroid-relative and are computed eagerly from integer
instance masks. Cellpose flow targets can be precomputed and streamed.

## Checkpoint invariants

Supported adapter checkpoints persist:

- `adapter_dim_buf == 256`;
- `head_type_code == 0` for flow or `1` for HoVer;
- the five-level LN-SPM key schema beginning with `spm.stem_s1`;
- optional `out_class`/`W3` tensors whose shape determines `nclasses`.

Inference is strict. Every key, shape, dtype and architecture metadata value must
match before any tensor is loaded. Missing `head_type_code` is rejected. Old
metadata-free artifacts require an explicit new checkpoint revision or retraining;
there is no runtime migration or architecture-inference fallback.

Training first exact-loads the complete donor, including heads that will later be
replaced. Only an explicitly changed segmentation or classification head may be
replaced. This deliberately allows a foundation flow head to become a fresh
HoVer head, or a new class head to be initialized from the learned flow
foreground logit, without accepting a truncated encoder/adapter. The resulting
saved checkpoint is again a complete, strict release artifact.

An explicit bare DINOv3 safetensors artifact can initialize the backbone alone.
Pairing it with a complete class-agnostic adapter donor validates that donor first,
then replaces only its backbone. Fresh initialization without an explicit artifact
uses timm's official pretrained backbone. Network construction and exact inference
loading never download weights implicitly.

The loader rejects BN-SPM, U-Net, CoMer-specific, ConvNeXt, foundation-model
conditioning, boundary-head, auxiliary-head and wide-adapter schemas. They are
not supported checkpoint formats.

## Tiled inference

`histoforge_models.core.run_net` performs overlapping tiled forward passes and restores a
full-resolution output field. Flow modules decode the stitched flow/cell-probability
maps with Cellpose dynamics. HoVer modules stitch NP/HV outputs first and run one
watershed on the complete maps; they do not watershed each tile independently.
Module recipes must therefore release their tile size, overlap, and decode
thresholds with the weights.

Flow cleanup preserves upstream Cellpose order: filter undersized masks, fill
holes, then filter again. `min_instance_pixels=0` disables both size filters but
not hole filling. HoVer retains its own watershed and final-area filtering policy.

`pathology.cellpose_window_decode` owns flow integration for a bounded window of
a larger target. Its trajectories use complete-target coordinates; substituting
ordinary window-local dynamics can change integer destinations. Full-halo labels
remain provisional until the caller checks instance ownership and context. This
decoder does not own scheduling, source storage or geometry publication.

The flow, HoVer and semantic wrappers return diagnostic maps by default for
research consumers. Production callers use `include_diagnostics=False`: instance
predictions retain the complete label mask and every instance's class vote;
semantic predictions retain the complete all-channel class labels. Background and
training-only channels still compete. Emitted-class filtering belongs to the
geometry consumer, not prediction. This option changes retention, not decoding.

Coarse wall-clock logs separate field prediction from decoding/classification.
Field prediction includes normalization, tiling, network execution, device
transfers and blending; it is not a pure GPU-kernel measurement. These spans are
not additive with enclosing request timers and do not introduce per-tile device
synchronization.

`pathology.network_execution.forward_network` owns inference precision for
fixed-field, variable-field and window callers. Its default `fp32` delegates to
the unchanged upstream `_forward`. Explicit `bf16` applies CUDA autocast only to
the network call, retaining FP32 checkpoint parameters and returning FP32 arrays
before blending and decoding. It requires native CUDA BF16 support and fails
instead of silently falling back. This option does not change the training loop
or select a serving policy; the consuming application owns that selection.

The standalone research evaluator deliberately retains complete-image
normalization through `CellposeModel.eval` for flow and its research HoVer helper.
The shared module wrappers normalize the complete supplied real-pixel-supported
image before neural tiling. Moving
files does not equate these different evaluation protocols or change recorded
research results.

## Source ownership and standalone use

`histoforge_models/pathology/` owns the shared architecture, strict loading, field geometry,
normalization, decoding and `managed_training/` scientific primitives. HistoForge
vendors that runtime together with the required root Cellpose primitives, preserving relative
paths and exact bytes from one source commit. No backend-specific import rewriting
or duplicate algorithm implementation is required. The managed optimizer engine
and flow/HoVer execution binding belong to the backend, not this scientific package.
The standalone research loop remains source-owned and independent.

The upstream-owned function boundary is pinned to Cellpose commit
`a54cb48849b7e225a81e8e43dcb042d42427f543`. Within that boundary, formatting and
package-relative imports may differ, but function ASTs must remain equivalent:

| source | upstream-owned functions used by HistoForge |
|---|---|
| `histoforge_models/core.py` | `_forward` |
| `histoforge_models/dynamics.py` | every top-level function |
| `histoforge_models/transforms.py` | `_taper_mask`, `make_tiles`, `average_tiles`, `get_pad_yx` |
| `histoforge_models/utils.py` | `fill_holes_and_remove_small_masks` |
| `histoforge_models/metrics.py` | every top-level function |
| `histoforge_models/pathology/managed_training/_flow_targets.py` | `_extend_centers_gpu`, `center_of_mass`, `get_centers`, `masks_to_flows_gpu` |

`histoforge_models.core.run_net` is the one bounded modification to that execution path. It
retains upstream field placement, forwarding, batching, tapering and reconstruction,
and adds an optional network-batch callback plus the overlap clamp needed to keep
allocation consistent with `make_tiles`. The shared field wrapper owns
normalization before tiling. Variable-field placement and
CoreHalo ownership remain separate HistoForge compositions around these primitives.
Managed training carries exact copies of the four upstream two-dimensional flow-target
functions because that package is also usable as a self-contained vendor boundary.
After upstream diffusion returns, the managed target builder explicitly zeroes the
direction of complete one-pixel instances; this product policy is not folded into the
upstream functions. This contract does not claim that every unrelated function in the
fork is identical to the pinned upstream commit.

`histoforge_models/pathology/training/` is source-only: explicit initialization, research
data/augmentation, W&B orchestration and evaluation. Entry points are:

```bash
python -m histoforge_models.pathology.training.train_seg --help
python -m histoforge_models.pathology.training.eval --help
```

The upstream Cellpose package/GUI remains separate and independently usable.
Retired HistoForge architecture switches and `histoforge_models.custom` aliases are absent.
Source compatibility does not establish model quality or checkpoint rights.

## Dependency and licence boundary

DINOv3 initialization is fetched through timm from the gated Meta checkpoint.
The repository does not bundle those weights. Before redistributing a derivative
checkpoint, preserve this repository's Apache/BSD notices, include the applicable
DINOv3 terms, and audit all warm-start checkpoints and datasets in its lineage.
