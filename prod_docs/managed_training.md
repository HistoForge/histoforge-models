# Scientific primitives for managed segmentation training

`histoforge_models.pathology.managed_training` owns target construction, masked losses,
objective combination and real-pixel appearance operators. It does not own an
optimizer loop, dataset lifecycle or cloud integration.

HistoForge owns managed optimization in `histoforge/cloud/training/engine.py`
and binds these scientific functions in `training/trainers/instance/loop.py`.
The standalone training CLI and `training/train_seg_loop.py` are independent
of the backend engine. Flow and HoVer training import these scientific
primitives directly.

## Boundary and data flow

1. The host prepares approved instance/class labels, split membership and field
   sampling. For crossing objects, it rasterizes complete geometry on a bounded
   object-local grid, calls `build_managed_targets`, then crops the resulting
   arrays. Cropping the object first would change its flow center/HV centroid.
2. The host supplies already augmented RGB fields and `ManagedSegmentationBatch`
   tensors: directional targets `[B,2,H,W]`, binary foreground `[B,H,W]`, boolean
   supervision and real-pixel masks, and optional integer class targets.
3. The host validates the batch before model forward. It passes float32 predictions
   to `managed_segmentation_loss`, which returns `ManagedLossResult`: the
   differentiable `total_sum`, scientific `task_sums`, diagnostic `component_sums`
   and actual `field_count`. Diagnostic terms may overlap and must not be summed
   again into the objective.
4. The host owns backward, optimizer steps, LR scheduling, validation, selection,
   reports, cancellation state, checkpoint persistence and publication. These
   functions do not load datasets or weights, write checkpoints or contact W&B.

`head_type="flow"` predicts `[dY,dX,foreground_logit]`; `head_type="hovernet"`
predicts `[V,H,np_background,np_foreground]`. Optional classification logits
precede those channels. `class_count=0` disables classification; otherwise it is
the number of class channels including background zero. Class 255 is permitted
only outside effective support. No simultaneous flow-plus-HoVer prediction is
implied by sharing an engine.

Effective support is `supervision_mask & real_pixel_mask`. Each field's loss is
reduced independently on that support. Microbatches return sums; accumulation
divides once by the actual number of fields, including a partial final group.
Flow uses the established scaled-flow MSE and foreground BCE; HoVer uses
HV-MSE, derivative MSE and foreground CE/Dice. Managed HoVer invalidates
derivative neighborhoods crossing masked pixels or the tensor exterior; standalone
research callers retain their existing exterior convention by default.

Classification supports CE, focal Tversky or both, with resolved class weighting.
`ManagedObjective` combines fixed task losses or learned homoscedastic task
weights. Its trainable parameters must belong to the host's optimizer. In
HistoForge, the backend engine normalizes and finite-checks all owned gradients;
optional clipping retains the model-only norm. It rejects nonfinite losses,
gradients and invalid schedules without retry or a completion callback, but does
not promise rollback of earlier steps or arbitrary optimizer side effects.
These are backend engine guarantees, not behavior supplied by this package.

## Lifecycle and reuse

Managed QuPath training follows the backend's frozen training contract: no W&B
or TensorBoard account, no `latest` resume checkpoint, and only the final and
best-primary-validation candidates. Independent research runs retain their
normal W&B, receipt and semantic checkpoint policies. Do not substitute one
lifecycle or loss-reduction policy for the other.

The directory is self-contained: Python 3.10+, NumPy, SciPy, PyTorch and
`fastremap` are its runtime dependencies. Vendor the complete directory as part of
the shared runtime from one exact commit, together with upstream `LICENSE`, `NOTICE` and a checked
per-file SHA-256 manifest. Do not use a mutable sibling checkout or mix files
from different commits. Product-side adoption and serving validation are separate
from this repository's unit/regression tests.

## Classification weighting

The reusable loss supports weighting CE alone or both CE and focal Tversky.
CE uses its weighted-mean denominator. Weighted Tversky multiplies each
per-class focal loss by its class weight and averages over all class channels,
not the sum of weights. The host chooses and records the policy.

`flow_loss_weighting = pixel | equal_instance` is Cellpose-only, with `pixel`
as default. Sampling, class-count estimation, validation and checkpoint
selection belong to the caller, not these scientific primitives.
