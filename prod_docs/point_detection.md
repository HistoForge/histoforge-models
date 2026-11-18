# Center-point detection

The shared library defines the network, targets, loss and decoder for
center-point detection. HistoForge backend 0.1.5 supports finite FP32 inference.
Managed training and a trained MIDOG module are outside this support.

## Network and targets

Construct the untrained network with
`common.build({"head_type": "point_detection", "num_classes": class_count})`.
It reuses the fixed DINOv3 adapter, stride-4 features and full-resolution class
readout. It has no flow output, background channel or segmentation-based class
initializer. Backbone initialization and checkpoint selection remain explicit
training responsibilities; construction downloads no weights.

`managed_training.build_point_targets` accepts every point and its zero-based
class index, an explicit Gaussian spread in model pixels, and a boolean mask
containing only reviewed real image pixels. Transform points with the RGB image
before building targets. Context and padding must be excluded from that mask.

Coordinates use pixel edges: `(0, 0)` is the image corner; the first pixel center
is `(0.5, 0.5)`. Each mark's containing pixel receives target 1. Gaussian
shoulders are max-combined within each class and remain below 1. They control
loss weighting near a center, not object size. Out-of-bounds marks, marks outside
support and same-class marks occupying one output pixel fail explicitly.

## Loss

`managed_training.point_heatmap_loss` accepts BCHW logits and float32 targets,
plus boolean BHW support. It returns one loss per field. Callers own batch and
gradient-accumulation reduction. Its CenterNet-style modified focal terms are:

```text
p = sigmoid(logit)
positive (target = 1): -log(p) * (1 - p)^2
negative (target < 1): -log(1 - p) * p^2 * (1 - target)^4
field loss = sum of supported terms / max(number of centers, 1)
```

Loss arithmetic uses float32 and stable log-sigmoid operations. Empty reviewed
fields provide negative supervision; context and padding have zero gradient.
This follows the center-heatmap objective in [CenterNet (2019)](https://arxiv.org/abs/1904.07850),
without adopting its box-size or coordinate-offset heads.

## Inference

`point_inference.run_point_inference` uses an explicit existing variable-field
profile. It normalizes real pixels once, blends overlapping raw logits with
the shared field executor, then calls `decode_point_logits` once. This entry
point uses FP32; it does not qualify BF16 or large-region inference.

The decoder requires an explicit score threshold, square peak-neighborhood
radius and result-count limit. It finds class-specific maxima on raw logits,
before sigmoid rounding can flatten unequal values. A connected equal-valued
plateau yields its first pixel in row-major order. A plateau touching a higher
value, or with a higher value in any member's declared neighborhood, is rejected.
Unsupported pixels neither emit nor suppress peaks. Scores equal to the
threshold are retained; exceeding the count limit raises
`PointDetectionLimitExceeded` without truncation. Other invalid inputs remain
ordinary `ValueError` failures, so callers can distinguish a result limit from
invalid model output.

Results contain `xy` (float64 pixel centers), `class_indices` (int64) and `scores`
(float32 independent sigmoid values, evaluated in float64 then rounded once).
They sort by descending score, class,
y and x. Scores have not been calibrated. The application owns slide-coordinate
transforms, exact ROI/parent admission and native point materialization.

The spread, threshold and neighborhood are experiment settings, not MIDOG's
7.5-micrometer scoring tolerance. Choose them using development validation and
retain the settings with the selected checkpoint. Closely spaced figures and
tile boundaries require explicit evaluation. At radius 1, same-class centers
in adjacent pixels compete or form one plateau, so both cannot be returned at
those locations. Training retains those marks; check their frequency at the
chosen model resolution before freezing the decoder. A center detector
provides no cell-outline measurements.

## Checkpoints and verification

Complete point checkpoints have head identity 2 and mandatory class-readout
tensors. `common.build_and_load` checks all keys, shapes, dtypes and metadata
before loading. Existing flow/HoVer checkpoint layouts remain unchanged.

The focused tests use synthetic marks/images, a local lightweight trunk with
the actual adapter/head, one optimizer update, strict safetensors reload and
ordinary tiled prediction. They also cover empty fields, masked gradients,
close/tied peaks, result overflow and existing segmentation readouts. These
checks establish code behavior, not MIDOG accuracy or product availability.
