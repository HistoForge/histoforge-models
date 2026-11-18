"""Scientific contracts for center targets, masked loss and scored peaks."""

import math

import numpy as np
import pytest
import torch

from histoforge_models.pathology import point_inference
from histoforge_models.pathology.managed_training.point_losses import point_heatmap_loss
from histoforge_models.pathology.managed_training.point_targets import (
    build_point_targets,
)
from histoforge_models.pathology.point_inference import (
    PointDetectionLimitExceeded,
    decode_point_logits,
    run_point_inference,
)


def _targets(points=((2.3, 1.8),), classes=(0,), support=None, **options):
    return build_point_targets(
        np.asarray(points, dtype=np.float64).reshape(-1, 2),
        np.asarray(classes, dtype=np.int64),
        support_mask=np.ones((5, 9), dtype=bool) if support is None else support,
        **{"class_count": 1, "sigma_model_px": 1.0, **options},
    )


def _decode(logits, **options):
    return decode_point_logits(
        logits,
        **{
            "score_threshold": 0.5,
            "peak_radius_model_px": 1,
            "max_detections": 100,
            **options,
        },
    )


def test_fractional_marks_have_exact_positive_cells_and_max_combined_shoulders():
    points = np.array([[2.3, 1.8], [5.9, 1.1]])
    saved = points.copy()
    target = build_point_targets(
        points,
        np.zeros(2, dtype=np.int64),
        class_count=1,
        sigma_model_px=1,
        support_mask=np.ones((5, 9), dtype=bool),
    )
    yy, xx = np.indices((5, 9))
    expected = np.maximum(
        np.exp(-((xx - 2) ** 2 + (yy - 1) ** 2) / 2),
        np.exp(-((xx - 5) ** 2 + (yy - 1) ** 2) / 2),
    )
    np.testing.assert_allclose(target[0], expected, rtol=1e-6, atol=1e-4)
    assert target.dtype == np.float32
    assert np.count_nonzero(target == 1) == 2
    np.testing.assert_array_equal(points, saved)


def test_pixel_edges_channel_zero_and_independent_colocated_classes():
    target = _targets(
        points=((0, 0), (8.999, 4.999), (0.1, 0.1)),
        classes=(0, 0, 1),
        class_count=2,
    )
    assert target.shape == (2, 5, 9)
    assert target[0, 0, 0] == target[0, 4, 8] == target[1, 0, 0] == 1
    assert np.count_nonzero(target == 1) == 3


def test_targets_exclude_context_and_allow_no_positive_marks():
    support = np.ones((5, 9), dtype=bool)
    support[1, 3] = False
    target = _targets(support=support)
    assert target[0, 1, 3] == 0
    negative = _targets(points=(), classes=(), support=support)
    assert np.count_nonzero(negative) == 0


@pytest.mark.parametrize("sigma", [1e-300, 1e300])
def test_extreme_valid_sigma_still_has_only_the_real_center(sigma):
    result = _targets(sigma_model_px=sigma)
    assert np.isfinite(result).all()
    assert np.count_nonzero(result == 1) == 1


@pytest.mark.parametrize(
    "points,classes,options,match",
    [
        (((2.1, 1.1), (2.9, 1.9)), (0, 0), {}, "collide"),
        (((-0.01, 0),), (0,), {}, "pixel edges"),
        (((9, 1),), (0,), {}, "pixel edges"),
        (((1, 5),), (0,), {}, "pixel edges"),
        (((math.nan, 1),), (0,), {}, "finite"),
        (((1, math.inf),), (0,), {}, "finite"),
        (((1, 1),), (1,), {}, "channel"),
        (((1, 1),), (-1,), {}, "channel"),
        (((1, 1),), (), {}, "channel"),
        (((1, 1),), (0,), {"class_count": True}, "class_count"),
        (((1, 1),), (0,), {"sigma_model_px": 0}, "sigma"),
        (((1, 1),), (0,), {"sigma_model_px": math.inf}, "sigma"),
        (((1, 1),), (0,), {"sigma_model_px": True}, "sigma"),
    ],
)
def test_invalid_point_targets_fail_instead_of_repairing(
    points, classes, options, match
):
    with pytest.raises(ValueError, match=match):
        _targets(points=points, classes=classes, **options)


def test_center_in_a_support_hole_fails():
    support = np.ones((5, 9), dtype=bool)
    support[1, 2] = False
    with pytest.raises(ValueError, match="outside supervised"):
        _targets(support=support)
    with pytest.raises(ValueError, match="supervised real pixels"):
        _targets(points=(), classes=(), support=np.zeros((5, 9), dtype=bool))


def test_modified_focal_loss_matches_hand_computation_at_zero_logits():
    logits = torch.zeros(1, 1, 1, 4, requires_grad=True)
    target = torch.tensor([[[[1.0, 0.5, 0.0, 0.0]]]])
    support = torch.tensor([[[True, True, True, False]]])
    value = point_heatmap_loss(logits, target, support)
    expected = -math.log(0.5) * 0.5**2 * (1 + (1 - 0.5) ** 4 + 1)
    torch.testing.assert_close(value, torch.tensor([expected]))
    value.sum().backward()
    assert logits.grad[0, 0, 0, 0] < 0
    assert logits.grad[0, 0, 0, 1] > 0
    assert logits.grad[0, 0, 0, 3] == 0


def test_positive_empty_and_mixed_losses_are_finite_and_partition_independent():
    targets = torch.from_numpy(np.stack([_targets(), _targets(points=(), classes=())]))
    logits = (
        torch.linspace(-2, 2, targets.numel()).reshape(targets.shape).requires_grad_()
    )
    support = torch.ones(2, 5, 9, dtype=torch.bool)
    support[:, 3:] = False
    full = point_heatmap_loss(logits, targets, support)
    partitioned = torch.cat(
        [
            point_heatmap_loss(
                logits[i : i + 1], targets[i : i + 1], support[i : i + 1]
            )
            for i in range(2)
        ]
    )
    torch.testing.assert_close(full, partitioned, rtol=0, atol=0)
    full.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1].sum() > 0
    assert torch.count_nonzero(logits.grad[:, :, 3:]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_extreme_logits_use_stable_float32_loss(dtype):
    logits = torch.tensor([[[[-90, 90, 90, -90]]]], dtype=dtype, requires_grad=True)
    targets = torch.tensor([[[[1.0, 0, 1, 0]]]])
    value = point_heatmap_loss(logits, targets, torch.ones(1, 1, 4, dtype=torch.bool))
    assert value.dtype == torch.float32
    torch.testing.assert_close(value, torch.tensor([90.0]))
    value.sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("which", ["logits", "targets"])
def test_nonfinite_loss_inputs_fail(which):
    logits = torch.zeros(1, 1, 1, 2)
    target = logits.clone()
    (logits if which == "logits" else target)[0, 0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        point_heatmap_loss(logits, target, torch.ones(1, 1, 2, dtype=torch.bool))


def test_loss_requires_supported_fields_and_valid_targets():
    logits = torch.zeros(1, 1, 1, 2)
    support = torch.ones(1, 1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="support"):
        point_heatmap_loss(logits, logits, ~support)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        point_heatmap_loss(logits, torch.full_like(logits, 1.1), support)
    with pytest.raises(ValueError, match="BCHW"):
        point_heatmap_loss(logits, logits[:, 0], support)


def test_close_peaks_classes_scores_and_deterministic_order():
    logits = np.full((5, 9, 2), -10, dtype=np.float32)
    logits[1, 2, 0] = logits[1, 5, 0] = 2
    logits[1, 2, 1] = 3
    result = _decode(logits)
    np.testing.assert_array_equal(result["xy"], [[2.5, 1.5], [2.5, 1.5], [5.5, 1.5]])
    np.testing.assert_array_equal(result["class_indices"], [1, 0, 0])
    np.testing.assert_allclose(
        result["scores"], [1 / (1 + math.exp(-3)), *([1 / (1 + math.exp(-2))] * 2)]
    )
    for name, values in result.items():
        np.testing.assert_array_equal(values, _decode(logits)[name])


def test_tied_plateau_emits_first_pixel_but_a_lower_flat_shoulder_is_not_a_peak():
    logits = np.full((5, 12, 1), -10, dtype=np.float32)
    logits[1:3, 1:4, 0] = 2
    logits[3, 6:10, 0] = 1
    logits[3, 10, 0] = 3
    result = _decode(logits)
    np.testing.assert_array_equal(result["xy"], [[10.5, 3.5], [1.5, 1.5]])


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 5, 1), (5, 1, 1), (2, 3, 1)])
def test_constant_plateau_including_single_pixel_has_one_defined_location(shape):
    result = _decode(np.ones(shape, dtype=np.float32))
    np.testing.assert_array_equal(result["xy"], [[0.5, 0.5]])


def test_logit_comparison_prevents_sigmoid_saturation_from_merging_unequal_peaks():
    logits = np.array([[[40.0], [50.0], [40.0]]], dtype=np.float32)
    result = _decode(logits)
    np.testing.assert_array_equal(result["xy"], [[1.5, 0.5]])
    np.testing.assert_array_equal(result["scores"], [1.0])


def test_float16_maps_are_promoted_without_changing_peaks_or_scores():
    raw = np.full((3, 7, 1), -10, dtype=np.float16)
    raw[1, (1, 5), 0] = (2, 3)
    expected = _decode(raw.astype(np.float32))
    actual = _decode(raw)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_upcasting_identical_float32_logits_cannot_change_threshold_admission():
    value = np.float32(-1.0000000116860974e-7)
    raw = np.array([[[value], [value]]], dtype=np.float32)
    for threshold in (0.0, 0.5):
        expected = _decode(raw, score_threshold=threshold)
        actual = _decode(raw.astype(np.float64), score_threshold=threshold)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])


def test_support_excludes_high_context_and_preserves_boundary_peak():
    logits = np.array([[[0.0], [20.0], [-10.0], [-10.0], [1.0]]], dtype=np.float32)
    support = np.array([[True, False, True, True, True]])
    saved = logits.copy()
    logits.flags.writeable = False
    result = _decode(logits, support_mask=support)
    np.testing.assert_array_equal(result["xy"], [[4.5, 0.5], [0.5, 0.5]])
    assert result["scores"][1] == 0.5  # Inclusive threshold.
    np.testing.assert_array_equal(logits, saved)


def test_peak_radius_rejects_whole_plateau_if_any_part_has_a_higher_neighbor():
    logits = np.full((1, 9, 1), -10, dtype=np.float32)
    logits[0, 1:4, 0] = 2
    logits[0, 5, 0] = 3
    result = _decode(logits, peak_radius_model_px=2)
    np.testing.assert_array_equal(result["xy"], [[5.5, 0.5]])


def test_empty_population_and_limit_failures_never_truncate():
    empty = _decode(np.full((3, 5, 1), -10, dtype=np.float32))
    assert empty["xy"].shape == (0, 2)
    assert empty["class_indices"].shape == empty["scores"].shape == (0,)
    logits = np.full((3, 5, 1), -10, dtype=np.float32)
    logits[1, (0, 4), 0] = 1
    assert len(_decode(logits, max_detections=2)["xy"]) == 2
    with pytest.raises(PointDetectionLimitExceeded, match="max_detections"):
        _decode(logits, max_detections=1)


@pytest.mark.parametrize(
    "options",
    [
        {"score_threshold": -0.1},
        {"score_threshold": math.nan},
        {"score_threshold": True},
        {"peak_radius_model_px": 0},
        {"peak_radius_model_px": 1.0},
        {"max_detections": False},
    ],
)
def test_invalid_decode_settings_fail(options):
    with pytest.raises(ValueError):
        _decode(np.zeros((1, 1, 1), dtype=np.float32), **options)


def test_nonfinite_network_maps_fail_even_outside_support():
    with pytest.raises(ValueError, match="finite"):
        _decode(np.array([[[math.nan]]]), support_mask=np.zeros((1, 1), dtype=bool))


FIELD_PROFILE = {
    "planner": "variable-field-v2",
    "minSideModelPx": 32,
    "maxSideModelPx": 64,
    "preferredTileSideModelPx": 32,
    "alignmentModelPx": 16,
    "outerArtificialPaddingModelPx": 8,
    "smallRoiContext": False,
    "tileOverlap": 0.25,
    "aggregation": "cellpose-tapered-mean-v1",
    "preprocessing": "normalize99-real-image-v1",
}


class PixelCenterNetwork(torch.nn.Module):
    """Known RGB-to-logit mapping through the actual field execution path."""

    head_type = "point_detection"
    nclasses = 1
    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, images):
        self.calls.append(tuple(images.shape))
        return 20 * images[:, :1] - 10, images.new_zeros((len(images), 256))


def test_real_field_blending_emits_one_point_per_event_without_tile_duplicates():
    image = np.zeros((40, 75, 3), dtype=np.float32)
    image[19, 31, 0] = 1  # One event seen by overlapping neural fields.
    image[35, 70, 0] = 1
    image[0, 0, 0] = 1
    support = np.ones(image.shape[:2], dtype=bool)
    support[0, 0] = False
    network = PixelCenterNetwork()
    result = run_point_inference(
        network,
        image,
        field_profile=FIELD_PROFILE,
        score_threshold=0.9,
        peak_radius_model_px=1,
        max_detections=2,
        batch_size=2,
        normalize=False,
        real_pixel_mask=support,
    )
    assert len(network.calls) > 1
    assert all(shape[1:] == (3, 64, 32) for shape in network.calls)
    np.testing.assert_array_equal(result["xy"], [[31.5, 19.5], [70.5, 35.5]])


def test_inference_validates_head_and_options_before_forward(monkeypatch):
    monkeypatch.setattr(
        point_inference, "run_model_fields", lambda *a, **k: pytest.fail("forward")
    )
    with pytest.raises(ValueError, match="point-detection head"):
        run_point_inference(
            object(),
            None,
            field_profile=FIELD_PROFILE,
            score_threshold=0.5,
            peak_radius_model_px=1,
            max_detections=10,
        )


def test_filtering_uses_the_returned_float32_score_without_weak_scalar_rounding():
    raw = np.array([[[1e-10]]], dtype=np.float64)
    # Its sigmoid exceeds 0.5 in float64 but rounds to exactly 0.5 when returned.
    assert len(_decode(raw, score_threshold=0.5)["xy"]) == 1
    assert len(_decode(raw, score_threshold=0.5 + 1e-12)["xy"]) == 0


def test_no_supported_pixels_and_a_radius_larger_than_the_image():
    raw = np.ones((3, 5, 1), dtype=np.float32)
    assert len(_decode(raw, support_mask=np.zeros((3, 5), dtype=bool))["xy"]) == 0
    result = _decode(raw, peak_radius_model_px=10**50)
    np.testing.assert_array_equal(result["xy"], [[0.5, 0.5]])


def test_loss_rejects_low_precision_targets_that_can_turn_shoulders_into_centers():
    logits = torch.zeros(1, 1, 1, 2)
    with pytest.raises(ValueError, match="float32 targets"):
        point_heatmap_loss(
            logits, logits.bfloat16(), torch.ones(1, 1, 2, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="score_threshold"):
        run_point_inference(
            PixelCenterNetwork(),
            None,
            field_profile=FIELD_PROFILE,
            score_threshold=-1,
            peak_radius_model_px=1,
            max_detections=10,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_non_fp32_metadata_fails_before_inference(monkeypatch, dtype):
    monkeypatch.setattr(
        point_inference, "run_model_fields", lambda *a, **k: pytest.fail("forward")
    )
    network = PixelCenterNetwork()
    network.dtype = dtype
    with pytest.raises(ValueError, match="FP32 network weights"):
        run_point_inference(
            network,
            None,
            field_profile=FIELD_PROFILE,
            score_threshold=0.5,
            peak_radius_model_px=1,
            max_detections=10,
        )


def test_point_validation_disables_ambient_autocast_only_during_its_forward():
    class ConvolutionalCenterNetwork(PixelCenterNetwork):
        def __init__(self):
            super().__init__()
            self.readout = torch.nn.Conv2d(3, 1, 1)
            with torch.no_grad():
                self.readout.weight.zero_()
                self.readout.weight[0, 0, 0, 0] = 20
                self.readout.bias.fill_(-10)
            self.forward_dtypes = []

        def forward(self, images):
            output = self.readout(images)
            self.forward_dtypes.append(output.dtype)
            return output, images.new_zeros((len(images), 256))

    network = ConvolutionalCenterNetwork()
    image = np.zeros((10, 12, 3), dtype=np.float32)
    image[3, 7, 0] = 1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = run_point_inference(
            network,
            image,
            field_profile=FIELD_PROFILE,
            score_threshold=0.9,
            peak_radius_model_px=1,
            max_detections=1,
            normalize=False,
        )
        assert torch.is_autocast_enabled("cpu")
    assert network.forward_dtypes and set(network.forward_dtypes) == {torch.float32}
    np.testing.assert_array_equal(result["xy"], [[7.5, 3.5]])


def test_adjacent_centers_are_retained_as_labels_but_compete_during_peak_decoding():
    target = _targets(points=((2.3, 1.8), (3.1, 1.9)), classes=(0, 0))
    assert np.count_nonzero(target == 1) == 2
    result = _decode((10 * target - 5).transpose(1, 2, 0))
    np.testing.assert_array_equal(result["xy"], [[2.5, 1.5]])
