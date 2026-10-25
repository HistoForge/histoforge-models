"""Variable-field execution with independent geometry and blending oracles."""

import numpy as np
import pytest

from histoforge_models import core
from histoforge_models.pathology.field_inference import (
    OUTER_ARTIFICIAL_PADDING_MODEL_PX,
    accumulate_variable_field_window,
    run_model_fields,
)
from histoforge_models.pathology.normalization import normalize99


def _profile(**changes):
    """Small executable test profile, not a qualified model default."""
    return {
        "planner": "variable-field-v2",
        "minSideModelPx": 32,
        "maxSideModelPx": 64,
        "preferredTileSideModelPx": 32,
        "alignmentModelPx": 16,
        "outerArtificialPaddingModelPx": OUTER_ARTIFICIAL_PADDING_MODEL_PX,
        "smallRoiContext": False,
        "tileOverlap": 0.25,
        "aggregation": "cellpose-tapered-mean-v1",
        "preprocessing": "normalize99-real-image-v1",
        **changes,
    }


def _run(image, **kwargs):
    return run_model_fields(None, image, fixed_field_defaults=(32, 0.1), **kwargs)


def _raw(batch):
    """Context-dependent raw channels with no cross-sample interaction."""
    raw = batch.copy()
    raw[:, 1] **= 2
    raw[:, 2] = batch.mean(axis=(1, 2, 3), keepdims=True)[:, 0]
    return raw


def _normalized_image(image, support):
    """Independent per-channel oracle over the complete supplied real image."""
    pixels = image.astype(np.float32)
    result = np.zeros_like(pixels)
    for channel in range(3):
        values = pixels[..., channel][support]
        if values.size:
            low, high = np.percentile(values, (1, 99))
            if high - low > 0.001:
                result[..., channel][support] = (values - low) / (high - low)
    return result


def _taper(height, width):
    """Independent Cellpose kernel oracle; production reuses transforms._taper_mask."""
    side = max(224, height, width)
    distance = np.abs(np.arange(side) - (side - 1) / 2)
    axis = 1 / (1 + np.exp((distance - (side / 2 - 20)) / 7.5))
    y0, x0 = side // 2 - height // 2, side // 2 - width // 2
    return np.outer(axis, axis)[y0 : y0 + height, x0 : x0 + width]


CASES = [
    ((9, 22), {}, [(-8, -8, 38, 25)]),
    ((33, 17), {}, [(-8, -8, 33, 49)]),
    (
        (64, 64),
        {},
        [(x, y, 32, 32) for y in (-8, 16, 40) for x in (-8, 16, 40)],
    ),
    ((17, 131), {}, [(x, -8, 32, 33) for x in (-8, 16, 40, 64, 88, 107)]),
    ((131, 17), {}, [(-8, y, 33, 32) for y in (-8, 16, 40, 64, 88, 107)]),
    (
        (97, 85),
        {},
        [(x, y, 32, 32) for y in (-8, 16, 40, 64, 73) for x in (-8, 16, 40, 61)],
    ),
    ((17, 65), {"tileOverlap": 0}, [(x, -8, 32, 33) for x in (-8, 24, 41)]),
    (
        (17, 65),
        {"tileOverlap": 0.75},
        [(x, -8, 32, 33) for x in (-8, 0, 8, 16, 24, 32, 40, 41)],
    ),
    (
        (601, 769),
        {"minSideModelPx": 256, "maxSideModelPx": 640, "preferredTileSideModelPx": 512},
        [(-8, -8, 512, 617), (265, -8, 512, 617)],
    ),
]


@pytest.mark.parametrize("shape,changes,rectangles", CASES)
@pytest.mark.parametrize("batch_size", [1, 3, 8])
@pytest.mark.parametrize("normalize", [False, True])
def test_exact_fields_padding_normalization_and_blending(
    monkeypatch, shape, changes, rectangles, batch_size, normalize
):
    """Verify the complete result, forward pixels, padded taper, and batch order."""
    image = np.random.default_rng(87).integers(0, 256, (*shape, 3), np.uint8)
    image[:, shape[1] // 2 :] //= 4
    original = image.copy()
    support = np.ones(shape, bool)
    support[:3, :7] = False
    profile = _profile(**changes)
    calls = []

    def forward(_net, batch):
        calls.append(batch.copy())
        return _raw(batch), np.zeros((len(batch), 256), np.float32)

    monkeypatch.setattr(core, "_forward", forward)
    actual = _run(
        image,
        field_profile=profile,
        batch_size=batch_size,
        normalize=normalize,
        real_pixel_mask=support,
    )
    expected = np.zeros((3, *shape), np.float32)
    weights = np.zeros(shape, np.float64)
    expected_fields = []
    prepared_image = (
        _normalized_image(image, support)
        if normalize
        else np.where(support[..., None], image, 0).astype(np.float32)
    )
    for x, y, w, h in rectangles:
        padded_h = 16 * int(np.ceil(max(h, profile["minSideModelPx"]) / 16))
        padded_w = 16 * int(np.ceil(max(w, profile["minSideModelPx"]) / 16))
        top, left = (padded_h - h) // 2, (padded_w - w) // 2
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(shape[1], x + w), min(shape[0], y + h)
        field = np.zeros((h, w, 3), dtype=np.float32)
        field_y = slice(y0 - y, y1 - y)
        field_x = slice(x0 - x, x1 - x)
        field[field_y, field_x] = prepared_image[y0:y1, x0:x1]
        padded = np.zeros((3, padded_h, padded_w), np.float32)
        padded[:, top : top + h, left : left + w] = field.transpose(2, 0, 1)
        expected_fields.append(padded)
        raw = _raw(padded[None])[0, :, top : top + h, left : left + w]
        taper = _taper(padded_h, padded_w)[top : top + h, left : left + w]
        expected[:, y0:y1, x0:x1] += raw[:, field_y, field_x] * taper[field_y, field_x]
        weights[y0:y1, x0:x1] += taper[field_y, field_x]
    expected /= weights
    np.testing.assert_array_equal(np.concatenate(calls), np.stack(expected_fields))
    np.testing.assert_array_equal(actual, expected.transpose(1, 2, 0))
    np.testing.assert_array_equal(image, original)
    assert actual.dtype == np.float32
    assert [len(batch) for batch in calls] == [
        min(batch_size, len(rectangles) - offset)
        for offset in range(0, len(rectangles), batch_size)
    ]
    assert weights.min() > 0


def test_single_field_keeps_centered_padding_and_ignores_artificial_values(monkeypatch):
    """An all-padding field remains zero; real black/white pixels still participate."""
    image = np.arange(9 * 17 * 3, dtype=np.float32).reshape(9, 17, 3)
    image[0] = 0
    image[1] = 255
    support = np.ones((9, 17), bool)
    support[-2:] = False
    image[-2:] = np.nan
    calls = []

    def forward(_net, batch):
        calls.append(batch.copy())
        return batch, np.zeros((len(batch), 256), np.float32)

    monkeypatch.setattr(core, "_forward", forward)
    _run(image, field_profile=_profile(), real_pixel_mask=support)
    expected = np.pad(
        normalize99(image, real_pixel_mask=support).transpose(2, 0, 1),
        ((0, 0), (11, 12), (15, 16)),
    )
    np.testing.assert_array_equal(calls[0][0], expected)
    assert not _run(
        image, field_profile=_profile(), real_pixel_mask=np.zeros_like(support)
    ).any()


@pytest.mark.parametrize("batch_size", [1, 4])
def test_exact_maximum_matches_historical_fixed_field_execution(
    monkeypatch, batch_size
):
    """The v2 border reproduces fixed 1024 field preparation and raw blending."""
    side = 1024
    image = np.random.default_rng(710).integers(0, 256, (side, side, 3), dtype=np.uint8)
    support = np.ones((side, side), dtype=bool)
    support[:11, :7] = False
    calls = []

    def capture(_net, batch):
        calls.append(batch.copy())
        return _raw(batch), np.zeros((len(batch), 256), np.float32)

    monkeypatch.setattr(core, "_forward", capture)
    fixed = run_model_fields(
        None,
        image,
        fixed_field_defaults=(side, 0.1),
        bsize=side,
        tile_overlap=0.1,
        batch_size=batch_size,
        real_pixel_mask=support,
    )
    fixed_fields = np.concatenate(calls)
    calls.clear()
    variable = run_model_fields(
        None,
        image,
        fixed_field_defaults=(side, 0.1),
        field_profile=_profile(
            minSideModelPx=512,
            maxSideModelPx=side,
            preferredTileSideModelPx=side,
        ),
        batch_size=batch_size,
        real_pixel_mask=support,
    )
    variable_fields = np.concatenate(calls)

    np.testing.assert_array_equal(variable_fields, fixed_fields)
    np.testing.assert_array_equal(variable, fixed)
    assert variable_fields.shape == (4, 3, side, side)


@pytest.mark.parametrize(
    "shape,expected_count,expected_field_shape",
    [
        ((512, 512), 1, (528, 528)),
        ((512, 768), 1, (528, 784)),
        ((1008, 1008), 1, (1024, 1024)),
        ((1009, 1009), 4, (1024, 1024)),
        ((768, 1024), 2, (784, 1024)),
        ((1024, 2048), 6, (1024, 1024)),
    ],
)
def test_outer_border_field_counts_and_shapes(
    monkeypatch, shape, expected_count, expected_field_shape
):
    calls = []

    def capture(_net, batch):
        calls.append(batch.shape)
        return np.zeros((len(batch), 1, *batch.shape[2:]), dtype=np.float32), None

    monkeypatch.setattr(core, "_forward", capture)
    result = _run(
        np.zeros((*shape, 3), dtype=np.uint8),
        field_profile=_profile(
            minSideModelPx=512,
            maxSideModelPx=1024,
            preferredTileSideModelPx=1024,
        ),
        batch_size=3,
        normalize=False,
    )

    assert sum(batch_shape[0] for batch_shape in calls) == expected_count
    assert {batch_shape[2:] for batch_shape in calls} == {expected_field_shape}
    assert result.shape == (*shape, 1)


@pytest.mark.parametrize("wrapper", ["flow", "hovernet", "semantic"])
def test_wrappers_blend_every_raw_channel_then_decode_once(monkeypatch, wrapper):
    """Both segmentation families and class probabilities use the same raw raster."""
    import torch
    from histoforge_models import dynamics
    from histoforge_models.pathology import flow_inference, hovernet_inference, semantic_inference

    calls, decoded = [], []
    channels = 6 if wrapper == "hovernet" else 5

    def forward(_net, batch):
        calls.append(batch.copy())
        raw = np.zeros((len(batch), channels, *batch.shape[2:]), np.float32)
        raw[:, 1] = 10
        raw[:, -1] = batch[:, 0]
        return raw, np.zeros((len(batch), 256), np.float32)

    def decode(values, *args, **kwargs):
        decoded.append(values.copy())
        return np.ones(values.shape[-2:], np.int32)

    monkeypatch.setattr(core, "_forward", forward)
    monkeypatch.setattr(dynamics, "compute_masks", decode)
    monkeypatch.setattr(hovernet_inference, "watershed_from_hv_maps", decode)
    function = {
        "flow": flow_inference.run_flow_inference,
        "hovernet": hovernet_inference.run_hovernet_inference,
        "semantic": semantic_inference.run_semantic_inference,
    }[wrapper]
    image = np.random.default_rng(72).random((17, 131, 3)).astype(np.float32)
    net = torch.nn.Conv2d(3, channels, 1)
    result = function(net, image, num_classes=2, field_profile=_profile(), batch_size=4)
    assert [len(batch) for batch in calls] == [4, 2]
    assert result["class_map"].shape == image.shape[:2]
    assert np.all(result["class_map"] == 1)
    if wrapper == "semantic":
        assert decoded == []
        np.testing.assert_allclose(result["probs"].sum(axis=0), 1)
    else:
        assert len(decoded) == 1
        assert decoded[0].shape[-2:] == image.shape[:2]
        assert result["votes_per_instance"] == {1: 1}


@pytest.mark.parametrize(
    "changes",
    [
        {"planner": "variable-field-v1"},
        {"planner": "future"},
        {"outerArtificialPaddingModelPx": 0},
        {"outerArtificialPaddingModelPx": 7},
        {"outerArtificialPaddingModelPx": True},
        {"aggregation": "different"},
        {"preprocessing": "other"},
        {"preprocessing": "normalize99-real-model-field-v1"},
        {"minSideModelPx": 0},
        {"maxSideModelPx": 48.5},
        {"alignmentModelPx": 0},
        {"minSideModelPx": 48},
        {"maxSideModelPx": 63},
        {"preferredTileSideModelPx": 31},
        {"preferredTileSideModelPx": 80},
        {"tileOverlap": 1},
        {"tileOverlap": float("nan")},
    ],
)
def test_unsupported_contract_fails_before_forward(monkeypatch, changes):
    monkeypatch.setattr(core, "_forward", lambda *_: pytest.fail("unexpected forward"))
    with pytest.raises(ValueError):
        _run(np.zeros((17, 33, 3)), field_profile=_profile(**changes))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bsize": 32},
        {"tile_overlap": 0.25},
        {"batch_size": 0},
        {"batch_size": True},
        {"real_pixel_mask": np.ones((17, 33))},
        {"real_pixel_mask": np.ones((17, 32), bool)},
    ],
)
def test_conflicting_parameters_and_bad_support_fail_before_forward(
    monkeypatch, kwargs
):
    monkeypatch.setattr(core, "_forward", lambda *_: pytest.fail("unexpected forward"))
    with pytest.raises(ValueError):
        _run(np.zeros((17, 33, 3)), field_profile=_profile(), **kwargs)


def test_missing_outer_artificial_padding_fails_before_forward(monkeypatch):
    monkeypatch.setattr(core, "_forward", lambda *_: pytest.fail("unexpected forward"))
    profile = _profile()
    del profile["outerArtificialPaddingModelPx"]
    with pytest.raises(ValueError, match="outerArtificialPaddingModelPx"):
        _run(np.zeros((17, 33, 3)), field_profile=profile)


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((1, 3, 32, 31), np.float32),
        np.zeros((2, 3, 32, 32), np.float32),
        np.zeros((1, 0, 32, 32), np.float32),
        np.full((1, 3, 32, 32), np.nan, np.float32),
    ],
)
def test_bad_network_output_is_not_decoded_as_a_valid_result(monkeypatch, output):
    monkeypatch.setattr(core, "_forward", lambda *_: (output, None))
    with pytest.raises(ValueError, match="model output"):
        _run(np.zeros((17, 17, 3)), field_profile=_profile())


@pytest.mark.parametrize("overlap", [0.0, 0.75])
def test_offset_window_matches_complete_variable_fields_with_full_context(overlap):
    """Isolate tiling: already-prepared pixels agree with sufficient context."""
    image = np.random.default_rng(912).random((131, 131, 3)).astype(np.float32)
    support = np.ones(image.shape[:2], dtype=bool)
    profile = _profile(tileOverlap=overlap)

    def forward(fields):
        return _raw(fields)

    complete = accumulate_variable_field_window(
        image,
        field_profile=profile,
        target_height=131,
        target_width=131,
        window_x=0,
        window_y=0,
        real_pixel_mask=support,
        batch_size=3,
        forward_fields=forward,
        normalize=False,
    )
    window = accumulate_variable_field_window(
        image[24:104, 24:104],
        field_profile=profile,
        target_height=131,
        target_width=131,
        window_x=24,
        window_y=24,
        real_pixel_mask=support[24:104, 24:104],
        batch_size=3,
        forward_fields=forward,
        normalize=False,
    )

    np.testing.assert_array_equal(window[24:56, 24:56], complete[48:80, 48:80])


def test_window_normalization_uses_its_image_not_neural_tiles_or_whole_target():
    """Different surroundings rescale even the same central tissue patch."""
    image = np.random.default_rng(15).uniform(20, 80, (131, 131, 3)).astype(np.float32)
    image[:24] = 200
    support = np.ones(image.shape[:2], bool)
    support[40:43, 40:43] = False
    image[~support] = np.nan
    results = []
    for start, end in ((0, 131), (24, 104)):
        window = image[start:end, start:end]
        real = support[start:end, start:end]
        result = accumulate_variable_field_window(
            window,
            field_profile=_profile(),
            target_height=131,
            target_width=131,
            window_x=start,
            window_y=start,
            real_pixel_mask=real,
            batch_size=3,
            forward_fields=lambda batch: batch,
        )
        np.testing.assert_allclose(
            result, _normalized_image(window, real), atol=2e-7, rtol=2e-7
        )
        assert not result[~real].any()
        results.append(result)
    assert not np.allclose(results[0][48:80, 48:80], results[1][24:56, 24:56])


def test_offset_window_preserves_complete_target_batch_slots():
    """Skipping distant fields does not repack fields into different batch slots."""
    image = np.arange(131 * 131 * 3, dtype=np.float32).reshape(131, 131, 3)
    profile = _profile(tileOverlap=0.75)

    def slot_sensitive_forward(fields):
        slots = np.arange(len(fields), dtype=np.float32)[:, None, None, None]
        return fields[:, :1] + slots * 10_000.0

    complete = accumulate_variable_field_window(
        image,
        field_profile=profile,
        target_height=131,
        target_width=131,
        window_x=0,
        window_y=0,
        real_pixel_mask=None,
        batch_size=3,
        forward_fields=slot_sensitive_forward,
        normalize=False,
    )
    window = accumulate_variable_field_window(
        image[24:104, 24:104],
        field_profile=profile,
        target_height=131,
        target_width=131,
        window_x=24,
        window_y=24,
        real_pixel_mask=None,
        batch_size=3,
        forward_fields=slot_sensitive_forward,
        normalize=False,
    )

    np.testing.assert_array_equal(window[24:56, 24:56], complete[48:80, 48:80])


def test_window_execution_does_not_enumerate_the_complete_cartesian_grid():
    """One bounded window forwards only nearby complete-target field batches."""
    forwarded = []

    def observe(fields):
        forwarded.append(len(fields))
        return np.zeros((len(fields), 1, *fields.shape[2:]), dtype=np.float32)

    result = accumulate_variable_field_window(
        np.ones((64, 64, 3), dtype=np.float32),
        field_profile=_profile(),
        target_height=100_000,
        target_width=100_000,
        window_x=49_968,
        window_y=49_968,
        real_pixel_mask=np.ones((64, 64), dtype=bool),
        batch_size=3,
        forward_fields=observe,
    )

    assert result.shape == (64, 64, 1)
    assert not result.any()
    assert 0 < sum(forwarded) < 100
    assert all(1 <= size <= 3 for size in forwarded)
