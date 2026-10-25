"""Exact image preprocessing and fixed-tiler integration without model weights."""

import numpy as np
import pytest

from histoforge_models import core, transforms
from histoforge_models.pathology.field_inference import run_model_fields
from histoforge_models.pathology.normalization import normalize99


def _reference(image, support=None):
    """Independently retain the historical percentile/arithmetic order."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 2:
        values = values[..., None]
    output = np.zeros_like(values)
    for channel in range(values.shape[-1]):
        real = (
            values[..., channel] if support is None else values[..., channel][support]
        )
        if real.size:
            low, high = np.percentile(real, [1, 99])
            if high - low > 0.001:
                if support is None:
                    output[..., channel] = (real - low) / (high - low)
                else:
                    output[..., channel][support] = (real - low) / (high - low)
    return output.squeeze(-1) if output.shape[-1] == 1 else output


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(17, 23), (17, 23, 1), (17, 23, 3)])
def test_all_real_values_match_existing_normalize99_exactly(dtype, shape):
    """Extracting the helper and supplying an all-real mask cannot alter math."""
    image = np.random.default_rng(123).uniform(-20, 300, shape).astype(dtype)
    expected = _reference(image)
    np.testing.assert_array_equal(normalize99(image), expected)
    np.testing.assert_array_equal(
        normalize99(image, real_pixel_mask=np.ones(shape[:2], bool)), expected
    )
    assert expected.dtype == np.float32


def test_artificial_padding_is_ignored_but_real_black_and_white_are_included():
    """Pixel values never decide whether source pixels exist."""
    image = np.arange(300, dtype=np.float32).reshape(10, 10, 3) % 256
    image[:2] = 0
    image[-2:] = 255
    padded = np.full((23, 31, 3), -10000, dtype=np.float32)
    padded[5:15, 7:17] = image
    support = np.zeros(padded.shape[:2], bool)
    support[5:15, 7:17] = True
    original = padded.copy()
    normalized = normalize99(padded, real_pixel_mask=support)
    np.testing.assert_array_equal(
        normalized[5:15, 7:17], (image / np.float64(255)).astype(np.float32)
    )
    assert not normalized[~support].any()
    np.testing.assert_array_equal(padded, original)
    padded[~support] = np.nan
    np.testing.assert_array_equal(
        normalize99(padded, real_pixel_mask=support), normalized
    )


def test_empty_support_degenerate_range_and_unclipped_values():
    """Empty filler fields are zero, and valid tails are not clipped away."""
    image = np.repeat(np.arange(101, dtype=np.float32)[None, :, None], 3, axis=2)
    image[..., 1] = 5
    image[..., 2] *= 0.000001
    normalized = normalize99(image)
    assert normalized[0, 0, 0] < 0
    assert normalized[0, -1, 0] > 1
    assert not normalized[..., 1:].any()
    assert not normalize99(image, real_pixel_mask=np.zeros(image.shape[:2], bool)).any()
    np.testing.assert_array_equal(normalized, _reference(image))


@pytest.mark.parametrize("support", [np.ones((3, 5)), np.ones((3, 4), bool)])
def test_support_must_be_boolean_and_match_the_field(support):
    """Reject ambiguous support before it changes scientific inputs."""
    with pytest.raises(ValueError, match="real_pixel_mask"):
        normalize99(np.zeros((3, 5, 3)), real_pixel_mask=support)


def _planned_tiles(image, support, side, overlap, augment=False):
    """Use only the existing geometric primitives to form an independent oracle."""
    height, width = image.shape[:2]
    top, bottom, left, right = transforms.get_pad_yx(
        height, width, min_size=(side, side)
    )
    pads = ((0, 0), (top, bottom), (left, right))
    fields, ys, xs, padded_h, padded_w = transforms.make_tiles(
        np.pad(image.transpose(2, 0, 1), pads), side, augment, overlap
    )
    masks = transforms.make_tiles(np.pad(support[None], pads), side, augment, overlap)[
        0
    ]
    return (
        fields.reshape(-1, 3, side, side),
        masks.reshape(-1, side, side).astype(bool),
        ys,
        xs,
        padded_h,
        padded_w,
        top,
        left,
    )


@pytest.mark.parametrize("overlap", [0.0, 0.9])
def test_fixed_tiler_allocation_uses_the_same_clamped_overlap_as_tile_generation(
    monkeypatch, overlap
):
    """Formerly allocation counted raw overlap while make_tiles counted its clamp."""
    image = np.random.default_rng(93).normal(size=(65, 73, 3)).astype(np.float32)
    support = np.ones(image.shape[:2], dtype=bool)
    expected, *_ = _planned_tiles(image, support, 32, overlap)
    calls = []

    def forward(_net, batch):
        calls.extend(batch.copy())
        return batch, np.zeros((len(batch), 256), dtype=np.float32)

    monkeypatch.setattr(core, "_forward", forward)
    result, _ = core.run_net(
        object(), image[None], bsize=32, batch_size=3, tile_overlap=overlap
    )
    np.testing.assert_array_equal(np.stack(calls), expected)
    np.testing.assert_allclose(result[0], image, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("shape", [(11, 19), (32, 32), (53, 79)])
@pytest.mark.parametrize("batch_size", [1, 3, 16])
def test_image_is_normalized_before_unchanged_fixed_tiling(
    monkeypatch, shape, batch_size
):
    """Preserve tile order, extra border, batching, and tapered output merging."""
    image = np.random.default_rng(101).integers(0, 256, (*shape, 3), np.uint8)
    image[:, shape[1] // 2 :] //= 4
    support = np.ones(shape, bool)
    support[:3, :5] = False
    image[:3, :5] = 255
    fields, _masks, ys, xs, height, width, top, left = _planned_tiles(
        _reference(image, support), support, 32, 0.1
    )
    expected_fields = fields
    calls = []

    def forward(_net, batch):
        calls.append(batch.copy())
        return batch.copy(), np.zeros((len(batch), 256), np.float32)

    monkeypatch.setattr(core, "_forward", forward)
    output = run_model_fields(
        None,
        image,
        fixed_field_defaults=(32, 0.1),
        bsize=32,
        batch_size=batch_size,
        real_pixel_mask=support,
    )
    np.testing.assert_array_equal(np.concatenate(calls), expected_fields)
    assert [len(batch) for batch in calls] == [
        min(batch_size, len(fields) - start)
        for start in range(0, len(fields), batch_size)
    ]
    expected_output = transforms.average_tiles(expected_fields, ys, xs, height, width)
    np.testing.assert_array_equal(
        output,
        expected_output[:, top : top + shape[0], left : left + shape[1]].transpose(
            1, 2, 0
        ),
    )
    if shape == (32, 32):
        assert len(fields) == 4  # Preserve the existing extra 16-pixel border.


def test_unprocessed_callers_keep_the_existing_forward_inputs(monkeypatch):
    """The generic Cellpose tiler does not choose a new normalization policy."""
    image = np.random.default_rng(5).random((2, 19, 21, 3)).astype(np.float32)
    calls = []

    def forward(_net, batch):
        calls.append(batch.copy())
        return batch.copy(), np.zeros((len(batch), 256), np.float32)

    monkeypatch.setattr(core, "_forward", forward)
    core.run_net(None, image, bsize=32, batch_size=8)
    expected = np.concatenate(
        [
            _planned_tiles(item, np.ones(item.shape[:2], bool), 32, 0.1)[0]
            for item in image
        ]
    )
    np.testing.assert_array_equal(np.concatenate(calls), expected)


@pytest.mark.parametrize("augment", [False, True])
def test_multiple_images_and_flipped_fields_keep_independent_support(
    monkeypatch, augment
):
    """Batch peers and geometric flips cannot mix masks or percentile statistics."""
    image = np.random.default_rng(8).random((3, 9, 13, 3)).astype(np.float32)
    image[1] = 120 + 50 * image[1]
    image[2] = 255
    support = np.ones(image.shape[:3], bool)
    support[0, :4, :3] = False
    support[1, -2:, -5:] = False
    support[2] = False
    calls = []

    def forward(_net, batch):
        calls.append(batch.copy())
        return batch.copy(), np.zeros((len(batch), 256), np.float32)

    monkeypatch.setattr(core, "_forward", forward)
    core.run_net(
        None,
        np.stack(
            [
                normalize99(item, real_pixel_mask=mask)
                for item, mask in zip(image, support)
            ]
        ),
        bsize=32,
        batch_size=12,
        augment=augment,
    )
    expected = []
    for item, mask in zip(image, support):
        fields, *_ = _planned_tiles(_reference(item, mask), mask, 32, 0.1, augment)
        expected.extend(fields)
    np.testing.assert_array_equal(np.concatenate(calls), np.stack(expected))
    assert not calls[-1][-1].any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"real_pixel_mask": np.ones((9, 13))},
        {"real_pixel_mask": np.ones((1, 9, 13), bool)},
        {"real_pixel_mask": np.ones((9, 12), bool)},
    ],
)
def test_image_preparation_rejects_misaligned_source_support(monkeypatch, kwargs):
    """Source support must describe the exact grid, not a guessed resize."""

    def forward(*_args):
        pytest.fail("invalid source support reached the network")

    monkeypatch.setattr(core, "_forward", forward)
    with pytest.raises(ValueError):
        run_model_fields(
            None,
            np.zeros((9, 13, 3)),
            fixed_field_defaults=(32, 0.1),
            bsize=32,
            **kwargs,
        )


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("wrapper", ["flow", "hovernet", "semantic"])
def test_model_wrappers_normalize_complete_image_and_preserve_decode(
    monkeypatch, wrapper, normalize
):
    """Exercise all serving wrappers through the tiler with controlled raw outputs."""
    import torch
    from histoforge_models import dynamics
    from histoforge_models.pathology import (
        flow_inference,
        hovernet_inference,
        semantic_inference,
    )

    image = np.random.default_rng(2).integers(0, 256, (32, 32, 3), np.uint8)
    support = np.ones(image.shape[:2], bool)
    support[:4] = False
    calls = []
    channels = 6 if wrapper == "hovernet" else 5

    def forward(_net, batch):
        calls.append(batch.copy())
        output = np.zeros((len(batch), channels, 32, 32), np.float32)
        output[:, 1] = batch[:, 0]
        return output, np.zeros((len(batch), 256), np.float32)

    monkeypatch.setattr(core, "_forward", forward)
    decoded = []

    def decode(values, *args, **kwargs):
        decoded.append((values.copy(), kwargs))
        return np.ones(values.shape[-2:], np.int32)

    monkeypatch.setattr(dynamics, "compute_masks", decode)
    monkeypatch.setattr(hovernet_inference, "watershed_from_hv_maps", decode)
    net = torch.nn.Conv2d(3, channels, 1)
    function = {
        "flow": flow_inference.run_flow_inference,
        "hovernet": hovernet_inference.run_hovernet_inference,
        "semantic": semantic_inference.run_semantic_inference,
    }[wrapper]
    result = function(
        net,
        image,
        num_classes=2,
        bsize=32,
        batch_size=3,
        tile_overlap=0.1,
        real_pixel_mask=support,
        normalize=normalize,
    )
    expected, *_ = _planned_tiles(
        _reference(image, support) if normalize else image, support, 32, 0.1
    )
    np.testing.assert_array_equal(np.concatenate(calls), expected)
    assert result["class_map"].shape == (32, 32)
    if wrapper != "semantic":
        assert len(decoded) == 1  # Decode the blended output, never individual tiles.
        assert result["votes_per_instance"].keys() == {1}
        assert decoded[0][0].shape[-2:] == (32, 32)


@pytest.mark.parametrize("wrapper", ["flow", "hovernet", "semantic"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_complete_image_rejects_nonfinite_maps_before_decoding(
    monkeypatch, wrapper, invalid
):
    import torch
    from histoforge_models import dynamics
    from histoforge_models.pathology import (
        flow_inference,
        hovernet_inference,
        semantic_inference,
    )

    channels = 6 if wrapper == "hovernet" else 5

    def forward(_net, batch):
        output = np.zeros((len(batch), channels, 32, 32), np.float32)
        output[:, 1] = invalid
        return output, np.zeros((len(batch), 256), np.float32)

    def decode(*_args, **_kwargs):
        pytest.fail("nonfinite network output reached decoding")

    monkeypatch.setattr(core, "_forward", forward)
    monkeypatch.setattr(dynamics, "compute_masks", decode)
    monkeypatch.setattr(hovernet_inference, "watershed_from_hv_maps", decode)
    monkeypatch.setattr(semantic_inference, "class_labels_from_maps", decode)
    function = {
        "flow": flow_inference.run_flow_inference,
        "hovernet": hovernet_inference.run_hovernet_inference,
        "semantic": semantic_inference.run_semantic_inference,
    }[wrapper]

    with pytest.raises(ValueError, match="finite"):
        function(
            torch.nn.Conv2d(3, channels, 1),
            np.zeros((32, 32, 3), np.uint8),
            num_classes=2,
            bsize=32,
            batch_size=3,
        )
