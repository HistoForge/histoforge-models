"""Check tile placement, batching and map blending for image windows."""

import numpy as np
import pytest
import torch

from histoforge_models.core import run_net
from histoforge_models.pathology.normalization import normalize99
from histoforge_models.pathology.window_maps import (
    predict_window_maps,
)


def _contextual_forward(tiles):
    means = tiles.mean(axis=(2, 3), keepdims=True)
    return tiles[:, :1] + means[:, :1]


class _ContextualNet(torch.nn.Module):
    device = torch.device("cpu")
    dtype = torch.float32

    def forward(self, tiles):
        means = tiles.mean(dim=(2, 3), keepdim=True)
        output = tiles[:, :1] + means[:, :1]
        return output, torch.zeros((tiles.shape[0], 256), dtype=tiles.dtype)


@pytest.mark.parametrize("batch_size", [1, 3, 16])
def test_complete_window_matches_the_existing_run_net_tile_contract(batch_size):
    rng = np.random.default_rng(8120)
    image = rng.normal(size=(39, 47, 3)).astype(np.float32)
    support = np.ones(image.shape[:2], bool)
    support[:3] = False
    image[:3] = 255
    net = _ContextualNet()
    expected, _ = run_net(
        net,
        normalize99(image, real_pixel_mask=support)[np.newaxis],
        bsize=16,
        tile_overlap=0.2,
        batch_size=batch_size,
        augment=False,
    )
    actual = predict_window_maps(
        image,
        real_pixel_mask=support,
        target_height=39,
        target_width=47,
        window_x=0,
        window_y=0,
        field_options={"bsize": 16, "tile_overlap": 0.2},
        batch_size=batch_size,
        forward_tiles=lambda tiles: net(torch.from_numpy(tiles))[0].detach().numpy(),
    )

    np.testing.assert_array_equal(actual, expected[0])


def test_target_global_tiles_match_with_sufficient_context_and_equal_statistics():
    # Every aligned window repeats the same population, so percentiles are equal.
    # Geometry is exact under equal inputs; arbitrary windows need not normalize alike.
    pattern = np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    image = np.tile(pattern, (8, 8, 1))
    full = predict_window_maps(
        image,
        real_pixel_mask=np.ones(image.shape[:2], bool),
        target_height=64,
        target_width=64,
        window_x=0,
        window_y=0,
        field_options={"bsize": 16, "tile_overlap": 0.25},
        batch_size=3,
        forward_tiles=_contextual_forward,
    )
    crop = image[8:56, 8:56]
    window = predict_window_maps(
        crop,
        real_pixel_mask=np.ones(crop.shape[:2], bool),
        target_height=64,
        target_width=64,
        window_x=8,
        window_y=8,
        field_options={"bsize": 16, "tile_overlap": 0.25},
        batch_size=3,
        forward_tiles=_contextual_forward,
    )

    np.testing.assert_array_equal(window[16:32, 16:32], full[24:40, 24:40])
    assert not np.array_equal(window[:4, :4], full[8:12, 8:12])


def test_target_global_window_preserves_complete_target_batch_slots():
    image = np.arange(64 * 64 * 3, dtype=np.float32).reshape(64, 64, 3)

    def slot_sensitive_forward(tiles):
        slots = np.arange(tiles.shape[0], dtype=np.float32)[:, None, None, None]
        return np.zeros_like(tiles[:, :1]) + slots * 10_000.0

    full = predict_window_maps(
        image,
        real_pixel_mask=np.ones(image.shape[:2], bool),
        target_height=64,
        target_width=64,
        window_x=0,
        window_y=0,
        field_options={"bsize": 16, "tile_overlap": 0.25},
        batch_size=3,
        forward_tiles=slot_sensitive_forward,
    )
    crop = image[8:56, 8:56]
    window = predict_window_maps(
        crop,
        real_pixel_mask=np.ones(crop.shape[:2], bool),
        target_height=64,
        target_width=64,
        window_x=8,
        window_y=8,
        field_options={"bsize": 16, "tile_overlap": 0.25},
        batch_size=3,
        forward_tiles=slot_sensitive_forward,
    )

    np.testing.assert_array_equal(window[16:32, 16:32], full[24:40, 24:40])


@pytest.mark.parametrize("variable", [False, True])
def test_core_halo_normalizes_each_complete_processing_image(variable):
    """All tiles share their window's statistics; the surrounding window matters."""
    from tests.test_variable_field_inference import _profile

    image = np.random.default_rng(19).uniform(20, 80, (131, 131, 3)).astype(np.float32)
    image[:24] = 200
    support = np.ones(image.shape[:2], bool)
    support[40:43, 40:43] = False
    image[~support] = np.nan
    results = []
    for start, end in ((0, 131), (24, 104)):
        pixels = image[start:end, start:end]
        real = support[start:end, start:end]
        result = predict_window_maps(
            pixels,
            real_pixel_mask=real,
            target_height=131,
            target_width=131,
            window_x=start,
            window_y=start,
            field_options=(
                {"field_profile": _profile()}
                if variable
                else {"bsize": 32, "tile_overlap": 0.25}
            ),
            batch_size=3,
            forward_tiles=lambda batch: batch,
        )
        expected = np.zeros_like(pixels)
        for channel in range(3):
            values = pixels[..., channel][real]
            low, high = np.percentile(values, (1, 99))
            expected[..., channel][real] = (values - low) / (high - low)
        np.testing.assert_allclose(result, expected, atol=2e-7, rtol=2e-7)
        assert not result[~real].any()
        results.append(result)
    assert not np.allclose(results[0][48:80, 48:80], results[1][24:56, 24:56])


def test_target_global_tiles_are_batch_partition_invariant():
    image = np.arange(39 * 47 * 3, dtype=np.float32).reshape(39, 47, 3)
    one = predict_window_maps(
        image,
        real_pixel_mask=np.ones(image.shape[:2], bool),
        target_height=39,
        target_width=47,
        window_x=0,
        window_y=0,
        field_options={"bsize": 16, "tile_overlap": 0.2},
        batch_size=1,
        forward_tiles=_contextual_forward,
    )
    many = predict_window_maps(
        image,
        real_pixel_mask=np.ones(image.shape[:2], bool),
        target_height=39,
        target_width=47,
        window_x=0,
        window_y=0,
        field_options={"bsize": 16, "tile_overlap": 0.2},
        batch_size=11,
        forward_tiles=_contextual_forward,
    )

    np.testing.assert_array_equal(one, many)


def test_target_global_window_does_not_scan_or_forward_the_complete_wsi_grid():
    image = np.ones((64, 64, 3), dtype=np.float32)
    forwarded_batch_sizes = []

    def observe(tiles):
        forwarded_batch_sizes.append(tiles.shape[0])
        return np.zeros((tiles.shape[0], 1, 16, 16), dtype=np.float32)

    result = predict_window_maps(
        image,
        real_pixel_mask=np.ones(image.shape[:2], bool),
        target_height=100_000,
        target_width=100_000,
        window_x=49_968,
        window_y=49_968,
        field_options={"bsize": 16, "tile_overlap": 0.25},
        batch_size=3,
        forward_tiles=observe,
    )

    assert result.shape == (64, 64, 1)
    assert np.count_nonzero(result) == 0
    assert 0 < len(forwarded_batch_sizes) < 32
    assert all(1 <= size <= 3 for size in forwarded_batch_sizes)


@pytest.mark.parametrize("variable", [False, True])
def test_large_offset_keeps_real_pixels_and_support_on_the_local_grid(variable):
    from tests.test_variable_field_inference import _profile

    pixels = np.random.default_rng(123).uniform(10, 230, (47, 61, 3)).astype(np.float32)
    support = np.ones(pixels.shape[:2], bool)
    support[:5, :7] = False
    pixels[~support] = np.nan
    result = predict_window_maps(
        pixels,
        real_pixel_mask=support,
        target_height=1_000_003,
        target_width=900_019,
        window_x=850_007,
        window_y=910_021,
        field_options=(
            {"field_profile": _profile()}
            if variable
            else {"bsize": 32, "tile_overlap": 0.25}
        ),
        batch_size=3,
        forward_tiles=lambda batch: batch,
    )
    expected = np.zeros_like(pixels)
    for channel in range(3):
        values = pixels[..., channel][support]
        low, high = np.percentile(values, (1, 99))
        expected[..., channel][support] = (values - low) / (high - low)
    np.testing.assert_allclose(result, expected, atol=2e-7, rtol=2e-7)
    assert not result[~support].any()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"window_x": -1}, "window_x"),
        ({"window_x": 50}, "target_width"),
        (
            {"field_options": {"bsize": 16, "tile_overlap": float("nan")}},
            "tile_overlap",
        ),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_target_global_tiles_reject_invalid_geometry_and_limits(kwargs, message):
    values = {
        "real_pixel_mask": np.ones((16, 16), bool),
        "target_height": 32,
        "target_width": 32,
        "window_x": 0,
        "window_y": 0,
        "field_options": {"bsize": 16, "tile_overlap": 0.25},
        "batch_size": 2,
        "forward_tiles": _contextual_forward,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        predict_window_maps(np.zeros((16, 16, 3), np.float32), **values)


def test_target_global_tiles_reject_malformed_forward_output():
    with pytest.raises(ValueError, match="NCHW"):
        predict_window_maps(
            np.zeros((16, 16, 3), np.float32),
            real_pixel_mask=np.ones((16, 16), bool),
            target_height=16,
            target_width=16,
            window_x=0,
            window_y=0,
            field_options={"bsize": 16, "tile_overlap": 0.25},
            batch_size=2,
            forward_tiles=lambda tiles: tiles[:, 0],
        )


@pytest.mark.parametrize("variable", [False, True], ids=["fixed", "variable"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_window_rejects_nonfinite_network_maps(variable, invalid):
    from tests.test_variable_field_inference import _profile

    def forward(tiles):
        output = np.zeros_like(tiles)
        output[:, 1] = invalid
        return output

    with pytest.raises(ValueError, match="finite"):
        predict_window_maps(
            np.zeros((16, 16, 3), np.float32),
            real_pixel_mask=np.ones((16, 16), bool),
            target_height=32,
            target_width=32,
            window_x=8,
            window_y=8,
            field_options=(
                {"field_profile": _profile()}
                if variable
                else {"bsize": 16, "tile_overlap": 0.25}
            ),
            batch_size=2,
            forward_tiles=forward,
        )
