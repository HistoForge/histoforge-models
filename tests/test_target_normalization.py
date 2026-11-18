"""Exact target percentiles and their fixed/variable window application."""

import json

import numpy as np
import pytest

from histoforge_models.pathology.normalization import (
    normalize99,
    percentiles_from_uint8_histogram,
)
from histoforge_models.pathology.window_maps import predict_window_maps


def _histogram(pixels, support, strip_height):
    counts = np.zeros((pixels.shape[-1], 256), np.uint64)
    for top in range(0, pixels.shape[0], strip_height):
        strip = pixels[top : top + strip_height]
        mask = support[top : top + strip_height]
        for channel in range(pixels.shape[-1]):
            counts[channel] += np.bincount(
                strip[..., channel][mask], minlength=256
            ).astype(np.uint64)
    return counts


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("strip_height", [1, 7, 64])
def test_histograms_reproduce_numpy_percentiles_and_normalized_pixel_bytes(
    seed, strip_height
):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (31, 43, 3), dtype=np.uint8)
    support = rng.random((31, 43)) > 0.25
    counts = _histogram(pixels, support, strip_height)
    percentiles = percentiles_from_uint8_histogram(counts)
    expected = np.percentile(pixels.astype(np.float32)[support], (1, 99), axis=0).T
    assert percentiles.dtype == np.float64
    np.testing.assert_array_equal(percentiles, expected)
    # The CPU worker transports JSON-compatible binary64 values to prediction.
    transported = json.loads(json.dumps(percentiles.tolist()))
    original = pixels.copy()
    actual = normalize99(
        pixels, real_pixel_mask=support, channel_percentiles=transported
    )
    reference = normalize99(pixels, real_pixel_mask=support)
    assert actual.tobytes() == reference.tobytes()
    np.testing.assert_array_equal(pixels, original)


@pytest.mark.parametrize("kind", ["empty", "single", "constant", "rank-ties"])
def test_degenerate_support_and_percentile_rank_boundaries(kind):
    pixels = np.zeros((1, 102, 3), np.uint8)
    support = np.ones((1, 102), bool)
    if kind == "empty":
        support[:] = False
    elif kind == "single":
        support[:, 1:] = False
        pixels[0, 0] = [3, 125, 254]
    elif kind == "constant":
        pixels[:] = [7, 127, 255]
    else:
        pixels[:, 2:100] = [128, 63, 175]
        pixels[:, 100:] = 255
    values = percentiles_from_uint8_histogram(_histogram(pixels, support, 1))
    actual = normalize99(pixels, real_pixel_mask=support, channel_percentiles=values)
    assert actual.tobytes() == normalize99(pixels, real_pixel_mask=support).tobytes()
    if kind == "empty":
        assert not values.any()


def test_large_histogram_counts_preserve_fractional_ranks_without_large_allocation():
    size = 2**35 + 17
    rank = (size - 1) * np.float64(0.01)
    lower_count = int(np.floor(rank)) + 1
    counts = np.zeros((1, 256), np.uint64)
    counts[0, 37] = lower_count
    counts[0, 129] = size - lower_count
    actual = percentiles_from_uint8_histogram(counts)
    # normalize99 requests a pair. A scalar query has different NumPy 2 dtype
    # promotion and would round the oracle to float32 before comparison.
    expected = np.quantile(np.array([37, 129], np.float32), [rank % 1, 1.0])
    np.testing.assert_array_equal(actual[0], expected)


def test_order_statistic_lookup_does_not_round_integer_counts_to_float():
    counts = np.zeros((1, 256), np.uint64)
    counts[0, 37] = 17_834_254_524_387_181
    counts[0, 129] = (2**54 + 17) - int(counts[0, 37])
    np.testing.assert_array_equal(percentiles_from_uint8_histogram(counts), [[37, 37]])


def test_signed_index_boundary_and_overflow_are_explicit():
    counts = np.zeros((1, 256), np.uint64)
    counts[0, 255] = np.iinfo(np.int64).max
    np.testing.assert_array_equal(
        percentiles_from_uint8_histogram(counts), [[255, 255]]
    )
    counts[0, 255] += np.uint64(1)
    with pytest.raises(ValueError, match="signed index range"):
        percentiles_from_uint8_histogram(counts)


@pytest.mark.parametrize(
    "counts",
    [np.zeros((3, 256)), np.zeros((3, 255), np.uint64), np.zeros(256, np.uint64)],
)
def test_histogram_contract_rejects_non_integer_counts_or_wrong_bins(counts):
    with pytest.raises(ValueError, match="uint64"):
        percentiles_from_uint8_histogram(counts)


@pytest.mark.parametrize(
    "values",
    [
        [[0, 255]],
        [[0, 255], [0, 255], [float("nan"), 255]],
        [[0, 255], [0, 255], [0, float("inf")]],
        [[0, 255], [0, 255], [100, 50]],
    ],
)
def test_supplied_percentiles_reject_invalid_channel_ranges(values):
    with pytest.raises(ValueError, match="finite ordered pair"):
        normalize99(np.zeros((9, 11, 3), np.uint8), channel_percentiles=values)


@pytest.mark.parametrize("variable", [False, True], ids=["fixed", "variable"])
def test_windows_use_shared_statistics_before_unchanged_field_planning(variable):
    from tests.test_variable_field_inference import _profile

    pixels = np.random.default_rng(19).integers(20, 80, (131, 131, 3), np.uint8)
    pixels[:24] = 200
    support = np.ones((131, 131), bool)
    support[40:43, 40:43] = False
    pixels[~support] = 255
    values = percentiles_from_uint8_histogram(_histogram(pixels, support, 7))
    options = (
        {"field_profile": _profile()}
        if variable
        else {"bsize": 32, "tile_overlap": 0.25}
    )
    shared = []
    for start, end in ((0, 131), (24, 104)):
        crop = pixels[start:end, start:end]
        mask = support[start:end, start:end]
        result = predict_window_maps(
            crop,
            real_pixel_mask=mask,
            target_height=131,
            target_width=131,
            window_x=start,
            window_y=start,
            field_options=options,
            batch_size=3,
            forward_tiles=lambda tiles: tiles,
            channel_percentiles=values,
        )
        expected = normalize99(crop, real_pixel_mask=mask, channel_percentiles=values)
        np.testing.assert_allclose(result, expected, rtol=2e-7, atol=2e-7)
        assert not result[~mask].any()
        shared.append(result)
    np.testing.assert_array_equal(shared[0][48:80, 48:80], shared[1][24:56, 24:56])
    assert not np.array_equal(
        normalize99(pixels, real_pixel_mask=support)[48:80, 48:80],
        normalize99(pixels[24:104, 24:104], real_pixel_mask=support[24:104, 24:104])[
            24:56, 24:56
        ],
    )
