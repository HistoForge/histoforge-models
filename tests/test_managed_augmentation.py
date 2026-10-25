"""Masked appearance contracts, using synthetic RGB only (no dataset/TEST)."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pytest

from histoforge_models.pathology.managed_training.augmentation import (
    augment_blur,
    augment_brightness_contrast,
    augment_hed,
    augment_noise,
)


# These are test parameters, not recommended or qualified model presets.
OPERATORS = [
    (augment_hed, {"sigma_range": 0.2, "mean_intensity_range": (0.15, 0.85)}),
    (
        augment_brightness_contrast,
        {"brightness_range": (0.9, 1.1), "contrast_range": (0.9, 1.1)},
    ),
    (augment_blur, {"sigma_range": (0.5, 2.0)}),
    (augment_noise, {"sigma": 0.03}),
]


def _image():
    return np.random.default_rng(21).integers(20, 235, (19, 27, 3), dtype=np.uint8)


def _apply(operator, settings, image, mask=None, *, probability=1, seed=219):
    return operator(
        image,
        real_pixel_mask=np.ones(image.shape[:2], bool) if mask is None else mask,
        rng=np.random.default_rng(seed),
        probability=probability,
        **settings,
    )


@pytest.mark.parametrize("operator,settings", OPERATORS)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
def test_repeatable_finite_detached_raw_domain(operator, settings, dtype):
    pixels = _image().astype(dtype)
    original = pixels.copy()
    result = _apply(operator, settings, pixels)
    repeated = _apply(operator, settings, pixels)
    reference_uint8 = _apply(operator, settings, _image())
    np.testing.assert_array_equal(result, repeated)
    np.testing.assert_array_equal(result, reference_uint8)
    np.testing.assert_array_equal(pixels, original)
    assert not np.shares_memory(result, pixels)
    assert result.shape == pixels.shape and result.dtype == np.float32
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 255
    assert np.any(result != pixels)


@pytest.mark.parametrize("operator,settings", OPERATORS)
def test_padding_never_changes_real_pixels_statistics_or_draws(operator, settings):
    image = _image()
    mask = np.ones(image.shape[:2], bool)
    mask[5:8, 7:11] = False
    original_mask = mask.copy()
    expected = _apply(operator, settings, image, mask)
    for artificial_value in (-10_000.0, 255.0, np.nan):
        padded = np.full((38, 49, 3), artificial_value, np.float32)
        padded[8:27, 11:38] = image
        padded_mask = np.zeros(padded.shape[:2], bool)
        padded_mask[8:27, 11:38] = mask
        padded[~padded_mask] = artificial_value
        result = _apply(operator, settings, padded, padded_mask)
        np.testing.assert_allclose(result[8:27, 11:38][mask], expected[mask], atol=2e-5)
        np.testing.assert_array_equal(result[~padded_mask], padded[~padded_mask])
        np.testing.assert_array_equal(padded_mask[8:27, 11:38], original_mask)
    np.testing.assert_array_equal(mask, original_mask)


@pytest.mark.parametrize("operator,settings", OPERATORS)
def test_disabled_or_empty_operators_are_identity_copies(operator, settings):
    image = _image()
    for mask, probability in [(None, 0), (np.zeros(image.shape[:2], bool), 1)]:
        result = _apply(operator, settings, image, mask, probability=probability)
        np.testing.assert_array_equal(result, image)
        assert not np.shares_memory(result, image)


@pytest.mark.parametrize("operator,settings", OPERATORS)
@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_probabilities_fail_before_mutation(operator, settings, probability):
    image = _image()
    with pytest.raises(ValueError, match="probability"):
        _apply(operator, settings, image, probability=probability)
    np.testing.assert_array_equal(image, _image())


@pytest.mark.parametrize("operator,settings", OPERATORS)
@pytest.mark.parametrize("bad_value", [-1, 256, np.nan, np.inf])
def test_invalid_real_rgb_is_not_hidden_by_disable(operator, settings, bad_value):
    pixels = _image().astype(np.float32)
    pixels[0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="finite raw"):
        _apply(operator, settings, pixels, probability=0)


@pytest.mark.parametrize("operator,settings", OPERATORS)
def test_masks_and_numeric_domain_are_explicit(operator, settings):
    image = _image()
    with pytest.raises(ValueError, match="boolean"):
        _apply(operator, settings, image, np.ones(image.shape[:2], np.uint8))
    with pytest.raises(ValueError, match="HW shape"):
        _apply(operator, settings, image, np.ones((3, 4), bool))
    with pytest.raises(ValueError, match="uint8, float32, or float64"):
        _apply(operator, settings, image.astype(np.uint16))
    with pytest.raises(ValueError, match="HWC RGB"):
        _apply(operator, settings, image.transpose(2, 0, 1))
    with pytest.raises(ValueError, match="Generator"):
        operator(
            image,
            real_pixel_mask=np.ones(image.shape[:2], bool),
            rng=None,
            probability=1,
            **settings,
        )


def test_noise_has_exact_draw_order_scale_and_clipping():
    image = _image().astype(np.float32)
    mask = np.ones(image.shape[:2], bool)
    mask[3:7] = False
    rng = np.random.default_rng(219)
    rng.random()  # first draw always determines whether to apply this operator
    noise = rng.normal(0.0, 0.03 * 255, (int(mask.sum()), 3))
    expected = image.copy()
    expected[mask] = np.clip(image[mask].astype(np.float64) + noise, 0, 255)
    np.testing.assert_array_equal(
        _apply(augment_noise, {"sigma": 0.03}, image, mask), expected
    )


def test_hed_matches_independent_existing_od_equations_without_integer_quantization():
    image = _image()
    values = image.astype(np.float64).reshape(-1, 3) / 255
    matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
    concentrations = -np.log(np.maximum(values, 1e-6)) @ np.linalg.inv(matrix)
    rng = np.random.default_rng(219)
    rng.random()
    for channel in range(3):
        scale, bias = rng.uniform(-0.2, 0.2, size=2)
        concentrations[:, channel] *= 1 + scale
        concentrations[:, channel] += bias
    expected = np.clip(np.exp(-(concentrations @ matrix)) * 255, 0, 255)
    actual = _apply(*OPERATORS[0], image)
    np.testing.assert_allclose(
        actual, expected.reshape(image.shape).astype(np.float32), rtol=0, atol=2e-5
    )


def test_hed_cutoff_counts_only_real_context_including_real_white():
    white = np.full((19, 27, 3), 255, np.uint8)
    np.testing.assert_array_equal(_apply(*OPERATORS[0], white), white)
    black = np.zeros_like(white)
    np.testing.assert_array_equal(_apply(*OPERATORS[0], black), black)
    # Both genuinely dark and bright real pixels participate in the cutoff.
    mixed = white.copy()
    mixed[:10] = 0
    assert np.any(_apply(*OPERATORS[0], mixed) != mixed)


def test_brightness_contrast_matches_ordered_luminance_reference():
    image = _image().astype(np.float32)
    settings = {"brightness_range": (1.3, 1.3), "contrast_range": (0.8, 0.8)}
    bright = np.clip(image.astype(np.float64) * 1.3, 0, 255)
    mean = (bright @ np.array([0.2989, 0.5870, 0.1140])).mean()
    expected = np.clip(mean + 0.8 * (bright - mean), 0, 255).astype(np.float32)
    np.testing.assert_array_equal(
        _apply(augment_brightness_contrast, settings, image), expected
    )


def test_masked_blur_matches_explicit_truncated_gaussian_sum():
    image = _image().astype(np.float64)
    mask = np.ones(image.shape[:2], bool)
    mask[8:12, 12:15] = False
    sigma = 0.8
    radius = int(4 * sigma + 0.5)
    actual = _apply(augment_blur, {"sigma_range": (sigma, sigma)}, image, mask)
    for y, x in ((0, 0), (7, 12), (13, 14), (18, 26)):
        total, mass = np.zeros(3), 0.0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < 19 and 0 <= xx < 27 and mask[yy, xx]:
                    weight = np.exp(-(dy * dy + dx * dx) / (2 * sigma * sigma))
                    total += image[yy, xx] * weight
                    mass += weight
        np.testing.assert_allclose(actual[y, x], total / mass, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize(
    "operator,settings",
    [
        (augment_hed, {"sigma_range": 0, "mean_intensity_range": (0, 1)}),
        (
            augment_brightness_contrast,
            {"brightness_range": (1, 1), "contrast_range": (1, 1)},
        ),
        (augment_blur, {"sigma_range": (0, 0)}),
        (augment_noise, {"sigma": 0}),
    ],
)
def test_zero_strength_is_exact_identity(operator, settings):
    image = _image().astype(np.float32)
    image[:3] = 0
    image[3:6] = 255
    np.testing.assert_array_equal(_apply(operator, settings, image), image)


@pytest.mark.parametrize(
    "operator,settings",
    [
        (augment_hed, {"sigma_range": -0.1, "mean_intensity_range": (0, 1)}),
        (augment_hed, {"sigma_range": 1.1, "mean_intensity_range": (0, 1)}),
        (augment_hed, {"sigma_range": 0.2, "mean_intensity_range": (0.5, 0.5)}),
        (augment_hed, {"sigma_range": 0.2, "mean_intensity_range": (-1, 1)}),
        (augment_blur, {"sigma_range": (2, 1)}),
        (augment_blur, {"sigma_range": (-1, 1)}),
        (augment_blur, {"sigma_range": (0, np.inf)}),
        (augment_noise, {"sigma": np.nan}),
        (augment_noise, {"sigma": -1}),
        (augment_noise, {"sigma": 2}),
        (
            augment_brightness_contrast,
            {"brightness_range": (-1, 1), "contrast_range": (1, 1)},
        ),
        (
            augment_brightness_contrast,
            {"brightness_range": (1, 1), "contrast_range": (1, np.inf)},
        ),
    ],
)
def test_bad_parameters_reject_even_when_disabled(operator, settings):
    with pytest.raises(ValueError):
        _apply(operator, settings, _image(), probability=0)


def _composed_sample(index):
    """Test caller owns order and independent operator seeds, not the worker."""
    image = _image()
    for operator_index, (operator, settings) in enumerate(OPERATORS):
        image = _apply(operator, settings, image, seed=500 * index + operator_index)
    return image


def test_spawned_workers_match_serial_composition_without_global_rng_changes():
    state = np.random.get_state()
    serial = [_composed_sample(index) for index in range(4)]
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        parallel = list(pool.map(_composed_sample, range(4)))
    for expected, actual in zip(serial, parallel):
        np.testing.assert_array_equal(actual, expected)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
