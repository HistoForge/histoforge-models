"""Explicit, real-pixel appearance operations for managed brightfield training.

Inputs and outputs are HWC sRGB on the raw [0, 255] scale, never normalized
network inputs. Outputs are detached float32 arrays; conversion does not infer
a scale from dtype or observed values. All real pixels participate, including
context outside supervision. Artificial pixels are preserved but never sampled
or included in statistics/filtering. Targets are deliberately not an argument.

The embedding application owns compatible presets, probabilities/ranges, order,
and a separately seeded NumPy Generator for each sample/operator. These functions
provide no dataset lookup, scientific defaults, random global state, silent
fallback, or claim of qualified stain/task settings. Apply them before normalize99
and model/batch padding. An existing source/fixed-field mask is also supported.

HED uses the optical-density algorithm and Ruifrok/Johnston absorption matrix
already used by this repository's research augmentation. It is not suitable for
intensity-defined IHC or a substitute for a PAS-specific transform. This masked,
floating-point API is not the historical quantized batch-augmentation recipe.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

_RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
    dtype=np.float64,
)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)
_LUMINANCE = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)


def augment_hed(
    pixels: np.ndarray,
    *,
    real_pixel_mask: np.ndarray,
    rng: np.random.Generator,
    probability: float,
    sigma_range: float,
    mean_intensity_range: tuple[float, float],
) -> np.ndarray:
    """Scale and shift each H/E/D channel, with explicit real-pixel cutoff.

    After the probability draw and cutoff check, draw scale offset then bias
    uniformly from [-sigma_range, sigma_range] for H, E, D in that order. Work
    in float64 optical density with RGB floor 1e-6. A zero magnitude is exact
    identity, not a lossy deconvolution round trip. The cutoff is mean real RGB
    intensity on [0, 1]; it never inspects class labels or padding colours.
    """
    amount = _number(sigma_range, "sigma_range")
    if not 0 <= amount <= 1:
        raise ValueError("sigma_range must be in [0, 1]")
    lower, upper = _range(mean_intensity_range, "mean_intensity_range")
    if not 0 <= lower < upper <= 1:
        raise ValueError("mean_intensity_range must satisfy 0 <= lower < upper <= 1")
    output, mask, selected = _prepare(pixels, real_pixel_mask, rng, probability)
    if not selected or amount == 0:
        return output
    values = output[mask].astype(np.float64) / 255.0
    if not lower <= values.mean() <= upper:
        return output
    hed = -np.log(np.maximum(values, 1e-6)) @ _HED_FROM_RGB
    for channel in range(3):
        offset = rng.uniform(-amount, amount)
        bias = rng.uniform(-amount, amount)
        hed[:, channel] = hed[:, channel] * (1 + offset) + bias
    # Clamping negative OD before exp is equivalent to clipping RGB at one,
    # but cannot overflow when a concentration perturbation yields negative OD.
    values = np.exp(-np.maximum(hed @ _RGB_FROM_HED, 0.0))
    return _assign_real(output, mask, values * 255.0)


def augment_brightness_contrast(
    pixels: np.ndarray,
    *,
    real_pixel_mask: np.ndarray,
    rng: np.random.Generator,
    probability: float,
    brightness_range: tuple[float, float],
    contrast_range: tuple[float, float],
) -> np.ndarray:
    """Apply brightness, then contrast about mean real-pixel luminance.

    Draw multiplicative brightness then contrast uniformly from their supplied
    nonnegative ranges, even for collapsed ranges. Brightness is clipped before
    computing luminance (0.2989 R + 0.5870 G + 0.1140 B), then contrast is clipped.
    Identity factors preserve pixels exactly. Per-field percentile normalization
    can cancel unclipped affine changes; this does not imply accuracy improvement.
    """
    brightness = _range(brightness_range, "brightness_range", nonnegative=True)
    contrast = _range(contrast_range, "contrast_range", nonnegative=True)
    output, mask, selected = _prepare(pixels, real_pixel_mask, rng, probability)
    if not selected:
        return output
    brightness_factor = rng.uniform(*brightness)
    contrast_factor = rng.uniform(*contrast)
    if brightness_factor == 1 and contrast_factor == 1:
        return output
    values = output[mask].astype(np.float64)
    with np.errstate(over="raise", invalid="raise"):
        values = np.clip(values * brightness_factor, 0.0, 255.0)
        if contrast_factor != 1:
            mean = (values @ _LUMINANCE).mean()
            values = mean + contrast_factor * (values - mean)
    return _assign_real(output, mask, values)


def augment_blur(
    pixels: np.ndarray,
    *,
    real_pixel_mask: np.ndarray,
    rng: np.random.Generator,
    probability: float,
    sigma_range: tuple[float, float],
) -> np.ndarray:
    """Gaussian blur in model pixels, independently per RGB channel.

    Draw one sigma uniformly from the explicit range. Use normalized convolution
    G(real * RGB) / G(real), with a four-sigma kernel and zero exterior support.
    Only real samples contribute, so neither pre-existing padding nor adding more
    padding darkens real boundaries. Unknown label support is not image padding.
    A zero sigma and an empty real mask are exact identity. This is deliberately
    not the legacy integer/reflect-boundary research blur implementation.
    """
    bounds = _range(sigma_range, "sigma_range", nonnegative=True)
    output, mask, selected = _prepare(pixels, real_pixel_mask, rng, probability)
    if not selected:
        return output
    sigma = rng.uniform(*bounds)
    if sigma == 0:
        return output
    weights = gaussian_filter(mask.astype(np.float64), sigma, mode="constant")
    values = np.where(mask[..., None], output, 0.0).astype(np.float64)
    values = gaussian_filter(values, (sigma, sigma, 0), mode="constant")
    # Positive real pixels have at least the central kernel mass.
    if np.any(weights[mask] <= 0):
        raise FloatingPointError("Gaussian blur lost real-pixel support")
    return _assign_real(output, mask, values[mask] / weights[mask, None])


def augment_noise(
    pixels: np.ndarray,
    *,
    real_pixel_mask: np.ndarray,
    rng: np.random.Generator,
    probability: float,
    sigma: float,
) -> np.ndarray:
    """Add independent Gaussian RGB noise; sigma is on the [0, 1] scale.

    Draw only real pixels in row-major order, channels R/G/B. Noise on artificial
    pixels cannot shift draws on later real pixels. Multiply sigma by 255 on the
    raw input scale and clip after addition. A zero sigma is exact identity.
    """
    deviation = _number(sigma, "sigma")
    if not 0 <= deviation <= 1:
        raise ValueError("noise sigma must be in [0, 1]")
    output, mask, selected = _prepare(pixels, real_pixel_mask, rng, probability)
    if not selected or deviation == 0:
        return output
    values = output[mask].astype(np.float64)
    values += rng.normal(0.0, deviation * 255.0, values.shape)
    return _assign_real(output, mask, values)


def _prepare(pixels, real_pixel_mask, rng, probability):
    """Validate the explicit numeric contract before drawing or mutating anything."""
    probability = _number(probability, "probability")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicitly supplied numpy.random.Generator")
    image, mask = np.asarray(pixels), np.asarray(real_pixel_mask)
    if image.ndim != 3 or image.shape[-1] != 3 or not image.size:
        raise ValueError("appearance augmentation requires nonempty HWC RGB")
    if image.dtype not in (np.dtype("uint8"), np.dtype("float32"), np.dtype("float64")):
        raise ValueError("raw RGB must be uint8, float32, or float64 on [0, 255]")
    if mask.dtype != np.bool_ or mask.shape != image.shape[:2]:
        raise ValueError("real_pixel_mask must be boolean with the image HW shape")
    real_values = image[mask]
    if (
        not np.isfinite(real_values).all()
        or np.any(real_values < 0)
        or np.any(real_values > 255)
    ):
        raise ValueError("real RGB must contain finite raw values in [0, 255]")
    output = image.astype(np.float32, copy=True)
    selected = bool(mask.any()) and rng.random() < probability
    return output, mask, selected


def _assign_real(output, mask, values):
    """Fail visibly on invalid arithmetic; preserve every artificial pixel."""
    if not np.isfinite(values).all():
        raise FloatingPointError("appearance augmentation produced non-finite RGB")
    output[mask] = np.clip(values, 0.0, 255.0).astype(np.float32)
    return output


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _range(value, name, *, nonnegative=False):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain lower and upper bounds")
    lower, upper = (_number(item, name) for item in value)
    if lower > upper or (nonnegative and lower < 0):
        raise ValueError(
            f"{name} must have ordered{' nonnegative' if nonnegative else ''} bounds"
        )
    return lower, upper
