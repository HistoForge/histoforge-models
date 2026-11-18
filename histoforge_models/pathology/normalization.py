"""Shared real-image pixel normalization for training and inference."""

import numpy as np


DEFAULT_PERCENTILES = (1.0, 99.0)


def normalize99(
    x: np.ndarray,
    lower: float = DEFAULT_PERCENTILES[0],
    upper: float = DEFAULT_PERCENTILES[1],
    *,
    real_pixel_mask: np.ndarray | None = None,
    channel_percentiles: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize each channel over real pixels and leave artificial padding zero.

    Accept one HW or HWC image. Real context and background participate, even
    outside the supervised ROI. The mask describes pixel availability, never
    label support. Without a mask, every supplied pixel is real. Empty support
    or a channel with percentile range at most 1e-3 produces zeros. Values are
    not clipped to [0, 1]. Percentiles use NumPy's linear interpolation and the
    established float32 input/arithmetic order.

    Without ``channel_percentiles``, estimate ``lower``/``upper`` over this image.
    Training and finite inference retain that behavior. A large-input caller may
    instead supply one float64 low/high pair per channel, estimated over its
    complete logical target. Supplied pairs override local estimation, not the
    arithmetic, support or padding rules. The caller owns their source identity
    and scope; separate images or logical targets must not pool statistics.
    """

    pixels = np.asarray(x, dtype=np.float32)
    if pixels.ndim not in (2, 3):
        raise ValueError("normalization requires one HW or HWC field")
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    if real_pixel_mask is not None:
        real_pixel_mask = np.asarray(real_pixel_mask)
        if (
            real_pixel_mask.dtype != np.bool_
            or real_pixel_mask.shape != pixels.shape[:2]
        ):
            raise ValueError(
                "real_pixel_mask must be boolean with the field's HW shape"
            )
    if channel_percentiles is not None:
        channel_percentiles = np.array(channel_percentiles, dtype=np.float64, copy=True)
        if (
            channel_percentiles.shape != (pixels.shape[-1], 2)
            or not np.isfinite(channel_percentiles).all()
            or np.any(channel_percentiles[:, 1] < channel_percentiles[:, 0])
        ):
            raise ValueError(
                "channel_percentiles requires one finite ordered pair per channel"
            )
    output = np.zeros_like(pixels)
    for channel in range(pixels.shape[-1]):
        values = pixels[..., channel]
        real_values = values if real_pixel_mask is None else values[real_pixel_mask]
        if real_values.size == 0:
            continue
        lo, hi = (
            np.percentile(real_values, (lower, upper))
            if channel_percentiles is None
            else channel_percentiles[channel]
        )
        if hi - lo > 1e-3:
            if real_pixel_mask is None:
                output[..., channel] = (values - lo) / (hi - lo)
            else:
                output[..., channel][real_pixel_mask] = (real_values - lo) / (hi - lo)
    return output.squeeze(-1) if output.shape[-1] == 1 else output


def percentiles_from_uint8_histogram(counts: np.ndarray) -> np.ndarray:
    """Recover exact NumPy-linear P1/P99 pairs from channel-by-256 uint64 counts.

    Each real target pixel contributes once per channel. Callers may sum disjoint
    block histograms without retaining the whole image. These counts describe
    uint8 raster values, not augmented float training images. Returned float64
    pairs retain NumPy's interpolation order and must remain float64 on reuse.
    """

    if (
        not isinstance(counts, np.ndarray)
        or counts.dtype != np.uint64
        or counts.ndim != 2
        or counts.shape[0] == 0
        or counts.shape[1] != 256
    ):
        raise ValueError("counts must be a channel-by-256 uint64 array")
    result = np.zeros((counts.shape[0], 2), dtype=np.float64)
    for channel, bins in enumerate(counts):
        size = sum(int(value) for value in bins)
        if size == 0:
            continue
        if size > np.iinfo(np.int64).max:
            raise ValueError("histogram population exceeds the signed index range")
        ranks = (size - 1) * (np.asarray(DEFAULT_PERCENTILES) / 100.0)
        lower_rank = np.floor(ranks).astype(np.int64)
        upper_rank = np.ceil(ranks).astype(np.int64)
        # Keep count/rank comparisons integer-exact, including above 2**53.
        # The checked total bounds every nonnegative cumulative count to int64.
        cumulative = bins.cumsum(dtype=np.int64)
        low = np.searchsorted(cumulative, lower_rank, side="right").astype(np.float64)
        high = np.searchsorted(cumulative, upper_rank, side="right").astype(np.float64)
        fraction = ranks - lower_rank
        difference = high - low
        result[channel] = np.where(
            fraction >= 0.5,
            high - difference * (1.0 - fraction),
            low + difference * fraction,
        )
    return result
