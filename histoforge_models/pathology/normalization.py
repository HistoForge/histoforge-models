"""Shared real-image pixel normalization for training and inference."""

import numpy as np


def normalize99(
    x: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
    *,
    real_pixel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize each channel over real pixels and leave artificial padding zero.

    Accept one HW or HWC image. Real context and background participate, even
    outside the supervised ROI. The mask describes pixel availability, never
    label support. Without a mask, every supplied pixel is real. Empty support
    or a channel with percentile range at most 1e-3 produces zeros. Values are
    not clipped to [0, 1]. Percentiles use NumPy's linear interpolation and the
    established float32 input/arithmetic order.

    Managed training calls this on each sampled image after augmentation and
    before model/batch padding. Inference calls it once on the complete supplied
    ROI/crop or bounded core-plus-halo processing image, before neural tiling.
    Neither path pools statistics across images or includes artificial padding.
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
    output = np.zeros_like(pixels)
    for channel in range(pixels.shape[-1]):
        values = pixels[..., channel]
        real_values = values if real_pixel_mask is None else values[real_pixel_mask]
        if real_values.size == 0:
            continue
        lo, hi = np.percentile(real_values, (lower, upper))
        if hi - lo > 1e-3:
            if real_pixel_mask is None:
                output[..., channel] = (values - lo) / (hi - lo)
            else:
                output[..., channel][real_pixel_mask] = (real_values - lo) / (hi - lo)
    return output.squeeze(-1) if output.shape[-1] == 1 else output
