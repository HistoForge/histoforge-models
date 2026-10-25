"""Shared fixed/variable model-field execution before head-specific decoding.

Inputs already have their physical resolution and permitted image context. This
module owns only neural field placement, normalization, padding, and raw blending,
not ROI clipping, instance decoding, or the outer WSI processing strategy.
"""

from bisect import bisect_left, bisect_right
from functools import partial

import numpy as np

from .field_geometry import centered_field_padding, variable_field_axis
from .normalization import normalize99
from .network_execution import NetworkPrecision


VARIABLE_FIELD_PLANNER = "variable-field-v2"
OUTER_ARTIFICIAL_PADDING_MODEL_PX = 8


def run_model_fields(
    net,
    image: np.ndarray,
    *,
    fixed_field_defaults: tuple[int, float],
    bsize: int | None = None,
    tile_overlap: float | None = None,
    batch_size: int = 8,
    normalize: bool = True,
    real_pixel_mask: np.ndarray | None = None,
    field_profile: dict | None = None,
    network_precision: NetworkPrecision = "fp32",
) -> np.ndarray:
    """Return float32 HWC raw predictions for one complete supplied image.

    Without ``field_profile``, use the unchanged fixed Cellpose tiler and the
    calling head's historical defaults. A variable profile is the module's exact
    ``input.fieldProfile``; it cannot coexist with explicit fixed-size settings.
    No dimensions or preprocessing policy are guessed from available memory.
    ``normalize=False`` retains the existing diagnostic/preprocessed-input API;
    normal serving normalizes the complete supplied image before neural tiling.
    """
    from .. import core
    from .network_execution import forward_network

    pixels = np.asarray(image)
    if pixels.ndim == 2:
        pixels = np.stack([pixels, pixels, pixels], axis=-1)
    if pixels.shape[-1] != 3:
        pixels = pixels[..., :3]
    if field_profile is None:
        if normalize:
            pixels = normalize99(pixels, real_pixel_mask=real_pixel_mask)
        output, _ = core.run_net(
            net,
            pixels.astype(np.float32)[None],
            bsize=fixed_field_defaults[0] if bsize is None else bsize,
            tile_overlap=(
                fixed_field_defaults[1] if tile_overlap is None else tile_overlap
            ),
            batch_size=batch_size,
            augment=False,
            forward=partial(forward_network, network_precision=network_precision),
        )
        if not np.isfinite(output).all():
            raise ValueError("model returned nonfinite network maps")
        return output[0]
    if bsize is not None or tile_overlap is not None:
        raise ValueError("field_profile cannot coexist with bsize or tile_overlap")
    return _run_variable_fields(
        net,
        pixels,
        field_profile,
        batch_size,
        normalize,
        real_pixel_mask,
        network_precision,
    )


def _run_variable_fields(
    net, pixels, profile, batch_size, normalize, support, network_precision
):
    """Blend one complete input through the shared target-global field path."""
    from .network_execution import forward_network

    return accumulate_variable_field_window(
        pixels,
        field_profile=profile,
        target_height=pixels.shape[0],
        target_width=pixels.shape[1],
        window_x=0,
        window_y=0,
        real_pixel_mask=support,
        batch_size=batch_size,
        forward_fields=lambda batch: forward_network(
            net, batch, network_precision=network_precision
        )[0],
        normalize=normalize,
    )


def accumulate_variable_field_window(
    pixels,
    *,
    field_profile,
    target_height,
    target_width,
    window_x,
    window_y,
    real_pixel_mask,
    batch_size,
    forward_fields,
    normalize=True,
):
    """Blend raw fields for one window on the complete target's field grid.

    ``pixels`` contains only the supplied HWC window. The target dimensions and
    window origin preserve field positions and row-major batch slots from a
    complete-input run. The logical target is centered inside the profile's
    artificial outer border before fields are planned. Pixels in that border, or
    outside the supplied window, remain unavailable: they are zero and excluded
    from normalization. Statistics come from the complete supplied window, not
    its neural fields or the whole target. Only that logical window is returned.
    This function does not own outer processing windows, decoding, or result
    ownership.
    """
    from .. import transforms

    window = np.asarray(pixels)
    if window.ndim != 3 or window.shape[-1] != 3 or not window.size:
        raise ValueError("variable-field inference requires RGB pixels")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if real_pixel_mask is not None:
        real_pixel_mask = np.asarray(real_pixel_mask)
        if (
            real_pixel_mask.dtype != np.bool_
            or real_pixel_mask.shape != window.shape[:2]
        ):
            raise ValueError("real_pixel_mask must be boolean with the input HW shape")
    _require_window_geometry(
        window.shape[:2], target_height, target_width, window_x, window_y
    )
    if (
        field_profile["planner"] != VARIABLE_FIELD_PLANNER
        or field_profile["aggregation"] != "cellpose-tapered-mean-v1"
        or field_profile["preprocessing"] != "normalize99-real-image-v1"
    ):
        raise ValueError("unsupported variable-field execution contract")
    outer_padding = field_profile.get("outerArtificialPaddingModelPx")
    if (
        isinstance(outer_padding, bool)
        or not isinstance(outer_padding, int)
        or outer_padding != OUTER_ARTIFICIAL_PADDING_MODEL_PX
    ):
        raise ValueError(
            "outerArtificialPaddingModelPx must equal "
            f"{OUTER_ARTIFICIAL_PADDING_MODEL_PX}"
        )

    if normalize:
        window = normalize99(window, real_pixel_mask=real_pixel_mask)
    elif real_pixel_mask is not None:
        window = np.where(real_pixel_mask[..., None], window, 0)

    # Plan in one virtual coordinate frame. Translating both the complete target
    # and every supplied CoreHalo/finite window keeps the artificial border at
    # the true target edge without allocating or normalizing those pixels.
    target_height += 2 * outer_padding
    target_width += 2 * outer_padding
    window_x += outer_padding
    window_y += outer_padding

    minimum = field_profile["minSideModelPx"]
    maximum = field_profile["maxSideModelPx"]
    tile = field_profile["preferredTileSideModelPx"]
    alignment = field_profile["alignmentModelPx"]
    y_starts, field_h = variable_field_axis(
        target_height,
        maximum_side=maximum,
        tile_side=tile,
        overlap=field_profile["tileOverlap"],
    )
    x_starts, field_w = variable_field_axis(
        target_width,
        maximum_side=maximum,
        tile_side=tile,
        overlap=field_profile["tileOverlap"],
    )
    left, right, top, bottom = centered_field_padding(
        field_h, field_w, minimum_side=minimum, alignment=alignment
    )
    if minimum > tile or maximum % alignment or tile % alignment:
        raise ValueError("field sides must be aligned and minimum <= tile <= maximum")
    padded_h, padded_w = field_h + top + bottom, field_w + left + right
    crop = (slice(top, top + field_h), slice(left, left + field_w))
    taper = transforms._taper_mask(padded_h, padded_w)[crop]

    window_right = window_x + window.shape[1]
    window_bottom = window_y + window.shape[0]
    y_indices = _intersecting_field_indices(y_starts, field_h, window_y, window_bottom)
    x_indices = _intersecting_field_indices(x_starts, field_w, window_x, window_right)
    if not y_indices or not x_indices:
        raise RuntimeError("variable field grid does not cover the supplied window")

    x_count = len(x_starts)
    total_fields = len(y_starts) * x_count
    batch_starts = sorted(
        {
            ((y_index * x_count + x_index) // batch_size) * batch_size
            for y_index in y_indices
            for x_index in x_indices
        }
    )
    weights = np.zeros(window.shape[:2], dtype=np.float64)
    output = None
    for batch_start in batch_starts:
        origins = [
            (
                x_starts[field_index % x_count],
                y_starts[field_index // x_count],
            )
            for field_index in range(
                batch_start, min(batch_start + batch_size, total_fields)
            )
        ]
        intersections = [
            _intersection(
                x,
                y,
                x + field_w,
                y + field_h,
                window_x,
                window_y,
                window_right,
                window_bottom,
            )
            for x, y in origins
        ]
        batch = np.zeros((len(origins), 3, padded_h, padded_w), np.float32)
        for index, ((field_x, field_y), intersection) in enumerate(
            zip(origins, intersections, strict=True)
        ):
            if intersection is None:
                continue
            x0, y0, x1, y1 = intersection
            source_y = slice(y0 - window_y, y1 - window_y)
            source_x = slice(x0 - window_x, x1 - window_x)
            field_y_slice = slice(top + y0 - field_y, top + y1 - field_y)
            field_x_slice = slice(left + x0 - field_x, left + x1 - field_x)
            batch[index, :, field_y_slice, field_x_slice] = window[
                source_y, source_x
            ].transpose(2, 0, 1)

        raw = np.asarray(forward_fields(batch), dtype=np.float32)
        if (
            raw.ndim != 4
            or raw.shape[0] != len(origins)
            or raw.shape[1] == 0
            or raw.shape[2:] != (padded_h, padded_w)
            or not np.isfinite(raw).all()
        ):
            raise ValueError(
                "model output must be finite BCHW on the padded field grid"
            )
        if output is None:
            output = np.zeros(
                (raw.shape[1], window.shape[0], window.shape[1]), dtype=np.float32
            )
        elif raw.shape[1] != output.shape[0]:
            raise ValueError("model output channel count changed between fields")
        for field, ((field_x, field_y), intersection) in zip(
            raw, zip(origins, intersections, strict=True), strict=True
        ):
            if intersection is None:
                continue
            x0, y0, x1, y1 = intersection
            output_y = slice(y0 - window_y, y1 - window_y)
            output_x = slice(x0 - window_x, x1 - window_x)
            field_y_slice = slice(y0 - field_y, y1 - field_y)
            field_x_slice = slice(x0 - field_x, x1 - field_x)
            local_taper = taper[field_y_slice, field_x_slice]
            unpadded = field[:, crop[0], crop[1]]
            output[:, output_y, output_x] += (
                unpadded[:, field_y_slice, field_x_slice] * local_taper
            )
            weights[output_y, output_x] += local_taper
    if output is None or np.any(weights <= 0):
        raise RuntimeError("variable field accumulation left uncovered window pixels")
    output /= weights
    return output.transpose(1, 2, 0)


def _require_window_geometry(
    window_shape, target_height, target_width, window_x, window_y
):
    """Validate one nonempty half-open window inside its complete target."""

    for value, name in (
        (target_height, "target_height"),
        (target_width, "target_width"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for value, name in ((window_x, "window_x"), (window_y, "window_y")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if window_x + window_shape[1] > target_width:
        raise ValueError("window pixels exceed target_width")
    if window_y + window_shape[0] > target_height:
        raise ValueError("window pixels exceed target_height")


def _intersecting_field_indices(starts, field_side, window_start, window_end):
    """Return indices whose half-open fields intersect one half-open window."""

    first = bisect_right(starts, window_start - field_side)
    last = bisect_left(starts, window_end)
    return range(first, last)


def _intersection(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
    """Return one half-open intersection, or ``None`` when disjoint."""

    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    return None if x0 >= x1 or y0 >= y1 else (x0, y0, x1, y1)
