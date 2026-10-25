"""Predict and blend neural-network maps for one image window.

The supplied image is a rectangular crop of a larger ROI or slide. In large-image
inference, this window contains one core plus its surrounding halo. Neural tiles
keep the positions and batch grouping defined for the complete ROI or slide.
Only batches containing tiles that intersect the window are evaluated.

Normalize the real pixels of the supplied image once before neural tiling;
artificial padding stays zero. Return blended maps for every output channel.
The caller chooses the windows, loads their pixels, and converts the maps into
instances or semantic regions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..transforms import _taper_mask, get_pad_yx
from .normalization import normalize99


def predict_window_maps(
    image: Any,
    *,
    real_pixel_mask: np.ndarray,
    target_height: int,
    target_width: int,
    window_x: int,
    window_y: int,
    field_options: Mapping[str, Any],
    batch_size: int,
    forward_tiles: Callable[[np.ndarray], Any],
) -> np.ndarray:
    """Return blended HWC maps for the supplied image window.

    ``target_height`` and ``target_width`` describe the complete model-resolution
    ROI or slide. ``window_x`` and ``window_y`` locate this HWC image within it.
    ``real_pixel_mask`` marks real image pixels, excluding any source padding.

    ``field_options`` selects fixed or variable neural tiles. ``forward_tiles``
    receives float32 NCHW tiles and returns NCHW network outputs. A tile may
    extend beyond the supplied image; those unavailable pixels stay zero.
    This function does not read additional pixels from the ROI or slide.
    """

    if not isinstance(field_options, Mapping):
        raise ValueError("field_options must be a mapping")
    if set(field_options) == {"field_profile"}:
        from .field_inference import (
            accumulate_variable_field_window,
        )

        return accumulate_variable_field_window(
            image,
            field_profile=field_options["field_profile"],
            target_height=target_height,
            target_width=target_width,
            window_x=window_x,
            window_y=window_y,
            real_pixel_mask=real_pixel_mask,
            batch_size=batch_size,
            forward_fields=forward_tiles,
        )
    if set(field_options) != {"bsize", "tile_overlap"}:
        raise ValueError("field_options must select fixed or variable neural tiles")
    bsize = field_options["bsize"]
    tile_overlap = field_options["tile_overlap"]

    window = np.asarray(image, dtype=np.float32)
    if window.ndim != 3 or window.shape[2] <= 0:
        raise ValueError("window image must be a nonempty HWC array")
    support = np.asarray(real_pixel_mask)
    if support.dtype != np.bool_ or support.shape != window.shape[:2]:
        raise ValueError("real_pixel_mask must be boolean with the window's HW shape")
    target_height = _positive_int(target_height, "target_height")
    target_width = _positive_int(target_width, "target_width")
    window_x = _nonnegative_int(window_x, "window_x")
    window_y = _nonnegative_int(window_y, "window_y")
    bsize = _positive_int(bsize, "bsize")
    batch_size = _positive_int(batch_size, "batch_size")
    if window_x + window.shape[1] > target_width:
        raise ValueError("window image exceeds target_width")
    if window_y + window.shape[0] > target_height:
        raise ValueError("window image exceeds target_height")
    overlap = float(tile_overlap)
    if not math.isfinite(overlap):
        raise ValueError("tile_overlap must be finite")
    overlap = min(0.5, max(0.05, overlap))
    window = normalize99(window, real_pixel_mask=support)

    ypad1, ypad2, xpad1, xpad2 = get_pad_yx(
        target_height,
        target_width,
        min_size=(bsize, bsize),
    )
    padded_height = target_height + ypad1 + ypad2
    padded_width = target_width + xpad1 + xpad2
    y_starts = _axis_starts(padded_height, bsize, overlap)
    x_starts = _axis_starts(padded_width, bsize, overlap)

    padded_window = (
        window_y + ypad1,
        window_x + xpad1,
        window_y + ypad1 + window.shape[0],
        window_x + xpad1 + window.shape[1],
    )
    y_indices = np.flatnonzero(
        (y_starts < padded_window[2]) & (y_starts + bsize > padded_window[0])
    )
    x_indices = np.flatnonzero(
        (x_starts < padded_window[3]) & (x_starts + bsize > padded_window[1])
    )
    if not y_indices.size or not x_indices.size:
        raise RuntimeError("the complete input's neural-tile grid misses this window")

    # A slide can contain many neural tiles, but this window needs only a few
    # batches. Find those batches without iterating over every tile in the slide.
    # Keep each tile's original batch and position within that batch.
    x_count = len(x_starts)
    total_tiles = len(y_starts) * x_count
    intersecting_flat_indices = (
        y_indices[:, np.newaxis] * x_count + x_indices[np.newaxis, :]
    ).ravel()
    batch_starts = np.unique((intersecting_flat_indices // batch_size) * batch_size)

    taper = _taper_mask(ly=bsize, lx=bsize)
    weights = np.zeros(window.shape[:2], dtype=np.float64)
    accumulated: np.ndarray | None = None
    for start_value in batch_starts:
        start = int(start_value)
        origins = [
            (
                int(y_starts[flat_index // x_count]),
                int(x_starts[flat_index % x_count]),
            )
            for flat_index in range(start, min(start + batch_size, total_tiles))
        ]
        intersections = [
            (
                _intersection(
                    tile_y,
                    tile_x,
                    tile_y + bsize,
                    tile_x + bsize,
                    *padded_window,
                )
                if _intersects(
                    tile_y,
                    tile_x,
                    tile_y + bsize,
                    tile_x + bsize,
                    *padded_window,
                )
                else None
            )
            for tile_y, tile_x in origins
        ]
        if not any(intersection is not None for intersection in intersections):
            continue

        # Do not pack intersecting tiles into smaller batches: changing batch
        # shape can change GPU rounding and, consequently, decoded predictions.
        # Keep missing tiles as zeros in their original positions and discard
        # their outputs. No pixels outside the supplied window are read.
        tiles = np.zeros(
            (len(origins), window.shape[2], bsize, bsize),
            dtype=np.float32,
        )
        for index, ((tile_y, tile_x), intersection) in enumerate(
            zip(origins, intersections, strict=True)
        ):
            if intersection is None:
                continue
            top, left, bottom, right = intersection
            source_y = slice(top - padded_window[0], bottom - padded_window[0])
            source_x = slice(left - padded_window[1], right - padded_window[1])
            tile_slice_y = slice(top - tile_y, bottom - tile_y)
            tile_slice_x = slice(left - tile_x, right - tile_x)
            tiles[index, :, tile_slice_y, tile_slice_x] = window[
                source_y, source_x
            ].transpose(2, 0, 1)

        outputs = np.asarray(forward_tiles(tiles), dtype=np.float32)
        if (
            outputs.ndim != 4
            or outputs.shape[0] != len(origins)
            or outputs.shape[2:] != (bsize, bsize)
            or outputs.shape[1] <= 0
        ):
            raise ValueError("forward_tiles returned an invalid NCHW output")
        if not np.isfinite(outputs).all():
            raise ValueError("forward_tiles returned nonfinite network maps")
        if accumulated is None:
            accumulated = np.zeros(
                (outputs.shape[1], window.shape[0], window.shape[1]),
                dtype=np.float32,
            )
        elif accumulated.shape[0] != outputs.shape[1]:
            raise ValueError("forward_tiles changed its output channel count")

        for index, ((tile_y, tile_x), intersection) in enumerate(
            zip(origins, intersections, strict=True)
        ):
            if intersection is None:
                continue
            top, left, bottom, right = intersection
            window_y_slice = slice(top - padded_window[0], bottom - padded_window[0])
            window_x_slice = slice(left - padded_window[1], right - padded_window[1])
            tile_y_slice = slice(top - tile_y, bottom - tile_y)
            tile_x_slice = slice(left - tile_x, right - tile_x)
            local_taper = taper[tile_y_slice, tile_x_slice]
            accumulated[:, window_y_slice, window_x_slice] += (
                outputs[index, :, tile_y_slice, tile_x_slice] * local_taper
            )
            weights[window_y_slice, window_x_slice] += local_taper

    if accumulated is None or np.any(weights <= 0):
        raise RuntimeError("neural-tile blending left uncovered window pixels")
    accumulated /= weights[np.newaxis]
    return accumulated.transpose(1, 2, 0)


def _axis_starts(length: int, bsize: int, overlap: float) -> np.ndarray:
    count = (
        1 if length <= bsize else int(np.ceil((1.0 + 2.0 * overlap) * length / bsize))
    )
    return np.linspace(0, length - bsize, count).astype(int)


def _intersection(
    top_a: int,
    left_a: int,
    bottom_a: int,
    right_a: int,
    top_b: int,
    left_b: int,
    bottom_b: int,
    right_b: int,
) -> tuple[int, int, int, int]:
    top = max(top_a, top_b)
    left = max(left_a, left_b)
    bottom = min(bottom_a, bottom_b)
    right = min(right_a, right_b)
    if top >= bottom or left >= right:
        raise ValueError("rectangles do not intersect")
    return top, left, bottom, right


def _intersects(*values: int) -> bool:
    top_a, left_a, bottom_a, right_a, top_b, left_b, bottom_b, right_b = values
    return max(top_a, top_b) < min(bottom_a, bottom_b) and max(left_a, left_b) < min(
        right_a, right_b
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value
