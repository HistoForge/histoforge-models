"""Model-field placement shared by managed training and reference inference.

Variable fields preserve natural dimensions. The separate fixed planner exposes
the historical Cellpose grid without allocating a complete padded image or tiles.
"""

import math
from collections.abc import Iterator


def centered_field_padding(
    height: int, width: int, *, minimum_side: int, alignment: int
) -> tuple[int, int, int, int]:
    """Return left, right, top, bottom padding for one already selected field.

    Each axis reaches the aligned maximum of its natural size and the minimum.
    Odd remainders go on the bottom/right. Apply these same offsets to RGB,
    targets, and support masks, and remove them before inference aggregation.
    This function neither tiles oversized inputs nor reads real image context.
    Callers enforce their accepted maximum before allocating a padded tensor.
    """

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (height, width, minimum_side, alignment)
    ):
        raise ValueError(
            "field dimensions, minimum, and alignment must be positive integers"
        )
    if minimum_side % alignment:
        raise ValueError("minimum field side must be divisible by alignment")
    padded_height = (
        (max(height, minimum_side) + alignment - 1) // alignment
    ) * alignment
    padded_width = ((max(width, minimum_side) + alignment - 1) // alignment) * alignment
    extra_y, extra_x = padded_height - height, padded_width - width
    top, left = extra_y // 2, extra_x // 2
    return left, extra_x - left, top, extra_y - top


def variable_field_rectangles(
    height: int,
    width: int,
    *,
    maximum_side: int,
    tile_side: int,
    overlap: float,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield row-major ``(x, y, width, height)`` fields on the natural input.

    Each axis at or below the maximum remains whole. An oversized axis uses the
    preferred tile side with integer step ``floor(tile_side * (1 - overlap))``,
    at least one pixel. The final tile is anchored to the far edge exactly once;
    it may overlap its predecessor more. Minimum/alignment padding happens after
    selecting a field and never shifts this grid or adds source context.
    """

    ys, field_height = variable_field_axis(
        height,
        maximum_side=maximum_side,
        tile_side=tile_side,
        overlap=overlap,
    )
    xs, field_width = variable_field_axis(
        width,
        maximum_side=maximum_side,
        tile_side=tile_side,
        overlap=overlap,
    )
    return ((x, y, field_width, field_height) for y in ys for x in xs)


def variable_field_axis(
    extent: int,
    *,
    maximum_side: int,
    tile_side: int,
    overlap: float,
) -> tuple[tuple[int, ...] | range, int]:
    """Return ordered starts and the natural field size for one target axis.

    This is the indexed form of :func:`variable_field_rectangles`. A bounded
    CoreHalo caller can inspect only starts intersecting its supplied window
    without enumerating the complete target's Cartesian field grid.
    """

    _require_dimensions(extent, maximum_side, tile_side)
    _require_overlap(overlap)
    if tile_side > maximum_side:
        raise ValueError("tile side cannot exceed maximum field side")
    if extent <= maximum_side:
        return (0,), extent
    step = max(1, math.floor(tile_side * (1 - overlap)))
    last = extent - tile_side
    starts = range(0, last + 1, step)
    if starts[-1] == last:
        return starts, tile_side
    return (*starts, last), tile_side


def fixed_field_rectangles(
    height: int, width: int, *, side: int, overlap: float
) -> Iterator[tuple[int, int, int, int]]:
    """Expose the existing non-augmented ``core.run_net`` placement exactly.

    Coordinates refer to the unpadded logical input and may extend beyond it.
    Callers fill that overhang with artificial pixels, not additional image
    context. This deliberately retains the extra stride-16 border, historical
    overlap clamp, linspace placement, and fixed square fields.
    """

    import numpy as np

    from ..transforms import get_pad_yx

    _require_dimensions(height, width, side)
    _require_overlap(overlap)
    top, bottom, left, right = get_pad_yx(height, width, min_size=(side, side))
    overlap = min(0.5, max(0.05, overlap))

    def axis(extent: int, before: int, after: int) -> list[int]:
        padded = extent + before + after
        count = 1 if padded <= side else math.ceil((1 + 2 * overlap) * padded / side)
        return (np.linspace(0, padded - side, count).astype(int) - before).tolist()

    return (
        (x, y, side, side)
        for y in axis(height, top, bottom)
        for x in axis(width, left, right)
    )


def _require_dimensions(*values: int) -> None:
    """Reject invalid public geometry inputs before planning or allocation."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("field dimensions must be positive integers")


def _require_overlap(value: float) -> None:
    """Validate the requested overlap without silently changing its value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value < 1
    ):
        raise ValueError("field overlap must be finite and in [0, 1)")
