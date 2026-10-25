"""Exact variable-field padding, independent of historic fixed-field planning."""

import numpy as np
import pytest

from histoforge_models.pathology.field_geometry import (
    centered_field_padding,
    fixed_field_rectangles,
    variable_field_rectangles,
)


@pytest.mark.parametrize(
    "shape,minimum,expected",
    [
        ((32, 48), 16, (0, 0, 0, 0)),
        ((9, 22), 32, (5, 5, 11, 12)),
        ((33, 17), 16, (7, 8, 7, 8)),
        ((1, 1), 16, (7, 8, 7, 8)),
        ((48, 5), 32, (13, 14, 0, 0)),
    ],
)
def test_centered_padding_preserves_pixels_and_has_exact_odd_placement(
    shape, minimum, expected
):
    """Real pixels stay in order and removing model padding restores their frame."""
    padding = centered_field_padding(*shape, minimum_side=minimum, alignment=16)
    assert padding == expected
    left, right, top, bottom = padding
    values = np.arange(np.prod(shape)).reshape(shape)
    padded = np.pad(values, ((top, bottom), (left, right)), constant_values=-1)
    assert all(side >= minimum and side % 16 == 0 for side in padded.shape)
    np.testing.assert_array_equal(
        padded[top : top + shape[0], left : left + shape[1]], values
    )


def test_padding_rule_is_independent_on_each_axis():
    """A rectangular field never inherits its long side as a square minimum."""
    for height in range(1, 130):
        for width in (1, 17, 32, 65, 128):
            left, right, top, bottom = centered_field_padding(
                height, width, minimum_side=32, alignment=16
            )
            assert left <= right <= left + 1
            assert top <= bottom <= top + 1
            assert height + top + bottom == 16 * int(np.ceil(max(height, 32) / 16))
            assert width + left + right == 16 * int(np.ceil(max(width, 32) / 16))


@pytest.mark.parametrize(
    "height,width,minimum,alignment",
    [
        (0, 16, 16, 16),
        (16, -1, 16, 16),
        (16, 16, 0, 16),
        (16, 16, 16, 0),
        (True, 16, 16, 16),
        (16.5, 16, 16, 16),
        (16, 16, 17, 16),
    ],
)
def test_invalid_geometric_contract_is_rejected(height, width, minimum, alignment):
    """Reject malformed geometry instead of silently rounding profile parameters."""
    with pytest.raises(ValueError):
        centered_field_padding(height, width, minimum_side=minimum, alignment=alignment)


@pytest.mark.parametrize("shape", [(1, 1), (9, 22), (33, 17), (64, 64)])
def test_variable_roi_within_maximum_is_one_natural_rectangular_field(shape):
    assert list(
        variable_field_rectangles(*shape, maximum_side=64, tile_side=32, overlap=0.25)
    ) == [(0, 0, shape[1], shape[0])]


def test_variable_oversized_axis_uses_requested_overlap_and_far_edge_anchor():
    fields = list(
        variable_field_rectangles(17, 163, maximum_side=64, tile_side=32, overlap=0.25)
    )
    assert fields == [(x, 0, 32, 17) for x in (0, 24, 48, 72, 96, 120, 131)]
    transposed = list(
        variable_field_rectangles(163, 17, maximum_side=64, tile_side=32, overlap=0.25)
    )
    assert transposed == [(0, y, 17, 32) for y in (0, 24, 48, 72, 96, 120, 131)]


@pytest.mark.parametrize("overlap", [0.0, 0.01, 0.25, 0.7, 0.95])
def test_variable_fields_cover_every_pixel_once_or_more_without_resizing(overlap):
    for height in (1, 31, 64, 65, 97):
        for width in (7, 32, 63, 67, 99):
            fields = list(
                variable_field_rectangles(
                    height, width, maximum_side=64, tile_side=32, overlap=overlap
                )
            )
            coverage = np.zeros((height, width), dtype=np.int32)
            assert len(fields) == len(set(fields))
            assert fields == sorted(fields, key=lambda value: (value[1], value[0]))
            for x, y, w, h in fields:
                assert 0 <= x <= width - w and 0 <= y <= height - h
                assert w == (width if width <= 64 else 32)
                assert h == (height if height <= 64 else 32)
                coverage[y : y + h, x : x + w] += 1
            assert coverage.min() >= 1


@pytest.mark.parametrize("shape", [(1, 5), (17, 29), (32, 32), (53, 79), (128, 33)])
@pytest.mark.parametrize("overlap", [0.0, 0.05, 0.1, 0.25, 0.5, 0.9])
def test_fixed_planner_matches_actual_historical_tile_pixels_and_positions(
    shape, overlap
):
    from histoforge_models import transforms

    height, width = shape
    pixels = np.arange(height * width).reshape(1, height, width).astype(np.float32) + 1
    top, bottom, left, right = transforms.get_pad_yx(height, width, min_size=(32, 32))
    old_tiles, ys, xs, _, _ = transforms.make_tiles(
        np.pad(pixels, ((0, 0), (top, bottom), (left, right))),
        bsize=32,
        tile_overlap=overlap,
    )
    old_tiles = old_tiles.reshape(-1, 32, 32)
    fields = list(fixed_field_rectangles(height, width, side=32, overlap=overlap))
    assert len(fields) == len(old_tiles)
    for rectangle, expected, old_y, old_x in zip(fields, old_tiles, ys, xs):
        x, y, w, h = rectangle
        assert (x, y, w, h) == (old_x[0] - left, old_y[0] - top, 32, 32)
        actual = np.zeros((h, w), dtype=np.float32)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
        actual[y0 - y : y1 - y, x0 - x : x1 - x] = pixels[0, y0:y1, x0:x1]
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "changes",
    [
        {"height": 0},
        {"width": -1},
        {"height": True},
        {"width": 3.5},
        {"maximum_side": 0},
        {"tile_side": 65},
        {"overlap": -0.01},
        {"overlap": 1},
        {"overlap": float("nan")},
        {"overlap": True},
    ],
)
def test_variable_planner_rejects_invalid_profile_instead_of_clamping(changes):
    with pytest.raises(ValueError):
        variable_field_rectangles(
            **{
                "height": 32,
                "width": 32,
                "maximum_side": 64,
                "tile_side": 32,
                "overlap": 0.25,
                **changes,
            }
        )
