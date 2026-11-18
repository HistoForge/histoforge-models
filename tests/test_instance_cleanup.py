"""Keep upstream Cellpose cleanup ordering and provisional-window protection."""

import numpy as np
import pytest

from histoforge_models.pathology.instance_cleanup import cleanup_cellpose_instances


def _annulus():
    mask = np.zeros((9, 9), dtype=np.int32)
    mask[2:7, 2:7] = 31
    mask[3:6, 3:6] = 0  # raw area 16, filled area 25
    return mask


def test_small_ring_removed_before_filling():
    assert not cleanup_cellpose_instances(_annulus(), 20).any()


def test_zero_keeps_hole_filling_without_size_filter():
    result = cleanup_cellpose_instances(_annulus(), 0)
    assert np.count_nonzero(result) == 25
    assert result[4, 4] > 0


def test_protected_id_survives_relabeling_and_both_filters():
    mask = _annulus()
    mask[0, 0] = 2
    mask[8, 8] = 99
    result = cleanup_cellpose_instances(mask, 30, protected_instance_ids={31, 99})
    assert result[0, 0] == 0
    assert result[4, 4] > 0 and result[8, 8] > 0
    assert len(np.unique(result[result > 0])) == 2


@pytest.mark.parametrize("threshold,expected", [(16, 16), (17, 0)])
def test_exact_threshold_matches_upstream_rule(threshold, expected):
    mask = np.zeros((6, 6), dtype=np.int32)
    mask[1:5, 1:5] = 71
    assert np.count_nonzero(cleanup_cellpose_instances(mask, threshold)) == expected


def test_background_free_mask_still_filters_its_lowest_positive_label():
    mask = np.array([[5, 9, 9], [9, 9, 9]], dtype=np.int32)

    np.testing.assert_array_equal(
        cleanup_cellpose_instances(mask, 2),
        np.array([[0, 1, 1], [1, 1, 1]], dtype=np.int32),
    )


def test_background_frame_survives_hole_fill_for_second_size_pass():
    mask = np.array(
        [
            [1, 3, 3, 3, 3],
            [3, 2, 2, 2, 3],
            [3, 2, 1, 2, 3],
            [3, 2, 0, 2, 3],
            [3, 2, 2, 2, 3],
        ],
        dtype=np.int32,
    )

    result = cleanup_cellpose_instances(mask, 2)

    assert result[0, 0] == 0
    np.testing.assert_array_equal(np.unique(result), np.array([0, 1, 2]))


def test_flow_decode_calls_upstream_cleanup_once(monkeypatch):
    from histoforge_models import dynamics, utils
    from histoforge_models.pathology import flow_inference

    mask = _annulus()
    monkeypatch.setattr(
        flow_inference,
        "run_model_fields",
        lambda *_args, **_kwargs: np.zeros((9, 9, 3), np.float32),
    )

    def decode(*_args, **kwargs):
        assert kwargs["min_size"] == -1
        return mask.copy()

    monkeypatch.setattr(dynamics, "compute_masks", decode)
    original_cleanup = utils.fill_holes_and_remove_small_masks
    cleanup_calls = []

    def track_cleanup(*args, **kwargs):
        cleanup_calls.append(kwargs["min_size"])
        return original_cleanup(*args, **kwargs)

    monkeypatch.setattr(utils, "fill_holes_and_remove_small_masks", track_cleanup)

    result = flow_inference.run_flow_inference(
        None,
        np.zeros((9, 9, 3), np.uint8),
        device="cpu",
        min_instance_pixels=0,
    )

    assert cleanup_calls == [1]
    assert np.count_nonzero(result["instance_mask"]) == 25


def test_does_not_modify_input():
    mask = _annulus()
    before = mask.copy()
    cleanup_cellpose_instances(mask, 0)
    np.testing.assert_array_equal(mask, before)
