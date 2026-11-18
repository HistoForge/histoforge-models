"""Target-coordinate decoding retains the original numerical window contract."""

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from histoforge_models import dynamics
from histoforge_models.pathology.cellpose_window_decode import (
    _target_coordinate_flow_steps,
    compute_cellpose_masks_in_target_coordinates,
)


def test_complete_target_decode_matches_ordinary_dynamics_exactly():
    height = width = 96
    yy, xx = np.mgrid[:height, :width]
    center_y, center_x = 49, 47
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    probability = np.where(radius <= 13, 1.0, -1.0).astype(np.float32)
    denominator = np.maximum(radius, 1.0)
    flows = np.stack(
        [(center_y - yy) / denominator * 5.0, (center_x - xx) / denominator * 5.0]
    ).astype(np.float32)
    expected = dynamics.compute_masks(
        flows,
        probability,
        niter=100,
        cellprob_threshold=0.0,
        flow_threshold=0.0,
        min_size=1,
        device=torch.device("cpu"),
    )
    actual = compute_cellpose_masks_in_target_coordinates(
        flows,
        probability,
        target_height=height,
        target_width=width,
        window_x=0,
        window_y=0,
        niter=100,
        cellprob_threshold=0.0,
        flow_threshold=0.0,
        min_instance_pixels=1,
        device="cpu",
    )
    np.testing.assert_array_equal(actual, expected)


def test_trajectories_match_complete_target_not_window_local_coordinates():
    height, width = 80, 92
    window_y, window_x = 12, 14
    window_bottom, window_right = 68, 78
    rng = np.random.default_rng(21)
    flows = (
        gaussian_filter(
            rng.normal(size=(2, height, width)).astype(np.float32), sigma=(0, 3, 3)
        )
        * 2.0
    )
    global_indices = (
        rng.integers(window_y + 10, window_bottom - 10, size=500),
        rng.integers(window_x + 10, window_right - 10, size=500),
    )
    local_indices = (
        global_indices[0] - window_y,
        global_indices[1] - window_x,
    )
    expected = dynamics.steps_interp(
        flows, global_indices, 100, device=torch.device("cpu")
    )
    anchored = _target_coordinate_flow_steps(
        flows[:, window_y:window_bottom, window_x:window_right],
        local_indices,
        100,
        height,
        width,
        window_y,
        window_x,
        torch.device("cpu"),
    )
    ordinary_local = dynamics.steps_interp(
        flows[:, window_y:window_bottom, window_x:window_right],
        local_indices,
        100,
        device=torch.device("cpu"),
    ) + torch.tensor([[window_y], [window_x]])
    # Algebraically equivalent transforms may differ by a few float32 ULPs
    # across PyTorch builds. Integer destinations are the decoder contract.
    torch.testing.assert_close(anchored, expected, atol=2e-5, rtol=0.0)
    torch.testing.assert_close(anchored.int(), expected.int(), atol=0, rtol=0)
    assert torch.count_nonzero(ordinary_local.int() != expected.int()) > 0


@pytest.mark.parametrize("target_width,expected_area", [(16, 0), (32, 4)])
def test_size_exemption_is_only_for_artificial_window_edges(
    monkeypatch, target_width, expected_area
):
    labels = np.zeros((16, 16), np.int32)
    labels[7:9, -2:] = 41
    monkeypatch.setattr(dynamics, "get_masks_torch", lambda *_a, **_k: labels.copy())
    result = compute_cellpose_masks_in_target_coordinates(
        np.zeros((2, 16, 16), np.float32),
        np.ones((16, 16), np.float32),
        target_height=16,
        target_width=target_width,
        window_x=0,
        window_y=0,
        niter=0,
        flow_threshold=0,
        min_instance_pixels=5,
        device="cpu",
    )
    assert np.count_nonzero(result) == expected_area
