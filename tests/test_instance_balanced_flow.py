"""Contracts for the opt-in equal-instance Cellpose-flow objective."""

import sys

import numpy as np
import pytest
import torch
from torch import nn

from histoforge_models.pathology.managed_training import (
    ManagedSegmentationBatch,
    managed_segmentation_loss,
)
from histoforge_models.train import (
    _inverse_instance_area_map,
    _loss_fn_seg,
    _normalized_instance_flow_weights,
    _prepare_flow_training_label,
    _split_flow_weight_channel,
)


def _raw_flow_label(instance_mask):
    """Return a synthetic ``[instance, cellprob, dY, dX]`` label."""

    label = np.zeros((4, *instance_mask.shape), dtype=np.float32)
    label[0] = instance_mask
    label[1] = instance_mask > 0
    return label


def test_pixel_mode_preserves_historical_label_envelope_exactly():
    raw = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)

    actual = _prepare_flow_training_label(raw, "pixel")

    np.testing.assert_array_equal(actual, raw[1:])
    assert np.shares_memory(actual, raw)


def test_inverse_area_map_gives_each_complete_instance_unit_mass():
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[0:2, 0:2] = 1
    mask[3:7, 2:7] = 9

    weights = _inverse_instance_area_map(mask)

    assert weights.dtype == np.float32
    assert weights[mask == 0].sum() == 0
    assert weights[mask == 1].sum() == pytest.approx(1.0)
    assert weights[mask == 9].sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "mask,message",
    [
        (np.array([[0.0, 1.5]], dtype=np.float32), "non-negative integers"),
        (np.array([[0.0, np.nan]], dtype=np.float32), "finite numeric"),
        (np.array([[0, -1]], dtype=np.int32), "non-negative integers"),
    ],
)
def test_inverse_area_map_rejects_invalid_instance_ids(mask, message):
    with pytest.raises(ValueError, match=message):
        _inverse_instance_area_map(mask)


def test_equal_instance_label_round_trip_keeps_weight_map_before_flow_pair():
    mask = np.zeros((6, 7), dtype=np.int32)
    mask[1:3, 1:3] = 1
    mask[2:6, 4:7] = 2
    raw = _raw_flow_label(mask)
    raw[2] = 17.0
    raw[3] = 23.0

    prepared = _prepare_flow_training_label(raw, "equal_instance")
    restored, weights = _split_flow_weight_channel(
        torch.from_numpy(prepared)[None], "equal_instance"
    )

    assert prepared.shape == (4, 6, 7)
    np.testing.assert_array_equal(prepared[0], raw[1])
    assert prepared[1][mask == 1].sum() == pytest.approx(1.0)
    assert prepared[1][mask == 2].sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(prepared[-2], raw[-2])
    np.testing.assert_array_equal(prepared[-1], raw[-1])
    np.testing.assert_array_equal(restored[0].numpy(), raw[1:])
    np.testing.assert_array_equal(weights[0].numpy(), prepared[1])


def test_spatial_augmentation_uses_nearest_neighbour_for_weight_map():
    from histoforge_models.transforms import random_rotate_and_resize

    mask = np.zeros((32, 32), dtype=np.int32)
    mask[4:12, 4:12] = 1
    mask[16:28, 16:28] = 2
    raw = _raw_flow_label(mask)
    raw[2] = 17.0
    raw[3] = 23.0
    prepared = _prepare_flow_training_label(raw, "equal_instance")

    rng_state = np.random.get_state()
    try:
        np.random.seed(123)
        _, augmented, _ = random_rotate_and_resize(
            [np.zeros((3, 32, 32), dtype=np.float32)],
            Y=[prepared],
            scale_range=0.4,
            xy=(32, 32),
            do_flip=False,
            rotate=True,
        )
    finally:
        np.random.set_state(rng_state)

    np.testing.assert_array_equal(
        np.unique(augmented[0, -3]), np.unique(prepared[-3])
    )
    assert len(np.unique(augmented[0, -2])) > len(np.unique(prepared[-2]))


def test_equal_instance_label_rejects_non_instance_supervision():
    raw = _raw_flow_label(np.zeros((4, 4), dtype=np.int32))
    raw[1, 1:3, 1:3] = 1

    with pytest.raises(ValueError, match="instance IDs and foreground"):
        _prepare_flow_training_label(raw, "equal_instance")


def test_normalized_weights_preserve_foreground_and_background_mass():
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[0:2, 0:2] = 1
    mask[3:7, 2:7] = 2
    source = torch.from_numpy(_inverse_instance_area_map(mask))[None]
    foreground = torch.from_numpy(mask > 0)[None]

    normalized = _normalized_instance_flow_weights(source, foreground)

    n_foreground = int(foreground.sum())
    assert normalized[foreground].sum().item() == pytest.approx(n_foreground)
    assert normalized[0][torch.from_numpy(mask == 1)].sum().item() == (
        pytest.approx(n_foreground / 2)
    )
    assert normalized[0][torch.from_numpy(mask == 2)].sum().item() == (
        pytest.approx(n_foreground / 2)
    )
    assert torch.all(normalized[~foreground] == 1)


def test_crop_fragment_keeps_fractional_source_instance_mass():
    full = np.zeros((8, 12), dtype=np.int32)
    full[0:4, 0:4] = 1
    full[0:4, 6:10] = 2
    source = _inverse_instance_area_map(full)
    crop_foreground = np.zeros_like(full, dtype=bool)
    crop_foreground[0:2, 0:2] = True
    crop_foreground[0:4, 6:10] = True
    cropped_source = np.where(crop_foreground, source, 0.0)

    normalized = _normalized_instance_flow_weights(
        torch.from_numpy(cropped_source)[None],
        torch.from_numpy(crop_foreground)[None],
    )[0]

    fragment_mass = normalized[
        torch.from_numpy((full == 1) & crop_foreground)
    ].sum().item()
    complete_mass = normalized[
        torch.from_numpy((full == 2) & crop_foreground)
    ].sum().item()
    assert fragment_mass / complete_mass == pytest.approx(0.25)


def test_foreground_without_positive_instance_weights_fails_closed():
    source = torch.zeros(1, 4, 4)
    foreground = torch.zeros(1, 4, 4, dtype=torch.bool)
    foreground[:, 1:3, 1:3] = True

    with pytest.raises(ValueError, match="positive instance-flow weight"):
        _normalized_instance_flow_weights(source, foreground)


def test_pixel_loss_is_bit_exact_historical_arithmetic():
    torch.manual_seed(3)
    label = torch.randn(2, 3, 7, 9)
    label[:, 0] = (label[:, 0] > 0).float()
    prediction = torch.randn(2, 3, 7, 9)
    expected = nn.MSELoss(reduction="mean")(
        prediction[:, -3:-1], 5.0 * label[:, -2:]
    )
    expected /= 2.0
    expected += nn.BCEWithLogitsLoss(reduction="mean")(
        prediction[:, -1], (label[:, -3] > 0.5).to(prediction.dtype)
    )

    actual = _loss_fn_seg(label, prediction, torch.device("cpu"))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_equal_instance_weights_only_flow_mse_and_matches_managed_loss():
    mask = np.zeros((4, 8), dtype=np.int32)
    mask[0:2, 0:2] = 1
    mask[:, 4:8] = 2
    foreground = torch.from_numpy(mask > 0)[None].float()
    weights = torch.from_numpy(_inverse_instance_area_map(mask))[None]
    directions = torch.zeros((1, 2, 4, 8))
    prediction = torch.zeros((1, 3, 4, 8), requires_grad=True)
    prediction.data[:, 0, mask == 1] = 2.0
    prediction.data[:, 0, mask == 2] = 1.0
    prediction.data[:, 2] = torch.linspace(-1.0, 1.0, 32).reshape(4, 8)
    label = torch.cat([foreground[:, None], directions], dim=1)

    normalized = _normalized_instance_flow_weights(weights, foreground.bool())
    expected_flow = (
        prediction[:, :2].square() * normalized[:, None]
    ).mean() / 2.0
    expected_bce = nn.BCEWithLogitsLoss(reduction="mean")(
        prediction[:, 2], foreground
    )
    research = _loss_fn_seg(
        label,
        prediction,
        torch.device("cpu"),
        flow_pixel_weights=weights,
    )
    managed_batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 4, 8)),
        directional_targets=directions,
        foreground_targets=foreground,
        supervision_mask=torch.ones((1, 4, 8), dtype=torch.bool),
        real_pixel_mask=torch.ones((1, 4, 8), dtype=torch.bool),
        instance_flow_weights=weights,
    )
    managed = managed_segmentation_loss(
        prediction,
        managed_batch,
        head_type="flow",
        flow_loss_weighting="equal_instance",
    )

    torch.testing.assert_close(research, expected_flow + expected_bce)
    torch.testing.assert_close(managed.total_sum, research)
    torch.testing.assert_close(managed.component_sums["foreground_bce"], expected_bce)
    research.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_cli_exposes_generic_setting_with_legacy_default(monkeypatch):
    from histoforge_models.pathology.training.train_seg import parse_args

    required = [
        "train_seg",
        "--train_images",
        "images",
        "--train_labels",
        "labels",
        "--save_path",
        "output",
    ]
    monkeypatch.setattr(sys, "argv", required)
    assert parse_args().flow_loss_weighting == "pixel"

    monkeypatch.setattr(
        sys,
        "argv",
        [*required, "--flow_loss_weighting", "equal_instance"],
    )
    assert parse_args().flow_loss_weighting == "equal_instance"
