import pytest
import torch

from histoforge_models.pathology.managed_training._hover_losses import HVMapLoss, MSGELossMaps


def _inputs(size=12):
    torch.manual_seed(4)
    hv_pred = torch.randn(1, 2, size, size, requires_grad=True)
    np_pred = torch.randn(1, 2, size, size, requires_grad=True)
    hv_target = torch.randn(1, 2, size, size)
    fg_target = torch.zeros(1, size, size, dtype=torch.long)
    fg_target[:, 3:9, 3:9] = 1
    return hv_pred, np_pred, hv_target, fg_target


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "channel,axis,expected_slope",
    [(0, "y", 12.0), (0, "x", 0.0), (1, "x", 12.0), (1, "y", 0.0)],
)
def test_msge_differentiates_vertical_y_and_horizontal_x(
    dtype, channel, axis, expected_slope
):
    # [V,H] is the target and decoder contract. For a unit own-axis ramp,
    # the unnormalized 5x5 HoVer kernel has response 12; cross-axis response is 0.
    # Center the ramps to reduce cancellation from a constant offset.
    y, x = torch.meshgrid(
        torch.arange(-4, 5, dtype=dtype),
        torch.arange(-5, 6, dtype=dtype),
        indexing="ij",
    )
    prediction = torch.zeros(1, 2, 9, 11, dtype=dtype)
    prediction[0, channel] = y if axis == "y" else x
    prediction.requires_grad_()
    focus = torch.zeros(1, 1, 9, 11, dtype=dtype)
    focus[:, :, 2:-2, 2:-2] = 1
    loss_fn = MSGELossMaps()

    gradient = loss_fn._gradient_vh(prediction)[:, :, 2:-2, 2:-2]
    expected = torch.zeros_like(gradient)
    expected[:, channel] = expected_slope
    torch.testing.assert_close(gradient, expected, rtol=1e-6, atol=1e-6)

    loss = loss_fn(prediction, torch.zeros_like(prediction), focus)
    # 35 focused pixels in each of two channels; only one channel has a slope.
    expected_loss = expected_slope ** 2 * 35 / (70 + 1.0e-8)
    assert loss.item() == pytest.approx(expected_loss, rel=1e-6, abs=1e-10)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[:, 1 - channel]) == 0


def test_all_valid_mask_preserves_existing_hover_loss():
    loss_fn = HVMapLoss()
    hv_pred, np_pred, hv_target, fg_target = _inputs()
    focus = fg_target.unsqueeze(1).to(hv_pred.dtype)
    legacy = {
        "hv_mse": loss_fn.mse(hv_pred, hv_target),
        "hv_msge": loss_fn.msge(hv_pred, hv_target, focus),
        "np_ce": loss_fn.ce(np_pred, fg_target),
        "np_dice": loss_fn.dice(np_pred, fg_target),
    }
    legacy["total"] = sum(legacy.values())
    explicit = loss_fn(
        hv_pred,
        np_pred,
        hv_target,
        fg_target,
        valid_mask=torch.ones_like(fg_target, dtype=torch.bool),
    )
    for name in legacy:
        assert explicit[name].item() == pytest.approx(legacy[name].item())


def test_ignore_pixels_have_zero_gradient_for_every_prediction_channel():
    loss_fn = HVMapLoss()
    hv_pred, np_pred, hv_target, fg_target = _inputs()
    fg_target[:, :4, :] = 255

    loss_fn(hv_pred, np_pred, hv_target, fg_target)["total"].backward()

    assert torch.count_nonzero(hv_pred.grad[:, :, :4, :]) == 0
    assert torch.count_nonzero(np_pred.grad[:, :, :4, :]) == 0
    assert torch.count_nonzero(hv_pred.grad[:, :, 6:, :]) > 0
    assert torch.count_nonzero(np_pred.grad[:, :, 6:, :]) > 0


def test_ignored_prediction_values_do_not_change_loss():
    loss_fn = HVMapLoss()
    hv_pred, np_pred, hv_target, fg_target = _inputs()
    fg_target[:, :, :3] = 255
    first = loss_fn(hv_pred, np_pred, hv_target, fg_target)

    changed_hv = hv_pred.detach().clone()
    changed_np = np_pred.detach().clone()
    changed_hv[:, :, :, :3] = 1.0e6
    changed_np[:, :, :, :3] = -1.0e6
    second = loss_fn(changed_hv, changed_np, hv_target, fg_target)

    for name in first:
        assert second[name].item() == pytest.approx(first[name].item())


def test_msge_erodes_support_by_kernel_radius_and_image_edge():
    msge = MSGELossMaps()
    valid = torch.ones(1, 9, 9, dtype=torch.bool)
    valid[:, 4, 4] = False
    eroded = msge.erode_valid_support(valid)

    assert eroded.shape == (1, 1, 9, 9)
    assert eroded[:, :, 0, 0].all()
    assert eroded[:, :, -1, -1].all()
    assert not eroded[:, :, 2:7, 2:7].any()


def test_managed_msge_treats_tensor_exterior_as_invalid():
    loss_fn = HVMapLoss()
    torch.manual_seed(5)
    full_hv_pred = torch.randn(1, 2, 9, 9)
    full_np_pred = torch.randn(1, 2, 9, 9)
    full_hv_target = torch.randn(1, 2, 9, 9)
    full_fg_target = torch.ones(1, 9, 9, dtype=torch.long)
    full_valid = torch.zeros(1, 9, 9, dtype=torch.bool)
    full_valid[:, 2:7, 2:7] = True

    full = loss_fn(
        full_hv_pred,
        full_np_pred,
        full_hv_target,
        full_fg_target,
        valid_mask=full_valid,
    )
    cropped = loss_fn(
        full_hv_pred[:, :, 2:7, 2:7],
        full_np_pred[:, :, 2:7, 2:7],
        full_hv_target[:, :, 2:7, 2:7],
        full_fg_target[:, 2:7, 2:7],
        valid_mask=torch.ones(1, 5, 5, dtype=torch.bool),
        tensor_exterior_is_invalid=True,
    )

    for name in full:
        assert cropped[name].item() == pytest.approx(full[name].item())


def test_no_valid_pixels_produces_finite_zero_loss_and_gradients():
    loss_fn = HVMapLoss()
    hv_pred, np_pred, hv_target, fg_target = _inputs()
    fg_target.fill_(255)
    result = loss_fn(hv_pred, np_pred, hv_target, fg_target)

    assert result["total"].item() == pytest.approx(0.0)
    result["total"].backward()
    assert torch.count_nonzero(hv_pred.grad) == 0
    assert torch.count_nonzero(np_pred.grad) == 0
