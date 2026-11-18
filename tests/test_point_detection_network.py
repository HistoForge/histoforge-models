"""Actual adapter/head execution with a small, local DINOv3 trunk fixture."""

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from test_point_detection import FIELD_PROFILE
from torch import nn
from torch.nn import functional as F

from histoforge_models.pathology import common, dinov3, point_inference
from histoforge_models.pathology.class_head import initialize_decoupled_class_head
from histoforge_models.pathology.managed_training.point_losses import point_heatmap_loss
from histoforge_models.pathology.managed_training.point_targets import (
    build_point_targets,
)
from histoforge_models.pathology.optimization import build_param_groups
from histoforge_models.pathology.point_inference import run_point_inference
from histoforge_models.pathology.training.initialization import create_cellpose_model


class TokenBlock(nn.Module):
    def forward(self, value, rope=None):
        return value


class PatchEmbedding(nn.Conv2d):
    def __init__(self):
        super().__init__(3, 1024, kernel_size=16, stride=16)

    def forward(self, value):
        return super().forward(value).permute(0, 2, 3, 1)


class LocalTrunk(nn.Module):
    """Preserve DINOv3's feature interface without downloading or constructing ViT-L."""

    num_prefix_tokens = 5

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.blocks = nn.ModuleList([TokenBlock() for _ in range(24)])
        self.norm_pre = nn.Identity()

    def _pos_embed(self, value):
        tokens = value.reshape(value.shape[0], -1, 1024)
        prefix = value.new_zeros((value.shape[0], 5, 1024))
        return torch.cat((prefix, tokens), dim=1), None


@pytest.fixture
def point_network(monkeypatch):
    def construct(name, *, pretrained, num_classes):
        assert name == dinov3.DINOV3_MODEL
        assert pretrained is False and num_classes == 0
        return LocalTrunk()

    monkeypatch.setattr(dinov3.timm, "create_model", construct)
    return common.build({"head_type": "point_detection", "num_classes": 1})


def test_point_head_has_no_flow_branch_and_keeps_every_full_resolution_subpixel(
    point_network,
):
    network = point_network
    assert network.head_type_code.item() == 2
    assert network.nout == 0 and network.nclasses == 1
    assert not hasattr(network, "out") and not hasattr(network, "W2")
    assert network.out_class.weight.shape == (16, 256, 1, 1)
    assert not network.W3.requires_grad
    features = torch.randn(1, 256, 2, 3)
    raw = network.out_class(features)
    torch.testing.assert_close(
        network._readout(features), F.pixel_shuffle(raw, 4), rtol=0, atol=0
    )
    # Legal nonidentity W3 values remain part of the readout/checkpoint contract.
    with torch.no_grad():
        network.W3.mul_(2)
    torch.testing.assert_close(
        network._readout(features), 2 * F.pixel_shuffle(raw, 4), rtol=0, atol=0
    )


def test_optimizer_assigns_the_center_head_once_at_the_head_learning_rate(
    point_network,
):
    groups = build_param_groups(
        point_network,
        base_lr=1e-5,
        llrd_factor=0.8,
        patch_embed_lr=1e-6,
        weight_decay=0.1,
        head_lr=2e-3,
    )
    parameters = [parameter for group in groups for parameter in group["params"]]
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert {id(parameter) for parameter in parameters} == {
        id(parameter)
        for parameter in point_network.parameters()
        if parameter.requires_grad
    }
    for parameter in point_network.out_class.parameters():
        group = next(
            group
            for group in groups
            if any(item is parameter for item in group["params"])
        )
        assert group["lr"] == 2e-3


def test_rgb_update_strict_safetensors_reload_and_ordinary_prediction(
    point_network, tmp_path
):
    torch.manual_seed(73)
    network = point_network
    images = torch.rand(2, 3, 32, 32)
    support = np.ones((32, 32), dtype=bool)
    targets = torch.from_numpy(
        np.stack(
            [
                build_point_targets(
                    np.array([[11.2, 14.8]]),
                    np.array([0]),
                    class_count=1,
                    sigma_model_px=1.5,
                    support_mask=support,
                ),
                build_point_targets(
                    np.empty((0, 2)),
                    np.empty(0, dtype=np.int64),
                    class_count=1,
                    sigma_model_px=1.5,
                    support_mask=support,
                ),
            ]
        )
    )
    before = network.out_class.weight.detach().clone()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    logits, style = network(images)
    assert logits.shape == (2, 1, 32, 32) and style.shape == (2, 256)
    loss = point_heatmap_loss(
        logits, targets, torch.from_numpy(np.stack([support, support]))
    ).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert network.out_class.weight.grad is not None
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    assert not torch.equal(before, network.out_class.weight)

    path = tmp_path / "point.safetensors"
    save_file(network.state_dict(), path)
    state = load_file(path)
    assert common.detect_head_type_from_state(state) == "point_detection"
    assert common.detect_num_classes_from_state(state) == 1
    restored = common.build_and_load(
        {"head_type": "point_detection", "num_classes": 1}, state
    )
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    options = dict(
        field_profile=FIELD_PROFILE,
        score_threshold=0.5,
        peak_radius_model_px=1,
        max_detections=1024,
        batch_size=1,
    )
    image = images[0].permute(1, 2, 0).numpy()
    expected = run_point_inference(network, image, **options)
    actual = run_point_inference(restored, image, **options)
    assert len(expected["xy"]) > 0
    for name in expected:
        np.testing.assert_array_equal(expected[name], actual[name])


@pytest.mark.parametrize(
    "change", ["missing", "shape", "dtype", "metadata", "flow_tensor"]
)
def test_point_checkpoint_mismatch_never_partially_loads(point_network, change):
    network = point_network
    state = {name: value.clone() for name, value in network.state_dict().items()}
    state["out_class.bias"].fill_(88)
    before = network.out_class.bias.detach().clone()
    if change == "missing":
        del state["out_class.weight"]
    elif change == "shape":
        state["out_class.weight"] = torch.zeros(32, 256, 1, 1)
    elif change == "dtype":
        state["out_class.weight"] = state["out_class.weight"].double()
    elif change == "metadata":
        state["head_type_code"] = torch.tensor(0)
    else:
        state["out.weight"] = torch.zeros(48, 256, 1, 1)
    with pytest.raises(common.StateDictIdentityError):
        common.load_state_dict_exact(network, state)
    torch.testing.assert_close(before, network.out_class.bias, rtol=0, atol=0)


def test_segmentation_initializer_and_cellpose_api_reject_point_checkpoints(
    point_network, tmp_path
):
    with pytest.raises(ValueError, match="no segmentation foreground"):
        initialize_decoupled_class_head(point_network, 2)
    path = tmp_path / "point.pt"
    torch.save(point_network.state_dict(), path)
    with pytest.raises(ValueError, match="Cellpose evaluation API"):
        create_cellpose_model(pretrained_model=path, gpu=False)


@pytest.mark.parametrize("classes", [None, 0, -1, True, 1.5])
def test_invalid_point_class_count_fails_before_trunk_construction(
    monkeypatch, classes
):
    monkeypatch.setattr(
        dinov3.timm, "create_model", lambda *a, **k: pytest.fail("trunk")
    )
    with pytest.raises(ValueError, match="positive num_classes"):
        common.build({"head_type": "point_detection", "num_classes": classes})


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
def test_old_head_readouts_match_the_original_order_and_operations(
    point_network, head_type
):
    network = common.build({"head_type": head_type})
    common._attach_cls_head(network, 3)
    with torch.no_grad():
        network.W3.normal_()
    features = torch.randn(1, 256, 3, 5)
    segmentation = F.conv_transpose2d(network.out(features), network.W2, stride=4)
    classification = F.conv_transpose2d(
        network.out_class(features), network.W3, stride=4
    )
    expected = torch.cat((classification, segmentation), dim=1)
    torch.testing.assert_close(network._readout(features), expected, rtol=0, atol=0)
    assert common.detect_head_type_from_state(network.state_dict()) == head_type


def test_inference_checks_actual_parameter_precision_after_tensor_conversion(
    point_network, monkeypatch
):
    point_network.to(dtype=torch.bfloat16)
    # Module.to does not invoke the existing dtype property setter.
    assert point_network.dtype == torch.float32
    monkeypatch.setattr(
        point_inference, "run_model_fields", lambda *a, **k: pytest.fail("forward")
    )
    with pytest.raises(ValueError, match="FP32 network weights"):
        run_point_inference(
            point_network,
            None,
            field_profile=FIELD_PROFILE,
            score_threshold=0.5,
            peak_radius_model_px=1,
            max_detections=10,
        )


def test_multiple_point_classes_and_module_prefix_strictly_reload(point_network):
    network = common.build({"head_type": "point_detection", "num_classes": 3})
    state = {"module." + name: value for name, value in network.state_dict().items()}
    assert common.detect_head_type_from_state(state) == "point_detection"
    assert common.detect_num_classes_from_state(state) == 3
    restored = common.build_and_load(
        {"head_type": "point_detection", "num_classes": 3}, state
    )
    features = torch.rand(1, 256, 2, 3)
    torch.testing.assert_close(
        network._readout(features), restored._readout(features), rtol=0, atol=0
    )
