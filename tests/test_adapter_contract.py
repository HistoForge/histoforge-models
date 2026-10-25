"""Exact checkpoint and explicit-transfer contracts for the fixed adapter."""

import pytest
import torch
from torch import nn

from conftest import TinyAdapter
from histoforge_models.pathology import common
from histoforge_models.pathology.training import initialization


def test_spatial_prior_shapes_and_layer_norm():
    from histoforge_models.pathology.dinov3 import LayerNorm2d, SPM

    spm = SPM(3, 64, 256)
    outputs = spm(torch.randn(1, 3, 128, 128))
    assert [tuple(output.shape) for output in outputs] == [
        (1, 256, 32, 32),
        (1, 256, 16, 16),
        (1, 256, 8, 8),
    ]
    assert any(isinstance(module, LayerNorm2d) for module in spm.modules())
    assert not any(isinstance(module, nn.BatchNorm2d) for module in spm.modules())


@pytest.fixture()
def tiny_factory(monkeypatch):
    monkeypatch.setattr(common, "DinoV3AdapterBackbone", TinyAdapter)
    return initialization.create_cellpose_model


def _save_donor(tmp_path, *, head_type="flow", nclasses=0):
    net = TinyAdapter(head_type=head_type)
    initialization.initialize_decoupled_class_head(net, nclasses)
    with torch.no_grad():
        for name, value in net.state_dict().items():
            if value.is_floating_point():
                value.fill_(9.0 if name.startswith("backbone.") else 7.0)
    path = tmp_path / "donor.pt"
    torch.save(net.state_dict(), path)
    return net, path


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
def test_exact_load_reconstructs_head_and_classes(tiny_factory, tmp_path, head_type):
    donor, path = _save_donor(tmp_path, head_type=head_type, nclasses=3)
    model = tiny_factory(gpu=False, pretrained_model=path)
    assert model.net.head_type == head_type
    assert model.net.nclasses == 3
    for name, tensor in model.net.state_dict().items():
        torch.testing.assert_close(tensor, donor.state_dict()[name])


@pytest.mark.parametrize("key", ["head_type_code", "spm.weight", "out.bias"])
def test_incomplete_donor_rejected_before_transfer(tiny_factory, tmp_path, key):
    donor, path = _save_donor(tmp_path)
    state = donor.state_dict()
    del state[key]
    torch.save(state, path)
    with pytest.raises(common.StateDictIdentityError, match=key):
        tiny_factory(
            gpu=False,
            pretrained_model=path,
            warm_start=True,
            head_type="hovernet",
            nclasses=3,
        )


@pytest.mark.parametrize("mutation", ["unexpected", "shape", "dtype", "metadata"])
def test_exact_loader_rejects_drift_without_mutation(mutation):
    net = TinyAdapter()
    before = {key: value.clone() for key, value in net.state_dict().items()}
    state = {key: value.clone() for key, value in before.items()}
    state["backbone.weight"].fill_(42)
    if mutation == "unexpected":
        state["retired.weight"] = torch.zeros(1)
    elif mutation == "shape":
        state["out.weight"] = torch.zeros(64, 256, 1, 1)
    elif mutation == "dtype":
        state["out.weight"] = state["out.weight"].double()
    else:
        state["adapter_dim_buf"] = torch.tensor(1024)
    with pytest.raises(common.StateDictIdentityError):
        common.load_state_dict_exact(net, state)
    for name, value in net.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_consistent_module_prefix_supported(tiny_factory, tmp_path):
    donor, path = _save_donor(tmp_path, head_type="hovernet", nclasses=3)
    state = {"module." + key: value for key, value in donor.state_dict().items()}
    torch.save(state, path)
    assert common.detect_head_type_from_state(state) == "hovernet"
    assert common.detect_num_classes_from_state(state) == 3
    model = tiny_factory(gpu=False, pretrained_model=path)
    torch.testing.assert_close(model.net.out_class.weight, donor.out_class.weight)


def test_mixed_prefix_rejected():
    with pytest.raises(common.StateDictIdentityError, match="mixes"):
        common.detect_head_type_from_state(
            {
                "module.head_type_code": torch.tensor(0),
                "W2": torch.zeros(1),
            }
        )


@pytest.mark.parametrize("options", [{"head_type": "hovernet"}, {"nclasses": 3}])
def test_exact_load_cannot_change_heads(tiny_factory, tmp_path, options):
    _, path = _save_donor(tmp_path)
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        tiny_factory(gpu=False, pretrained_model=path, **options)


def test_explicit_task_and_class_transfer(tiny_factory, tmp_path):
    donor, path = _save_donor(tmp_path)
    net = tiny_factory(
        gpu=False,
        pretrained_model=path,
        warm_start=True,
        head_type="hovernet",
        nclasses=3,
    ).net
    torch.testing.assert_close(net.spm.weight, donor.spm.weight)
    torch.testing.assert_close(net.backbone.weight, donor.backbone.weight)
    assert net.head_type == "hovernet" and net.head_type_code.item() == 1
    assert net.nout == 4 and net.W2.shape == (64, 4, 4, 4)
    assert net.out.weight.shape == (64, 256, 1, 1)
    torch.testing.assert_close(net.out_class.weight[:16], -0.5 * net.out.weight[-16:])
    torch.testing.assert_close(net.out_class.weight[16:32], 0.5 * net.out.weight[-16:])
    torch.testing.assert_close(net.out_class.bias[16:32], net.out.bias[-16:])
    assert net.warm_start_report["replace_seg_head"]
    assert net.warm_start_report["replace_class_head"]


def test_matching_class_head_keeps_learned_weights(tiny_factory, tmp_path):
    donor, path = _save_donor(tmp_path, nclasses=3)
    net = tiny_factory(
        gpu=False, pretrained_model=path, warm_start=True, nclasses=3
    ).net
    torch.testing.assert_close(net.out_class.weight, donor.out_class.weight)
    assert not net.warm_start_report["replace_class_head"]


def test_explicit_class_removal(tiny_factory, tmp_path):
    _, path = _save_donor(tmp_path, nclasses=3)
    net = tiny_factory(
        gpu=False, pretrained_model=path, warm_start=True, nclasses=0
    ).net
    assert net.nclasses == 0
    assert not hasattr(net, "out_class") and not hasattr(net, "W3")


def test_raw_backbone_plus_complete_adapter_donor(tiny_factory, tmp_path):
    from safetensors.torch import save_file

    donor, path = _save_donor(tmp_path)
    raw_path = tmp_path / "raw.safetensors"
    raw = {
        key: torch.ones_like(value)
        for key, value in donor.backbone.state_dict().items()
    }
    save_file(raw, raw_path)
    net = tiny_factory(
        gpu=False,
        adapter_checkpoint=path,
        dinov3_backbone_checkpoint=raw_path,
        warm_start=True,
    ).net
    for key, value in net.backbone.state_dict().items():
        torch.testing.assert_close(value, raw[key])
    torch.testing.assert_close(net.spm.weight, donor.spm.weight)
    torch.testing.assert_close(net.out.weight, donor.out.weight)
    assert net.adapter_warm_start_report["preserved_raw_backbone"]
    assert len(net.adapter_warm_start_report["transfer_manifest_sha256"]) == 64


def test_adapter_donor_requires_raw_backbone(tiny_factory):
    with pytest.raises(ValueError, match="requires dinov3_backbone_checkpoint"):
        tiny_factory(gpu=False, warm_start=True, adapter_checkpoint="not-opened.pt")


def test_raw_backbone_artifact_is_strict(tmp_path):
    from safetensors.torch import save_file

    net = nn.Linear(2, 2)
    path = tmp_path / "raw.safetensors"
    state = net.state_dict()
    state["adapter.weight"] = torch.zeros(1)
    save_file(state, path)
    with pytest.raises(ValueError, match="unexpected=.*adapter.weight"):
        initialization._load_dinov3_backbone_checkpoint(net, path)


def test_shell_does_not_load_cpsam(monkeypatch):
    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("implicit Cellpose constructor/model read")

    monkeypatch.setattr(initialization.CellposeModel, "__init__", forbidden_constructor)
    model = initialization._new_cellpose_model_shell(gpu=False)
    assert model.device.type == "cpu"
    assert model.net is None and model.pretrained_model is None
