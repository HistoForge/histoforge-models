"""Class changes preserve learned tensors and reconstruct through normal loading."""

import pytest
import torch

from conftest import TinyAdapter
from histoforge_models.pathology import common
from histoforge_models.pathology.class_head import (
    adapt_class_head,
    initialize_decoupled_class_head,
)
from histoforge_models.pathology.dinov3 import DinoV3AdapterBackbone


def _source(nclasses=3, head_type="flow"):
    net = TinyAdapter(head_type=head_type)
    initialize_decoupled_class_head(net, nclasses)
    if nclasses:
        with torch.no_grad():
            net.out_class.weight.normal_()
            net.out_class.bias.normal_()
            # Legal nonidentity spatial readout must not be silently reset.
            net.W3.normal_()
    return net


def _state(net):
    return {name: tensor.clone() for name, tensor in net.state_dict().items()}


def _assert_nonclass_unchanged(net, before):
    for name, tensor in net.state_dict().items():
        if name != "W3" and not name.startswith("out_class."):
            assert torch.equal(tensor, before[name]), name


def _assert_initialized(net, first=0):
    area = net.ps**2
    for channel in range(first, net.nclasses):
        rows = slice(channel * area, (channel + 1) * area)
        sign = -0.5 if channel == 0 else 0.5
        assert torch.equal(net.out_class.weight[rows], sign * net.out.weight[-area:])
        assert torch.equal(net.out_class.bias[rows], net.out.bias[-area:])
    assert not net.W3.requires_grad


@pytest.mark.parametrize("nclasses", [0, 3])
def test_use_preserves_parameters_and_rng(nclasses):
    net = _source(nclasses)
    before = _state(net)
    parameters = tuple(net.parameters())
    rng = torch.get_rng_state()
    adapt_class_head(net, nclasses, mode="use")
    assert tuple(net.parameters()) == parameters
    assert torch.equal(torch.get_rng_state(), rng)
    for name, tensor in net.state_dict().items():
        assert torch.equal(tensor, before[name])


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
def test_add_preserves_old_logits_and_full_readout(head_type):
    net = _source(head_type=head_type)
    before = _state(net)
    features = torch.randn(2, 256, 3, 5)
    expected = DinoV3AdapterBackbone._readout(net, features).detach()
    adapt_class_head(net, 5, mode="add")
    actual = DinoV3AdapterBackbone._readout(net, features).detach()
    rows = 3 * net.ps**2
    assert torch.equal(net.out_class.weight[:rows], before["out_class.weight"])
    assert torch.equal(net.out_class.bias[:rows], before["out_class.bias"])
    assert torch.equal(net.W3[:rows, :3], before["W3"])
    assert torch.count_nonzero(net.W3[:rows, 3:]) == 0
    assert torch.count_nonzero(net.W3[rows:, :3]) == 0
    assert torch.equal(net.W3[rows:, 3:].reshape(32, 32), torch.eye(32))
    torch.testing.assert_close(actual[:, :3], expected[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 5:], expected[:, 3:], rtol=0, atol=0)
    _assert_initialized(net, first=3)
    _assert_nonclass_unchanged(net, before)


@pytest.mark.parametrize("target_count", [2, 3, 5])
def test_define_resets_background_and_foreground_even_at_same_count(target_count):
    net = _source()
    before = _state(net)
    adapt_class_head(net, target_count, mode="define")
    assert net.nclasses == target_count
    _assert_initialized(net)
    _assert_nonclass_unchanged(net, before)
    size = target_count * net.ps**2
    assert torch.equal(net.W3.reshape(size, size), torch.eye(size))
    assert not torch.equal(net.out_class.bias[:16], before["out_class.bias"][:16])


@pytest.mark.parametrize("mode", ["add", "define"])
def test_segmentation_source_has_no_class_outputs_to_preserve(mode):
    net = _source(0)
    before = _state(net)
    adapt_class_head(net, 2, mode=mode)
    _assert_initialized(net)
    _assert_nonclass_unchanged(net, before)


@pytest.mark.parametrize("mode,count", [
    ("unknown", 4),
    ("use", 4),
    ("add", 3),
    ("add", 2),
    ("define", 0),
    ("define", 1),
    ("define", True),
    ("define", 3.0),
])
def test_invalid_mode_or_count_fails_before_mutation(mode, count):
    net = _source()
    before = _state(net)
    with pytest.raises(ValueError):
        adapt_class_head(net, count, mode=mode)
    for name, tensor in net.state_dict().items():
        assert torch.equal(tensor, before[name])


@pytest.mark.parametrize("mode,count", [("use", 3), ("add", 5), ("define", 3)])
def test_adapted_state_strictly_reloads_and_trains(monkeypatch, tmp_path, mode, count):
    from safetensors.torch import load_file, save_file

    monkeypatch.setattr(common, "DinoV3AdapterBackbone", TinyAdapter)
    source = _source()
    net = common.build_and_load({
        "head_type": "flow",
        "num_classes": 3
    }, source.state_dict())
    torch.manual_seed(1937)
    adapt_class_head(net, count, mode=mode)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    features = torch.randn(1, 256, 2, 3)
    loss = DinoV3AdapterBackbone._readout(net, features).square().mean()
    loss.backward()
    assert net.out_class.weight.grad is not None
    optimizer.step()
    path = tmp_path / "adapted.safetensors"
    save_file(net.state_dict(), path)
    restored = common.build_and_load({
        "head_type": "flow",
        "num_classes": count
    }, load_file(path))
    assert torch.equal(
        DinoV3AdapterBackbone._readout(net, features),
        DinoV3AdapterBackbone._readout(restored, features),
    )


def test_initializer_preserves_device_and_dtype():
    net = _source().to(dtype=torch.float64)
    adapt_class_head(net, 4, mode="add")
    assert net.out_class.weight.dtype == net.out.weight.dtype == torch.float64
    assert net.W3.dtype == torch.float64
    assert net.W3.device == net.out.weight.device
    _assert_initialized(net, first=3)
