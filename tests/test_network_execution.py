"""Precision scope and routing checks, without a GPU or scientific dataset."""

from contextlib import nullcontext

import numpy as np
import pytest
import torch

from histoforge_models import core
from histoforge_models.pathology import network_execution
from histoforge_models.pathology.field_inference import run_model_fields
from tests.test_variable_field_inference import _profile


class TinyNetwork(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 5, 1)
        self.device = torch.device(device)
        self.observed = []

    @property
    def dtype(self):
        return self.layer.weight.dtype

    def forward(self, images):
        self.observed.append(
            (torch.is_grad_enabled(), torch.is_autocast_enabled("cpu"), self.training)
        )
        prediction = self.layer(images)
        return prediction, prediction.mean(dim=(2, 3))


def _cpu_autocast_substitute(monkeypatch):
    """Exercise real autocast on CPU while substituting only CUDA allocation."""
    autocast = torch.autocast
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda *, including_emulation: True
    )
    monkeypatch.setattr(
        core, "_to_device", lambda images, **kwargs: torch.from_numpy(images)
    )
    monkeypatch.setattr(
        torch, "autocast", lambda device, **kwargs: autocast("cpu", **kwargs)
    )


def test_default_and_explicit_fp32_match_upstream_bitwise():
    network = TinyNetwork()
    images = np.random.default_rng(21).random((2, 3, 8, 8), dtype=np.float32)
    expected = core._forward(network, images)
    for options in ({}, {"network_precision": "fp32"}):
        actual = network_execution.forward_network(network, images, **options)
        for observed, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(observed, reference)
            assert observed.dtype == np.float32
    assert network.observed == [(False, False, False)] * 3


def test_bf16_is_only_forward_and_preserves_weights_and_callable(monkeypatch):
    _cpu_autocast_substitute(monkeypatch)
    network = TinyNetwork("cuda:1")
    original = network.forward
    parameters = {name: value.clone() for name, value in network.state_dict().items()}
    images = np.ones((1, 3, 8, 8), np.float32)
    converted = []
    convert = core._from_device

    def from_device(value):
        assert not torch.is_autocast_enabled("cpu")
        converted.append(value.dtype)
        return convert(value)

    monkeypatch.setattr(core, "_from_device", from_device)
    prediction, style = network_execution.forward_network(
        network, images, network_precision="bf16"
    )
    assert network.observed == [(False, True, False)]
    assert converted == [torch.bfloat16, torch.bfloat16]
    assert prediction.dtype == style.dtype == np.float32
    assert network.forward == original
    assert not torch.is_autocast_enabled("cpu")
    assert torch.is_grad_enabled()
    for name, value in network.state_dict().items():
        assert value.dtype == torch.float32
        assert torch.equal(value, parameters[name])


def test_failed_forward_restores_context(monkeypatch):
    _cpu_autocast_substitute(monkeypatch)
    network = TinyNetwork("cuda")

    def fail(_images):
        assert torch.is_autocast_enabled("cpu")
        raise RuntimeError("deliberate network failure")

    monkeypatch.setattr(network, "forward", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        network_execution.forward_network(
            network, np.ones((1, 3, 8, 8), np.float32), network_precision="bf16"
        )
    assert not torch.is_autocast_enabled("cpu")
    assert torch.is_grad_enabled()
    assert network.forward is fail


def test_unknown_precision_fails_without_forward():
    with pytest.raises(ValueError, match="Unsupported network precision"):
        network_execution.forward_network(None, None, network_precision="fp16")


def test_bf16_rejects_cpu_instead_of_falling_back():
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        network_execution.forward_network(TinyNetwork(), None, network_precision="bf16")


def test_bf16_checks_selected_cuda_device_and_rejects_emulation(monkeypatch):
    devices = []
    monkeypatch.setattr(
        torch.cuda, "device", lambda device: (devices.append(device), nullcontext())[1]
    )

    def unsupported(*, including_emulation):
        assert including_emulation is False
        return False

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", unsupported)
    with pytest.raises(RuntimeError, match="native BF16"):
        network_execution.forward_network(
            TinyNetwork("cuda:1"), None, network_precision="bf16"
        )
    assert devices == [torch.device("cuda:1")]


def test_bf16_does_not_convert_non_fp32_weights(monkeypatch):
    _cpu_autocast_substitute(monkeypatch)
    network = TinyNetwork("cuda").half()
    with pytest.raises(ValueError, match="unchanged FP32"):
        network_execution.forward_network(network, None, network_precision="bf16")
    assert network.dtype == torch.float16


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("profile", [None, _profile()])
def test_fixed_and_variable_fields_use_shared_precision_owner(
    monkeypatch, precision, profile
):
    calls = []

    def forward(network, batch, *, network_precision):
        assert network is None
        calls.append(network_precision)
        return batch.copy(), np.zeros((len(batch), 256), np.float32)

    monkeypatch.setattr(network_execution, "forward_network", forward)
    actual = run_model_fields(
        None,
        np.ones((17, 65, 3), np.uint8),
        fixed_field_defaults=(32, 0.1),
        field_profile=profile,
        normalize=False,
        network_precision=precision,
    )
    assert calls and set(calls) == {precision}
    assert actual.dtype == np.float32
    # Existing FP32 weighted accumulation can round by one float32 ULP at one.
    np.testing.assert_allclose(actual, 1.0, rtol=0, atol=np.finfo(np.float32).eps)
