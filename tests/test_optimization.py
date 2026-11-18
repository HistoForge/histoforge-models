"""DINO parameter groups retain their source order, rates and decay rules."""

import copy

import pytest
import torch
from torch import nn

from histoforge_models.pathology.optimization import build_param_groups


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.cls_token = nn.Parameter(torch.ones(1, 1, 2))
        self.backbone.reg_token = nn.Parameter(torch.ones(1, 4, 2))
        self.backbone.mask_token = nn.Parameter(torch.ones(1, 1, 2))
        self.backbone.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        self.backbone.patch_embed = nn.Linear(2, 2)
        self.backbone.norm = nn.LayerNorm(2)
        self.spm = nn.Linear(2, 2)
        self.cti_modules = nn.ModuleList([nn.Linear(2, 2)])
        self.neck = nn.Linear(2, 2)
        self.out = nn.Linear(2, 2)
        self.out_class = nn.Linear(2, 2)
        self.other = nn.Linear(2, 2)
        self.W2 = nn.Parameter(torch.eye(2), requires_grad=False)
        self.frozen = nn.Linear(2, 2).requires_grad_(False)


def expected_groups(model, adapter_rate):
    rates = {
        "backbone.blocks.0": 0.0025,
        "backbone.blocks.1": 0.005,
        "backbone.blocks.2": 0.01,
        "backbone.patch_embed": 0.02,
        "spm": adapter_rate,
        "cti_modules.0": adapter_rate,
        "neck": 0.04,
        "out": 0.05,
        "out_class": 0.05,
    }
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        rate = rates.get(name.rpartition(".")[0], 0.01)
        no_decay = name.endswith("bias") or name in {
            "backbone.cls_token", "backbone.reg_token", "backbone.norm.weight"
        }
        decay = 0.0 if no_decay else 0.2
        groups.setdefault((rate, decay), []).append(parameter)
    return [{
        "params": values,
        "lr": rate,
        "weight_decay": decay
    } for (rate, decay), values in groups.items()]


@pytest.mark.parametrize("adapter_lr,adapter_rate", [
    (None, 0.04), (0.0, 0.04), (0.03, 0.03)
])
def test_parameter_group_order_ownership_and_optimizer_step(adapter_lr, adapter_rate):
    model = Adapter()
    reference = copy.deepcopy(model)
    original = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    actual = build_param_groups(model, 0.01, 0.5, 0.02, 0.2,
                                neck_lr=0.04, head_lr=0.05,
                                adapter_lr=adapter_lr)
    expected = expected_groups(model, adapter_rate)
    assert len(actual) == len(expected)
    for group, wanted in zip(actual, expected):
        assert group["lr"] == wanted["lr"]
        assert group["weight_decay"] == wanted["weight_decay"]
        assert [id(p) for p in group["params"]] == [id(p) for p in wanted["params"]]
    owned = [id(p) for group in actual for p in group["params"]]
    assert len(owned) == len(set(owned))
    assert set(owned) == {id(p) for p in model.parameters() if p.requires_grad}

    optimizer = torch.optim.AdamW(actual, foreach=False)
    reference_optimizer = torch.optim.AdamW(expected_groups(reference, adapter_rate),
                                           foreach=False)
    for step in range(3):
        for module in (model, reference):
            for index, parameter in enumerate(module.parameters()):
                if parameter.requires_grad:
                    parameter.grad = torch.full_like(parameter,
                                                     (index + 1) / (step + 2))
        optimizer.step()
        reference_optimizer.step()
    for (name, actual_parameter), reference_parameter in zip(
            model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(actual_parameter, reference_parameter, rtol=0, atol=0)
        if not actual_parameter.requires_grad:
            torch.testing.assert_close(actual_parameter, original[name], rtol=0, atol=0)
