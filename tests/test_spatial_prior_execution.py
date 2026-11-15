"""Skipping the unused spatial prior preserves actual adapter and AdamW behavior."""

import copy
from types import MethodType

import pytest
import torch
from torch import nn

from histoforge_models.pathology import dinov3
from histoforge_models.pathology.optimization import build_param_groups
from histoforge_models.pathology.training.initialization import (
    initialize_decoupled_class_head,
)


class _IdentityBlock(nn.Module):
    def forward(self, tokens, *, rope):
        return tokens


class _SmallBackbone(nn.Module):
    """Replace only the expensive pretrained trunk, not SPM, CTI or readout."""

    num_prefix_tokens = 5

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 1024, 16, stride=16)
        self.blocks = nn.ModuleList([_IdentityBlock() for _ in range(24)])
        self.norm_pre = nn.Identity()

    def patch_embed(self, value):
        return self.projection(value).permute(0, 2, 3, 1)

    def _pos_embed(self, value):
        tokens = value.flatten(1, 2)
        prefix = tokens.new_zeros(tokens.shape[0], 5, tokens.shape[2])
        return torch.cat((prefix, tokens), dim=1), None


def _prior_forward_with_discarded_levels(self, value):
    """Original five-level execution and adapter selection, before this change."""
    stride_1 = self.stem_s1(value)
    stride_2 = self.stem_s2(value)
    stride_4 = self.stage1_s4(stride_2)
    stride_8 = self.stage2(stride_4)
    stride_16 = self.stage3(stride_8)
    return [stride_1, stride_2, stride_4, stride_8, stride_16][2:]


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("head,classes", [("flow", 0), ("flow", 3), ("hovernet", 3)])
def test_real_adapter_output_gradients_and_optimizer_match_old_execution(
    monkeypatch, training, head, classes
):
    monkeypatch.setattr(dinov3.timm, "create_model", lambda *a, **k: _SmallBackbone())
    thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(73)
        actual = dinov3.DinoV3AdapterBackbone(head_type=head)
        initialize_decoupled_class_head(actual, classes)
        reference = copy.deepcopy(actual)
        reference.spm.forward = MethodType(
            _prior_forward_with_discarded_levels, reference.spm
        )
        # Strict loading still includes all three unused tensors (928 values).
        actual.load_state_dict(reference.state_dict(), strict=True)
        unused = list(actual.spm.stem_s1.parameters())
        assert len(unused) == 3 and sum(p.numel() for p in unused) == 928
        assert all(p.requires_grad for p in unused)
        assert list(actual.state_dict()) == list(reference.state_dict())
        initial_unused = [p.detach().clone() for p in unused]
        calls = []
        actual.spm.stem_s1.register_forward_hook(lambda *a: calls.append("unused"))

        actual.train(training)
        reference.train(training)
        value = torch.randn(1, 3, 32, 32, requires_grad=True)
        other_value = value.detach().clone().requires_grad_()
        rng_before = torch.get_rng_state()
        output, style = actual(value)
        rng_after_actual = torch.get_rng_state()
        expected, expected_style = reference(other_value)
        assert calls == []
        assert torch.equal(rng_before, rng_after_actual)
        assert torch.equal(rng_after_actual, torch.get_rng_state())
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(style, expected_style, rtol=0, atol=0)
        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(value.grad, other_value.grad, rtol=0, atol=0)
        for (name, p), (other_name, q) in zip(
            actual.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == other_name
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        groups = [
            build_param_groups(net, 1e-4, 0.9, 1e-4, 0.1) for net in (actual, reference)
        ]
        names = [
            {id(p): n for n, p in net.named_parameters()} for net in (actual, reference)
        ]
        for a, b in zip(*groups, strict=True):
            assert (a["lr"], a["weight_decay"]) == (b["lr"], b["weight_decay"])
            assert [names[0][id(p)] for p in a["params"]] == [
                names[1][id(p)] for p in b["params"]
            ]
        optimizers = [torch.optim.AdamW(g, foreach=False) for g in groups]
        for optimizer in optimizers:
            optimizer.step()
        for p, q in zip(actual.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            if p in optimizers[0].state:
                for key, state in optimizers[0].state[p].items():
                    torch.testing.assert_close(
                        state, optimizers[1].state[q][key], rtol=0, atol=0
                    )
        for p, initial in zip(unused, initial_unused, strict=True):
            assert p.grad is None and p not in optimizers[0].state
            torch.testing.assert_close(p, initial, rtol=0, atol=0)
    finally:
        torch.set_num_threads(thread_count)
