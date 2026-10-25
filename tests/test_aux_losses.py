"""Tests for retained classification and calibration losses."""

import numpy as np
import pytest
import torch
from torch import nn


class TestLogitAdjustmentClass:

    def _make_inputs(self, nclasses=3, batch=2, height=4, width=4, seed=0):
        torch.manual_seed(seed)
        y = torch.randn(batch, nclasses + 3, height, width)
        lbl = torch.zeros(batch, 4, height, width)
        lbl[:, 0] = torch.randint(0, nclasses, (batch, height, width)).float()
        return lbl, y

    def test_tau_zero_equals_standard_ce(self):
        from histoforge_models.train import _loss_fn_class
        lbl, y = self._make_inputs()
        plain = _loss_fn_class(lbl, y)
        priors = torch.tensor([-1.5, -0.5, -0.7])
        adjusted = _loss_fn_class(lbl, y, log_priors=priors, tau=0.0)
        assert torch.allclose(plain, adjusted)

    def test_flat_priors_equals_standard_ce(self):
        from histoforge_models.train import _loss_fn_class
        lbl, y = self._make_inputs()
        flat = torch.full((3,), np.log(1.0 / 3))
        plain = _loss_fn_class(lbl, y)
        adjusted = _loss_fn_class(lbl, y, log_priors=flat, tau=1.0)
        assert torch.allclose(plain, adjusted, atol=1e-6)

    def test_non_flat_priors_shift_loss(self):
        from histoforge_models.train import _loss_fn_class
        lbl, y = self._make_inputs()
        priors = torch.log(torch.tensor([0.001, 0.499, 0.5]))
        assert not torch.allclose(_loss_fn_class(lbl, y),
                                  _loss_fn_class(lbl, y, log_priors=priors, tau=1.0))

    def test_hover_channel_slice(self):
        from histoforge_models.train import _loss_fn_class
        torch.manual_seed(1)
        nclasses = 5
        y = torch.randn(2, nclasses + 4, 4, 4)
        lbl = torch.zeros(2, 5, 4, 4)
        lbl[:, 0] = torch.randint(0, nclasses, (2, 4, 4)).float()
        loss = _loss_fn_class(lbl, y, nclasses_slice=-4)
        reference = nn.CrossEntropyLoss(ignore_index=255)(y[:, :nclasses],
                                                          lbl[:, 0].long())
        assert torch.allclose(loss, reference)


class TestLogitAdjustmentTversky:

    def _make_inputs(self, nclasses=3, seed=2):
        torch.manual_seed(seed)
        y = torch.randn(2, nclasses + 3, 4, 4)
        lbl = torch.zeros(2, 4, 4, 4)
        lbl[:, 0] = torch.randint(0, nclasses, (2, 4, 4)).float()
        return lbl, y

    def test_tau_zero_equals_standard_tversky(self):
        from histoforge_models.train import _loss_fn_tversky
        lbl, y = self._make_inputs()
        priors = torch.tensor([-1.5, -0.5, -0.7])
        assert torch.allclose(
            _loss_fn_tversky(lbl, y, n_classes=3),
            _loss_fn_tversky(lbl, y, n_classes=3, log_priors=priors, tau=0.0))

    def test_flat_priors_equal_standard_tversky(self):
        from histoforge_models.train import _loss_fn_tversky
        lbl, y = self._make_inputs()
        flat = torch.full((3,), np.log(1.0 / 3))
        assert torch.allclose(
            _loss_fn_tversky(lbl, y, n_classes=3),
            _loss_fn_tversky(lbl, y, n_classes=3, log_priors=flat, tau=1.0), atol=1e-5)

    def test_hover_channel_slice(self):
        from histoforge_models.train import _loss_fn_tversky
        torch.manual_seed(3)
        nclasses = 5
        y = torch.randn(2, nclasses + 4, 4, 4)
        lbl = torch.zeros(2, 5, 4, 4)
        lbl[:, 0] = torch.randint(0, nclasses, (2, 4, 4)).float()
        loss = _loss_fn_tversky(lbl, y, n_classes=nclasses, nclasses_slice=-4)
        assert torch.isfinite(loss)


class TestMl1AceRelativeOrdering:

    @staticmethod
    def _peaked_logits(target, scale=10.0):
        classes = max(int(target.max().item()) + 1, 2)
        batch, height, width = target.shape
        logits = torch.full((batch, classes, height, width), -scale,
                            dtype=torch.float32)
        for class_index in range(classes):
            selected = target == class_index
            logits[:, class_index][selected] = scale
        return logits

    @pytest.mark.parametrize("variant", ["hard", "soft"])
    def test_well_calibrated_lower_than_overconfident(self, variant):
        from histoforge_models.train import mL1_ACE_loss
        torch.manual_seed(0)
        target = torch.randint(0, 3, (2, 32, 32))
        good = self._peaked_logits(target)
        wrong = self._peaked_logits((target + 1) % 3)
        assert mL1_ACE_loss(wrong, target,
                            variant=variant) > mL1_ACE_loss(good, target,
                                                            variant=variant)

    def test_ignore_index_excludes_pixels(self):
        from histoforge_models.train import mL1_ACE_loss
        target = torch.full((1, 8, 8), 255, dtype=torch.long)
        logits = torch.randn(1, 3, 8, 8)
        assert mL1_ACE_loss(logits, target).item() == 0.0

    def test_gradient_flows_soft(self):
        from histoforge_models.train import mL1_ACE_loss
        torch.manual_seed(2)
        logits = torch.randn(1, 3, 8, 8, requires_grad=True)
        target = torch.randint(0, 3, (1, 8, 8))
        mL1_ACE_loss(logits, target, variant="soft").backward()
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum().item() > 0


class TestMultiTaskLoss:

    def test_retained_tasks_backprop(self):
        from histoforge_models.pathology.training.train_seg_loop import MultiTaskLoss
        torch.manual_seed(0)
        loss_module = MultiTaskLoss(num_tasks=4, learnable=True)
        losses = [torch.randn(1, requires_grad=True).abs() + 0.5 for _ in range(4)]
        loss_module(*losses).backward()
        assert loss_module.log_sigma_sq.grad is not None
        assert (loss_module.log_sigma_sq.grad.abs() > 0).all()


class TestClassificationLossSelection:

    @pytest.mark.parametrize(("choice", "expected"), [
        ("ce", ("cls_ce",)),
        ("tversky", ("cls_tversky",)),
        ("ce+tversky", ("cls_ce", "cls_tversky")),
    ])
    def test_requested_terms_are_exact(self, choice, expected):
        from histoforge_models.pathology.training.train_seg_loop import (
            _classification_task_names,)

        assert _classification_task_names(choice) == expected

    def test_segmentation_only_has_no_classification_terms(self):
        from histoforge_models.pathology.training.train_seg_loop import (
            _classification_task_names,)

        assert _classification_task_names("ce+tversky", False) == ()

    def test_invalid_choice_fails_closed(self):
        from histoforge_models.pathology.training.train_seg_loop import (
            _classification_task_names,)

        with pytest.raises(ValueError, match="cls_loss_fn"):
            _classification_task_names("dice")
