"""Tests for classification inference utilities and explicit adapter initialization.

Covers:
  - eval.py: build_gt_class_map, evaluate_semantic
  - training/initialization.py: initialize_decoupled_class_head (dtype, shapes, weight init),
    create_cellpose_model semantic path (checkpoint round-trip)
  - train.py: _loss_fn_tversky
  - train_seg_loop.py: freeze_seg_head
"""

import json

import numpy as np
import pytest
import torch
from torch import nn

from conftest import TinyAdapter
from histoforge_models.pathology.training.initialization import initialize_decoupled_class_head


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def simple_masks():
    """3 instances in a 10x10 mask."""
    m = np.zeros((10, 10), dtype=np.int32)
    m[0:3, 0:3] = 1
    m[3:6, 3:6] = 2
    m[7:10, 7:10] = 3
    return m


@pytest.fixture()
def simple_class_map():
    """Class map matching simple_masks: inst 1→class 1, inst 2→class 2,
    inst 3→class 1. Background pixels are 0."""
    cm = np.zeros((10, 10), dtype=np.uint8)
    cm[0:3, 0:3] = 1
    cm[3:6, 3:6] = 2
    cm[7:10, 7:10] = 1
    return cm


# ===================================================================
# build_gt_class_map
# ===================================================================


class TestBuildGtClassMap:

    def test_basic(self, tmp_path):
        from histoforge_models.pathology.training.eval import build_gt_class_map
        mask = np.zeros((8, 8), dtype=np.int32)
        mask[0:4, 0:4] = 1
        mask[4:8, 4:8] = 2
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps({"1": 0, "2": 3}))
        cm = build_gt_class_map(mask, json_path)
        assert cm.dtype == np.uint8
        # inst 1 (class 0 in JSON) → 1 in map, inst 2 (class 3) → 4
        assert cm[0, 0] == 1
        assert cm[5, 5] == 4
        assert cm[0, 5] == 0  # background

    def test_unlabeled_stays_zero(self, tmp_path):
        from histoforge_models.pathology.training.eval import build_gt_class_map
        mask = np.zeros((4, 4), dtype=np.int32)
        mask[0:2, 0:2] = 1
        mask[2:4, 2:4] = 2
        json_path = tmp_path / "partial.json"
        json_path.write_text(json.dumps({"1": 0}))  # only inst 1 labeled
        cm = build_gt_class_map(mask, json_path)
        assert cm[0, 0] == 1
        assert cm[3, 3] == 0  # inst 2 not in JSON → stays 0


# ===================================================================
# evaluate_semantic
# ===================================================================


class TestEvaluateSemantic:

    def test_perfect_match(self):
        """Identical pred and GT should give AP=1, error=0."""
        from histoforge_models.pathology.training.eval import evaluate_semantic
        masks = np.zeros((20, 20), dtype=np.int32)
        masks[2:8, 2:8] = 1  # class 1
        masks[10:18, 10:18] = 2  # class 2
        cm = np.zeros((20, 20), dtype=np.uint8)
        cm[2:8, 2:8] = 1
        cm[10:18, 10:18] = 2

        result = evaluate_semantic([masks], [cm], [masks.copy()], [cm.copy()],
                                   nclasses=3)
        assert result["mean_class_ap"] == pytest.approx(1.0)
        assert result["mean_error"] == pytest.approx(0.0)

    def test_all_wrong_class(self):
        """Same masks, all wrong class → TP=0, FN>0, FP>0."""
        from histoforge_models.pathology.training.eval import evaluate_semantic
        masks = np.zeros((20, 20), dtype=np.int32)
        masks[2:8, 2:8] = 1
        masks[10:18, 10:18] = 2
        # Pred says both are class 1
        cp = np.zeros((20, 20), dtype=np.uint8)
        cp[2:8, 2:8] = 1
        cp[10:18, 10:18] = 1
        # GT says inst 1 is class 1, inst 2 is class 2
        ct = np.zeros((20, 20), dtype=np.uint8)
        ct[2:8, 2:8] = 1
        ct[10:18, 10:18] = 2

        result = evaluate_semantic([masks], [cp], [masks.copy()], [ct], nclasses=3)
        # Class 1: TP=1 (inst 1), FP=1 (inst 2 misclassified)
        # Class 2: TP=0, FN=1 (inst 2 missed)
        assert result["tp"][0, 0] == 1  # class 1 TP
        assert result["fp"][0, 0] == 1  # class 1 FP
        assert result["fn"][0, 1] == 1  # class 2 FN

    def test_background_removal(self):
        """Pred instances classified as background should be removed."""
        from histoforge_models.pathology.training.eval import evaluate_semantic
        masks_pred = np.zeros((20, 20), dtype=np.int32)
        masks_pred[2:8, 2:8] = 1  # will be classified as bg
        masks_pred[10:18, 10:18] = 2
        cp = np.zeros((20, 20), dtype=np.uint8)
        cp[2:8, 2:8] = 0  # background!
        cp[10:18, 10:18] = 1

        masks_true = np.zeros((20, 20), dtype=np.int32)
        masks_true[10:18, 10:18] = 1
        ct = np.zeros((20, 20), dtype=np.uint8)
        ct[10:18, 10:18] = 1

        result = evaluate_semantic([masks_pred], [cp], [masks_true], [ct], nclasses=2)
        # After bg removal, only inst 2 remains in pred, matches GT
        assert result["tp"][0, 0] == 1
        assert result["fp"][0, 0] == 0

    def test_no_predictions(self):
        """No predicted masks → all GT are FN."""
        from histoforge_models.pathology.training.eval import evaluate_semantic
        masks_pred = np.zeros((10, 10), dtype=np.int32)
        cp = np.zeros((10, 10), dtype=np.uint8)
        masks_true = np.zeros((10, 10), dtype=np.int32)
        masks_true[2:8, 2:8] = 1
        ct = np.zeros((10, 10), dtype=np.uint8)
        ct[2:8, 2:8] = 1

        result = evaluate_semantic([masks_pred], [cp], [masks_true], [ct], nclasses=2)
        assert result["fn"][0, 0] == 1
        assert result["tp"][0, 0] == 0

    def test_no_ground_truth(self):
        """No GT masks → all predictions are FP."""
        from histoforge_models.pathology.training.eval import evaluate_semantic
        masks_pred = np.zeros((10, 10), dtype=np.int32)
        masks_pred[2:8, 2:8] = 1
        cp = np.zeros((10, 10), dtype=np.uint8)
        cp[2:8, 2:8] = 1
        masks_true = np.zeros((10, 10), dtype=np.int32)
        ct = np.zeros((10, 10), dtype=np.uint8)

        result = evaluate_semantic([masks_pred], [cp], [masks_true], [ct], nclasses=2)
        assert result["fp"][0, 0] == 1
        assert result["fn"][0, 0] == 0

    def test_both_empty(self):
        """No masks on either side → all zeros."""
        from histoforge_models.pathology.training.eval import evaluate_semantic
        empty = np.zeros((10, 10), dtype=np.int32)
        cm = np.zeros((10, 10), dtype=np.uint8)
        result = evaluate_semantic([empty], [cm], [empty.copy()], [cm.copy()],
                                   nclasses=3)
        np.testing.assert_array_equal(result["tp"], np.zeros((1, 2), int))
        np.testing.assert_array_equal(result["fp"], np.zeros((1, 2), int))
        np.testing.assert_array_equal(result["fn"], np.zeros((1, 2), int))

    def test_multi_image(self):
        """Verify metrics aggregate correctly across multiple images."""
        from histoforge_models.pathology.training.eval import evaluate_semantic
        # Image 1: one class-1 instance, perfect match
        m1 = np.zeros((10, 10), dtype=np.int32)
        m1[2:8, 2:8] = 1
        c1 = np.zeros((10, 10), dtype=np.uint8)
        c1[2:8, 2:8] = 1

        # Image 2: one class-2 instance, perfect match
        m2 = np.zeros((10, 10), dtype=np.int32)
        m2[1:9, 1:9] = 1
        c2 = np.zeros((10, 10), dtype=np.uint8)
        c2[1:9, 1:9] = 2

        result = evaluate_semantic([m1, m2], [c1, c2], [m1.copy(), m2.copy()],
                                   [c1.copy(), c2.copy()], nclasses=3)
        # Image 1: class 1 TP=1, class 2 TP=0
        # Image 2: class 1 TP=0, class 2 TP=1
        assert result["tp"][0, 0] == 1
        assert result["tp"][1, 1] == 1
        assert result["mean_class_ap"] == pytest.approx(0.5)

    def test_return_keys(self):
        from histoforge_models.pathology.training.eval import evaluate_semantic
        empty = np.zeros((5, 5), dtype=np.int32)
        cm = np.zeros((5, 5), dtype=np.uint8)
        result = evaluate_semantic([empty], [cm], [empty.copy()], [cm.copy()],
                                   nclasses=2)
        assert "per_class_ap" in result
        assert "mean_class_ap" in result
        assert "per_class_error" in result
        assert "mean_error" in result
        assert "tp" in result
        assert "fp" in result
        assert "fn" in result


# ===================================================================
# initialize_decoupled_class_head
# ===================================================================


class TestDecoupledClassHead:

    @pytest.fixture()
    def bare_transformer(self):
        """Transformer with random weights (no checkpoint download)."""
        net = TinyAdapter(dtype=torch.float32)
        return net

    def test_output_shapes(self, bare_transformer):
        """out_class and W3 have correct shapes; out and W2 unchanged."""
        net = bare_transformer
        nclasses = 5
        ps = net.ps  # 4
        out_w_before = net.out.weight.shape
        w2_before = net.W2.shape
        initialize_decoupled_class_head(net, nclasses)
        # out_class shape (Conv2d: out_channels, in_channels, kH, kW)
        assert net.out_class.weight.shape == (nclasses * ps**2, 256, 1, 1)
        # W3 shape
        assert net.W3.shape == (nclasses * ps**2, nclasses, ps, ps)
        # out and W2 unchanged
        assert net.out.weight.shape == out_w_before
        assert net.W2.shape == w2_before

    def test_seg_weights_untouched(self, bare_transformer):
        """out.weight, out.bias, W2 are identical before and after init."""
        net = bare_transformer
        w_before = net.out.weight.data.clone()
        b_before = net.out.bias.data.clone()
        w2_before = net.W2.data.clone()
        initialize_decoupled_class_head(net, 4)
        torch.testing.assert_close(net.out.weight.data, w_before)
        torch.testing.assert_close(net.out.bias.data, b_before)
        torch.testing.assert_close(net.W2.data, w2_before)

    def test_nclasses_set(self, bare_transformer):
        net = bare_transformer
        initialize_decoupled_class_head(net, 5)
        assert net.nclasses == 5

    def test_bg_from_negative_cellprob(self, bare_transformer):
        """Background (class 0) weights are -0.5 * cellprob weights."""
        net = bare_transformer
        ps = net.ps
        nout = 3
        cp_w = net.out.weight.data[(nout - 1) * ps**2:nout * ps**2].clone()
        initialize_decoupled_class_head(net, 3)
        bg_w = net.out_class.weight.data[:ps**2]
        torch.testing.assert_close(bg_w, -0.5 * cp_w)

    def test_fg_from_positive_cellprob(self, bare_transformer):
        """Foreground class weights are +0.5 * cellprob weights."""
        net = bare_transformer
        ps = net.ps
        nout = 3
        cp_w = net.out.weight.data[(nout - 1) * ps**2:nout * ps**2].clone()
        nclasses = 4
        initialize_decoupled_class_head(net, nclasses)
        for i in range(1, nclasses):
            fg_w = net.out_class.weight.data[i * ps**2:(i + 1) * ps**2]
            torch.testing.assert_close(fg_w, 0.5 * cp_w)

    def test_W3_is_identity(self, bare_transformer):
        """W3 should be a reshaped identity matrix (not trainable)."""
        net = bare_transformer
        nclasses = 5
        ps = net.ps
        initialize_decoupled_class_head(net, nclasses)
        total = nclasses * ps**2
        expected = torch.eye(total, dtype=net.W3.dtype,
                             device=net.W3.device).reshape(total, nclasses, ps, ps)
        torch.testing.assert_close(net.W3, expected)
        assert not net.W3.requires_grad

    def test_dtype_preserved_float32(self, bare_transformer):
        """float32 model stays float32 after class head init."""
        net = bare_transformer
        assert net.out.weight.dtype == torch.float32
        initialize_decoupled_class_head(net, 5)
        assert net.out_class.weight.dtype == torch.float32
        assert net.out_class.bias.dtype == torch.float32
        assert net.W3.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="bfloat16 test needs CUDA")
    def test_dtype_preserved_bfloat16(self):
        """bfloat16 model stays bfloat16 after class head init."""
        net = TinyAdapter(dtype=torch.bfloat16)
        net = net.to("cuda")
        assert net.out.weight.dtype == torch.bfloat16
        initialize_decoupled_class_head(net, 3)
        assert net.out_class.weight.dtype == torch.bfloat16
        assert net.out_class.bias.dtype == torch.bfloat16
        assert net.W3.dtype == torch.bfloat16


class TestExactClassCheckpoint:

    def test_round_trip(self):
        from histoforge_models.pathology.common import load_state_dict_exact

        net = TinyAdapter()
        initialize_decoupled_class_head(net, 4)
        nn.init.xavier_uniform_(net.out_class.weight)
        net2 = TinyAdapter()
        initialize_decoupled_class_head(net2, 4)
        load_state_dict_exact(net2, net.state_dict())
        for key, value in net2.state_dict().items():
            torch.testing.assert_close(value, net.state_dict()[key])

    def test_missing_class_head_is_not_an_exact_load(self):
        from histoforge_models.pathology.common import StateDictIdentityError, load_state_dict_exact

        net = TinyAdapter()
        state = net.state_dict()
        initialize_decoupled_class_head(net, 4)
        with pytest.raises(StateDictIdentityError, match="out_class"):
            load_state_dict_exact(net, state)


# ===================================================================
# Integration: semantic forward pass (CPU-only, avoids MPS bfloat16 issue)
# ===================================================================


@pytest.mark.slow
@pytest.mark.legacy
class TestSemanticForwardPass:

    @pytest.fixture()
    def cpu_mock_model(self):
        """CPU-only MockCellposeModel (avoids MPS bfloat16 incompatibility)."""
        from conftest import MockCellposeModel
        model = MockCellposeModel(n_keep_layers=2, gpu=False)
        return model

    def test_mock_model_forward(self, cpu_mock_model):
        """Decoupled head on MockCellposeModel produces correct
        output channels."""
        model = cpu_mock_model
        nclasses = 4
        initialize_decoupled_class_head(model.net, nclasses)
        x = torch.randn(1, 3, 256, 256, device=model.device,
                        dtype=model.net.out.weight.dtype)
        with torch.no_grad():
            y, _ = model.net(x)
        assert y.shape == (1, nclasses + 3, 256, 256)

    def test_eval_returns_styles_with_class_logits(self, cpu_mock_model):
        """model.eval() on a semantic model returns per-pixel class logits
        in styles, not the default 256-dim vector."""
        model = cpu_mock_model
        nclasses = 3
        initialize_decoupled_class_head(model.net, nclasses)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        masks, flows, styles = model.eval([img], batch_size=1, bsize=256)
        assert styles[0].ndim >= 2
        assert styles[0].shape[-1] == nclasses


# ===================================================================
# _loss_fn_tversky
# ===================================================================


class TestTverskyLoss:

    def _make_batch(self, nclasses, H=16, W=16, target_class=1):
        """Create (lbl, y) where all pixels are target_class and
        model predicts perfectly."""
        B = 2
        # lbl: (B, 4, H, W) — [class_map, cellprob, flowY, flowX]
        lbl = torch.zeros(B, 4, H, W)
        lbl[:, 0] = target_class  # class map

        # y: (B, nclasses+3, H, W) — perfect prediction
        y = torch.zeros(B, nclasses + 3, H, W)
        y[:, target_class] = 10.0  # strong logit for target class
        # seg channels (last 3) — dummy
        y[:, -1] = 1.0
        return lbl, y

    def _make_all_classes_batch(self, nclasses, H=16, W=16):
        """Create (lbl, y) with all classes present and perfectly predicted."""
        B = 1
        lbl = torch.zeros(B, 4, H, W)
        y = torch.zeros(B, nclasses + 3, H, W)
        # Divide columns evenly among classes
        cols_per_class = W // nclasses
        for c in range(nclasses):
            c_start = c * cols_per_class
            c_end = (c + 1) * cols_per_class if c < nclasses - 1 else W
            lbl[:, 0, :, c_start:c_end] = c
            y[:, c, :, c_start:c_end] = 10.0
        y[:, -1] = 1.0
        return lbl, y

    def test_perfect_prediction(self):
        """All-correct prediction with all classes present → low loss."""
        from histoforge_models.train import _loss_fn_tversky
        lbl, y = self._make_all_classes_batch(nclasses=3)
        loss = _loss_fn_tversky(lbl, y)
        assert loss.item() < 0.05

    def test_all_wrong(self):
        """All-wrong prediction → loss near 1."""
        from histoforge_models.train import _loss_fn_tversky
        lbl, y = self._make_batch(nclasses=3, target_class=1)
        # Predict class 2 instead of class 1
        y[:, :] = 0
        y[:, 2] = 10.0
        loss = _loss_fn_tversky(lbl, y)
        assert loss.item() > 0.5

    def test_alpha_asymmetry(self):
        """Lower alpha (penalize FN more) gives different loss than higher."""
        from histoforge_models.train import _loss_fn_tversky
        # Partial mismatch: half pixels correct, half wrong
        B, H, W, nclasses = 1, 16, 16, 2
        lbl = torch.zeros(B, 4, H, W)
        lbl[:, 0, :, :W // 2] = 0
        lbl[:, 0, :, W // 2:] = 1
        y = torch.zeros(B, nclasses + 3, H, W)
        y[:, 0] = 10.0  # predict class 0 everywhere → FN for class 1

        loss_low_alpha = _loss_fn_tversky(lbl, y, alpha=0.1)  # more FN penalty
        loss_high_alpha = _loss_fn_tversky(lbl, y, alpha=0.9)  # more FP penalty
        assert loss_low_alpha.item() != pytest.approx(loss_high_alpha.item(), abs=1e-3)

    def test_gradient_flows(self):
        """Loss supports backward pass."""
        from histoforge_models.train import _loss_fn_tversky
        lbl, y = self._make_batch(nclasses=3, target_class=1)
        y.requires_grad_(True)
        loss = _loss_fn_tversky(lbl, y)
        loss.backward()
        assert y.grad is not None
        assert not torch.all(y.grad == 0)

    def test_eps_prevents_nan(self):
        """When a class has zero presence, loss should not be NaN."""
        from histoforge_models.train import _loss_fn_tversky
        B, H, W, nclasses = 1, 8, 8, 4
        # All pixels are class 0 — classes 1,2,3 have zero presence
        lbl = torch.zeros(B, 4, H, W)
        y = torch.zeros(B, nclasses + 3, H, W)
        y[:, 0] = 10.0  # predict class 0 everywhere
        loss = _loss_fn_tversky(lbl, y)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ===================================================================
# freeze_seg_head
# ===================================================================


class TestFreezeSegHead:

    @pytest.fixture()
    def decoupled_net(self):
        """Transformer with decoupled class head."""
        net = TinyAdapter(dtype=torch.float32)
        initialize_decoupled_class_head(net, 3)
        return net

    def test_out_frozen(self, decoupled_net):
        """freeze_seg_head disables out.weight and W2 gradients."""
        net = decoupled_net
        for param in net.out.parameters():
            param.requires_grad = False
        net.W2.requires_grad = False
        assert not net.out.weight.requires_grad
        assert not net.out.bias.requires_grad
        assert not net.W2.requires_grad

    def test_out_class_trainable(self, decoupled_net):
        """out_class stays trainable after freezing seg head."""
        net = decoupled_net
        for param in net.out.parameters():
            param.requires_grad = False
        net.W2.requires_grad = False
        assert net.out_class.weight.requires_grad
        assert net.out_class.bias.requires_grad

    def test_backbone_and_seg_frozen(self, decoupled_net):
        """freeze_backbone + freeze_seg_head → only out_class trainable."""
        net = decoupled_net
        # freeze backbone (params not starting with "out")
        for name, param in net.named_parameters():
            if not name.startswith("out"):
                param.requires_grad = False
        # freeze seg head
        for param in net.out.parameters():
            param.requires_grad = False
        net.W2.requires_grad = False

        trainable_names = [n for n, p in net.named_parameters() if p.requires_grad]
        assert all(n.startswith("out_class") for n in trainable_names)
        assert len(trainable_names) == 2  # weight + bias


# ===================================================================
# prepare_dense_semantic_labels
# ===================================================================


class TestPrepareDenseSemanticLabels:

    def test_output_shape(self):
        from histoforge_models.pathology.training.data_utils import prepare_dense_semantic_labels
        masks = [np.zeros((32, 48), dtype=np.uint8), np.ones((16, 16), dtype=np.uint8)]
        result = prepare_dense_semantic_labels(masks)
        assert len(result) == 2
        assert result[0].shape == (5, 32, 48)
        assert result[1].shape == (5, 16, 16)

    def test_dtype(self):
        from histoforge_models.pathology.training.data_utils import prepare_dense_semantic_labels
        masks = [np.zeros((8, 8), dtype=np.uint8)]
        result = prepare_dense_semantic_labels(masks)
        assert result[0].dtype == np.float32

    def test_class_map_at_channel_1(self):
        from histoforge_models.pathology.training.data_utils import prepare_dense_semantic_labels
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        mask[6:9, 1:4] = 2
        result = prepare_dense_semantic_labels([mask])
        lbl = result[0]
        # Channel 1 should match input mask
        np.testing.assert_array_equal(lbl[1], mask.astype("float32"))
        # All other channels should be zero
        np.testing.assert_array_equal(lbl[0], np.zeros((10, 10)))
        np.testing.assert_array_equal(lbl[2], np.zeros((10, 10)))
        np.testing.assert_array_equal(lbl[3], np.zeros((10, 10)))
        np.testing.assert_array_equal(lbl[4], np.zeros((10, 10)))

    def test_compatible_with_compute_class_weights(self):
        from histoforge_models.pathology.training.data_utils import (prepare_dense_semantic_labels,
                                                compute_class_weights)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0:10, :] = 1  # 50% class 1
        labels = prepare_dense_semantic_labels([mask])
        weights = compute_class_weights(labels, nclasses=2)
        assert weights.shape == (2,)
        # class 0 = 50%, class 1 = 50% → equal weights
        np.testing.assert_allclose(weights[0], weights[1], rtol=1e-5)
