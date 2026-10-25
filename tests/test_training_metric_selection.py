"""Run the real standalone loop with synthetic batches and a tiny CPU network."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from histoforge_models.pathology.training import hovernet_eval, train_seg_loop as loop
from histoforge_models.pathology.training.segmentation_metrics import evaluate_instance_masks


class _Net(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.out = torch.nn.Linear(1, 1)
        self.W2 = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.diam_mean = torch.nn.Parameter(torch.tensor([30.0]), requires_grad=False)
        self.diam_labels = torch.nn.Parameter(torch.tensor([30.0]), requires_grad=False)
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.channels = channels

    def forward(self, images):
        values = torch.zeros((len(images), self.channels, *images.shape[2:]))
        values[:, 2] = 2  # class 2 predictions on background: false positives
        values[:, 1, 0, 0] = 3  # one correctly classified class-1 target
        return values + self.anchor, None


def _batches(monkeypatch, *, classified=False):
    images = [np.zeros((3, 1, 10), np.float32)]
    labels = [np.zeros((5 if classified else 4, 1, 10), np.float32)]
    labels[0][0, 0, 0] = 1
    if classified:
        labels[0][1, 0, 0] = 1

    def prepare(**kwargs):
        return (images, labels, None, None, None, np.ones(1),
                images, labels, None, None, None, np.ones(1), True)

    def get_batch(indices, **kwargs):
        return [images[i] for i in indices], [labels[i][1:] for i in indices]

    monkeypatch.setattr(loop, "_process_train_test", prepare)
    monkeypatch.setattr(loop, "_get_batch", get_batch)
    monkeypatch.setattr(loop, "random_rotate_and_resize",
                        lambda images, Y, **k: (np.stack(images), np.stack(Y)))
    monkeypatch.setattr(loop.torch, "autocast", lambda **k: nullcontext())
    monkeypatch.setattr(loop, "_loss_fn_seg", lambda labels, y, *a, **k: y.square().mean())
    monkeypatch.setattr(loop, "_loss_fn_class", lambda labels, y, **k: y.square().mean())


@pytest.mark.parametrize("head", ["flow", "hovernet"])
@pytest.mark.parametrize("mode,selected_epochs", [("pooled", [0, 1]), ("per_image", [0])])
def test_checkpoint_selection_and_dataset_logging_use_selected_counts(
    monkeypatch, tmp_path, head, mode, selected_epochs
):
    _batches(monkeypatch)
    net = _Net(3 if head == "flow" else 4)
    epoch = -1
    saved = []
    logged = []
    truth = [np.array([[0, 1]], np.int32), np.arange(10, dtype=np.int32)[None]]

    def evaluate(*args, **kwargs):
        nonlocal epoch
        epoch += 1
        assert kwargs["segmentation_metric_aggregation"] == mode
        prediction = (
            [truth[0], np.zeros_like(truth[1])] if epoch == 0 else
            [np.zeros_like(truth[0]), np.where(truth[1] <= 5, truth[1], 0)]
        )
        return evaluate_instance_masks(truth, prediction, mode)

    monkeypatch.setattr(loop, "compute_seg_ap", evaluate)
    monkeypatch.setattr(hovernet_eval, "compute_hovernet_seg_ap", evaluate)
    monkeypatch.setattr(loop.torch, "save", lambda state, path: saved.append((str(path), epoch)))
    loop.train_seg_wandb(
        net, save_path=tmp_path, model_name="tiny", n_epochs=3, bsize=16,
        min_train_masks=0, ap_every=1, cellpose_model=object(), head_type=head,
        test_names=["synthetic__small", "synthetic__large"],
        segmentation_metric_aggregation=mode,
        wandb_run=SimpleNamespace(log=lambda values, **k: logged.append(values.copy())),
    )
    assert [at for path, at in saved if path.endswith("_best_ap")] == selected_epochs
    assert epoch == 2  # no additional inference for the comparison metric
    assert logged[0]["val/AP@0.5/synthetic"] == (0.1 if mode == "pooled" else 0.5)
    assert logged[0]["val/AP@0.5_pooled"] == 0.1
    assert logged[0]["val/AP@0.5_per_image"] == 0.5
    assert not any("elf" in key for row in logged for key in row)
    assert Path(saved[-1][0]).name == "tiny"  # final still saved


@pytest.mark.parametrize("logging", [False, True])
def test_pixel_f1_includes_prediction_only_class_and_selects_without_wandb(
    monkeypatch, tmp_path, logging
):
    _batches(monkeypatch, classified=True)
    saved, logged = [], []
    monkeypatch.setattr(loop.torch, "save", lambda state, path: saved.append(str(path)))
    loop.train_seg_wandb(
        _Net(7), save_path=tmp_path, model_name="tiny", n_epochs=2, bsize=16,
        min_train_masks=0, nclasses=4, freeze_seg_head=True,
        wandb_run=SimpleNamespace(log=lambda values, **k: logged.append(values.copy()))
        if logging else None,
    )
    assert sum(path.endswith("_best_f1") for path in saved) == 1  # earliest tie
    if logging:
        assert logged[0]["val/pixel_macro_f1"] == 0.5
        assert logged[0]["val/pixel_f1_class_1"] == 1
        assert logged[0]["val/pixel_f1_class_2"] == 0
        assert "val/pixel_f1_class_3" not in logged[0]
