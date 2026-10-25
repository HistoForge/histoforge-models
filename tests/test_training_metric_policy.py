"""Standalone flag, common matcher, and checkpoint wiring on synthetic arrays."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from histoforge_models import metrics
from histoforge_models.pathology.training import eval as evaluation
from histoforge_models.pathology.training import hovernet_eval
from histoforge_models.pathology.training.segmentation_metrics import (
    THRESHOLDS,
    evaluate_instance_masks,
)


def _masks():
    small = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
    many = np.arange(10, dtype=np.int32)[None]
    empty = np.zeros_like(small)
    return [small, many, empty], [small, empty, empty]


@pytest.mark.parametrize("mode,expected", [("pooled", 0.1), ("per_image", 0.5)])
def test_common_evaluator_preserves_cellpose_array_and_reports_both(mode, expected):
    truth, prediction = _masks()
    result = evaluate_instance_masks(truth, prediction, mode)
    with np.errstate(invalid="ignore"):
        ap, tp, fp, fn = metrics.average_precision(truth, prediction, THRESHOLDS)
    for key, wanted in (("ap_array", ap), ("tp", tp), ("fp", fp), ("fn", fn)):
        np.testing.assert_array_equal(result[key], wanted)
    assert result["AP@0.5"] == expected
    assert result["mSA"] == pytest.approx(expected)
    assert result["AP@0.5_pooled"] == 0.1
    assert result["AP@0.5_per_image"] == 0.5
    assert result["mSA_per_image"] == pytest.approx(float(np.nanmean(ap)))


@pytest.mark.parametrize("mode", ["pooled", "per_image"])
def test_flow_and_hover_share_counts_without_elf_or_extra_inference(monkeypatch, mode):
    from histoforge_models import core

    truth, prediction = _masks()
    calls = []

    def flow_eval(images, **kwargs):
        calls.append(("flow", len(images)))
        return prediction, None, None

    def run_net(net, stack, **kwargs):
        calls.append(("hover", len(stack)))
        return np.zeros((*stack.shape[:3], 4), dtype=np.float32), None

    def forbidden_elf(*args, **kwargs):
        pytest.fail("ELF must not participate in validation")

    monkeypatch.setitem(sys.modules, "elf.evaluation",
                        SimpleNamespace(mean_segmentation_accuracy=forbidden_elf))
    monkeypatch.setattr(core, "run_net", run_net)
    decoded = iter(prediction)
    monkeypatch.setattr(hovernet_eval, "watershed_from_hv_maps", lambda *a, **k: next(decoded))
    images = [np.zeros((*mask.shape, 3), dtype=np.uint8) for mask in truth]
    flow = evaluation.compute_seg_ap(SimpleNamespace(eval=flow_eval), images, truth,
                                     segmentation_metric_aggregation=mode)
    hover = hovernet_eval.compute_hovernet_seg_ap(None, images, truth,
                                                 segmentation_metric_aggregation=mode)
    assert flow.keys() == hover.keys()
    assert not any("elf" in key for key in hover)
    for key in flow:
        if isinstance(flow[key], np.ndarray):
            np.testing.assert_array_equal(flow[key], hover[key])
        else:
            assert flow[key] == hover[key]
    assert calls == [("flow", 3), ("hover", 1), ("hover", 1), ("hover", 1)]


def test_invalid_mode_fails_before_inference():
    with pytest.raises(ValueError, match="aggregation"):
        evaluation.compute_seg_ap(None, [], [], segmentation_metric_aggregation="elf")


@pytest.mark.parametrize("head", ["flow", "hovernet"])
@pytest.mark.parametrize("mode", ["pooled", "per_image"])
def test_evaluation_entrypoint_forwards_mode_and_records_it(monkeypatch, tmp_path, head, mode):
    truth, prediction = _masks()
    monkeypatch.setattr(evaluation, "get_paired_files", lambda *a: (["image"], ["label"]))
    monkeypatch.setattr(evaluation, "load_arrays", lambda paths: truth)
    model = SimpleNamespace(net=SimpleNamespace(head_type=head))
    monkeypatch.setattr(evaluation, "create_cellpose_model", lambda **k: model)
    received = []

    def compute(*args, **kwargs):
        received.append(kwargs["segmentation_metric_aggregation"])
        return evaluate_instance_masks(truth, prediction, received[-1])

    monkeypatch.setattr(evaluation, "compute_seg_ap", compute)
    monkeypatch.setattr(hovernet_eval, "compute_hovernet_seg_ap", compute)
    result = evaluation.evaluate_model("synthetic", "images", "labels", save_dir=tmp_path,
                                       segmentation_metric_aggregation=mode)
    assert received == [mode]
    assert result["segmentation_metric_aggregation"] == mode
    assert f"segmentation_metric_aggregation,{mode}" in (tmp_path / "eval_metrics.csv").read_text()


@pytest.mark.parametrize("mode", ["pooled", "per_image"])
def test_training_cli_records_and_forwards_mode_including_post_training(monkeypatch, tmp_path, mode):
    from histoforge_models.pathology.training import train_seg

    args = ["train", "--train_images", "synthetic-train", "--train_labels", "labels",
            "--val_images", "synthetic-val", "--val_labels", "labels",
            "--test_images", "synthetic-post-training", "--test_labels", "labels",
            "--save_path", str(tmp_path), "--segmentation_metric_aggregation", mode]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(train_seg.io, "logger_setup", lambda: (None, None))
    monkeypatch.setattr(train_seg, "get_paired_files", lambda *a: (["image"], ["mask"]))
    monkeypatch.setattr(train_seg, "load_arrays", lambda paths: [np.zeros((2, 2))])
    monkeypatch.setattr(train_seg, "create_cellpose_model",
                        lambda **k: SimpleNamespace(net=None, device="cpu"))
    captured = {}
    monkeypatch.setattr(train_seg, "_write_recipe",
                        lambda path, args, config, files: captured.update(recipe=config))

    def train(*args, **kwargs):
        captured["train"] = kwargs["segmentation_metric_aggregation"]
        return tmp_path / "synthetic-model", np.zeros(1), np.zeros(1)

    def evaluate(**kwargs):
        captured["evaluate"] = kwargs["segmentation_metric_aggregation"]
        return {}

    def init(**kwargs):
        captured["wandb"] = kwargs["config"]
        return SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)

    monkeypatch.setattr(train_seg, "train_seg_wandb", train)
    monkeypatch.setattr(evaluation, "evaluate_model", evaluate)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    train_seg.main()
    assert captured["recipe"]["segmentation_metric_aggregation"] == mode
    assert captured["wandb"]["segmentation_metric_aggregation"] == mode
    assert captured["train"] == captured["evaluate"] == mode


def test_both_clis_default_to_pooled(monkeypatch):
    from histoforge_models.pathology.training import train_seg

    monkeypatch.setattr(sys, "argv", ["eval", "--model_path", "synthetic",
                                      "--eval_images", "images", "--eval_labels", "labels"])
    assert evaluation.parse_args().segmentation_metric_aggregation == "pooled"
    monkeypatch.setattr(sys, "argv", ["train", "--train_images", "images",
                                      "--train_labels", "labels", "--save_path", "output"])
    assert train_seg.parse_args().segmentation_metric_aggregation == "pooled"


def test_existing_he_benchmark_pins_its_cellpose_comparison_protocol(monkeypatch):
    import scripts.he_seg_benchmark as benchmark

    monkeypatch.setattr(benchmark, "pair_test_files", lambda *a: (["synthetic"], ["mask"]))
    monkeypatch.setattr(benchmark, "load_arrays", lambda paths: [np.zeros((2, 2), np.uint8)])
    received = []

    def compute(*args, **kwargs):
        received.append(kwargs["segmentation_metric_aggregation"])
        return {"synthetic": True}

    monkeypatch.setattr(benchmark, "compute_seg_ap", compute)
    assert benchmark.evaluate_one_dataset(None, "unused", "unused", "", 256, 1) == {
        "synthetic": True}
    assert received == ["per_image"]
