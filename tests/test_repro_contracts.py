"""Small fail-closed tests for training provenance and benchmark completeness."""

import hashlib
import inspect
import sys
from types import SimpleNamespace

import pytest


def test_hover_decode_entrypoints_share_versioned_defaults(monkeypatch):
    import numpy as np
    from histoforge_models import core
    from histoforge_models.pathology import hovernet_inference
    from histoforge_models.pathology.decode_profile import hover_decode_profile
    from histoforge_models.pathology.hovernet_inference import (
        run_hovernet_inference,
    )
    from histoforge_models.pathology.training.hovernet_eval import compute_hovernet_seg_ap
    from histoforge_models.pathology.hv_decode import watershed_from_hv_maps

    profile = hover_decode_profile()
    tiled = inspect.signature(run_hovernet_inference).parameters
    metric = inspect.signature(compute_hovernet_seg_ap).parameters
    watershed = inspect.signature(watershed_from_hv_maps).parameters

    # The public wrapper now distinguishes omitted fixed settings from an
    # explicit variable profile. Verify the executed fixed defaults, not the
    # None marker in its signature.
    forwarded = {}

    def run_net(_net, images, **kwargs):
        forwarded.update(kwargs)
        return np.zeros((*images.shape[:3], 4), np.float32), None

    monkeypatch.setattr(core, "run_net", run_net)
    monkeypatch.setattr(hovernet_inference, "watershed_from_hv_maps",
                        lambda maps, **_kwargs: np.zeros(maps.shape[1:], np.int32))
    run_hovernet_inference(None, np.zeros((17, 33, 3), np.uint8))
    assert forwarded["tile_overlap"] == metric["tile_overlap"].default == profile[
        "tile_overlap"]
    assert forwarded["bsize"] == 1024
    assert tiled["edge_threshold"].default == metric[
        "edge_threshold"].default == watershed["edge_threshold"].default == profile[
            "edge_threshold"]
    assert tiled["object_size"].default == metric["object_size"].default == watershed[
        "object_size"].default == profile["object_size"]
    assert tiled["fg_threshold"].default == metric["fg_threshold"].default == watershed[
        "fg_threshold"].default == profile["fg_threshold"]
    assert tiled["morph_kernel"].default == metric["morph_kernel"].default == watershed[
        "morph_kernel"].default == profile["morph_kernel"]
    assert tiled["ksize"].default == watershed["ksize"].default == profile[
        "sobel_kernel"]


def test_file_set_fingerprint_uses_contents(tmp_path):
    from histoforge_models.pathology.training.train_seg import _file_set_summary

    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    first.write_bytes(b"AAAA")
    second.write_bytes(b"BBBB")
    before = _file_set_summary([second, first])

    first.write_bytes(b"CCCC")  # same path and byte count, different corpus
    after = _file_set_summary([first, second])

    assert before["count"] == after["count"] == 2
    assert before["total_bytes"] == after["total_bytes"] == 8
    assert before["content_sha256"] != after["content_sha256"]


def test_lazy_flow_requires_hash_of_current_mask(tmp_path):
    from histoforge_models.pathology.training.train_seg import _resolve_flow_files

    images = tmp_path / "train" / "images"
    masks = tmp_path / "train" / "instance_masks"
    flows = tmp_path / "train" / "flows"
    images.mkdir(parents=True)
    masks.mkdir()
    flows.mkdir()
    image = images / "sample.tif"
    mask = masks / "sample.tif"
    flow = flows / "sample_flows.tif"
    sidecar = flows / "sample_flows.mask.sha256"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask-v1")
    flow.write_bytes(b"derived-flow")
    sidecar.write_text(hashlib.sha256(mask.read_bytes()).hexdigest() + "\n")

    assert _resolve_flow_files([str(image)], str(images), [str(mask)]) == [str(flow)]

    mask.write_bytes(b"mask-v2")
    with pytest.raises(ValueError, match="mask hash mismatch"):
        _resolve_flow_files([str(image)], str(images), [str(mask)])


def test_benchmark_fails_when_a_requested_dataset_fails(monkeypatch, tmp_path):
    import scripts.he_seg_benchmark as benchmark

    monkeypatch.setattr(benchmark, "create_cellpose_model", lambda **_kwargs: object())

    def fail_dataset(*_args, **_kwargs):
        raise FileNotFoundError("held-out split is incomplete")

    monkeypatch.setattr(benchmark, "evaluate_one_dataset", fail_dataset)
    monkeypatch.setitem(sys.modules, "wandb",
                        SimpleNamespace(init=lambda **_kwargs: None))

    with pytest.raises(RuntimeError, match="benchmark incomplete"):
        benchmark.run_benchmark(
            model_path="explicit-checkpoint.pt",
            model_label="contract-test",
            bsize=256,
            batch_size=1,
            nuclei_root=str(tmp_path),
            datasets=["conic"],
            wandb_project="disabled",
        )
