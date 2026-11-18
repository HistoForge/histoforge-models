"""Exact label and probability contracts for dense semantic inference."""

import numpy as np
import pytest
import weakref

from histoforge_models.pathology import semantic_inference


def _class_logits(dtype):
    return np.array(
        [
            [[10, 1, 2, 3], [0, 3, 1, 2], [0, 1, 7, 2], [0, 1, 2, 12]],
            [[4, 4, 1, 0], [-9, 3, 3, -5], [1000] * 4, [-1000] * 3 + [-999]],
        ],
        dtype=dtype,
    )


def _expected_probabilities(class_logits):
    """Evaluate the existing float32 softmax independently for each pixel."""
    height, width, classes = class_logits.shape
    expected = np.empty((classes, height, width), dtype=np.float32)
    for row in range(height):
        for column in range(width):
            values = class_logits[row, column].astype(np.float32)
            exponentials = np.exp(values - values.max())
            expected[:, row, column] = exponentials / exponentials.sum()
    return expected


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("segmentation_channels", [0, 3, 4])
def test_labels_and_probabilities_match_pre_extraction_baseline(
    monkeypatch, dtype, segmentation_channels
):
    class_logits = _class_logits(dtype)
    tail = np.full((2, 4, segmentation_channels), 1_000_000, dtype=dtype)
    output = np.concatenate([class_logits, tail], axis=-1)
    original = output.copy()
    monkeypatch.setattr(semantic_inference, "run_model_fields", lambda *a, **k: output)

    result = semantic_inference.run_semantic_inference(
        None, np.zeros((2, 4, 3), dtype=np.uint8), num_classes=4
    )

    assert result.keys() == {"class_map", "probs"}
    # Background and any class a caller excludes from emission still compete.
    # Equal logits select the lowest channel index; segmentation channels do not.
    expected_labels = np.array([[0, 1, 2, 3], [0, 1, 0, 3]], dtype=np.int32)
    np.testing.assert_array_equal(result["class_map"], expected_labels)
    assert result["class_map"].dtype == np.int32
    labels = semantic_inference.class_labels_from_maps(output, num_classes=4)
    np.testing.assert_array_equal(labels, expected_labels)
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(
        result["probs"], _expected_probabilities(class_logits)
    )
    assert result["probs"].dtype == np.float32
    np.testing.assert_array_equal(output, original)


def test_label_selection_precedes_float32_probability_conversion(monkeypatch):
    output = np.array([[[1, 1 + 1e-12, 0, 1_000_000]]], dtype=np.float64)
    monkeypatch.setattr(semantic_inference, "run_model_fields", lambda *a, **k: output)

    result = semantic_inference.run_semantic_inference(
        None, np.zeros((1, 1, 3), dtype=np.uint8), num_classes=3
    )

    np.testing.assert_array_equal(result["class_map"], np.array([[1]], dtype=np.int32))
    np.testing.assert_array_equal(
        semantic_inference.class_labels_from_maps(output, num_classes=3),
        result["class_map"],
    )
    assert result["probs"][0, 0, 0] == result["probs"][1, 0, 0]
    np.testing.assert_array_equal(
        result["probs"], _expected_probabilities(output[..., :3])
    )


@pytest.mark.parametrize("num_classes", [0, -1, -4])
def test_nonpositive_class_count_fails_before_prediction(monkeypatch, num_classes):
    monkeypatch.setattr(
        semantic_inference,
        "run_model_fields",
        lambda *a, **k: pytest.fail("invalid class count reached prediction"),
    )
    with pytest.raises(
        ValueError,
        match=f"run_semantic_inference requires num_classes > 0, got {num_classes}",
    ):
        semantic_inference.run_semantic_inference(None, None, num_classes)


@pytest.mark.parametrize("mode", ["defaults", "fixed", "variable"])
def test_raw_prediction_receives_existing_options_unchanged(monkeypatch, mode):
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    support = np.ones((2, 4), dtype=bool)
    net = object()
    options = {}
    if mode != "defaults":
        options = {
            "batch_size": 3,
            "normalize": False,
            "device": "cpu",
            "real_pixel_mask": support,
        }
    if mode == "fixed":
        options.update(bsize=64, tile_overlap=0.3)
    elif mode == "variable":
        options["field_profile"] = {
            "planner": "variable-field-v2",
            "minSideModelPx": 32,
            "maxSideModelPx": 64,
            "preferredTileSideModelPx": 32,
            "alignmentModelPx": 16,
            "outerArtificialPaddingModelPx": 8,
            "smallRoiContext": False,
            "tileOverlap": 0.25,
            "aggregation": "cellpose-tapered-mean-v1",
            "preprocessing": "normalize99-real-image-v1",
        }
    calls = []

    def predict(supplied_net, supplied_image, **kwargs):
        assert supplied_net is net
        assert supplied_image is image
        calls.append(kwargs)
        return np.full((2, 4, 4), -10, dtype=np.float32)

    monkeypatch.setattr(semantic_inference, "run_model_fields", predict)
    result = semantic_inference.run_semantic_inference(
        net, image, num_classes=1, **options
    )

    assert len(calls) == 1
    expected_options = {
        "fixed_field_defaults": (1024, 0.1),
        "bsize": options.get("bsize"),
        "tile_overlap": options.get("tile_overlap"),
        "batch_size": options.get("batch_size", 8),
        "normalize": options.get("normalize", True),
        "real_pixel_mask": options.get("real_pixel_mask"),
        "field_profile": options.get("field_profile"),
        "network_precision": "fp32",
    }
    assert calls[0].keys() == expected_options.keys()
    for name, expected in expected_options.items():
        if name in ("real_pixel_mask", "field_profile"):
            assert calls[0][name] is expected
        else:
            assert calls[0][name] == expected
    np.testing.assert_array_equal(result["class_map"], np.zeros((2, 4), dtype=np.int32))
    np.testing.assert_array_equal(result["probs"], np.ones((1, 2, 4), dtype=np.float32))


def test_class_labels_need_no_prediction_or_probability_allocation(monkeypatch):
    output = _class_logits(np.float32)
    output.flags.writeable = False
    monkeypatch.setattr(
        semantic_inference,
        "run_model_fields",
        lambda *a, **k: pytest.fail("raw maps were already predicted"),
    )
    monkeypatch.setattr(
        np, "exp", lambda *a, **k: pytest.fail("labels need no softmax")
    )

    labels = semantic_inference.class_labels_from_maps(output, num_classes=4)

    np.testing.assert_array_equal(
        labels, np.array([[0, 1, 2, 3], [0, 1, 0, 3]], dtype=np.int32)
    )
    assert labels.dtype == np.int32


def test_ordinary_inference_reuses_shared_label_selection(monkeypatch):
    output = _class_logits(np.float32)
    monkeypatch.setattr(semantic_inference, "run_model_fields", lambda *a, **k: output)
    select_labels = semantic_inference.class_labels_from_maps
    calls = []

    def select(supplied_output, num_classes):
        assert supplied_output is output
        calls.append(num_classes)
        return select_labels(supplied_output, num_classes)

    monkeypatch.setattr(semantic_inference, "class_labels_from_maps", select)
    result = semantic_inference.run_semantic_inference(
        None, np.zeros((2, 4, 3), dtype=np.uint8), num_classes=4
    )

    assert calls == [4]
    np.testing.assert_array_equal(result["probs"], _expected_probabilities(output))


def test_compact_semantic_preserves_all_labels_without_softmax_or_map_retention(
    monkeypatch,
):
    arrays = []

    def predict(*args, **kwargs):
        output = _class_logits(np.float64)
        output[0, 0, :2] = [10, 10 + 1e-12]
        arrays.append(weakref.ref(output))
        return output

    monkeypatch.setattr(semantic_inference, "run_model_fields", predict)
    monkeypatch.setattr(np, "exp", lambda *a, **k: pytest.fail("unused softmax"))
    result = semantic_inference.run_semantic_inference(
        None,
        np.zeros((2, 4, 3), dtype=np.uint8),
        num_classes=4,
        include_diagnostics=False,
    )
    assert all(array() is None for array in arrays)
    assert result.keys() == {"class_map"}
    np.testing.assert_array_equal(
        result["class_map"], np.array([[1, 1, 2, 3], [0, 1, 0, 3]], dtype=np.int32)
    )
