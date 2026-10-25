"""Real-watershed regression fixtures for HoVer map decoding."""

import inspect
import weakref

import numpy as np
import pytest

from histoforge_models.pathology import hovernet_inference


def _disk_maps(num_classes):
    """Six disjoint nuclei, including one fragment on each image side.

    Each edge fragment has 153 pixels, the small interior nucleus has 253,
    and the large interior nucleus has 709. The listed order fixes the IDs
    produced by the existing watershed with its unchanged default settings.
    """
    rows, columns = np.indices((128, 128))
    instance_mask = np.zeros((128, 128), dtype=np.int32)
    class_map = np.zeros_like(instance_mask)
    vertical = np.zeros((128, 128), dtype=np.float32)
    horizontal = np.zeros_like(vertical)
    regions = (
        (1, 30, 9, 0),  # Top edge, background class remains a valid vote.
        (35, 126, 9, 3),  # Right edge, a class callers may choose not to emit.
        (70, 50, 9, 2),  # Small interior instance.
        (70, 1, 9, 2),  # Left edge.
        (90, 95, 15, 1),  # Large interior instance.
        (126, 90, 9, 1),  # Bottom edge.
    )
    for instance_id, (center_y, center_x, radius, class_id) in enumerate(regions, 1):
        pixels = (rows - center_y) ** 2 + (columns - center_x) ** 2 <= radius**2
        instance_mask[pixels] = instance_id
        class_map[pixels] = class_id
        vertical[pixels] = (rows[pixels] - center_y) / radius
        horizontal[pixels] = (columns[pixels] - center_x) / radius

    output = np.zeros((128, 128, num_classes + 4), dtype=np.float32)
    for class_id in range(num_classes):
        output[..., class_id] = np.where(class_map == class_id, 5, -3)
    output[..., -4] = vertical
    output[..., -3] = horizontal
    foreground = instance_mask > 0
    output[..., -2] = np.where(foreground, -8, 8)
    output[..., -1] = np.where(foreground, 8, -8)
    return output, instance_mask, class_map


def _retain_ids(instance_mask, retained_ids):
    expected = np.zeros_like(instance_mask)
    for new_id, old_id in enumerate(sorted(retained_ids), 1):
        expected[instance_mask == old_id] = new_id
    return expected


def _assert_result(
    result, output, original_mask, expected_mask, class_map, num_classes
):
    expected_keys = {"instance_mask", "hv_map", "np_prob"}
    if num_classes:
        expected_keys |= {"class_map", "votes_per_instance"}
    assert result.keys() == expected_keys
    np.testing.assert_array_equal(result["instance_mask"], expected_mask)
    assert result["instance_mask"].dtype == np.int32
    np.testing.assert_array_equal(
        result["hv_map"], np.moveaxis(output[..., -4:-2], -1, 0)
    )
    assert result["hv_map"].dtype == np.float32
    low = np.exp(np.float32(-16))
    denominator = np.float32(1) + low
    expected_probability = np.where(
        original_mask > 0, 1 / denominator, low / denominator
    )
    np.testing.assert_array_equal(result["np_prob"], expected_probability)
    assert result["np_prob"].dtype == np.float32
    if num_classes:
        np.testing.assert_array_equal(result["class_map"], class_map)
        assert result["class_map"].dtype == np.int32
        expected_votes = {
            int(instance_id): int(class_map[expected_mask == instance_id][0])
            for instance_id in np.unique(expected_mask)
            if instance_id > 0
        }
        assert result["votes_per_instance"] == expected_votes


@pytest.mark.parametrize("num_classes", [0, 4])
@pytest.mark.parametrize("min_instance_pixels", [0, 300])
def test_complete_field_defaults_match_real_watershed_baseline(
    monkeypatch, num_classes, min_instance_pixels
):
    output, original_mask, class_map = _disk_maps(num_classes)
    image = np.zeros((*original_mask.shape, 3), dtype=np.uint8)
    calls = []

    def predict(net, supplied_image, **kwargs):
        assert net is None
        assert supplied_image is image
        calls.append(kwargs)
        return output

    monkeypatch.setattr(hovernet_inference, "run_model_fields", predict)
    result = hovernet_inference.run_hovernet_inference(
        None,
        image,
        num_classes=num_classes,
        min_instance_pixels=min_instance_pixels,
    )

    assert calls == [
        {
            "fixed_field_defaults": (1024, 0.25),
            "bsize": None,
            "tile_overlap": None,
            "batch_size": 8,
            "normalize": True,
            "real_pixel_mask": None,
            "field_profile": None,
            "network_precision": "fp32",
        }
    ]
    retained_ids = range(1, 7) if min_instance_pixels == 0 else (5,)
    expected_mask = _retain_ids(original_mask, retained_ids)
    _assert_result(result, output, original_mask, expected_mask, class_map, num_classes)

    decoded = hovernet_inference.decode_hovernet_maps(
        output, num_classes, min_instance_pixels=min_instance_pixels
    )
    _assert_result(
        decoded, output, original_mask, expected_mask, class_map, num_classes
    )


def test_complete_field_batch_default_is_shared_with_window_callers():
    parameters = inspect.signature(hovernet_inference.run_hovernet_inference).parameters
    assert hovernet_inference.DEFAULT_BATCH_SIZE == 8
    assert parameters["batch_size"].default == hovernet_inference.DEFAULT_BATCH_SIZE


@pytest.mark.parametrize("num_classes", [0, 4])
@pytest.mark.parametrize(
    "protected_edges, retained_ids",
    [
        (("top",), (1, 5)),
        (("bottom",), (5, 6)),
        (("left",), (4, 5)),
        (("right",), (2, 5)),
        (("top", "left"), (1, 4, 5)),
        (("top", "bottom", "left", "right"), (1, 2, 4, 5, 6)),
    ],
)
def test_only_selected_artificial_edges_bypass_final_area_filter(
    num_classes, protected_edges, retained_ids
):
    output, original_mask, class_map = _disk_maps(num_classes)

    result = hovernet_inference.decode_hovernet_maps(
        output,
        num_classes,
        min_instance_pixels=300,
        protected_edges=protected_edges,
    )

    # Unselected sides stand for true target edges. Small interior instances
    # also remain subject to the final area threshold in every case.
    expected_mask = _retain_ids(original_mask, retained_ids)
    _assert_result(result, output, original_mask, expected_mask, class_map, num_classes)


@pytest.mark.parametrize("num_classes", [0, 4])
def test_protection_does_not_change_results_when_final_area_filter_is_disabled(
    num_classes,
):
    output, original_mask, class_map = _disk_maps(num_classes)
    result = hovernet_inference.decode_hovernet_maps(
        output,
        num_classes,
        protected_edges=("top", "bottom", "left", "right"),
    )
    _assert_result(result, output, original_mask, original_mask, class_map, num_classes)


def test_protection_cannot_restore_foreground_removed_by_watershed():
    output, _, _ = _disk_maps(0)
    # The existing watershed removes foreground components below ten pixels.
    # Protecting the top side must not bypass that earlier scientific step.
    output[:3, :3, -2] = -8
    output[:3, :3, -1] = 8

    result = hovernet_inference.decode_hovernet_maps(
        output, min_instance_pixels=300, protected_edges=("top",)
    )

    assert not result["instance_mask"][:3, :3].any()
    assert result["instance_mask"][0, 30] > 0


def test_complete_field_forwards_every_existing_decoder_setting(monkeypatch):
    output, _, _ = _disk_maps(4)
    monkeypatch.setattr(hovernet_inference, "run_model_fields", lambda *a, **k: output)
    watershed = hovernet_inference.watershed_from_hv_maps
    calls = []

    def decode(maps, **kwargs):
        calls.append(kwargs)
        return watershed(maps, **kwargs)

    monkeypatch.setattr(hovernet_inference, "watershed_from_hv_maps", decode)
    decoder_settings = {
        "fg_threshold": 0.7,
        "edge_threshold": 0.3,
        "ksize": 7,
        "morph_kernel": 3,
        "object_size": 4,
    }
    result = hovernet_inference.run_hovernet_inference(
        None,
        np.zeros((128, 128, 3), dtype=np.uint8),
        num_classes=4,
        min_instance_pixels=300,
        **decoder_settings,
    )

    assert calls == [decoder_settings]
    assert result["votes_per_instance"] == {1: 1}
    assert np.count_nonzero(result["instance_mask"]) == 709


@pytest.mark.parametrize("num_classes", [0, 4])
def test_compact_prediction_releases_maps_and_preserves_all_instance_votes(
    monkeypatch, num_classes
):
    output, _, _ = _disk_maps(num_classes)
    expected = hovernet_inference.decode_hovernet_maps(output, num_classes)
    arrays = []

    def predict(*args, **kwargs):
        raw = output.copy()
        arrays.append(weakref.ref(raw))
        return raw

    monkeypatch.setattr(hovernet_inference, "run_model_fields", predict)
    result = hovernet_inference.run_hovernet_inference(
        None,
        np.zeros((128, 128, 3), dtype=np.uint8),
        num_classes,
        include_diagnostics=False,
    )
    assert all(array() is None for array in arrays)
    assert result.keys() == (
        {"instance_mask", "votes_per_instance"} if num_classes else {"instance_mask"}
    )
    np.testing.assert_array_equal(result["instance_mask"], expected["instance_mask"])
    if num_classes:
        assert result["votes_per_instance"] == expected["votes_per_instance"]
        assert set(result["votes_per_instance"].values()) == {0, 1, 2, 3}
