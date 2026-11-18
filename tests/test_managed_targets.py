import numpy as np
import pytest

from histoforge_models.pathology.managed_training import build_managed_targets


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
def test_builds_both_directional_target_families(head_type):
    instances = np.zeros((9, 11), dtype=np.int32)
    instances[2:7, 3:9] = 1
    classes = np.zeros_like(instances)
    classes[instances == 1] = 2

    targets = build_managed_targets(
        instances, head_type=head_type, class_labels=classes
    )

    assert targets.directional.shape == (2, 9, 11)
    assert targets.directional.dtype == np.float32
    assert np.array_equal(targets.foreground, instances > 0)
    assert np.array_equal(targets.classes, classes)
    if head_type == "flow":
        assert targets.instance_flow_weights is not None
        assert targets.instance_flow_weights[instances == 1].sum() == (
            pytest.approx(1.0)
        )
        assert not np.any(targets.instance_flow_weights[instances == 0])
    else:
        assert targets.instance_flow_weights is None


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
def test_object_local_targets_match_a_larger_reference_before_field_crop(head_type):
    reference = np.zeros((31, 33), dtype=np.int32)
    reference[8:23, 9:26] = 7
    local = reference[5:26, 6:29]

    reference_targets = build_managed_targets(reference, head_type=head_type)
    local_targets = build_managed_targets(local, head_type=head_type)
    expected_field = reference_targets.directional[:, 10:21, 9:18]
    local_field = local_targets.directional[:, 5:16, 3:12]
    incorrectly_cropped_first = build_managed_targets(
        reference[10:21, 9:18], head_type=head_type
    )

    assert np.array_equal(local_field, expected_field)
    assert not np.array_equal(local_field, incorrectly_cropped_first.directional)


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
def test_empty_object_local_grid_produces_a_valid_negative_target(head_type):
    instances = np.zeros((9, 11), dtype=np.int32)

    targets = build_managed_targets(instances, head_type=head_type)

    assert targets.directional.shape == (2, 9, 11)
    assert not np.any(targets.directional)
    assert not np.any(targets.foreground)
    assert targets.classes is None
    if head_type == "flow":
        assert targets.instance_flow_weights is not None
        assert not np.any(targets.instance_flow_weights)
    else:
        assert targets.instance_flow_weights is None


def test_adjacent_singleton_keeps_supervision_but_has_no_direction():
    instances = np.zeros((7, 7), dtype=np.int32)
    instances[2:5, 2:5] = 4
    instances[3, 5] = 9
    classes = np.zeros_like(instances)
    classes[instances == 4] = 1
    classes[instances == 9] = 2

    targets = build_managed_targets(
        instances,
        head_type="flow",
        class_labels=classes,
    )

    singleton = instances == 9
    assert np.array_equal(targets.directional[:, singleton], np.zeros((2, 1)))
    assert targets.foreground[singleton].item() == 1
    assert targets.classes[singleton].item() == 2
    assert targets.instance_flow_weights[singleton].item() == pytest.approx(1.0)


def test_rejects_classes_on_background_pixels():
    instances = np.zeros((4, 4), dtype=np.int32)
    classes = np.zeros_like(instances)
    classes[0, 0] = 1

    with pytest.raises(ValueError, match="background pixels"):
        build_managed_targets(instances, head_type="flow", class_labels=classes)


def test_rejects_background_class_on_foreground_pixels():
    instances = np.zeros((4, 4), dtype=np.int32)
    instances[1:3, 1:3] = 1
    classes = np.zeros_like(instances)

    with pytest.raises(ValueError, match="foreground pixels"):
        build_managed_targets(instances, head_type="flow", class_labels=classes)


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
def test_noncontiguous_instance_ids_match_dense_ids(head_type):
    dense = np.zeros((9, 11), dtype=np.int32)
    dense[2:7, 3:9] = 1
    sparse = dense.copy()
    sparse[sparse == 1] = 7

    dense_targets = build_managed_targets(dense, head_type=head_type)
    sparse_targets = build_managed_targets(sparse, head_type=head_type)

    assert np.array_equal(sparse_targets.directional, dense_targets.directional)
    assert np.array_equal(sparse_targets.foreground, dense_targets.foreground)
    if head_type == "flow":
        assert np.array_equal(
            sparse_targets.instance_flow_weights,
            dense_targets.instance_flow_weights,
        )
