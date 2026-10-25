"""Pixel labels and raw logits share exact, non-mutating instance voting."""

import numpy as np
import pytest

from histoforge_models.pathology.common import per_instance_majority_vote


@pytest.mark.parametrize("sparse_ids", [False, True])
@pytest.mark.parametrize("ignore", [(), (0,), (0, 3)])
@pytest.mark.parametrize("strided", [False, True])
def test_votes_preserve_readonly_inputs_class_competition_and_ties(
    sparse_ids, ignore, strided
):
    classes = np.array([[0, 0, 3, 3], [1, 1, 2, 2]], dtype=np.int32)
    mask = np.array([[0, 7, 7, 7], [9, 9, 9, 9]], dtype=np.int32)
    if sparse_ids:
        mask[mask == 9] = 2_000_000_000
    if strided:
        classes, mask = classes[:, ::-1], mask[:, ::-1]
    original_classes, original_mask = classes.copy(), mask.copy()
    classes.flags.writeable = False
    mask.flags.writeable = False
    logits = np.eye(4, dtype=np.float32)[classes].transpose(2, 0, 1)
    logits.flags.writeable = False
    expected = {}
    for instance_id in np.unique(mask):
        if instance_id == 0:
            continue
        votes = [int(c) for c in classes[mask == instance_id] if c not in ignore]
        if votes:
            expected[int(instance_id)] = min(
                set(votes), key=lambda c: (-votes.count(c), c)
            )
    assert per_instance_majority_vote(classes, mask, ignore) == expected
    assert per_instance_majority_vote(logits, mask, ignore) == expected
    np.testing.assert_array_equal(classes, original_classes)
    np.testing.assert_array_equal(mask, original_mask)
