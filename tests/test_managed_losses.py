import pytest
import torch

from histoforge_models.pathology.managed_training import (
    ManagedSegmentationBatch,
    ManagedObjective,
    managed_segmentation_loss,
)


def _classification_batch(width=2):
    foreground = torch.zeros((1, 1, width))
    classes = torch.zeros((1, 1, width), dtype=torch.long)
    if width > 1:
        foreground[:, :, -1] = 1
        classes[:, :, -1] = 1
    return ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 1, width)),
        directional_targets=torch.zeros((1, 2, 1, width)),
        foreground_targets=foreground,
        supervision_mask=torch.ones((1, 1, width), dtype=torch.bool),
        real_pixel_mask=torch.ones((1, 1, width), dtype=torch.bool),
        class_targets=classes,
    )


def test_flow_loss_ignores_pixels_outside_effective_support():
    batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 1, 2)),
        directional_targets=torch.zeros((1, 2, 1, 2)),
        foreground_targets=torch.zeros((1, 1, 2)),
        supervision_mask=torch.tensor([[[True, False]]]),
        real_pixel_mask=torch.ones((1, 1, 2), dtype=torch.bool),
    )
    baseline = torch.zeros((1, 3, 1, 2))
    changed = baseline.clone()
    changed[:, :, :, 1] = 10_000

    first = managed_segmentation_loss(baseline, batch, head_type="flow")
    second = managed_segmentation_loss(changed, batch, head_type="flow")

    assert first.total_sum.item() == pytest.approx(second.total_sum.item())


def test_default_flow_loss_preserves_established_reduction_exactly():
    torch.manual_seed(19)
    support = torch.tensor([[[True, True, False], [True, False, True]]])
    directions = torch.randn((1, 2, 2, 3))
    foreground = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
    prediction = torch.randn((1, 3, 2, 3))
    batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 2, 3)),
        directional_targets=directions,
        foreground_targets=foreground,
        supervision_mask=support,
        real_pixel_mask=torch.ones_like(support),
    )
    support_float = support.float()
    expected_flow = (
        (prediction[:, :2] - 5.0 * directions).square()
        * support_float[:, None]
    ).sum() / (4.0 * support_float.sum())
    expected_bce = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            prediction[:, 2], foreground, reduction="none"
        )
        * support_float
    ).sum() / support_float.sum()

    result = managed_segmentation_loss(prediction, batch, head_type="flow")

    torch.testing.assert_close(result.component_sums["flow_mse"], expected_flow)
    torch.testing.assert_close(
        result.component_sums["foreground_bce"], expected_bce
    )
    torch.testing.assert_close(result.total_sum, expected_flow + expected_bce)


def test_equal_instance_flow_loss_requires_complete_instance_weights():
    batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 2, 2)),
        directional_targets=torch.zeros((1, 2, 2, 2)),
        foreground_targets=torch.ones((1, 2, 2)),
        supervision_mask=torch.ones((1, 2, 2), dtype=torch.bool),
        real_pixel_mask=torch.ones((1, 2, 2), dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="derived from complete instances"):
        managed_segmentation_loss(
            torch.zeros((1, 3, 2, 2)),
            batch,
            head_type="flow",
            flow_loss_weighting="equal_instance",
        )


def test_equal_instance_flow_loss_rejects_hover_head():
    batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 5, 5)),
        directional_targets=torch.zeros((1, 2, 5, 5)),
        foreground_targets=torch.zeros((1, 5, 5)),
        supervision_mask=torch.ones((1, 5, 5), dtype=torch.bool),
        real_pixel_mask=torch.ones((1, 5, 5), dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="only to the Cellpose flow head"):
        managed_segmentation_loss(
            torch.zeros((1, 4, 5, 5)),
            batch,
            head_type="hovernet",
            flow_loss_weighting="equal_instance",
        )


def test_equal_instance_setting_leaves_foreground_and_classification_unchanged():
    foreground = torch.tensor([[[1.0, 1.0, 0.0, 1.0]]])
    classes = torch.tensor([[[1, 1, 0, 1]]])
    weights = torch.tensor([[[0.5, 0.5, 0.0, 1.0]]])
    batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 1, 4)),
        directional_targets=torch.zeros((1, 2, 1, 4)),
        foreground_targets=foreground,
        supervision_mask=torch.ones((1, 1, 4), dtype=torch.bool),
        real_pixel_mask=torch.ones((1, 1, 4), dtype=torch.bool),
        class_targets=classes,
        instance_flow_weights=weights,
    )
    prediction = torch.tensor(
        [[
            [[0.3, -0.4, 0.7, -0.2]],
            [[-0.1, 0.8, -0.5, 0.4]],
            [[2.0, 0.0, 0.0, -1.0]],
            [[0.0, 0.0, 0.0, 3.0]],
            [[-0.8, 0.2, 1.1, -0.3]],
        ]],
        dtype=torch.float32,
    )

    pixel = managed_segmentation_loss(
        prediction, batch, head_type="flow", class_count=2
    )
    equal = managed_segmentation_loss(
        prediction,
        batch,
        head_type="flow",
        class_count=2,
        flow_loss_weighting="equal_instance",
    )

    torch.testing.assert_close(
        equal.component_sums["foreground_bce"],
        pixel.component_sums["foreground_bce"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        equal.component_sums["classification_cross_entropy"],
        pixel.component_sums["classification_cross_entropy"],
        rtol=0,
        atol=0,
    )
    assert equal.component_sums["flow_mse"] != pixel.component_sums["flow_mse"]


def test_hover_and_classification_share_the_same_effective_support():
    batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 5, 5)),
        directional_targets=torch.zeros((1, 2, 5, 5)),
        foreground_targets=torch.zeros((1, 5, 5)),
        supervision_mask=torch.ones((1, 5, 5), dtype=torch.bool),
        real_pixel_mask=torch.ones((1, 5, 5), dtype=torch.bool),
        class_targets=torch.zeros((1, 5, 5), dtype=torch.long),
    )
    batch.supervision_mask[:, 0, :] = False
    prediction = torch.zeros((1, 6, 5, 5))
    changed = prediction.clone()
    changed[:, :, 0, :] = 10_000

    first = managed_segmentation_loss(
        prediction,
        batch,
        head_type="hovernet",
        class_count=2,
        classification_objective="cross_entropy+tversky",
    )
    second = managed_segmentation_loss(
        changed,
        batch,
        head_type="hovernet",
        class_count=2,
        classification_objective="cross_entropy+tversky",
    )

    assert first.total_sum.item() == pytest.approx(second.total_sum.item())
    assert set(first.component_sums) == {
        "segmentation",
        "hover_mse",
        "hover_msge",
        "foreground_cross_entropy",
        "foreground_dice",
        "classification_cross_entropy",
        "classification_tversky",
    }


def test_masked_hover_and_classification_gradients_ignore_outside_support():
    support = torch.ones((1, 9, 9), dtype=torch.bool)
    support[:, :2, :] = False
    batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 9, 9)),
        directional_targets=torch.zeros((1, 2, 9, 9)),
        foreground_targets=torch.zeros((1, 9, 9)),
        supervision_mask=support,
        real_pixel_mask=torch.ones_like(support),
        class_targets=torch.zeros((1, 9, 9), dtype=torch.long),
    )
    baseline = torch.zeros((1, 6, 9, 9), requires_grad=True)
    changed = baseline.detach().clone()
    changed[:, :, :2, :] = 10_000
    changed.requires_grad_()

    first = managed_segmentation_loss(
        baseline,
        batch,
        head_type="hovernet",
        class_count=2,
        classification_objective="cross_entropy+tversky",
    )
    second = managed_segmentation_loss(
        changed,
        batch,
        head_type="hovernet",
        class_count=2,
        classification_objective="cross_entropy+tversky",
    )
    first.total_sum.backward()
    second.total_sum.backward()

    assert first.total_sum.item() == pytest.approx(second.total_sum.item())
    assert torch.equal(baseline.grad, changed.grad)
    assert torch.count_nonzero(baseline.grad[:, :, :2, :]) == 0


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (torch.tensor([[[255]]]), "invalid inside effective support"),
        (torch.tensor([[[0]]]), "agree on background"),
    ],
)
def test_rejects_invalid_class_supervision_on_foreground(invalid, message):
    batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 1, 1)),
        directional_targets=torch.zeros((1, 2, 1, 1)),
        foreground_targets=torch.ones((1, 1, 1)),
        supervision_mask=torch.ones((1, 1, 1), dtype=torch.bool),
        real_pixel_mask=torch.ones((1, 1, 1), dtype=torch.bool),
        class_targets=invalid,
    )

    with pytest.raises(ValueError, match=message):
        managed_segmentation_loss(
            torch.zeros((1, 5, 1, 1)),
            batch,
            head_type="flow",
            class_count=2,
        )


def test_class_255_is_allowed_only_outside_effective_support():
    batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 1, 2)),
        directional_targets=torch.zeros((1, 2, 1, 2)),
        foreground_targets=torch.zeros((1, 1, 2)),
        supervision_mask=torch.tensor([[[True, False]]]),
        real_pixel_mask=torch.ones((1, 1, 2), dtype=torch.bool),
        class_targets=torch.tensor([[[0, 255]]]),
    )

    result = managed_segmentation_loss(
        torch.zeros((1, 5, 1, 2)),
        batch,
        head_type="flow",
        class_count=2,
    )

    assert torch.isfinite(result.total_sum)


@pytest.mark.parametrize(("head_type", "tail_channels"), [("flow", 3), ("hovernet", 4)])
def test_class_prefix_and_segmentation_tail_are_not_conflated(head_type, tail_channels):
    size = 7 if head_type == "hovernet" else 2
    batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, size, size)),
        directional_targets=torch.zeros((1, 2, size, size)),
        foreground_targets=torch.zeros((1, size, size)),
        supervision_mask=torch.ones((1, size, size), dtype=torch.bool),
        real_pixel_mask=torch.ones((1, size, size), dtype=torch.bool),
        class_targets=torch.zeros((1, size, size), dtype=torch.long),
    )
    baseline = torch.zeros((1, 2 + tail_channels, size, size))
    changed_class = baseline.clone()
    changed_class[:, 0] = 2.0
    changed_segmentation = baseline.clone()
    changed_segmentation[:, 2:] = 1.0

    base_loss = managed_segmentation_loss(
        baseline, batch, head_type=head_type, class_count=2
    )
    class_loss = managed_segmentation_loss(
        changed_class, batch, head_type=head_type, class_count=2
    )
    segmentation_loss = managed_segmentation_loss(
        changed_segmentation, batch, head_type=head_type, class_count=2
    )

    assert class_loss.component_sums["segmentation"] == pytest.approx(
        base_loss.component_sums["segmentation"]
    )
    assert class_loss.component_sums["classification_cross_entropy"] != pytest.approx(
        base_loss.component_sums["classification_cross_entropy"]
    )
    assert segmentation_loss.component_sums[
        "classification_cross_entropy"
    ] == pytest.approx(base_loss.component_sums["classification_cross_entropy"])
    assert segmentation_loss.component_sums["segmentation"] != pytest.approx(
        base_loss.component_sums["segmentation"]
    )


def test_weighted_cross_entropy_matches_established_weighted_mean():
    predictions = torch.zeros((1, 5, 1, 2))
    batch = _classification_batch()
    weights = torch.tensor([1.0, 10.0])

    result = managed_segmentation_loss(
        predictions,
        batch,
        head_type="flow",
        class_count=2,
        class_weights=weights,
        weight_tversky_by_class=False,
    )

    expected = torch.nn.functional.cross_entropy(
        predictions[:, :2], batch.class_targets, weight=weights
    )
    assert result.component_sums[
        "classification_cross_entropy"
    ].item() == pytest.approx(expected.item())


def test_class_weights_can_be_limited_to_cross_entropy():
    predictions = torch.zeros((1, 5, 1, 2))
    batch = _classification_batch()
    weights = torch.tensor([1.0, 10.0])

    ce_only = managed_segmentation_loss(
        predictions,
        batch,
        head_type="flow",
        class_count=2,
        classification_objective="cross_entropy+tversky",
        class_weights=weights,
        weight_tversky_by_class=False,
    )
    both = managed_segmentation_loss(
        predictions,
        batch,
        head_type="flow",
        class_count=2,
        classification_objective="cross_entropy+tversky",
        class_weights=weights,
        weight_tversky_by_class=True,
    )

    assert ce_only.component_sums["classification_cross_entropy"].item() == (
        pytest.approx(both.component_sums["classification_cross_entropy"].item())
    )
    assert ce_only.component_sums["classification_tversky"].item() != (
        pytest.approx(both.component_sums["classification_tversky"].item())
    )


def test_uncertainty_weighting_matches_established_task_formula():
    predictions = torch.zeros((1, 5, 1, 2))
    batch = _classification_batch()
    objective = ManagedObjective(
        class_count=2,
        classification_objective="cross_entropy+tversky",
        classification_loss_weight=2.0,
        use_uncertainty_weighting=True,
    )
    with torch.no_grad():
        objective.log_variances.copy_(torch.tensor([0.2, -0.3, 0.4]))

    result = managed_segmentation_loss(
        predictions,
        batch,
        head_type="flow",
        class_count=2,
        classification_objective="cross_entropy+tversky",
        classification_loss_weight=2.0,
        objective=objective,
    )

    task_values = torch.stack([result.task_sums[name] for name in objective.task_names])
    scaled = task_values * objective.task_scales
    expected = (
        torch.exp(-objective.log_variances) * scaled + objective.log_variances
    ).sum()
    assert result.total_sum.item() == pytest.approx(expected.item())


def test_managed_loss_rejects_half_precision():
    batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 1, 1)),
        directional_targets=torch.zeros((1, 2, 1, 1)),
        foreground_targets=torch.zeros((1, 1, 1)),
        supervision_mask=torch.ones((1, 1, 1), dtype=torch.bool),
        real_pixel_mask=torch.ones((1, 1, 1), dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="requires float32"):
        managed_segmentation_loss(
            torch.zeros((1, 3, 1, 1), dtype=torch.float16),
            batch,
            head_type="flow",
        )
