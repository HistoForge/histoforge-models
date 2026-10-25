"""Independent numerical checks for the managed dense objective."""

from dataclasses import replace

from histoforge_models.pathology.managed_training import (
    ManagedDenseBatch,
    ManagedObjective,
    managed_dense_loss,
)
import pytest
import torch


def _batch():
    return ManagedDenseBatch(
        images=torch.zeros(2, 3, 2, 2),
        class_targets=torch.tensor([[[0, 1], [2, 255]], [[1, 255], [255, 255]]]),
        supervision_mask=torch.tensor(
            [[[True, True], [True, False]], [[True, False], [False, False]]]
        ),
        real_pixel_mask=torch.ones(2, 2, 2, dtype=torch.bool),
    )


def _oracle(logits, batch, weights):
    ce, tversky = logits.new_zeros(()), logits.new_zeros(())
    for index in range(len(logits)):
        labels = batch.class_targets[index][batch.effective_support[index]]
        scores = logits[index].permute(1, 2, 0)[batch.effective_support[index]]
        log_probabilities = scores - scores.logsumexp(dim=1, keepdim=True)
        ce = (
            ce
            - (
                log_probabilities[torch.arange(len(labels)), labels] * weights[labels]
            ).sum()
            / weights[labels].sum()
        )
        probabilities = log_probabilities.exp()
        for channel in range(len(weights)):
            truth = labels == channel
            tp = probabilities[truth, channel].sum()
            fp = probabilities[~truth, channel].sum()
            fn = (1 - probabilities[truth, channel]).sum()
            ratio = tp / (tp + 0.3 * fp + 0.7 * fn).clamp_min(
                torch.finfo(torch.float32).eps
            )
            tversky = tversky + weights[channel] * (1 - ratio).clamp(
                1e-6, 1 - 1e-6
            ).pow(1 / 1.33) / len(weights)
    return torch.stack([ce, tversky])


def test_weighted_losses_and_gradients_match_independent_oracle():
    batch = _batch()
    logits = torch.linspace(-2, 2, 24).reshape(2, 3, 2, 2).requires_grad_()
    oracle_logits = logits.detach().clone().requires_grad_()
    weights = torch.tensor([0.75, 1.5, 2.0])
    objective = ManagedObjective(
        class_count=3,
        classification_objective="cross_entropy+tversky",
        use_uncertainty_weighting=True,
        include_segmentation=False,
    )
    assert objective.task_names == (
        "classification_cross_entropy",
        "classification_tversky",
    )
    assert objective.log_variances.shape == (2,)
    with torch.no_grad():
        objective.log_variances.copy_(torch.tensor([0.2, -0.3]))
    result = managed_dense_loss(
        logits,
        batch,
        class_count=3,
        class_weights=weights,
        objective=objective,
    )
    expected_tasks = _oracle(oracle_logits, batch, weights)
    oracle_variances = objective.log_variances.detach().clone().requires_grad_()
    expected = (expected_tasks * (-oracle_variances).exp() + 2 * oracle_variances).sum()
    assert result.field_count == 2
    torch.testing.assert_close(
        torch.stack(list(result.task_sums.values())), expected_tasks
    )
    torch.testing.assert_close(result.total_sum, expected)
    result.total_sum.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, oracle_logits.grad)
    torch.testing.assert_close(objective.log_variances.grad, oracle_variances.grad)
    assert (
        torch.count_nonzero(logits.grad.permute(0, 2, 3, 1)[~batch.effective_support])
        == 0
    )


def test_unequal_fields_are_summed_without_pixel_population_weighting():
    batch = _batch()
    logits = torch.randn(2, 3, 2, 2, generator=torch.Generator().manual_seed(11))
    result = managed_dense_loss(logits, batch, class_count=3)
    individual = []
    for index in range(2):
        one = ManagedDenseBatch(
            **{
                name: getattr(batch, name)[index : index + 1]
                for name in batch.__dataclass_fields__
            }
        )
        individual.append(
            managed_dense_loss(logits[index : index + 1], one, class_count=3).total_sum
        )
    torch.testing.assert_close(result.total_sum, torch.stack(individual).sum())


def test_real_pixel_mask_excludes_context_and_background_only_is_valid():
    batch = _batch()
    batch = replace(batch, class_targets=torch.zeros_like(batch.class_targets))
    real = batch.real_pixel_mask.clone()
    real[0, 0, 1] = False
    batch = replace(batch, real_pixel_mask=real)
    logits = torch.zeros(2, 3, 2, 2, requires_grad=True)
    result = managed_dense_loss(logits, batch, class_count=3)
    assert torch.isfinite(result.total_sum)
    result.total_sum.backward()
    assert torch.count_nonzero(logits.grad[0, :, 0, 1]) == 0


@pytest.mark.parametrize(
    "change,match",
    [
        ({"class_targets": torch.zeros(2, 2, 2)}, "integer dtype"),
        ({"class_targets": torch.full((2, 2, 2), 255)}, "inside effective support"),
        ({"class_targets": torch.full((2, 2, 2), 4)}, "channel indices"),
        ({"supervision_mask": torch.zeros(2, 2, 2, dtype=torch.bool)}, "every field"),
        ({"real_pixel_mask": torch.ones(2, 2, 2)}, "boolean"),
    ],
)
def test_invalid_batches_are_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        replace(_batch(), **change).validate(3)


def test_dense_rejects_instance_objective_and_non_class_outputs():
    logits = torch.zeros(2, 3, 2, 2)
    with pytest.raises(ValueError, match="segmentation task"):
        managed_dense_loss(
            logits,
            _batch(),
            class_count=3,
            objective=ManagedObjective(
                class_count=3, classification_objective="cross_entropy+tversky"
            ),
        )
    with pytest.raises(ValueError, match="shape"):
        managed_dense_loss(torch.zeros(2, 6, 2, 2), _batch(), class_count=3)
    with pytest.raises(ValueError, match="float32"):
        managed_dense_loss(logits.half(), _batch(), class_count=3)
