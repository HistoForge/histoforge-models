"""Standalone scientific primitives have no backend or research-loop dependency."""

import ast
from pathlib import Path
import subprocess
import shutil
import sys

from histoforge_models.pathology import managed_training
from histoforge_models.pathology.managed_training import (
    ManagedDenseBatch,
    ManagedObjective,
    build_managed_targets,
    managed_dense_loss,
    managed_segmentation_loss,
)


def test_public_api_exposes_only_scientific_primitives():
    """Expose losses, objectives and targets, not the backend optimizer engine."""

    assert set(managed_training.__all__) == {
        "ClassificationObjective",
        "FlowLossWeighting",
        "HeadType",
        "ManagedDenseBatch",
        "ManagedLossResult",
        "ManagedObjective",
        "ManagedSegmentationBatch",
        "ManagedSegmentationTargets",
        "augment_blur",
        "augment_brightness_contrast",
        "augment_hed",
        "augment_noise",
        "build_managed_targets",
        "managed_dense_loss",
        "managed_segmentation_loss",
        "validate_managed_loss_configuration",
    }
    assert managed_training.ManagedObjective is ManagedObjective
    assert managed_training.ManagedDenseBatch is ManagedDenseBatch
    assert managed_training.managed_dense_loss is managed_dense_loss
    assert managed_training.build_managed_targets is build_managed_targets
    assert managed_training.managed_segmentation_loss is managed_segmentation_loss
    for retired in (
        "ManagedEpochResult",
        "ManagedTrainingCancelled",
        "optimize_managed",
        "train_managed_segmentation",
    ):
        assert not hasattr(managed_training, retired)


def test_public_package_imports_in_clean_interpreter(tmp_path):
    """Verify the public import without relying on pytest's loaded modules."""

    repository_root = Path(__file__).resolve().parents[1]
    script = f"""
import sys
sys.path.insert(0, {str(repository_root)!r})
from histoforge_models.pathology import managed_training
assert managed_training.__name__ == 'histoforge_models.pathology.managed_training'
"""
    subprocess.run(
        [sys.executable, "-B", "-I", "-c", script],
        cwd=tmp_path,
        check=True,
        timeout=30,
    )


def test_standalone_package_imports_targets_before_losses(tmp_path):
    """Preserve the standalone import mitigation without importing Cellpose."""

    package_parent = Path(managed_training.__file__).parent.parent
    script = f"""
import sys
sys.path.insert(0, {str(package_parent)!r})
import managed_training
assert 'cellpose' not in sys.modules
assert 'histoforge_models' not in sys.modules
assert managed_training.__name__ == 'managed_training'
loaded = list(sys.modules)
assert loaded.index('managed_training.targets') < loaded.index('managed_training.losses')
"""
    subprocess.run(
        [sys.executable, "-B", "-I", "-c", script],
        cwd=tmp_path,
        check=True,
        timeout=30,
    )


def test_copied_package_runs_appearance_targets_and_losses_without_parent(tmp_path):
    """Exercise the actual export boundary, not a sibling-checkout import."""
    package_root = Path(managed_training.__file__).parent
    destination = tmp_path / "managed_training"
    destination.mkdir()
    for source in package_root.glob("*.py"):
        shutil.copyfile(source, destination / source.name)
    script = f"""
import sys
sys.path.insert(0, {str(tmp_path)!r})
import managed_training as mt
import numpy as np
import torch
assert 'cellpose' not in sys.modules
assert 'histoforge_models' not in sys.modules
labels = np.zeros((16, 16), dtype=np.int32)
labels[3:12, 5:11] = 71
pixels = np.arange(16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3) % 256
pixels = mt.augment_noise(pixels.astype(np.uint8), real_pixel_mask=np.ones((16, 16), bool),
    rng=np.random.default_rng(6), probability=1, sigma=0.01)
for head in ('flow', 'hovernet'):
    targets = mt.build_managed_targets(labels, head_type=head)
    weights = targets.instance_flow_weights
    batch = mt.ManagedSegmentationBatch(
        images=torch.from_numpy(pixels.transpose(2, 0, 1).copy()[None] / 255),
        directional_targets=torch.from_numpy(targets.directional[None]),
        foreground_targets=torch.from_numpy(targets.foreground[None]),
        supervision_mask=torch.ones(1, 16, 16, dtype=torch.bool),
        real_pixel_mask=torch.ones(1, 16, 16, dtype=torch.bool),
        instance_flow_weights=None if weights is None else torch.from_numpy(weights[None]),
    )
    model = torch.nn.Conv2d(3, 3 if head == 'flow' else 4, kernel_size=1)
    weighting = 'equal_instance' if head == 'flow' else 'pixel'
    batch.validate(0, flow_loss_weighting=weighting)
    result = mt.managed_segmentation_loss(model(batch.images), batch,
        head_type=head, class_count=0, flow_loss_weighting=weighting)
    assert result.field_count == 1 and torch.isfinite(result.total_sum)
    (result.total_sum / result.field_count).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
assert 'cellpose' not in sys.modules
assert 'histoforge_models' not in sys.modules
assert not any(name == 'histoforge' or name.startswith('histoforge.')
               for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-B", "-I", "-c", script], cwd=tmp_path, check=True, timeout=60
    )


def test_managed_package_has_only_declared_external_dependencies():
    """Prevent hidden dependencies outside the directory that will be vendored."""

    package_root = Path(managed_training.__file__).parent
    allowed_external = {"fastremap", "numpy", "scipy", "torch"}
    violations = []
    for source_path in sorted(package_root.glob("*.py")):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").partition(".")[0]
                if node.level > 1 or (
                    node.level == 0
                    and root not in sys.stdlib_module_names
                    and root not in allowed_external
                ):
                    violations.append((source_path.name, node.lineno))
            elif isinstance(node, ast.Import):
                roots = {alias.name.partition(".")[0] for alias in node.names}
                if any(
                    root not in sys.stdlib_module_names and root not in allowed_external
                    for root in roots
                ):
                    violations.append((source_path.name, node.lineno))

    assert violations == []
