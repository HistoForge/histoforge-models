"""Keep retired architecture ablations out of the maintained shared surface."""

import inspect
from pathlib import Path

from histoforge_models.pathology.training.initialization import create_cellpose_model


def test_retired_implementations_and_launchers_stay_absent():
    """Historical run references are allowed; executable retired paths are not."""

    root = Path(__file__).parents[1]
    retired = (
        "tests/test_comer.py",
        "tests/test_unet_decoder.py",
        "histoforge_models/pathology/convnext_backbone.py",
        "histoforge_models/pathology/hv_losses.py",
        "histoforge_models/pathology/hv_transforms.py",
        "job_scripts/training/dinov3/train_dinov3_armB_comer.sh",
        "job_scripts/training/dinov3/train_dinov3_armB_comer_d1024.sh",
        "job_scripts/training/dinov3/train_dinov3_armB_unet.sh",
        "job_scripts/benchmarks/phase2_puma.sh",
    )
    assert not [name for name in retired if (root / name).exists()]
    assert not list((root / "histoforge_models/custom").rglob("*.py"))
    assert not list((root / "histoforge_models/histoforge").rglob("*.py"))
    parameters = inspect.signature(create_cellpose_model).parameters
    assert not set(parameters) & {
        "use_unet_decoder", "use_comer", "ps", "backbone", "sam_checkpoint",
        "hybrid_attention", "bsize",
    }
