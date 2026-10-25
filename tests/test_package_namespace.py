"""The standalone library must not depend on an installed Cellpose package."""

from pathlib import Path
import subprocess
import sys


def test_public_namespace_is_self_contained(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = f"""
import sys
sys.path.insert(0, {str(root)!r})
import histoforge_models
from histoforge_models import core, dynamics, metrics, models
from histoforge_models.pathology import common, class_head
from histoforge_models.pathology.training import initialization, segmentation_metrics
assert histoforge_models.__name__ == 'histoforge_models'
assert not any(name == 'cellpose' or name.startswith('cellpose.') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-B", "-I", "-c", script],
        cwd=tmp_path,
        check=True,
        timeout=60,
    )
