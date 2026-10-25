"""Source-release metadata, resources and documentation stay self-contained."""

from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _metadata():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_version_and_dependencies_do_not_depend_on_the_build_host():
    metadata = _metadata()
    project = metadata["project"]
    assert project["name"] == "histoforge-models"
    assert project["version"] == "0.1.0"
    assert not project.get("dynamic")
    assert "torch>=2.6" in project["dependencies"]
    assert "torchvision" in project["dependencies"]
    assert {"pyqt6", "pyqt6.sip"} <= set(
        project["optional-dependencies"]["gui"]
    )
    assert metadata["build-system"]["build-backend"] == "setuptools.build_meta"
    assert not (ROOT / "setup.py").exists()


def test_training_install_includes_its_logging_dependency():
    extras = _metadata()["project"]["optional-dependencies"]
    assert "wandb" in extras["training"]
    assert "wandb" in extras["all"]
    for relative in ("CONTRIBUTING.md", "prod_docs/training.md"):
        assert ".[dinov3,training,dev]" in (ROOT / relative).read_text()


def test_source_manifest_includes_root_test_fixtures():
    manifest = (ROOT / "MANIFEST.in").read_text().splitlines()
    assert "include conftest.py" in manifest
    assert "graft tests" in manifest


def test_package_keeps_gui_assets_and_legal_notices():
    for relative in (
        "histoforge_models/logo/logo.png",
        "histoforge_models/logo/cellpose.ico",
        "LICENSE",
        "NOTICE",
    ):
        assert (ROOT / relative).is_file(), relative
    assert list((ROOT / "histoforge_models/gui").glob("*.html"))
    project = _metadata()["project"]
    assert project["license"] == "Apache-2.0 AND BSD-3-Clause"
    assert project["license-files"] == ["LICENSE", "NOTICE"]


def test_release_tree_has_no_internal_working_directories():
    for relative in (
        "AGENTS.md", ".agents", "docs/research_logs", "job_scripts",
        "notebooks", "paper", "prod_docs/research-snapshot.md",
        "scripts/modal", ".readthedocs.yml",
    ):
        assert not (ROOT / relative).exists(), relative


def test_scorer_only_contains_the_public_dataset_preset():
    from scripts.score_cls import DATASETS

    assert set(DATASETS) == {"puma"}
    assert DATASETS["puma"]["radius_px"] == 15.0


def test_documentation_links_resolve_within_the_source_tree():
    paths = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    paths.extend(sorted((ROOT / "prod_docs").glob("*.md")))
    for source in paths:
        for target in re.findall(r"\]\(([^)]+)\)", source.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            target_path = (source.parent / target.split("#", 1)[0]).resolve()
            assert target_path.is_relative_to(ROOT), (source.name, target)
            assert target_path.is_file(), (source.name, target)


def test_documented_tools_are_shipped_with_the_source():
    for relative in (
        "scripts/score_cls.py", "scripts/he_seg_benchmark.py",
        "scripts/eval_instance_cls.py",
        "datasets/dataset_converters/precompute_flows.py",
    ):
        assert (ROOT / relative).is_file(), relative
