"""Test-runner isolation: never hide runtime conflicts or mutate user caches."""

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

import conftest as fixture_code
from histoforge_models import models


def test_archive_extraction_is_private_and_repeatable(tmp_path):
    archive = tmp_path / "original.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("data/2D/image.txt", "original fixture")
    original = archive.read_bytes()
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    fixture_code.extract_zip(archive, one)
    (one / "data/2D/image.txt").write_text("test result")
    fixture_code.extract_zip(archive, two)
    assert (two / "data/2D/image.txt").read_text() == "original fixture"
    assert archive.read_bytes() == original


@pytest.mark.parametrize(
    "name", ["../escape.txt", "/absolute.txt", "data/../../escape.txt"]
)
def test_archive_cannot_escape_private_storage(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("data/first.txt", "must not extract before validating")
        stream.writestr(name, "unsafe")
    destination = tmp_path / "extracted"
    destination.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        fixture_code.extract_zip(archive, destination)
    assert list(destination.iterdir()) == []


def test_cache_copy_never_shares_writable_weights_or_registry(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    for name in ("cpsam", "cyto3", "user-checkpoint", "gui_models.txt"):
        (source / name).write_bytes(name.encode())
    fixture_code.copy_cached_models(source, destination)
    assert sorted(path.name for path in destination.iterdir()) == ["cpsam", "cyto3"]
    (destination / "cpsam").write_bytes(b"changed by test")
    assert (source / "cpsam").read_bytes() == b"cpsam"
    assert (source / "gui_models.txt").read_bytes() == b"gui_models.txt"


def test_legacy_registry_and_child_environment_use_same_private_path(
    tmp_path, monkeypatch
):
    request = SimpleNamespace(
        node=SimpleNamespace(get_closest_marker=lambda _: object()),
        getfixturevalue=lambda name: tmp_path,
    )
    fixture_code.isolated_legacy_models.__wrapped__(request, monkeypatch)
    assert models.MODEL_DIR == tmp_path
    assert models.MODEL_LIST_PATH == str(tmp_path / "gui_models.txt")
    import os

    assert os.environ["CELLPOSE_LOCAL_MODELS_PATH"] == str(tmp_path)


@pytest.mark.parametrize(
    "run_slow,run_legacy", [(False, False), (True, False), (False, True), (True, True)]
)
def test_slow_option_cannot_bypass_legacy_opt_in(run_slow, run_legacy):
    markers = []
    item = SimpleNamespace(
        keywords={"legacy": True, "slow": True}, add_marker=markers.append
    )
    options = {"--runslow": run_slow, "--runlegacy": run_legacy}
    fixture_code.pytest_collection_modifyitems(
        SimpleNamespace(getoption=options.__getitem__), [item]
    )
    reasons = [marker.kwargs["reason"] for marker in markers]
    assert any("--runlegacy" in reason for reason in reasons) == (not run_legacy)
    assert any("--runslow" in reason for reason in reasons) == (not run_slow)


def test_default_setup_never_even_requests_a_model_cache(monkeypatch):
    def unexpected_cache_access(_):
        raise AssertionError("default test accessed model cache")

    request = SimpleNamespace(
        node=SimpleNamespace(get_closest_marker=lambda _: None),
        getfixturevalue=unexpected_cache_access,
    )
    fixture_code.isolated_legacy_models.__wrapped__(request, monkeypatch)


def test_collection_never_enables_unsafe_openmp_override():
    root = Path(__file__).parents[1]
    for path in [root / "conftest.py", *sorted((root / "tests").glob("test_*.py"))]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(
                        target.slice, ast.Constant
                    ):
                        assert target.slice.value != "KMP_DUPLICATE_LIB_OK", path


def test_data_fixture_reads_cached_archive_without_modifying_home(
    tmp_path, monkeypatch
):
    cached_home = tmp_path / "read-only-source"
    (cached_home / ".cellpose").mkdir(parents=True)
    archive = cached_home / ".cellpose/data.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("data/2D/example.txt", "sample")
    content = archive.read_bytes()
    monkeypatch.setattr(Path, "home", lambda: cached_home)
    destination = tmp_path / "test-storage"
    destination.mkdir()
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda _: False))
    data = fixture_code.data_dir.__wrapped__(request, destination)
    assert data == destination / "data"
    assert (data / "2D/example.txt").read_text() == "sample"
    assert archive.read_bytes() == content
    assert not (cached_home / ".cellpose/data").exists()


@pytest.mark.parametrize("run_legacy", [False, True])
def test_missing_sample_archive_never_downloads(tmp_path, monkeypatch, run_legacy):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "missing-home")
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda _: run_legacy))
    exception = pytest.fail.Exception if run_legacy else pytest.skip.Exception
    with pytest.raises(exception):
        fixture_code.data_dir.__wrapped__(request, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_verbose_cli_preserves_source_log_and_arguments_without_a_model(tmp_path):
    source_home = tmp_path / "synthetic-home"
    (source_home / ".cellpose").mkdir(parents=True)
    source_log = source_home / ".cellpose/run.log"
    source_log.write_text("preserve prior user log")
    log_directory = tmp_path / "private-logs"
    command = fixture_code.cellpose_cli_command(
        ["--image_path", "missing.png", "--train", "--verbose"], log_directory
    )
    # The deliberate invalid argument pair exits before model construction or
    # training. A fake home in this child proves that the actual logging path
    # is private; no real user log is opened by this regression.
    command[2] = (
        f"from pathlib import Path\nPath.home = lambda: Path({str(source_home)!r})\n"
        + command[2]
    )
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert "cannot train model with single image input" in result.stderr
    assert (log_directory / "run.log").is_file()
    assert source_log.read_text() == "preserve prior user log"
