"""Research stain augmentation contracts using synthetic pixels only."""

import builtins
from pathlib import Path

import numpy as np
from packaging.requirements import Requirement
import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from histoforge_models.pathology.training import augmentation


def _image(dtype=np.uint8):
    pixels = np.random.default_rng(21).integers(40, 210, (3, 32, 40), dtype=np.uint8)
    if dtype == np.uint16:
        return pixels.astype(dtype) * 257
    if np.issubdtype(dtype, np.floating):
        return pixels.astype(dtype) / 255
    return pixels


@pytest.fixture
def missing_sklearn(monkeypatch):
    real_import = builtins.__import__

    def import_without_sklearn(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sklearn)


def test_install_requires_nmf_dependency():
    metadata_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(metadata_path.read_text())
    requirements = {
        requirement.name: requirement
        for requirement in map(Requirement, metadata["project"]["dependencies"])
    }
    assert "scikit-learn" in requirements
    specifier = requirements["scikit-learn"].specifier
    assert "1.0" in specifier
    assert "0.24" not in specifier  # alpha_W and alpha_H were introduced in 1.0.


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("strategy", ["nmf", "all"])
def test_real_nmf_executes_and_preserves_array_contract(monkeypatch, dtype, strategy):
    from sklearn.decomposition import NMF

    fitted_shapes = []
    real_fit = NMF.fit

    def fit(self, values, *args, **kwargs):
        fitted_shapes.append(values.shape)
        return real_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(NMF, "fit", fit)
    pixels = _image(dtype)
    original = pixels.copy()
    results = []
    states = []
    for _ in range(2):
        # Seed 0 selects NMF in the all strategy and passes its application gate.
        rng = np.random.default_rng(0)
        if strategy == "nmf":
            result = augmentation.perturb_nmf(pixels, rng, p=1)
        else:
            result = augmentation.augment_batch([pixels], strategy, rng=rng)[0]
        results.append(result)
        states.append(rng.bit_generator.state)

    assert len(fitted_shapes) == 2
    assert all(shape[1] == 3 for shape in fitted_shapes)
    np.testing.assert_array_equal(results[0], results[1])
    assert states[0] == states[1]
    np.testing.assert_array_equal(pixels, original)
    result = results[0]
    assert result.shape == pixels.shape and result.dtype == pixels.dtype
    assert not np.shares_memory(result, pixels)
    assert np.isfinite(result).all() and np.any(result != pixels)
    upper = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
    assert result.min() >= 0 and result.max() <= upper


def test_applied_nmf_requires_dependency(missing_sklearn):
    pixels = _image()
    original = pixels.copy()
    with pytest.raises(ImportError, match="NMF.*scikit-learn") as error:
        augmentation.perturb_nmf(pixels, np.random.default_rng(0), p=1)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    np.testing.assert_array_equal(pixels, original)


@pytest.mark.parametrize("seed", [0, 2])  # NMF and HED branches, respectively.
def test_all_checks_dependency_before_stochastic_choice(missing_sklearn, seed):
    rng = np.random.default_rng(seed)
    state = rng.bit_generator.state
    pixels = _image()
    with pytest.raises(ImportError, match="NMF.*scikit-learn"):
        augmentation.augment_batch([pixels], "all", rng=rng)
    assert rng.bit_generator.state == state
    np.testing.assert_array_equal(pixels, _image())


@pytest.mark.parametrize("strategy", ["none", "color"])
def test_non_nmf_strategies_do_not_require_sklearn(missing_sklearn, strategy):
    pixels = _image()
    result = augmentation.augment_batch(
        [pixels], strategy, rng=np.random.default_rng(0)
    )[0]
    if strategy == "none":
        assert result is pixels
    else:
        assert np.any(result != pixels)
    np.testing.assert_array_equal(pixels, _image())


def test_skipped_direct_nmf_keeps_existing_probability_gate(missing_sklearn):
    pixels = _image()
    assert augmentation.perturb_nmf(pixels, np.random.default_rng(0), p=0) is pixels


def test_numerical_fit_failure_keeps_existing_fallback(monkeypatch):
    from sklearn.decomposition import NMF

    def fail_fit(*args, **kwargs):
        raise np.linalg.LinAlgError("synthetic numerical failure")

    monkeypatch.setattr(NMF, "fit", fail_fit)
    pixels = _image()
    assert augmentation.perturb_nmf(pixels, np.random.default_rng(0), p=1) is pixels
