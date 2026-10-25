import pytest


@pytest.mark.legacy
def test_cellpose_imports_without_error():
    import histoforge_models
    from histoforge_models import models, core
    model = models.CellposeModel()


@pytest.mark.legacy
def test_model_zoo_imports_without_error():
    from histoforge_models import models, denoise
    for model_name in models.MODEL_NAMES:
        model = models.CellposeModel(pretrained_model=model_name)

def test_gui_imports_without_error():
    from histoforge_models import gui


def test_gpu_check():
    from histoforge_models import core
    core.use_gpu()


def itest_model_dir():
    import os, pathlib
    import numpy as np
    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.fspath(
        pathlib.Path.home().joinpath('.cellpose'))

    from histoforge_models import models
    model = models.CellposeModel(pretrained_model='cpsam')
    masks = model.eval(np.random.randn(256, 256))[0]
    assert masks.shape == (256, 256)


@pytest.mark.legacy
def test_load_cp3_fail():
    from histoforge_models.models import CellposeModel, MODEL_DIR

    cyto3_model_path = (MODEL_DIR / 'cyto3').absolute()

    if not cyto3_model_path.exists():
        pytest.fail(f"legacy integration requires cached cyto3 weights: {cyto3_model_path}")

    with pytest.raises(ValueError):
        # using `pretrained_model=cyto3` just loads the cpsam model unless the path is given
        model = CellposeModel(pretrained_model=str(cyto3_model_path))
