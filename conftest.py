import time
import os
import shutil
import sys
import numpy as np
import pytest
from histoforge_models import models, vit_sam
import zipfile
import torch
import torch.nn.functional as F
from pathlib import Path


_CLI_TEST_BOOTSTRAP = """
import functools
import sys
from histoforge_models import io
log_directory = sys.argv.pop(1)
io.logger_setup = functools.partial(io.logger_setup, cp_path=log_directory)
from histoforge_models.__main__ import main
main()
"""


def cellpose_cli_command(arguments, log_directory):
    """Run the real CLI with only logging redirected to absolute test storage."""
    return [sys.executable, "-c", _CLI_TEST_BOOTSTRAP,
            str(log_directory.resolve()), *arguments]


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runlegacy", action="store_true", default=False,
        help="run integration tests with cached assets; may train locally (no downloads)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "legacy: cached-model or CLI integration test")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_legacy = pytest.mark.skip(reason="need --runlegacy option for cached-model/CLI integration")
    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
        if "legacy" in item.keywords and not config.getoption("--runlegacy"):
            item.add_marker(skip_legacy)


@pytest.fixture()
def image_names():
    image_names = ['rgb_2D_tif.tif', 'rgb_2D.png', 'gray_2D.png']
    return image_names


@pytest.fixture()
def image_names_3d():
    image_names_3d = ['rgb_3D.tif', 'gray_3D.tif']
    return image_names_3d


def extract_zip(cached_file, data_path):
    """Read an archive into a fresh test directory, never the user's data cache."""
    root = data_path.resolve()
    with zipfile.ZipFile(cached_file, "r") as archive:
        for name in archive.namelist():
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError(f"test archive path escapes temporary directory: {name}")
        archive.extractall(root)

@pytest.fixture()
def data_dir(request, tmp_path):
    archive = Path.home() / ".cellpose/data.zip"
    if not archive.is_file():
        if not request.config.getoption("--runlegacy"):
            pytest.skip("sample archive absent; no test assets are downloaded")
        pytest.fail(f"legacy integration requires the existing sample archive: {archive}")
    extract_zip(archive, tmp_path)
    return tmp_path / "data"


def copy_cached_models(source, destination):
    """Copy only standard test weights, not user models or their registry."""
    for name in ("cpsam", "cyto3"):
        cached = source / name
        if cached.is_file():
            shutil.copyfile(cached, destination / name)


@pytest.fixture(scope="session")
def legacy_model_directory(tmp_path_factory):
    if not (models.MODEL_DIR / "cpsam").is_file():
        pytest.fail(f"legacy integration requires cached cpsam weights in {models.MODEL_DIR}")
    destination = tmp_path_factory.mktemp("legacy-models")
    copy_cached_models(models.MODEL_DIR, destination)
    return destination


@pytest.fixture(autouse=True)
def isolated_legacy_models(request, monkeypatch):
    """Keep both in-process and child-CLI model/registry writes in test storage."""
    if request.node.get_closest_marker("legacy") is not None:
        destination = request.getfixturevalue("legacy_model_directory")
        monkeypatch.setattr(models, "MODEL_DIR", destination)
        monkeypatch.setattr(models, "MODEL_LIST_PATH", os.fspath(destination / "gui_models.txt"))
        monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", os.fspath(destination))

    
@pytest.fixture()
def cellposemodel_fixture_24layer():
    """ This is functionally identical to CellposeModel but uses mock class """
    use_gpu = torch.cuda.is_available()
    use_mps = 'mps' if torch.backends.mps.is_available() else False
    gpu = use_gpu or use_mps
    model = MockCellposeModel(24, gpu=gpu)
    yield model


@pytest.fixture()
def cellposemodel_fixture_2layer():
    """ This is only uses 2 transformer blocks for speed """
    use_gpu = torch.cuda.is_available()
    use_mps = 'mps' if torch.backends.mps.is_available() else False
    gpu = use_gpu or use_mps
    model = MockCellposeModel(n_keep_layers=2, gpu=gpu)
    yield model


class TinyAdapter(torch.nn.Module):
    """Small trunk with the real fixed-adapter task readout for contract tests."""

    def __init__(self, *, head_type="flow", dtype=torch.float32):
        super().__init__()
        self.ps = 4
        self.nclasses = 0
        self.backbone = torch.nn.Linear(2, 2)
        self.spm = torch.nn.Linear(2, 2)
        self.cti_modules = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.neck = torch.nn.Sequential(torch.nn.Conv2d(2, 256, 1))
        self.register_buffer("head_type_code", torch.tensor(0))
        self.register_buffer("adapter_dim_buf", torch.tensor(256))
        self.reset_segmentation_head(head_type)
        self.to(dtype=dtype)

    def reset_segmentation_head(self, head_type):
        from histoforge_models.pathology.dinov3 import DinoV3AdapterBackbone
        DinoV3AdapterBackbone.reset_segmentation_head(self, head_type)


class MockTransformer(vit_sam.Transformer):
    def __init__(self, use_layers: int):
        """ use_layers: the number of layers use starting from the first layer """
        super().__init__()

        self.use_layers = use_layers
        self.layer_idxs = np.linspace(0, 23, self.use_layers, dtype=int)

    def forward(self, x):
        x = self.encoder.patch_embed(x)
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed
        for layer_idx in self.layer_idxs:
            x = self.encoder.blocks[layer_idx](x)
        x = self.encoder.neck(x.permute(0, 3, 1, 2))
        x1 = self._readout(x)
        return x1, torch.randn((x.shape[0], 256), device=x.device)


class MockCellposeModel(models.CellposeModel):
    def __init__(self, n_keep_layers=2, gpu=False):
        super().__init__(gpu=gpu)

        self.net = MockTransformer(n_keep_layers)
        self.net.to(self.device)
        self.net.load_model(models.MODEL_DIR / 'cpsam', device=self.device)

    def eval(self, *args, **kwargs):
        tic = time.time()
        res = super().eval(*args, **kwargs)
        toc = time.time()

        print(f'eval() time elapsed: {toc-tic}')
        return res
