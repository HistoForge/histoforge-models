from histoforge_models import io, models, train
from subprocess import check_output, STDOUT
import shutil
import sys
import pytest
import torch
from pathlib import Path
from conftest import cellpose_cli_command


pytestmark = pytest.mark.legacy


def test_class_train(data_dir):
    train_dir = str(data_dir.joinpath('2D').joinpath('train'))
    model_dir = str(data_dir.joinpath('2D').joinpath('train').joinpath('models'))
    shutil.rmtree(model_dir, ignore_errors=True)
    output = io.load_train_test_data(train_dir, mask_filter='_cyto_masks')
    images, labels, image_names, test_images, test_labels, image_names_test = output
    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu)
    cpmodel_path = train.train_seg(model.net, images, labels, train_files=image_names,
                                   test_data=test_images, test_labels=test_labels,
                                   test_files=image_names_test,
                                   save_path=train_dir, n_epochs=3)[0]
    io.add_model(cpmodel_path)
    io.remove_model(cpmodel_path, delete=True)
    print('>>>> model trained and saved to %s' % cpmodel_path)


def test_cli_train(data_dir):
    # import sys
    # path_root = Path(__file__).parents[1]
    # sys.path.append(str(path_root))
    # print(Path(__file__).parents[0],Path(__file__).parents[1],Path(__file__).parents[2])
    train_dir = str(data_dir.joinpath('2D').joinpath('train'))
    model_dir = str(data_dir.joinpath('2D').joinpath('train').joinpath('models'))
    shutil.rmtree(model_dir, ignore_errors=True)
    use_gpu = torch.cuda.is_available()
    arguments = ["--train", "--n_epochs", "3",
           "--dir", train_dir, "--mask_filter", "_cyto_masks", "--pretrained_model", "None"]
    if use_gpu:
        arguments.append("--use_gpu")
    cmd = cellpose_cli_command(arguments, data_dir.parent / "cli-logs")
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT).decode()
    except Exception as e:
        print(e)
        raise ValueError(e)


def test_cli_make_train(data_dir):
    script_name = Path().resolve() / 'histoforge_models/gui/make_train.py'
    image_path = data_dir / '3D/gray_3D.tif'

    cmd = [sys.executable, str(script_name), "--image_path", str(image_path)]
    res = check_output(cmd, stderr=STDOUT)

    # there should be 30 slices: 
    files = [f for f in (data_dir / '3D/train/').iterdir() if 'gray_3D' in f.name]
    assert 30 == len(files)

    shutil.rmtree((data_dir / '3D/train'))
