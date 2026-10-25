"""PyTorch Dataset for parallel augmentation in cellpose training."""

import logging

import numpy as np
import torch
from histoforge_models import io
from histoforge_models.transforms import normalize_img, random_rotate_and_resize
from histoforge_models.train import _prepare_flow_training_label, _reshape_norm

logger = logging.getLogger(__name__)


class TrainingDataset(torch.utils.data.Dataset):
    """Map-style dataset that augments images on-the-fly.

    Each __getitem__ call applies the full augmentation pipeline:
    color/quality aug -> geometric transforms -> normalization.
    Use with torch.utils.data.DataLoader for parallel workers.
    """

    def __init__(self, data_array, label_array, diameter_array,
                 augmentation_strategy="none", diam_mean=30., rescale=False,
                 scale_range=0.5, bsize=256, normalize_params=None,
                 flow_loss_weighting="pixel"):
        self.data = data_array
        self.labels = label_array
        self.diams = diameter_array
        self.aug_strategy = augmentation_strategy
        self.diam_mean = diam_mean
        self.rescale = rescale
        self.scale_range = scale_range
        self.bsize = bsize
        self.norm_params = normalize_params or {"normalize": True}
        self.flow_loss_weighting = flow_loss_weighting

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        lbl = _prepare_flow_training_label(
            self.labels[idx], self.flow_loss_weighting)
        diam = self.diams[idx]

        rsc = diam / self.diam_mean if self.rescale else 1.0

        # Color augmentation BEFORE geometric transforms (needs raw pixels)
        if self.aug_strategy != "none":
            from histoforge_models.pathology.training.augmentation import augment_batch
            worker_info = torch.utils.data.get_worker_info()
            seed = ((worker_info.seed if worker_info else 0) + idx) % (2**32)
            rng = np.random.default_rng(seed)
            img = augment_batch([img], self.aug_strategy, rng=rng)[0]

        imgi, lbl = random_rotate_and_resize([img], Y=[lbl], rescale=[rsc],
                                             scale_range=self.scale_range,
                                             xy=(self.bsize, self.bsize))[:2]

        imgi = normalize_img(imgi[0], axis=0, **self.norm_params)
        return imgi, lbl[0]


class LazyTrainingDataset(torch.utils.data.Dataset):
    """Map-style dataset that reads each image + precomputed flow from disk.

    Memory-light alternative to TrainingDataset: it holds only file-path
    lists, so DataLoader workers (spawn start method) pickle a tiny object
    instead of the whole in-RAM corpus. flow_files are the {stem}_flows.tif
    written by datasets/dataset_converters/precompute_flows.py — a 4-channel
    [mask, cellprob, flowY, flowX] array. The instance-mask channel is normally
    dropped; equal-instance flow weighting first converts it into an
    inverse-full-instance-area map that is transformed with the targets.

    Augmentation is identical to TrainingDataset; only the source of the
    image/label arrays differs (disk read vs. in-RAM index).
    """

    def __init__(self, image_files, flow_files, diameter_array,
                 augmentation_strategy="none", diam_mean=30., rescale=False,
                 scale_range=0.5, bsize=256, normalize_params=None,
                 flow_loss_weighting="pixel"):
        self.image_files = image_files
        self.flow_files = flow_files
        self.diams = diameter_array
        self.aug_strategy = augmentation_strategy
        self.diam_mean = diam_mean
        self.rescale = rescale
        self.scale_range = scale_range
        self.bsize = bsize
        self.norm_params = normalize_params or {"normalize": True}
        self.flow_loss_weighting = flow_loss_weighting

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # io.imread is H,W,C — reshape to channels-first (3,H,W), as the eager
        # TrainingDataset has it (reshaped upstream in _process_train_test).
        img = _reshape_norm([io.imread(self.image_files[idx])],
                            normalize_params={"normalize": False})[0]
        raw = io.imread(self.flow_files[idx])
        lbl = _prepare_flow_training_label(raw, self.flow_loss_weighting)
        diam = self.diams[idx]

        rsc = diam / self.diam_mean if self.rescale else 1.0

        # Color augmentation BEFORE geometric transforms (needs raw pixels)
        if self.aug_strategy != "none":
            from histoforge_models.pathology.training.augmentation import augment_batch
            worker_info = torch.utils.data.get_worker_info()
            seed = ((worker_info.seed if worker_info else 0) + idx) % (2**32)
            rng = np.random.default_rng(seed)
            img = augment_batch([img], self.aug_strategy, rng=rng)[0]

        imgi, lbl = random_rotate_and_resize([img], Y=[lbl], rescale=[rsc],
                                             scale_range=self.scale_range,
                                             xy=(self.bsize, self.bsize))[:2]

        imgi = normalize_img(imgi[0], axis=0, **self.norm_params)
        return imgi, lbl[0]


def numpy_collate(batch):
    """Collate into numpy arrays (training loop converts to torch)."""
    imgs, lbls = zip(*batch)
    return np.stack(imgs), np.stack(lbls)


def seed_worker(worker_id):
    """Seed numpy RNG per DataLoader worker for augmentation variety."""
    info = torch.utils.data.get_worker_info()
    np.random.seed(info.seed % (2**32 - 1))
