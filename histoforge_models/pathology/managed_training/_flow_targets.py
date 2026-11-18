"""Pinned upstream Cellpose diffusion primitives used by managed training.

The managed package is intentionally self-contained, so it carries the exact
two-dimensional target functions from Cellpose commit
``a54cb48849b7e225a81e8e43dcb042d42427f543`` instead of importing the broader
runtime module. Product-specific target policy is applied by ``targets.py``
after these functions return.

Copyright 2025 Howard Hughes Medical Institute. Authored by Carsen Stringer,
Michael Rariden, and Marius Pachitariu. Distributed under this repository's
upstream Cellpose license terms.
"""

# These functions intentionally retain upstream locals and control flow.
# ruff: noqa: F841

import numpy as np
from scipy.ndimage import find_objects
import torch
import torch.nn.functional as F


def _extend_centers_gpu(
    neighbors, meds, isneighbor, shape, n_iter=200, device=torch.device("cpu")
):
    """Runs diffusion on GPU to generate flows for training images or quality control.

    Args:
        neighbors (torch.Tensor): 9 x pixels in masks.
        meds (torch.Tensor): Mask centers.
        isneighbor (torch.Tensor): Valid neighbor boolean 9 x pixels.
        shape (tuple): Shape of the tensor.
        n_iter (int, optional): Number of iterations. Defaults to 200.
        device (torch.device, optional): Device to run on. Defaults to torch.device("cpu").

    Returns:
        np.ndarray: Generated flows.
    """

    dtype = (
        torch.float32
        if torch.prod(torch.tensor(shape)) > 4e7 or device.type == "mps"
        else torch.float64
    )
    T_flat = torch.zeros(np.prod(shape), dtype=dtype, device=device)

    ndim = len(shape)
    Ly, Lx = shape[-2:]
    # speed up with flattened inds
    if ndim == 2:
        Ly, Lx = shape
        flat_neighbors = (neighbors[0] * Lx + neighbors[1]).long()
        flat_meds = (meds[:, 0] * Lx + meds[:, 1]).long()
    else:
        flat_neighbors = (
            neighbors[0] * (Ly * Lx) + neighbors[1] * Lx + neighbors[2]
        ).long()
        flat_meds = (meds[:, 0] * (Ly * Lx) + meds[:, 1] * Lx + meds[:, 2]).long()

    flat_center = flat_neighbors[0]

    nneigh = flat_neighbors.shape[0]
    for i in range(n_iter):
        T_flat[flat_meds] += 1
        Tneigh = T_flat[flat_neighbors]
        T_flat[flat_center] = (Tneigh * isneighbor).sum(dim=0) / nneigh
    del flat_meds, neighbors, meds, isneighbor, Tneigh

    if ndim == 2:
        grads = T_flat[flat_neighbors[[2, 1, 4, 3]]]
        dy = grads[0] - grads[1]
        dx = grads[2] - grads[3]
        del grads
        mu = np.stack((dy.cpu().numpy(), dx.cpu().numpy()), axis=0)
    else:
        grads = T_flat[flat_neighbors[1:]]
        dz = grads[0] - grads[1]
        dy = grads[2] - grads[3]
        dx = grads[4] - grads[5]
        del grads
        mu = np.stack((dz.cpu().numpy(), dy.cpu().numpy(), dx.cpu().numpy()), axis=0)

    return mu


def center_of_mass(mask):
    yi, xi = np.nonzero(mask)
    ymean = int(np.round(yi.sum() / len(yi)))
    xmean = int(np.round(xi.sum() / len(xi)))
    if not ((yi == ymean) * (xi == xmean)).sum():
        # center is closest point to (ymean, xmean) within mask
        imin = ((xi - xmean) ** 2 + (yi - ymean) ** 2).argmin()
        ymean = yi[imin]
        xmean = xi[imin]

    return ymean, xmean


def get_centers(masks, slices):
    centers = [center_of_mass(masks[slices[i]] == (i + 1)) for i in range(len(slices))]
    centers = np.array(
        [
            np.array(
                [centers[i][0] + slices[i][0].start, centers[i][1] + slices[i][1].start]
            )
            for i in range(len(slices))
        ]
    )
    exts = np.array(
        [
            (slc[0].stop - slc[0].start) + (slc[1].stop - slc[1].start) + 2
            for slc in slices
        ]
    )
    return centers, exts


def masks_to_flows_gpu(masks, device=torch.device("cpu"), niter=None):
    """Convert masks to flows using diffusion from center pixel.

    Center of masks where diffusion starts is defined by pixel closest to median within the mask.

    Args:
        masks (int, 2D or 3D array): Labelled masks. 0=NO masks; 1,2,...=mask labels.
        device (torch.device, optional): Device to run the computation on. Defaults to torch.device("cpu").
        niter (int, optional): Number of iterations for the diffusion process. Defaults to None.

    Returns:
        np.ndarray: A 4D array representing the flows for each pixel in Z, X, and Y.


    Returns:
        A tuple containing (mu, meds_p). mu is float 3D or 4D array of flows in (Z)XY.
        meds_p are cell centers.
    """
    if device is None:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    if masks.max() > 0:
        Ly0, Lx0 = masks.shape
        Ly, Lx = Ly0 + 2, Lx0 + 2

        masks_padded = torch.from_numpy(masks.astype("int64")).to(device)
        masks_padded = F.pad(masks_padded, (1, 1, 1, 1))
        shape = masks_padded.shape

        ### get mask pixel neighbors
        y, x = torch.nonzero(masks_padded, as_tuple=True)
        y = y.int()
        x = x.int()
        neighbors = torch.zeros((2, 9, y.shape[0]), dtype=torch.int, device=device)

        yxi = [[0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1]]
        for i in range(9):
            neighbors[0, i] = y + yxi[0][i]
            neighbors[1, i] = x + yxi[1][i]
        isneighbor = torch.ones((9, y.shape[0]), dtype=torch.bool, device=device)
        m0 = masks_padded[neighbors[0, 0], neighbors[1, 0]]
        for i in range(1, 9):
            isneighbor[i] = masks_padded[neighbors[0, i], neighbors[1, i]] == m0
        del m0, masks_padded

        ### get center-of-mass within cell
        slices = find_objects(masks)
        centers, ext = get_centers(masks, slices)
        meds_p = torch.from_numpy(centers).to(device).long()
        meds_p += 1  # for padding

        ### run diffusion
        n_iter = 2 * ext.max() if niter is None else niter
        mu = _extend_centers_gpu(
            neighbors, meds_p, isneighbor, shape, n_iter=n_iter, device=device
        )
        mu = mu.astype("float64")

        # new normalization
        mu /= 1e-60 + (mu**2).sum(axis=0) ** 0.5

        # put into original image
        mu0 = np.zeros((2, Ly0, Lx0))
        mu0[:, y.cpu().numpy() - 1, x.cpu().numpy() - 1] = mu
    else:
        # no masks, return empty flows
        mu0 = np.zeros((2, masks.shape[0], masks.shape[1]))
        slices = None

    return mu0, slices
