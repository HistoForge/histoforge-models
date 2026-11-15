"""HoVer-Net-style instance HV map generation.

Adapted from micro_sam/micro_sam/training/hv_transforms.py (Apache-2.0,
Constantin Pape et al.). Underlying methodology: HoVer-Net (Graham et al.,
Medical Image Analysis 2019, MIT license). Original reference:
github.com/vqdang/hover_net.

Convention: returns (V, H) order (vertical/y-axis first, horizontal/x-axis
last) so the maps drop into the last-2 channels of cellpose's label layout
and `random_rotate_and_resize`'s rotation of the last-2 vector channels
matches [flow_y, flow_x] semantics. Micro_sam uses the opposite [H, V] order.
"""

import numpy as np
from scipy.ndimage import center_of_mass


def _bounding_box(mask: np.ndarray) -> list:
    """Return [row_min, row_max, col_min, col_max] of a non-empty binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [rmin, rmax + 1, cmin, cmax + 1]


def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
    """Generate (V, H) gradient maps normalized per instance to [-1, 1].

    For each pixel inside an instance, the (V, H) values point away from
    the centroid, normalized separately per side (negative/positive) per
    axis so each instance spans [-1, 1] regardless of its shape.

    Args:
        inst_map: (H, W) integer instance mask. 0 = background.

    Returns:
        (2, H, W) float32:
            channel 0 = V (vertical / y-axis) gradient in [-1, 1]
            channel 1 = H (horizontal / x-axis) gradient in [-1, 1]
        Background pixels are 0 in both channels.
    """
    orig = inst_map.copy()
    h, w = orig.shape[:2]

    v_map = np.zeros((h, w), dtype=np.float32)
    h_map = np.zeros((h, w), dtype=np.float32)

    inst_ids = list(np.unique(orig))
    if 0 in inst_ids:
        inst_ids.remove(0)

    for inst_id in inst_ids:
        mask = (orig == inst_id).astype(np.uint8)
        box = _bounding_box(mask)
        # Expand bbox by 2 px safety margin, clipped to image bounds.
        box[0] = max(0, box[0] - 2)
        box[2] = max(0, box[2] - 2)
        box[1] = min(h, box[1] + 2)
        box[3] = min(w, box[3] + 2)
        crop = mask[box[0] : box[1], box[2] : box[3]]
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            continue

        com = list(center_of_mass(crop))
        com[0] = int(com[0] + 0.5)
        com[1] = int(com[1] + 0.5)

        y_range = np.arange(1, crop.shape[0] + 1) - com[0]
        x_range = np.arange(1, crop.shape[1] + 1) - com[1]
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        x_grid[crop == 0] = 0
        y_grid[crop == 0] = 0
        x_grid = x_grid.astype(np.float32)
        y_grid = y_grid.astype(np.float32)

        # Normalize negatives and positives separately so each instance
        # spans [-1, 1] regardless of aspect ratio.
        if np.min(x_grid) < 0:
            x_grid[x_grid < 0] /= -np.min(x_grid[x_grid < 0])
        if np.min(y_grid) < 0:
            y_grid[y_grid < 0] /= -np.min(y_grid[y_grid < 0])
        if np.max(x_grid) > 0:
            x_grid[x_grid > 0] /= np.max(x_grid[x_grid > 0])
        if np.max(y_grid) > 0:
            y_grid[y_grid > 0] /= np.max(y_grid[y_grid > 0])

        h_box = h_map[box[0] : box[1], box[2] : box[3]]
        h_box[crop > 0] = x_grid[crop > 0]
        v_box = v_map[box[0] : box[1], box[2] : box[3]]
        v_box[crop > 0] = y_grid[crop > 0]

    return np.stack([v_map, h_map])
