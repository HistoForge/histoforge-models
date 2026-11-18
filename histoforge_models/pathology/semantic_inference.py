"""Per-pixel semantic labels and probabilities from blended model outputs.

Class channels precede any segmentation channels. Label selection uses every
class channel; callers decide which winning classes to emit.
"""

import logging
from time import perf_counter
from typing import Dict, Optional, Union

import numpy as np
import torch

from .field_inference import run_model_fields
from .network_execution import NetworkPrecision

logger = logging.getLogger(__name__)


def class_labels_from_maps(output: np.ndarray, num_classes: int) -> np.ndarray:
    """Return int32 pixel labels from the leading class channels.

    ``output`` has shape ``(height, width, channels)``. Argmax uses the first
    ``num_classes`` channels, including background and any class a caller does
    not emit, and ignores later segmentation channels. Ties select the lowest
    channel index. This operation does not compute class probabilities.
    """
    return output[..., :num_classes].argmax(axis=-1).astype(np.int32)


@torch.no_grad()
def run_semantic_inference(
    net: torch.nn.Module,
    image: np.ndarray,
    num_classes: int,
    bsize: Optional[int] = None,
    tile_overlap: Optional[float] = None,
    batch_size: int = 8,
    normalize: bool = True,
    device: Optional[Union[str, torch.device]] = None,  # noqa: ARG001
    real_pixel_mask: Optional[np.ndarray] = None,
    field_profile: Optional[dict] = None,
    include_diagnostics: bool = True,
    network_precision: NetworkPrecision = "fp32",
) -> Dict[str, np.ndarray]:
    """Predict class labels and probabilities for one supplied image.

    Args:
        net: loaded network whose class channels come before segmentation channels.
        image: (H, W) or (H, W, 3) numpy array.
        num_classes: positive class-head channel count, including any background.
        bsize: fixed tile side in pixels, default 1024 without field_profile.
        tile_overlap: fixed overlap fraction, default 0.1 without field_profile.
        batch_size: tiles per forward pass.
        normalize: apply `normalize99` to the complete image before tiling.
        real_pixel_mask: boolean HW source support; artificial padding is excluded.
        field_profile: frozen variable input.fieldProfile, instead of bsize/overlap.
        device: unused; the forward pass uses the network's device.
        include_diagnostics: return probabilities as well as labels. False avoids
            allocating unused softmax arrays; all class channels still compete.
        network_precision: FP32 by default; BF16 changes only CUDA network forward.

    Returns:
        A dict containing ``class_map``, an (H, W) int32 argmax over all class
        channels. With diagnostics enabled, also return ``probs``, a
        (num_classes, H, W) float32 softmax over those same channels.
    """
    if num_classes <= 0:
        raise ValueError(
            f"run_semantic_inference requires num_classes > 0, got {num_classes}"
        )

    field_started = perf_counter()
    y = run_model_fields(
        net,
        image,
        fixed_field_defaults=(1024, 0.1),
        bsize=bsize,
        tile_overlap=tile_overlap,
        batch_size=batch_size,
        normalize=normalize,
        real_pixel_mask=real_pixel_mask,
        field_profile=field_profile,
        network_precision=network_precision,
    )
    logger.info(
        "stage=field_prediction elapsed_seconds=%.6f", perf_counter() - field_started
    )
    classification_started = perf_counter()

    # Segmentation channels do not participate in class labels or probabilities.
    cls_hwc = y[..., :num_classes]  # (H, W, C)
    class_map = class_labels_from_maps(y, num_classes)
    logger.info(
        "stage=classify elapsed_seconds=%.6f", perf_counter() - classification_started
    )
    if not include_diagnostics:
        return {"class_map": class_map}

    # Preserve float32 softmax after selecting labels from the original logits.
    cls_chw = np.transpose(cls_hwc, (2, 0, 1)).astype(np.float32)  # (C, H, W)
    e = np.exp(cls_chw - cls_chw.max(axis=0, keepdims=True))
    probs = (e / e.sum(axis=0, keepdims=True)).astype(np.float32)

    return {
        "class_map": class_map,
        "probs": probs,
    }
