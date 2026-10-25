"""HoVer map prediction, instance decoding, and optional class voting."""

from collections.abc import Collection
import logging
from time import perf_counter
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .common import per_instance_majority_vote
from .decode_profile import (
    HOVER_EDGE_THRESHOLD,
    HOVER_FG_THRESHOLD,
    HOVER_MORPH_KERNEL,
    HOVER_OBJECT_SIZE,
    HOVER_SOBEL_KERNEL,
    HOVER_TILE_OVERLAP,
)
from .dinov3 import DinoV3AdapterBackbone
from .field_inference import run_model_fields
from .hv_decode import watershed_from_hv_maps
from .instance_cleanup import remove_instances_below_area
from .network_execution import NetworkPrecision


DEFAULT_BATCH_SIZE = 8
logger = logging.getLogger(__name__)


@torch.no_grad()
def run_hovernet_inference(
    net: DinoV3AdapterBackbone,
    image: np.ndarray,
    num_classes: int = 0,
    bsize: Optional[int] = None,
    tile_overlap: Optional[float] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = True,
    device: Optional[Union[str, torch.device]] = None,  # noqa: ARG001
    fg_threshold: float = HOVER_FG_THRESHOLD,
    edge_threshold: float = HOVER_EDGE_THRESHOLD,
    ksize: int = HOVER_SOBEL_KERNEL,
    morph_kernel: int = HOVER_MORPH_KERNEL,
    object_size: int = HOVER_OBJECT_SIZE,
    min_instance_pixels: int = 0,
    real_pixel_mask: Optional[np.ndarray] = None,
    field_profile: Optional[dict] = None,
    include_diagnostics: bool = True,
    network_precision: NetworkPrecision = "fp32",
) -> Dict[str, Any]:
    """Predict HoVer maps for one image, then decode instances and class votes.

    Raw channels from overlapping model inputs are blended before watershed.
    The output layout is ``[classes..., vertical, horizontal, bg, fg]``.
    ``network_precision`` selects only network forward arithmetic, default FP32.
    """

    field_started = perf_counter()
    output = run_model_fields(
        net,
        image,
        fixed_field_defaults=(1024, HOVER_TILE_OVERLAP),
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
    return decode_hovernet_maps(
        output,
        num_classes,
        fg_threshold=fg_threshold,
        edge_threshold=edge_threshold,
        ksize=ksize,
        morph_kernel=morph_kernel,
        object_size=object_size,
        min_instance_pixels=min_instance_pixels,
        include_diagnostics=include_diagnostics,
    )


def decode_hovernet_maps(
    output: np.ndarray,
    num_classes: int = 0,
    *,
    fg_threshold: float = HOVER_FG_THRESHOLD,
    edge_threshold: float = HOVER_EDGE_THRESHOLD,
    ksize: int = HOVER_SOBEL_KERNEL,
    morph_kernel: int = HOVER_MORPH_KERNEL,
    object_size: int = HOVER_OBJECT_SIZE,
    min_instance_pixels: int = 0,
    protected_edges: Collection[str] = (),
    include_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Decode blended HoVer maps and vote on the retained instances.

    ``output`` has shape ``(height, width, channels)``, with channels ordered as
    ``[classes..., vertical, horizontal, bg, fg]``. The result contains a
    ``(height, width)`` instance mask, a ``(2, height, width)``
    vertical/horizontal map, the foreground probability at every pixel, and,
    when classes are present, the pixel class map and instance votes. Voting
    uses every supplied class channel after final area cleanup.

    ``protected_edges`` may name ``top``, ``bottom``, ``left``, or ``right``
    processing-image sides. Instances touching those sides bypass only the
    final area filter so a caller can check their completeness later. Callers
    must select artificial window edges, not true target-image boundaries.
    Watershed thresholds, marker cleanup, and morphology are unchanged.

    With ``include_diagnostics=False``, return only the complete instance mask
    and all instance votes. Neither background nor training-only winners are
    filtered here; geometry emission owns that projection.
    """

    decode_started = perf_counter()
    hv_map = np.stack([output[..., -4], output[..., -3]], axis=0).astype(
        np.float32, copy=False
    )
    nucleus_logits = output[..., -2:].astype(np.float32, copy=False)
    exponentials = np.exp(nucleus_logits - nucleus_logits.max(axis=-1, keepdims=True))
    nucleus_probability = (exponentials[..., 1] / exponentials.sum(axis=-1)).astype(
        np.float32, copy=False
    )

    decoder_input = np.stack([nucleus_probability, hv_map[0], hv_map[1]], axis=0)
    instance_mask = watershed_from_hv_maps(
        decoder_input,
        object_size=object_size,
        ksize=ksize,
        edge_threshold=edge_threshold,
        morph_kernel=morph_kernel,
        fg_threshold=fg_threshold,
    )
    protected_instance_ids = set()
    if protected_edges:
        edge_pixels = {
            "top": instance_mask[0, :],
            "bottom": instance_mask[-1, :],
            "left": instance_mask[:, 0],
            "right": instance_mask[:, -1],
        }
        protected_instance_ids = {
            int(instance_id)
            for edge in protected_edges
            for instance_id in np.unique(edge_pixels[edge])
            if instance_id > 0
        }
    instance_mask = remove_instances_below_area(
        instance_mask,
        min_instance_pixels,
        protected_instance_ids=protected_instance_ids,
    )

    result: Dict[str, Any] = {
        "instance_mask": instance_mask,
    }
    if include_diagnostics:
        result.update(hv_map=hv_map, np_prob=nucleus_probability)
    if num_classes > 0:
        class_logits = output[..., :num_classes]
        class_map = class_logits.argmax(axis=-1).astype(np.int32)
        result["votes_per_instance"] = per_instance_majority_vote(
            class_map, instance_mask
        )
        if include_diagnostics:
            result["class_map"] = class_map
    logger.info(
        "stage=decode_and_classify elapsed_seconds=%.6f",
        perf_counter() - decode_started,
    )
    return result
