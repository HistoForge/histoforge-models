"""Cellpose flow inference for the fixed HistoForge DINOv3 adapter.

Parallel to `hovernet_inference.run_hovernet_inference` and
`semantic_inference.run_semantic_inference`: shared raw field execution followed
by head-specific decoding. Fixed inputs retain `core.run_net`; a supplied
variable profile uses natural rectangular fields. `dynamics.compute_masks`
decodes the complete blended flow raster once.

Classification channels are read directly from the network output layout
`[cls_0..cls_{N-1}, dY, dX, cellprob]`.

Module-level imports stay `histoforge_models.core`-free (histoforge_models.core is only
imported lazily inside `run_flow_inference`) so this file is safe to
vendor into HistoForge without pulling the cellpose baseline runtime at
gateway-container start.
"""

import logging
from time import perf_counter
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .common import per_instance_majority_vote
from .field_inference import run_model_fields
from .instance_cleanup import cleanup_cellpose_instances, require_min_instance_pixels
from .network_execution import NetworkPrecision

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_flow_inference(
    net: torch.nn.Module,
    image: np.ndarray,
    num_classes: int = 0,
    bsize: Optional[int] = None,
    tile_overlap: Optional[float] = None,
    batch_size: int = 8,
    normalize: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    # Cellpose dynamics knobs:
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_instance_pixels: int = 15,
    niter: int = 200,
    real_pixel_mask: Optional[np.ndarray] = None,
    field_profile: Optional[dict] = None,
    max_size_fraction: float = 0.4,
    include_diagnostics: bool = True,
    network_precision: NetworkPrecision = "fp32",
) -> Dict[str, Any]:
    """End-to-end flow inference with tiling + optional cls.

    Args:
        net: `DinoV3AdapterBackbone` (or any cellpose-compatible net) with
            weights loaded, same construction as HoVer/semantic via
            `common.build_and_load`.
        image: (H, W) or (H, W, 3) numpy array; grayscale is broadcast to 3-ch.
        num_classes: >0 when the net has a cls head, enables `class_map`
            and `votes_per_instance` in the returned dict.
        bsize: fixed tile side in px, default 1024 without field_profile.
        tile_overlap: fixed overlap fraction, default 0.1 without field_profile.
        batch_size: tiles per forward pass.
        normalize: apply `normalize99` to the complete image before tiling.
        real_pixel_mask: boolean HW source support; artificial padding is excluded.
        field_profile: frozen variable input.fieldProfile, instead of bsize/overlap.
        device: forward-pass device; defaults to `next(net.parameters()).device`.
        flow_threshold, cellprob_threshold, niter:
            `dynamics.compute_masks` knobs. See the Cellpose baseline docs.
        min_instance_pixels: minimum model-grid area, checked before and after
            hole filling in upstream Cellpose order. Zero disables size
            filtering without disabling hole filling.
        max_size_fraction: remove masks larger than this fraction of the decoded
            raster, preserving the upstream default of 0.4. Set to 1.0 to disable
            this relative-area cutoff.
        include_diagnostics: retain flow, probability and pixel-class maps for
            research callers. False returns only the complete instance mask and
            all instance votes, without filtering any neural class.
        network_precision: FP32 by default; BF16 changes only CUDA network forward.

    Returns:
        Dict with the complete instance mask and, when classes are present,
        votes from every class channel. Diagnostic maps are included only when
        ``include_diagnostics`` is true:
            instance_mask: (H, W) int32
            dP: (2, H, W) float32, [dY, dX] flow field
            cellprob: (H, W) float32
            class_map: (H, W) int32 (only if num_classes > 0)
            votes_per_instance: {inst_id: class_id} (only if num_classes > 0)
    """
    from ..dynamics import compute_masks

    if device is None:
        device = next(net.parameters()).device
    if isinstance(device, str):
        device = torch.device(device)
    min_instance_pixels = require_min_instance_pixels(min_instance_pixels)

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
    decode_started = perf_counter()

    # Flow head owns the last three channels: [dY, dX, cellprob].
    dP = np.stack([y[..., -3], y[..., -2]], axis=0).astype(np.float32, copy=False)
    cellprob = y[..., -1].astype(np.float32)

    decoded_mask = compute_masks(
        dP,
        cellprob,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        # Preserve upstream dynamics and flow QC, while the shared HistoForge
        # adapter below owns one exact cleanup call for every inference path.
        min_size=-1,
        max_size_fraction=max_size_fraction,
        niter=niter,
        device=device,
    )
    instance_mask = cleanup_cellpose_instances(
        decoded_mask,
        min_instance_pixels,
    ).astype(np.int32, copy=False)

    out: Dict[str, Any] = {
        "instance_mask": instance_mask,
    }
    if include_diagnostics:
        out.update(dP=dP, cellprob=cellprob)

    if num_classes > 0:
        # Cls channels are the FIRST `num_classes` of the network output.
        cls_hwc = y[..., :num_classes]  # (H, W, C)
        class_map = cls_hwc.argmax(axis=-1).astype(np.int32)
        out["votes_per_instance"] = per_instance_majority_vote(class_map, instance_mask)
        if include_diagnostics:
            out["class_map"] = class_map

    logger.info(
        "stage=decode_and_classify elapsed_seconds=%.6f",
        perf_counter() - decode_started,
    )
    return out
