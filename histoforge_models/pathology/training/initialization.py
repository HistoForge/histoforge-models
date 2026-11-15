"""Standalone initialization for the fixed HistoForge adapter.

Inference loads complete checkpoints exactly. Training may then deliberately
replace a task/class head, or replace a donor backbone with an explicit official
DINOv3 artifact. No partial checkpoint or missing-metadata fallback is accepted.
"""

import hashlib
import json
from pathlib import Path

import torch

from ...core import assign_device
from ...models import CellposeModel
from ..class_head import initialize_decoupled_class_head
from ..common import (
    build,
    build_and_load,
    detect_head_type_from_state,
    detect_num_classes_from_state,
)
from ..dinov3 import DINOV3_MODEL


def _new_cellpose_model_shell(gpu):
    """Reuse the standalone Cellpose evaluation API without loading CPSAM."""
    model = CellposeModel.__new__(CellposeModel)
    model.device, model.gpu = assign_device(gpu=gpu)
    model.pretrained_model = None
    model.net = None
    model.timing = []
    return model


def _load_dinov3_backbone_checkpoint(backbone, path):
    """Strictly load one explicit official DINOv3 backbone artifact.

    The supported artifact is the bare timm/Hugging Face ``model.safetensors``
    state dict. It intentionally contains no HistoForge adapter or task-head
    tensors. Requiring an exact key and shape match prevents a renamed,
    truncated, or architecture-mismatched file from silently becoming a
    vanilla-initialization baseline.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DINOv3 backbone checkpoint is missing: {path}")
    if path.suffix != ".safetensors":
        raise ValueError(
            "explicit DINOv3 backbone initialization requires a .safetensors "
            f"artifact, got: {path}"
        )
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "explicit DINOv3 backbone initialization requires safetensors"
        ) from exc

    state = load_file(str(path), device="cpu")
    target = backbone.state_dict()
    missing = sorted(set(target) - set(state))
    unexpected = sorted(set(state) - set(target))
    mismatched = sorted(
        f"{key}: artifact={tuple(state[key].shape)} target={tuple(target[key].shape)}"
        for key in set(target) & set(state)
        if state[key].shape != target[key].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "DINOv3 backbone artifact does not exactly match the constructed "
            f"backbone: missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}"
        )
    backbone.load_state_dict(state, strict=True)
    return {
        "artifact": str(path),
        "format": "safetensors",
        "strict": True,
        "consumed_tensor_scope": "backbone_only_no_adapter_or_head_tensors",
        "loaded_keys": len(state),
        "missing": [],
        "unexpected": [],
        "mismatched": [],
    }


def _transfer_report(state, excluded_keys, *, preserve_backbone):
    transferred = {
        key: tensor
        for key, tensor in state.items()
        if key not in excluded_keys
        and not (preserve_backbone and key.startswith("backbone."))
    }
    manifest = [
        {
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
        }
        for key, tensor in sorted(transferred.items())
    ]
    return {
        "transferred_keys": len(transferred),
        "transferred_numel": sum(tensor.numel() for tensor in transferred.values()),
        "transfer_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "replaced_task_keys": sorted(excluded_keys),
        "preserved_raw_backbone": preserve_backbone,
    }


def create_cellpose_model(
    *,
    gpu=True,
    pretrained_model=None,
    nclasses=None,
    head_type=None,
    warm_start=False,
    dinov3_backbone_checkpoint=None,
    adapter_checkpoint=None,
):
    """Construct a standalone model with explicit exact-load/transfer semantics.

    ``pretrained_model`` is a complete current HistoForge checkpoint. An exact
    load reconstructs its heads; ``warm_start=True`` permits the requested heads
    to differ. ``adapter_checkpoint`` is a complete class-agnostic donor paired
    with an explicit raw backbone. Without either donor, initialize the official
    DINOv3 backbone (explicit artifact or timm's pretrained default) and fresh
    adapter/readout. All I/O stays here, outside the shared network constructor.
    """
    if head_type not in (None, "flow", "hovernet"):
        raise ValueError(f"unsupported head_type={head_type!r}")
    if nclasses is not None and (
        isinstance(nclasses, bool) or not isinstance(nclasses, int) or nclasses < 0
    ):
        raise ValueError("nclasses must be a nonnegative integer")
    if pretrained_model is not None and (
        dinov3_backbone_checkpoint is not None or adapter_checkpoint is not None
    ):
        raise ValueError(
            "pretrained_model is mutually exclusive with "
            "dinov3_backbone_checkpoint/adapter_checkpoint"
        )
    if adapter_checkpoint is not None:
        if dinov3_backbone_checkpoint is None:
            raise ValueError("adapter_checkpoint requires dinov3_backbone_checkpoint")
        if not warm_start:
            raise ValueError("adapter_checkpoint is a training-only warm start")

    donor_path = (
        pretrained_model if pretrained_model is not None else adapter_checkpoint
    )
    saved_head = None
    saved_classes = 0
    if donor_path is not None:
        state = torch.load(donor_path, map_location="cpu", weights_only=True)
        saved_head = detect_head_type_from_state(state)
        saved_classes = detect_num_classes_from_state(state)
        if adapter_checkpoint is not None and saved_classes:
            raise ValueError("adapter_checkpoint must be class agnostic")
        if not warm_start:
            if head_type is not None and head_type != saved_head:
                raise ValueError("head_type conflicts with checkpoint")
            if nclasses is not None and nclasses != saved_classes:
                raise ValueError("nclasses conflicts with checkpoint")
        # Validate every donor tensor before any deliberately replaced component
        # is discarded. A truncated donor never becomes a valid warm start.
        net = build_and_load(
            {"head_type": saved_head, "num_classes": saved_classes}, state
        )
    else:
        net = build({"head_type": head_type or "flow"})

    target_head = head_type or saved_head or "flow"
    target_classes = (
        nclasses if nclasses is not None else (saved_classes if not warm_start else 0)
    )
    excluded = set()
    if target_head != net.head_type:
        net.reset_segmentation_head(target_head)
        excluded.update({"out.weight", "out.bias", "W2", "head_type_code"})

    if dinov3_backbone_checkpoint is not None:
        net.backbone_initialization_report = _load_dinov3_backbone_checkpoint(
            net.backbone, dinov3_backbone_checkpoint
        )
    elif donor_path is None:
        import timm

        # Release the random trunk before constructing the pretrained one; do
        # not hold two ViT-L backbones or duplicate timm's checkpoint loader.
        del net.backbone
        net.backbone = timm.create_model(DINOV3_MODEL, pretrained=True, num_classes=0)

    if target_classes != saved_classes:
        initialize_decoupled_class_head(net, target_classes)
        excluded.update({"out_class.weight", "out_class.bias", "W3"})
    if donor_path is not None and warm_start:
        # The exact loader normalizes a consistent DataParallel prefix. Use the
        # normalized names for the report as well.
        from ..common import _normalize_state_dict_keys

        report = _transfer_report(
            _normalize_state_dict_keys(state),
            excluded,
            preserve_backbone=adapter_checkpoint is not None,
        )
        report.update(
            replace_seg_head=target_head != saved_head,
            replace_class_head=target_classes != saved_classes,
        )
        if adapter_checkpoint is not None:
            net.adapter_warm_start_report = report
        else:
            net.warm_start_report = report

    model = _new_cellpose_model_shell(gpu)
    model.net = net.to(model.device)
    return model
