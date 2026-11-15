"""Shared construction, checkpoint, and classification helpers.

HistoForge builds one fixed DINOv3 adapter architecture. Recipe parameters
select only the task head and class count; architectural variants are not a
runtime surface.
"""

from collections import OrderedDict
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from .dinov3 import (
    _HEAD_TYPE_CODE,
    _HEAD_TYPE_NAME,
    _HEAD_TYPE_NOUT,
    DinoV3AdapterBackbone,
)
from .normalization import normalize99  # noqa: F401


_W2_CLS_KEY = "out_class.weight"


class StateDictIdentityError(RuntimeError):
    """Checkpoint tensors do not match the constructed network exactly."""


def _normalize_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping):
        raise StateDictIdentityError("checkpoint state must be a tensor mapping")
    keys = list(state_dict)
    if not all(isinstance(key, str) for key in keys):
        raise StateDictIdentityError("checkpoint tensor names must be strings")
    prefixed = [key.startswith("module.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise StateDictIdentityError("checkpoint mixes module-prefixed and plain keys")
    if prefixed and all(prefixed):
        return OrderedDict((key[7:], value) for key, value in state_dict.items())
    return OrderedDict(state_dict.items())


def detect_num_classes_from_state(
    state_dict: Mapping[str, torch.Tensor],
) -> int:
    """Read the W2 classification-channel count from a checkpoint."""

    normalized = _normalize_state_dict_keys(state_dict)
    if _W2_CLS_KEY not in normalized:
        return 0
    weight = normalized[_W2_CLS_KEY]
    w2 = normalized.get("W2")
    if w2 is None or w2.ndim != 4 or w2.shape[-1] != w2.shape[-2]:
        raise StateDictIdentityError(
            "W2 is missing or malformed; class count cannot be derived"
        )
    patch_area = int(w2.shape[-1]) ** 2
    if weight.ndim != 4 or int(weight.shape[0]) % patch_area:
        raise StateDictIdentityError(
            "out_class.weight rows are not divisible by the W2 patch area"
        )
    return int(weight.shape[0]) // patch_area


def detect_head_type_from_state(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """Read and validate the required checkpoint head metadata."""

    normalized = _normalize_state_dict_keys(state_dict)
    code_tensor = normalized.get("head_type_code")
    if code_tensor is None:
        raise StateDictIdentityError("checkpoint requires head_type_code")
    try:
        code = int(code_tensor.item())
    except (AttributeError, RuntimeError, ValueError) as error:
        raise StateDictIdentityError("head_type_code must be one scalar") from error
    if code not in _HEAD_TYPE_NAME:
        raise StateDictIdentityError(f"unknown head_type_code={code}")

    head_type = _HEAD_TYPE_NAME[code]
    expected_outputs = _HEAD_TYPE_NOUT[head_type]
    w2 = normalized.get("W2")
    out_weight = normalized.get("out.weight")
    if w2 is None or w2.ndim != 4 or w2.shape[-1] != w2.shape[-2]:
        raise StateDictIdentityError(
            "W2 is missing or malformed; head type cannot be validated"
        )
    patch_side = int(w2.shape[-1])
    if int(w2.shape[1]) != expected_outputs:
        raise StateDictIdentityError(
            f"head_type_code declares {head_type}, but W2 has "
            f"{int(w2.shape[1])} output channels"
        )
    expected_rows = expected_outputs * patch_side**2
    if int(w2.shape[0]) != expected_rows:
        raise StateDictIdentityError("W2 has an inconsistent output-channel shape")
    if out_weight is None or out_weight.ndim != 4:
        raise StateDictIdentityError(
            "out.weight is missing or malformed; head type cannot be validated"
        )
    if int(out_weight.shape[0]) != expected_rows:
        raise StateDictIdentityError("out.weight and W2 disagree on output channels")
    return head_type


def load_state_dict_exact(
    network: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load only an exact current-architecture checkpoint.

    A consistent ``module.`` prefix is a serialization detail and is removed.
    Missing metadata, retired architecture tensors, partial checkpoints, and
    shape or dtype drift all fail before any parameter is mutated.
    """

    normalized = _normalize_state_dict_keys(state_dict)
    expected = network.state_dict()
    supplied_keys = set(normalized)
    expected_keys = set(expected)
    missing = sorted(expected_keys - supplied_keys)
    unexpected = sorted(supplied_keys - expected_keys)
    if missing or unexpected:
        raise StateDictIdentityError(
            "checkpoint tensor keys differ from the constructed model: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for name, tensor in normalized.items():
        target = expected[name]
        try:
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
        except AttributeError as error:
            raise StateDictIdentityError(
                f"checkpoint value {name!r} is not a tensor"
            ) from error
        if shape != tuple(target.shape):
            raise StateDictIdentityError(
                f"checkpoint tensor {name!r} has shape {shape}, "
                f"expected {tuple(target.shape)}"
            )
        if dtype != target.dtype:
            raise StateDictIdentityError(
                f"checkpoint tensor {name!r} has dtype {dtype}, expected {target.dtype}"
            )

    for name in ("head_type_code", "adapter_dim_buf"):
        if name in expected:
            supplied_value = int(normalized[name].item())
            expected_value = int(expected[name].item())
            if supplied_value != expected_value:
                raise StateDictIdentityError(
                    f"checkpoint {name}={supplied_value} does not match "
                    f"the constructed model value {expected_value}"
                )

    network.load_state_dict(normalized, strict=True)


def build(config: Mapping[str, object]) -> DinoV3AdapterBackbone:
    """Construct the fixed adapter with the requested task head."""

    unknown = sorted(set(config) - {"head_type", "num_classes", "dtype"})
    if unknown:
        raise ValueError(f"unsupported network construction keys: {unknown}")
    head_type = config.get("head_type")
    if head_type not in _HEAD_TYPE_CODE:
        raise ValueError(
            f"config['head_type']={head_type!r} must be in {list(_HEAD_TYPE_CODE)}"
        )
    dtype = config.get("dtype", torch.float32)
    if not isinstance(dtype, torch.dtype):
        raise ValueError("config['dtype'] must be a torch dtype")
    return DinoV3AdapterBackbone(head_type=head_type, dtype=dtype)


def _attach_cls_head(
    network: DinoV3AdapterBackbone,
    num_classes: int,
) -> None:
    """Attach the fixed W2 per-pixel classification head."""

    network.nclasses = num_classes
    if num_classes == 0:
        if hasattr(network, "out_class"):
            del network.out_class
            del network.W3
        return
    patch_side = network.ps
    device = network.out.weight.device
    dtype = network.out.weight.dtype
    network.out_class = torch.nn.Conv2d(
        256,
        num_classes * patch_side**2,
        kernel_size=1,
    ).to(device=device, dtype=dtype)
    network.W3 = torch.nn.Parameter(
        torch.eye(
            num_classes * patch_side**2,
            device=device,
            dtype=dtype,
        ).reshape(
            num_classes * patch_side**2,
            num_classes,
            patch_side,
            patch_side,
        ),
        requires_grad=False,
    )


def build_and_load(
    config: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
) -> DinoV3AdapterBackbone:
    """Build the recipe-declared network and load its exact checkpoint."""

    num_classes = config.get("num_classes")
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes < 0
    ):
        raise ValueError("config['num_classes'] must be a nonnegative integer")
    network = build(config)
    _attach_cls_head(network, num_classes)
    load_state_dict_exact(network, state_dict)
    return network


def per_instance_majority_vote(
    class_logits: torch.Tensor | np.ndarray,
    instance_mask: np.ndarray,
    ignore_classes: Sequence[int] = (),
) -> dict[int, int]:
    """Assign each nonzero instance its most frequent pixel class."""

    if torch.is_tensor(class_logits):
        classes = class_logits.detach().cpu().numpy()
    else:
        classes = np.asarray(class_logits)
    if classes.ndim == 4:
        classes = classes[0]
    if classes.ndim == 3:
        classes = classes.argmax(axis=0)
    classes = classes.astype(np.int32, copy=False)

    if classes.shape != instance_mask.shape:
        raise ValueError(
            f"class logits shape {classes.shape} does not match "
            f"instance mask shape {instance_mask.shape}"
        )

    mask = np.asarray(instance_mask)
    ignored_classes = {int(value) for value in ignore_classes}
    if mask.size == 0:
        return {}

    max_instance = int(mask.max())
    min_instance = int(mask.min())
    max_class = int(classes.max())
    min_class = int(classes.min())
    dense_count_cells = (max_instance + 1) * (max_class + 1)
    if (
        np.issubdtype(mask.dtype, np.integer)
        and min_instance >= 0
        and min_class >= 0
        and dense_count_cells <= 10_000_000
    ):
        class_count = max_class + 1
        counts = np.zeros((max_instance + 1, class_count), dtype=np.int64)
        flat_mask = mask.ravel()
        flat_classes = classes.ravel()
        ignored = np.fromiter(sorted(ignored_classes), dtype=np.int32)
        for start in range(0, flat_mask.size, 1_048_576):
            instance_chunk = flat_mask[start : start + 1_048_576]
            class_chunk = flat_classes[start : start + 1_048_576]
            keep = instance_chunk != 0
            if ignored.size:
                keep &= ~np.isin(class_chunk, ignored)
            if not np.any(keep):
                continue
            encoded = (
                instance_chunk[keep].astype(np.int64, copy=False) * class_count
                + class_chunk[keep]
            )
            counts += np.bincount(encoded, minlength=counts.size).reshape(counts.shape)
        totals = counts.sum(axis=1)
        winners = counts.argmax(axis=1)
        return {
            instance_id: int(winners[instance_id])
            for instance_id in range(1, max_instance + 1)
            if totals[instance_id] > 0
        }

    result: dict[int, int] = {}
    for instance_id in np.unique(mask):
        if instance_id == 0:
            continue
        pixels = classes[mask == instance_id]
        if ignored_classes:
            pixels = pixels[~np.isin(pixels, list(ignored_classes))]
        if pixels.size == 0:
            continue
        values, counts = np.unique(pixels, return_counts=True)
        result[int(instance_id)] = int(values[counts.argmax()])
    return result
