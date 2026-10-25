"""Network-only inference precision shared by finite and window predictors.

Normalization, tiling, blending and decoding stay outside this boundary. FP32
retains the upstream forward path. BF16 uses CUDA autocast for the network call
only, without converting stored weights; both paths return FP32 NumPy arrays.
Importing the policy type does not import Torch.
"""

from typing import Literal


NetworkPrecision = Literal["fp32", "bf16"]


def forward_network(net, images, *, network_precision: NetworkPrecision = "fp32"):
    """Return prediction and style arrays without changing network parameters.

    ``images`` contains a BCHW tile batch. BF16 requires FP32 weights on a CUDA
    device with native BF16 support. Unsupported requests fail before forwarding;
    there is no automatic precision fallback or change to TF32 settings.
    """
    from .. import core

    if network_precision == "fp32":
        return core._forward(net, images)
    if network_precision != "bf16":
        raise ValueError(f"Unsupported network precision: {network_precision!r}")

    import torch

    device = torch.device(net.device)
    if device.type != "cuda":
        raise RuntimeError("BF16 network inference requires a CUDA device")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("CUDA device does not support native BF16 inference")
    if net.dtype != torch.float32:
        raise ValueError("BF16 autocast requires unchanged FP32 network weights")

    inputs = core._to_device(images, device=device, dtype=net.dtype)
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction, style = net(inputs)[:2]
    del inputs
    return core._from_device(prediction), core._from_device(style)
