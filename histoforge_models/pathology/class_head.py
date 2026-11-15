"""Explicit class-readout adaptation after strict loading of a DINOv3 adapter.

This changes only classification tensors. Callers own class meaning, source
checkpoint validation, and seeding; they must adapt before building the optimizer.
Ordinary inference never initializes or repairs a checkpoint here.
"""

import torch

from .common import _attach_cls_head


def initialize_decoupled_class_head(net, nclasses):
    """Initialize class logits from the existing segmentation foreground.

    Preserve the standalone initializer exactly: background starts at -0.5
    times the last segmentation channel's weight, foreground classes at +0.5,
    and all classes copy its bias. W3 is the fixed identity spatial readout.
    """
    _attach_cls_head(net, nclasses)
    if nclasses == 0:
        return
    area = net.ps**2
    foreground_weights = net.out.weight[-area:]
    foreground_bias = net.out.bias[-area:]
    with torch.no_grad():
        for index in range(nclasses):
            region = slice(index * area, (index + 1) * area)
            sign = -0.5 if index == 0 else 0.5
            net.out_class.weight[region].copy_(sign * foreground_weights)
            net.out_class.bias[region].copy_(foreground_bias)


def adapt_class_head(net, nclasses, *, mode):
    """Keep, append to, or replace a strictly loaded classification readout.

    ``use`` keeps every tensor and requires the same channel count. ``add``
    preserves the old channel prefix, including background. ``define`` resets
    every class output even at the same count. A segmentation-only source has
    zero class channels; add/define create background plus foreground outputs.
    Neither operation removes a classifier or changes segmentation tensors.

    Added classes preserve the old logits, not their softmax probabilities:
    new outputs participate in the same competition after initialization.
    """
    if mode not in ("use", "add", "define"):
        raise ValueError("class initialization must be use, add, or define")
    if isinstance(nclasses, bool) or not isinstance(nclasses, int) or nclasses < 0:
        raise ValueError("nclasses must be a nonnegative integer")
    old_count = net.nclasses
    if mode == "use":
        if nclasses != old_count:
            raise ValueError("use requires the existing class count")
        return
    if nclasses < 2:
        raise ValueError("a new classifier requires background and foreground")
    if mode == "add" and nclasses <= old_count:
        raise ValueError("add requires more class channels than the source")

    old_head = net.out_class if old_count else None
    old_readout = net.W3 if old_count else None
    initialize_decoupled_class_head(net, nclasses)
    if mode == "add" and old_count:
        rows = old_count * net.ps**2
        with torch.no_grad():
            net.out_class.weight[:rows].copy_(old_head.weight)
            net.out_class.bias[:rows].copy_(old_head.bias)
            # A valid checkpoint may contain a nonidentity W3. Preserve its
            # entire old block; initialized cross-blocks stay zero.
            net.W3[:rows, :old_count].copy_(old_readout)
