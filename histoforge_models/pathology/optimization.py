"""DINOv3 parameter grouping shared by standalone and managed training."""


def build_param_groups(net, base_lr, llrd_factor, patch_embed_lr, weight_decay,
                       neck_lr=1e-3, head_lr=1e-3, adapter_lr=None):
    """AdamW groups with DINOv3 LLRD and explicit adapter/head rates."""
    adapter_rate = adapter_lr or neck_lr
    n_blocks = len(net.backbone.blocks)

    def lr_for(name):
        if name.startswith("backbone.blocks."):
            index = int(name.split(".")[2])
            return base_lr * (llrd_factor**(n_blocks - 1 - index))
        if name.startswith("backbone.patch_embed."):
            return patch_embed_lr
        if name.startswith("spm.") or name.startswith("cti_modules."):
            return adapter_rate
        if name.startswith("neck."):
            return neck_lr
        if name.startswith("out"):
            return head_lr
        return base_lr

    buckets = {}
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        is_token = "cls_token" in name or "reg_token" in name
        decay = 0.0 if (parameter.ndim < 2 or is_token) else weight_decay
        buckets.setdefault((lr_for(name), decay), []).append(parameter)

    return [{
        "params": parameters,
        "lr": lr,
        "weight_decay": decay
    } for (lr, decay), parameters in buckets.items()]
