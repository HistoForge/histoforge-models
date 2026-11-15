"""HoVer-Net-style losses for HistoForge Models.

Adapted from micro_sam/micro_sam/training/hv_losses.py (Apache-2.0,
Constantin Pape et al.). Underlying methodology: HoVer-Net (Graham et al.,
"HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in
Multi-Tissue Histology Images", Medical Image Analysis, 2019, MIT license).
Original reference implementation: github.com/vqdang/hover_net.

Convention divergence from micro_sam: HV maps here are stored as [V, H]
(vertical/y-axis first, horizontal/x-axis last) so they align with cellpose's
[flow_y, flow_x] rotational-pair semantics. ``random_rotate_and_resize`` rotates
the last-2 label channels using an image-coords rotation matrix. Kernel
assignments inside MSGELossMaps are swapped accordingly (v_dv on channel 0,
h_dh on channel 1). Micro_sam uses the opposite [H, V] order.
"""

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


IGNORE_INDEX = 255


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average values over true mask entries without dividing by padding."""

    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class MSELossMaps(nn.Module):
    """Per-pixel mean squared error on HV maps (both channels)."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return map MSE over all pixels or only explicitly valid pixels."""

        diff = pred - target
        squared = diff * diff
        if valid_mask is None:
            return squared.mean()
        valid = valid_mask.to(torch.bool)
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        return _masked_mean(squared, valid.expand_as(squared))


class MSGELossMaps(nn.Module):
    """Foreground-masked MSE on the HoVer gradient of HV maps.

    Uses HoVer-Net's gradient kernel `h / (h*h + v*v + eps)` at kernel_size=5
    (not standard Sobel; see Graham et al. 2019). Computes v_dv on channel 0
    (V-map) and h_dh on channel 1 (H-map). Cross-derivatives are NOT computed
    (matches HoVer-Net's original design).
    """

    KERNEL_SIZE = 5

    def _kernels(self, device: torch.device, dtype: torch.dtype):
        """Construct HoVer's horizontal and vertical finite-difference kernels."""

        s = self.KERNEL_SIZE
        r = torch.arange(-s // 2 + 1, s // 2 + 1, dtype=dtype, device=device)
        y, x = torch.meshgrid(r, r, indexing="ij")
        denom = x * x + y * y + 1.0e-15
        k_h = (x / denom).view(1, 1, s, s)
        k_v = (y / denom).view(1, 1, s, s)
        return k_h, k_v

    def _gradient_vh(self, vh: torch.Tensor) -> torch.Tensor:
        """vh: (B, 2, H, W) = [V, H]. Returns (B, 2, H, W) = [v_dv, h_dh]."""
        k_h, k_v = self._kernels(vh.device, vh.dtype)
        v_ch = vh[:, 0:1]
        h_ch = vh[:, 1:2]
        pad = self.KERNEL_SIZE // 2
        v_dv = F.conv2d(v_ch, k_v, padding=pad)
        h_dh = F.conv2d(h_ch, k_h, padding=pad)
        return torch.cat([v_dv, h_dh], dim=1)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, focus: torch.Tensor
    ) -> torch.Tensor:
        """pred/target: (B, 2, H, W) [V, H]. focus: (B, 1, H, W) fg mask."""
        true_grad = self._gradient_vh(target)
        pred_grad = self._gradient_vh(pred)
        focus_2 = torch.cat([focus, focus], dim=1).to(pred.dtype)
        diff = pred_grad - true_grad
        num = (focus_2 * diff * diff).sum()
        den = focus_2.sum() + 1.0e-8
        return num / den

    def erode_valid_support(
        self, valid_mask: torch.Tensor, *, exterior_is_invalid: bool = False
    ) -> torch.Tensor:
        """Remove pixels whose gradient kernel crosses invalid support."""
        valid = valid_mask.to(torch.bool)
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        radius = self.KERNEL_SIZE // 2
        invalid = (~valid).to(torch.float32)
        if exterior_is_invalid:
            invalid = F.pad(invalid, (radius, radius, radius, radius), value=1.0)
            padding = 0
        else:
            padding = radius
        crosses_invalid = F.max_pool2d(
            invalid, kernel_size=self.KERNEL_SIZE, stride=1, padding=padding
        )
        return crosses_invalid == 0


class XentropyLoss(nn.Module):
    """CrossEntropy on the NP (foreground vs background) logits."""

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return foreground cross-entropy over all or explicitly valid pixels."""

        if valid_mask is None:
            return F.cross_entropy(logits, target)
        valid = valid_mask.to(torch.bool)
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        per_pixel = F.cross_entropy(logits, safe_target, reduction="none")
        return _masked_mean(per_pixel, valid)


class DiceLoss(nn.Module):
    """Soft Dice on the NP logits: softmax -> one-hot -> per-class dice -> sum."""

    def __init__(self, smooth: float = 1.0e-3):
        """Create a soft Dice loss with the given denominator smoothing."""

        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return summed per-class soft Dice over valid pixels."""

        C = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        if valid_mask is None:
            valid = torch.ones_like(target, dtype=torch.bool)
            safe_target = target
        else:
            valid = valid_mask.to(torch.bool)
            safe_target = torch.where(valid, target, torch.zeros_like(target))
        oh = F.one_hot(safe_target, C).permute(0, 3, 1, 2).to(probs.dtype)
        valid = valid.unsqueeze(1).to(probs.dtype)
        probs = probs * valid
        oh = oh * valid
        inter = (probs * oh).sum(dim=(0, 2, 3))
        p_sum = probs.sum(dim=(0, 2, 3))
        o_sum = oh.sum(dim=(0, 2, 3))
        dice = 1.0 - (2.0 * inter + self.smooth) / (p_sum + o_sum + self.smooth)
        return dice.sum()


class HVMapLoss(nn.Module):
    """Combined MSE + MSGE + CE + Dice (equal-weighted sum)."""

    def __init__(self):
        """Create the four established equally weighted HoVer components."""

        super().__init__()
        self.mse = MSELossMaps()
        self.msge = MSGELossMaps()
        self.ce = XentropyLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        hv_pred: torch.Tensor,
        np_pred: torch.Tensor,
        hv_target: torch.Tensor,
        fg_target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        *,
        tensor_exterior_is_invalid: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hv_pred: (B, 2, H, W) [V, H] regression predictions.
            np_pred: (B, 2, H, W) NP logits (channel 0 = bg, channel 1 = fg).
            hv_target: (B, 2, H, W) [V_gt, H_gt] in [-1, 1] (bg = 0).
            fg_target: (B, H, W) long {0, 1, 255}; 255 is ignore.
            valid_mask: optional (B, H, W) boolean mask. If omitted, valid
                support is derived from ``fg_target != 255``.
            tensor_exterior_is_invalid: exclude gradient-loss positions whose
                finite-difference kernel would cross the tensor boundary.
        """
        derived_valid = fg_target != IGNORE_INDEX
        valid = (
            derived_valid
            if valid_mask is None
            else (valid_mask.to(torch.bool) & derived_valid)
        )
        safe_fg = torch.where(valid, fg_target, torch.zeros_like(fg_target))
        valid_values = safe_fg[valid]
        if valid_values.numel() and not torch.all(
            (valid_values == 0) | (valid_values == 1)
        ):
            raise ValueError("valid HoVer foreground targets must be 0 or 1")

        mse = self.mse(hv_pred, hv_target, valid_mask=valid)
        gradient_valid = self.msge.erode_valid_support(
            valid, exterior_is_invalid=tensor_exterior_is_invalid
        )
        focus = (safe_fg.unsqueeze(1) > 0) & gradient_valid
        msge = self.msge(hv_pred, hv_target, focus=focus)
        ce = self.ce(np_pred, safe_fg, valid_mask=valid)
        dice = self.dice(np_pred, safe_fg, valid_mask=valid)
        return {
            "total": mse + msge + ce + dice,
            "hv_mse": mse,
            "hv_msge": msge,
            "np_ce": ce,
            "np_dice": dice,
        }
