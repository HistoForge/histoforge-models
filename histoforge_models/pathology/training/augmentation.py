"""Stain color and image quality augmentations for histopathology training.

HED perturbation follows Tellez et al. 2019, using color deconvolution
from Ruifrok & Johnston 2001. NMF stain separation follows Vahadane et al.
2016. Quality augmentation parameters from StarDist CoNIC 2022
(Weigert & Schmidt, BSD-3-Clause, https://github.com/stardist/stardist).
HED/NMF combination strategy from Mandal et al. 2025 (Classpose).

All functions expect CHW numpy arrays and return CHW numpy arrays.
"""

import logging
import warnings

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HED color deconvolution (Ruifrok & Johnston 2001)
# ---------------------------------------------------------------------------

# Canonical stain absorption vectors for Hematoxylin, Eosin, DAB.
_RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29],
     [0.07, 0.99, 0.11],
     [0.27, 0.57, 0.78]], dtype=np.float64)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float64)


def _to_float64_hwc(img):
    """CHW numpy (any dtype) -> HWC float64 in [0, 1]."""
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"Expected 3-channel CHW, got shape {img.shape}")
    hwc = np.transpose(img, (1, 2, 0)).astype(np.float64)
    if img.dtype == np.uint8:
        hwc /= 255.0
    elif img.dtype == np.uint16:
        hwc /= 65535.0
    else:
        hwc = np.clip(hwc, 0.0, 1.0)
    return hwc


def _from_float64_hwc(hwc, original):
    """HWC float64 [0, 1] -> CHW with same dtype as *original*."""
    hwc = np.clip(hwc, 0.0, 1.0)
    if original.dtype == np.uint8:
        out = (hwc * 255.0).astype(np.uint8)
    elif original.dtype == np.uint16:
        out = (hwc * 65535.0).astype(np.uint16)
    else:
        out = hwc.astype(original.dtype)
    return np.transpose(out, (2, 0, 1))


# ---------------------------------------------------------------------------
# HED stain perturbation  (Tellez et al. 2019)
# ---------------------------------------------------------------------------

def perturb_hed(img, rng, sigma_range=0.2, cutoff=(0.15, 0.85)):
    """Randomly perturb Hematoxylin, Eosin, and DAB channels.

    Each channel is independently scaled and shifted:
        ch_new = ch * (1 + sigma) + bias
    where sigma, bias ~ Uniform(-sigma_range, +sigma_range).

    Patches with mean intensity outside *cutoff* are returned unchanged
    (avoids corrupting near-white background or near-black artifacts).

    Args:
        img: CHW numpy array (uint8, uint16, or float).
        rng: np.random.Generator.
        sigma_range: perturbation magnitude (default 0.2, Tellez 2019
            "HED-strong"; also StainTools default and Marini 2023 optimal).
        cutoff: (lo, hi) mean-intensity guard (default (0.15, 0.85),
            from DIAGNijmegen/pathology-he-auto-augment, MIDL 2021).
    """
    if not 0 < sigma_range <= 1.0:
        raise ValueError(f"sigma_range must be in (0, 1], got {sigma_range}")
    if not (0 <= cutoff[0] < cutoff[1] <= 1.0):
        raise ValueError(f"cutoff must satisfy 0 <= lo < hi <= 1, got {cutoff}")

    hwc = _to_float64_hwc(img)
    mean_intensity = hwc.mean()
    if not (cutoff[0] <= mean_intensity <= cutoff[1]):
        return img

    # Forward: RGB -> optical density -> HED
    od = -np.log(np.clip(hwc, 1e-6, None))
    hed = od.reshape(-1, 3) @ _HED_FROM_RGB
    hed = hed.reshape(hwc.shape)

    # Perturb each channel independently
    for ch in range(3):
        sigma = rng.uniform(-sigma_range, sigma_range)
        bias = rng.uniform(-sigma_range, sigma_range)
        hed[:, :, ch] = hed[:, :, ch] * (1.0 + sigma) + bias

    # Inverse: HED -> optical density -> RGB
    od_aug = hed.reshape(-1, 3) @ _RGB_FROM_HED
    rgb_aug = np.exp(-od_aug).reshape(hwc.shape)

    return _from_float64_hwc(rgb_aug, img)


# ---------------------------------------------------------------------------
# NMF stain decomposition  (Vahadane et al. 2016 / StarDist CoNIC 2022)
# ---------------------------------------------------------------------------

def _require_nmf():
    """Load NMF without silently changing the requested augmentation recipe."""
    try:
        from sklearn.decomposition import NMF
    except ImportError as exc:
        raise ImportError(
            "NMF stain augmentation requires scikit-learn>=1.0. "
            "Install the declared dependencies with: pip install -e ."
        ) from exc
    return NMF


def perturb_nmf(img, rng, matrix_amount=0.15, concentration_amount=0.4,
                p=0.9):
    """Perturb stain colors via per-image NMF decomposition.

    Decomposes the image into a learned stain matrix and concentration
    maps, perturbs both, and reconstructs.  Falls back to the original
    image on any numerical failure. A missing scikit-learn dependency is
    an error, not a numerical fallback.

    Parameters from StarDist CoNIC 2022 (BSD-3-Clause):
        matrix_amount=0.15, concentration_amount=0.4, p=0.9,
        subsample=128, tissue_threshold=200, NMF alpha_W=0.001.

    Args:
        img: CHW numpy array (uint8, uint16, or float).
        rng: np.random.Generator.
        matrix_amount: additive perturbation range for stain matrix.
        concentration_amount: multiplicative perturbation range for
            stain concentrations.
        p: probability of applying (0.9 = 90%).
    """
    if rng.random() > p:
        return img

    NMF = _require_nmf()

    try:
        # Convert to HWC uint8
        hwc = _to_float64_hwc(img)
        hwc_u8 = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)

        # Optical density
        od = np.maximum(-np.log(np.maximum(hwc_u8, 1).astype(np.float64)
                                / 255.0), 1e-6)

        # Tissue mask: LAB lightness < 200
        lab = cv2.cvtColor(hwc_u8, cv2.COLOR_RGB2LAB)
        tissue = lab[..., 0] < 200
        values = od[tissue]
        if len(values) == 0:
            values = od.reshape(-1, 3)

        # Fit NMF on subsampled tissue pixels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = NMF(n_components=2, init="random", random_state=0,
                        alpha_W=0.001, alpha_H=0, l1_ratio=1)
            step = 128
            model.fit(values[::step] if len(values) > step else values)

        H = model.components_                                   # (2, 3)
        H = H / np.linalg.norm(H, axis=1, keepdims=True)       # L2 normalize
        if H[0, 0] < H[1, 0]:                                  # H row first
            H = H[[1, 0]]
        stains = od.reshape(-1, 3) @ np.linalg.pinv(H)         # (N, 2)
        stains = stains.reshape(hwc.shape[:2] + (2,))

        # Perturb matrix and concentrations
        H_aug = H + rng.uniform(-1, 1, H.shape) * matrix_amount
        H_aug = np.maximum(H_aug, 0)
        s_aug = stains * (1 + rng.uniform(-1, 1, (1, 1, 2))
                          * concentration_amount)
        s_aug = np.maximum(s_aug, 0)

        # Reconstruct
        rgb_aug = np.clip(255.0 * np.exp(-(s_aug.reshape(-1, 2) @ H_aug)),
                          0, 255).astype(np.uint8)
        rgb_aug = rgb_aug.reshape(hwc_u8.shape)

        # Back to CHW, original dtype
        result = rgb_aug.astype(np.float64) / 255.0
        return _from_float64_hwc(result, img)

    except Exception:
        logger.debug("NMF augmentation failed, returning original", exc_info=True)
        return img


# ---------------------------------------------------------------------------
# Quality augmentations  (StarDist CoNIC 2022, BSD-3)
# ---------------------------------------------------------------------------

def apply_blur(img, rng, max_sigma=2.0, p=0.1):
    """Per-channel Gaussian blur.

    Args:
        img: CHW numpy array.
        rng: np.random.Generator.
        max_sigma: maximum blur sigma (StarDist CoNIC default: 2.0).
        p: probability of applying (StarDist CoNIC default: 0.1).
    """
    if rng.random() > p:
        return img
    sigma = rng.uniform(0, max_sigma)
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        out[c] = gaussian_filter(img[c], sigma=sigma)
    return out


def apply_noise(img, rng, sigma=0.01, p=0.8):
    """Additive Gaussian noise.

    Args:
        img: CHW numpy array.
        rng: np.random.Generator.
        sigma: noise standard deviation (StarDist CoNIC default: 0.01).
        p: probability of applying (StarDist CoNIC default: 0.8).
    """
    if rng.random() > p:
        return img
    # Work in float [0, 1]
    original_dtype = img.dtype
    if img.dtype == np.uint8:
        work = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        work = img.astype(np.float32) / 65535.0
    else:
        work = img.astype(np.float32)
    noise = rng.normal(0, sigma, work.shape).astype(np.float32)
    work = work + noise
    if original_dtype == np.uint8:
        return np.clip(work * 255, 0, 255).astype(np.uint8)
    if original_dtype == np.uint16:
        return np.clip(work * 65535, 0, 65535).astype(np.uint16)
    return np.clip(work, 0, 1).astype(original_dtype)


def apply_color_jitter(img, rng, hue=0.05, brightness=0.1,
                       saturation_range=(0.9, 1.1), contrast=0.1, p=0.9):
    """Hue, brightness, saturation, and contrast jitter via torchvision.

    Args:
        img: CHW numpy array (3 channels).
        rng: np.random.Generator.
        hue: hue shift range (our choice: ±0.05, conservative for H&E).
        brightness: brightness factor range (StarDist CoNIC default: ±0.1).
        saturation_range: (lo, hi) saturation multiplier (conventional ±10%).
        contrast: contrast factor range (our addition: ±0.1).
        p: probability of applying (StarDist CoNIC default: 0.9).
    """
    if rng.random() > p:
        return img

    import torch
    from torchvision.transforms.v2 import functional as tvf

    original_dtype = img.dtype

    # CHW numpy -> CHW float32 tensor in [0, 1]
    if img.dtype == np.uint8:
        t = torch.as_tensor(img.copy(), dtype=torch.float32) / 255.0
    elif img.dtype == np.uint16:
        t = torch.as_tensor(img.copy(), dtype=torch.float32) / 65535.0
    else:
        t = torch.as_tensor(np.clip(img, 0, 1).copy(), dtype=torch.float32)

    # Sample and apply
    if hue > 0:
        t = tvf.adjust_hue(t, float(rng.uniform(-hue, hue)))
    if brightness > 0:
        t = tvf.adjust_brightness(t, float(1 + rng.uniform(-brightness,
                                                            brightness)))
    if saturation_range[0] != saturation_range[1]:
        t = tvf.adjust_saturation(t, float(rng.uniform(*saturation_range)))
    if contrast > 0:
        t = tvf.adjust_contrast(t, float(1 + rng.uniform(-contrast, contrast)))

    # Back to numpy CHW, original dtype
    result = t.numpy()
    if original_dtype == np.uint8:
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    if original_dtype == np.uint16:
        return np.clip(result * 65535, 0, 65535).astype(np.uint16)
    return np.clip(result, 0, 1).astype(original_dtype)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def augment_batch(images, strategy="none", rng=None):
    """Apply color and quality augmentations to a list of images.

    Strategies:
        "none":  no augmentation (passthrough).
        "color": HED stain perturbation only (Tellez et al. 2019).
        "all":   50/50 HED vs NMF stain aug (Mandal et al. 2025) followed
                 by blur, noise, and color jitter (StarDist CoNIC 2022).
                 Requires scikit-learn even when the batch chooses HED.

    Args:
        images: list of CHW numpy arrays (raw, NOT normalized).
        strategy: one of "none", "color", "all".
        rng: np.random.Generator (created if None).

    Returns:
        List of augmented CHW numpy arrays (same dtypes as input).
    """
    if strategy == "none":
        return images
    if rng is None:
        rng = np.random.default_rng()

    if strategy == "color":
        return [perturb_hed(img, rng) for img in images]

    if strategy == "all":
        _require_nmf()
        # Per-batch: 50% HED, 50% NMF  (Mandal et al. 2025)
        use_hed = rng.random() < 0.5
        out = []
        for img in images:
            if use_hed:
                aug = perturb_hed(img, rng)
            else:
                aug = perturb_nmf(img, rng)
            aug = apply_blur(aug, rng)
            aug = apply_noise(aug, rng)
            aug = apply_color_jitter(aug, rng)
            out.append(aug)
        return out

    raise ValueError(f"Unknown augmentation strategy: {strategy!r}")
