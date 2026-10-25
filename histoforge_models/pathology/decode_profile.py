"""Versioned defaults shared by every supported HoVer decode path."""

HOVER_DECODE_PROFILE_VERSION = "hover-watershed-v1"
HOVER_TILE_OVERLAP = 0.25
HOVER_EDGE_THRESHOLD = 0.4
HOVER_OBJECT_SIZE = 10
HOVER_FG_THRESHOLD = 0.5
HOVER_MORPH_KERNEL = 5
HOVER_SOBEL_KERNEL = 21


def hover_decode_profile():
    """Return a JSON-serializable copy of the default decode contract."""
    return {
        "version": HOVER_DECODE_PROFILE_VERSION,
        "tile_overlap": HOVER_TILE_OVERLAP,
        "edge_threshold": HOVER_EDGE_THRESHOLD,
        "object_size": HOVER_OBJECT_SIZE,
        "fg_threshold": HOVER_FG_THRESHOLD,
        "morph_kernel": HOVER_MORPH_KERNEL,
        "sobel_kernel": HOVER_SOBEL_KERNEL,
    }
