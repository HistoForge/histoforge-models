"""Cellpose wrapper area settings and real CPU decoding regressions."""

import numpy as np
import pytest
import weakref

from histoforge_models import dynamics
from histoforge_models.pathology import flow_inference


@pytest.mark.parametrize(
    "options, expected_fraction",
    [({}, 0.4), ({"max_size_fraction": 0.4}, 0.4), ({"max_size_fraction": 1.0}, 1.0)],
)
def test_flow_wrapper_forwards_default_or_explicit_area_cutoff(
    monkeypatch, options, expected_fraction
):
    image = np.zeros((9, 9, 3), dtype=np.float32)
    monkeypatch.setattr(flow_inference, "run_model_fields", lambda *_, **__: image)
    received = []

    def decode(_flows, probability, **kwargs):
        received.append(kwargs["max_size_fraction"])
        return np.zeros(probability.shape, dtype=np.uint16)

    monkeypatch.setattr(dynamics, "compute_masks", decode)

    flow_inference.run_flow_inference(None, image, device="cpu", **options)

    assert received == [expected_fraction]


@pytest.mark.parametrize("radius", [12, 26, 64])
@pytest.mark.parametrize("options", [{}, {"max_size_fraction": 1.0}])
def test_real_cellpose_decode_retains_large_masks_only_with_explicit_override(
    monkeypatch, radius, options
):
    """Analytic inward flows decode small, majority-area and full-raster masks."""
    rows, columns = np.mgrid[:64, :64]
    foreground = (rows - 32) ** 2 + (columns - 32) ** 2 <= radius**2
    flows = np.stack((32 - rows, 32 - columns)).astype(np.float32)
    flows *= 5.0 / np.maximum(np.sqrt(np.sum(flows**2, axis=0)), 1.0)
    probability = np.where(foreground, 1.0, -1.0).astype(np.float32)
    raw = np.dstack((flows.transpose(1, 2, 0), probability))
    monkeypatch.setattr(flow_inference, "run_model_fields", lambda *_, **__: raw)

    result = flow_inference.run_flow_inference(
        None,
        np.zeros((64, 64, 3), dtype=np.uint8),
        device="cpu",
        # Isolate the relative-area filter from flow QC and minimum-area removal.
        flow_threshold=0,
        min_instance_pixels=0,
        **options,
    )

    expected = foreground if options or radius == 12 else np.zeros_like(foreground)
    np.testing.assert_array_equal(result["instance_mask"], expected.astype(np.int32))
    np.testing.assert_array_equal(result["dP"], flows)
    np.testing.assert_array_equal(result["cellprob"], probability)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compact_flow_preserves_class_ties_and_releases_dense_maps(monkeypatch, dtype):
    raw = np.zeros((8, 8, 7), dtype=dtype)
    raw[..., 0] = 1
    raw[:4, :4, 1] = 1  # Tie still selects background channel zero.
    raw[:4, 4:, 1] = 1 + 1e-12
    raw[4:, :4, 2] = 2
    raw[4:, 4:, 3] = 2  # Training-only winners still participate in votes.
    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[:4, :4], mask[:4, 4:], mask[4:, :4], mask[4:, 4:] = 1, 2, 3, 4
    arrays = []

    def predict(*args, **kwargs):
        output = raw.copy()
        arrays.append(weakref.ref(output))
        return output

    monkeypatch.setattr(flow_inference, "run_model_fields", predict)
    monkeypatch.setattr(dynamics, "compute_masks", lambda *a, **k: mask.copy())
    result = flow_inference.run_flow_inference(
        None,
        np.zeros((8, 8, 3), dtype=np.uint8),
        num_classes=4,
        min_instance_pixels=0,
        device="cpu",
        include_diagnostics=False,
    )
    assert all(array() is None for array in arrays)
    assert result.keys() == {"instance_mask", "votes_per_instance"}
    np.testing.assert_array_equal(result["instance_mask"], mask)
    assert result["votes_per_instance"] == {
        1: 0,
        2: int(dtype == np.float64),
        3: 2,
        4: 3,
    }
