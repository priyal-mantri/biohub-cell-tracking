"""
PyTorch and Spatial Unit Tests for 3D Regional Processing, Guard Zones, and Duplicate Reconciliation.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np

from src.config import SpatialResolution
from src.data import CoordinateTransformer
from src.regional import (
    BoundingBox3D,
    Region3D,
    RegionalCandidate,
    RegionPartitioner3D,
    PartitionConfig,
    CandidateReconciler,
    ReconciliationConfig,
    RegionalInferenceEngine,
)


@pytest.fixture
def volume_shape():
    return (64, 256, 256)


@pytest.fixture
def partitioner(volume_shape):
    config = PartitionConfig(core_shape=(32, 128, 128), guard_size=(8, 32, 32))
    return RegionPartitioner3D(volume_shape=volume_shape, config=config)


@pytest.fixture
def regions(partitioner):
    return partitioner.generate_regions()


def test_region_generation_count_and_types(regions):
    # (64/32) * (256/128) * (256/128) = 2 * 2 * 2 = 8 regions
    assert len(regions) == 8
    assert isinstance(regions[0], Region3D)


def test_core_and_guard_geometry(regions):
    r1 = regions[0]
    # Region 1 core: [0, 32), [0, 128), [0, 128)
    assert r1.core_bbox.slice_tuples == ((0, 32), (0, 128), (0, 128))
    # Region 1 guard (clipped at boundaries 0): [0, 40), [0, 160), [0, 160)
    assert r1.guard_bbox.slice_tuples == ((0, 40), (0, 160), (0, 160))


def test_boundary_clipping(partitioner):
    # Test volume where cores don't divide perfectly or sit at maximum boundaries
    custom_part = RegionPartitioner3D(volume_shape=(50, 100, 100), config=PartitionConfig((30, 60, 60), (10, 20, 20)))
    regs = custom_part.generate_regions()
    last_reg = regs[-1]

    # Guard bounds must be clipped to [0, 50), [0, 100), [0, 100)
    assert last_reg.guard_bbox.z_max <= 50
    assert last_reg.guard_bbox.y_max <= 100
    assert last_reg.guard_bbox.x_max <= 100


def test_complete_core_coverage(partitioner, regions):
    assert partitioner.verify_complete_core_coverage(regions) is True


def test_no_overlapping_authoritative_cores(partitioner, regions):
    assert partitioner.verify_no_overlapping_cores(regions) is True


def test_local_to_global_coordinate_conversion(regions):
    r = regions[7]  # Core at [32, 64), [128, 256), [128, 256)
    # Guard origin is [32-8=24, 128-32=96, 128-32=96]
    local_pos = (10, 40, 40)
    global_pos = r.local_to_global_voxel(local_pos)

    assert global_pos == (10 + 24, 40 + 96, 40 + 96)


def test_global_to_local_coordinate_conversion(regions):
    r = regions[7]
    global_pos = (34, 136, 136)
    local_pos = r.global_to_local_voxel(global_pos)

    assert local_pos == (34 - 24, 136 - 96, 136 - 96)


def test_core_membership_check(regions):
    r1 = regions[0]  # Core: [0, 32), [0, 128), [0, 128)
    assert r1.is_global_voxel_in_core((15, 50, 50)) is True
    assert r1.is_global_voxel_in_core((35, 50, 50)) is False


def test_guard_membership_check(regions):
    r1 = regions[0]  # Guard: [0, 40), [0, 160), [0, 160)
    # Voxel at (35, 140, 140) is outside core, but inside guard!
    assert r1.is_global_voxel_in_core((35, 140, 140)) is False
    assert r1.is_global_voxel_in_guard((35, 140, 140)) is True


def test_duplicate_candidate_reconciliation():
    reconciler = CandidateReconciler(config=ReconciliationConfig(reconciliation_distance_um=4.0))
    transformer = CoordinateTransformer()

    # Create two duplicate candidates from region 1 and region 2 observing same cell
    phys1 = (16.25, 40.625, 40.625)
    phys2 = (16.25, 41.000, 40.625)  # Distance = 0.375 um (< 4.0 um)

    c1 = RegionalCandidate(1, 1, 0, (10, 50, 50), (10, 100, 100), phys1, 0.90, True)
    c2 = RegionalCandidate(2, 2, 0, (10, 10, 10), (10, 101, 100), phys2, 0.85, False)

    reconciled = reconciler.reconcile_candidates([c1, c2])

    # Should merge 2 candidates into 1 reconciled candidate
    assert len(reconciled) == 1
    rec = reconciled[0]
    assert rec.confidence == 0.90
    assert rec.provenance["is_multi_region_detection"] is True
    assert sorted(rec.provenance["contributing_region_ids"]) == [1, 2]


def test_physical_distance_matching_under_anisotropy():
    transformer = CoordinateTransformer()
    # Z spacing = 1.625, Y spacing = 0.40625
    # 2 voxels dz = 3.25 um
    # 8 voxels dy = 3.25 um
    v1 = (10, 20, 20)
    v2 = (12, 20, 20)
    v3 = (10, 28, 20)

    d_z = transformer.anisotropic_voxel_distance(v1, v2)
    d_y = transformer.anisotropic_voxel_distance(v1, v3)

    assert np.isclose(d_z, 3.25)
    assert np.isclose(d_y, 3.25)


def test_regional_inference_on_tiny_synthetic_volume():
    class DummyDetector:
        def predict_heatmap(self, volume_3d):
            # Return dummy heatmap with a peak at center
            hm = np.zeros(volume_3d.shape, dtype=np.float32)
            z_c, y_c, x_c = volume_3d.shape[0] // 2, volume_3d.shape[1] // 2, volume_3d.shape[2] // 2
            hm[z_c, y_c, x_c] = 0.95
            return hm

    class DummyZarrLoader:
        shape = (10, 32, 64, 64)
        dtype = np.uint16
        def get_crop(self, t, z_range, y_range, x_range):
            dz = z_range[1] - z_range[0]
            dy = y_range[1] - y_range[0]
            dx = x_range[1] - x_range[0]
            return np.ones((dz, dy, dx), dtype=np.uint16) * 100

    engine = RegionalInferenceEngine(
        detector=DummyDetector(),
        partition_config=PartitionConfig(core_shape=(16, 32, 32), guard_size=(4, 8, 8))
    )

    cands, meta = engine.detect_frame_regional(DummyZarrLoader(), frame_idx=0)

    assert isinstance(cands, list)
    assert meta["num_regions"] == 8
