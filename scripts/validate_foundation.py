#!/usr/bin/env python3
"""
Validation script for BioHub Cell Tracking Foundation Layer.

Confirms:
1. Zarr volume opening and metadata verification.
2. GEFF tracking graph opening and metadata verification.
3. Expected shape [100, 64, 256, 256] and dtype uint16.
4. Expected node (861) and edge (845) counts.
5. Coordinate system transformations (voxel <-> physical).
6. Temporal range [0, 99] and spatial bounds.
7. Edge connectivity validity.
8. NetworkX temporal graph representation for STGNN / candidate generation compatibility.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DatasetConfig, SpatialResolution
from src.data import ZarrLoader, GeffLoader, CoordinateTransformer, physical_to_voxel, voxel_to_physical


def run_validation():
    print("==================================================")
    print("   BioHub Cell Tracking - Foundation Validation   ")
    print("==================================================")

    # 1. Configuration Check
    config = DatasetConfig.default_local_sample()
    print(f"\n[1/8] Configuration:")
    print(f"  Zarr Path: {config.zarr_path}")
    print(f"  GEFF Path: {config.geff_path}")
    print(f"  Z-spacing: {config.resolution.z_spacing} um/voxel")
    print(f"  Y-spacing: {config.resolution.y_spacing} um/voxel")
    print(f"  X-spacing: {config.resolution.x_spacing} um/voxel")
    print(f"  Anisotropy Ratio (Z / XY): {config.resolution.anisotropy_ratio_z_to_xy:.2f}")

    assert config.resolution.z_spacing == 1.625
    assert config.resolution.y_spacing == 0.40625
    assert config.resolution.x_spacing == 0.40625

    # 2. Zarr Loader Verification
    print(f"\n[2/8] Testing Zarr Loader (Lazy Access):")
    zarr_loader = ZarrLoader(config.zarr_path)
    shape = zarr_loader.shape
    dtype = zarr_loader.dtype
    chunks = zarr_loader.chunks

    print(f"  Shape (T, Z, Y, X): {shape}")
    print(f"  Data Type: {dtype}")
    print(f"  Chunk Grid: {chunks}")

    assert shape == (100, 64, 256, 256), f"Expected (100, 64, 256, 256), got {shape}"
    assert str(dtype) == "uint16", f"Expected uint16, got {dtype}"

    # Test lazy single-frame load
    frame_0 = zarr_loader.get_frame(0)
    print(f"  Frame 0 shape: {frame_0.shape}, dtype: {frame_0.dtype}, min: {frame_0.min()}, max: {frame_0.max()}")
    assert frame_0.shape == (64, 256, 256)

    # Test subvolume crop
    crop = zarr_loader.get_crop(t=0, z_range=(10, 20), y_range=(50, 100), x_range=(50, 100))
    print(f"  Subvolume crop shape: {crop.shape}")
    assert crop.shape == (10, 50, 50)

    # 3. GEFF Loader Verification
    print(f"\n[3/8] Testing GEFF Loader:")
    geff_loader = GeffLoader(config.geff_path)
    num_nodes = geff_loader.num_nodes
    num_edges = geff_loader.num_edges
    t_min, t_max = geff_loader.temporal_range

    print(f"  Total Nodes: {num_nodes}")
    print(f"  Total Edges: {num_edges}")
    print(f"  Temporal Frame Range: [{t_min}, {t_max}]")

    assert num_nodes == 861, f"Expected 861 nodes, got {num_nodes}"
    assert num_edges == 845, f"Expected 845 edges, got {num_edges}"
    assert t_min == 0 and t_max == 99, f"Expected [0, 99], got [{t_min}, {t_max}]"

    # 4. Node Coordinates Verification
    print(f"\n[4/8] Testing Node Coordinates & Spatial Bounds:")
    all_nodes = geff_loader.get_all_nodes()
    z_coords = [n.voxel_pos[0] for n in all_nodes]
    y_coords = [n.voxel_pos[1] for n in all_nodes]
    x_coords = [n.voxel_pos[2] for n in all_nodes]

    print(f"  Z Voxel Range: [{min(z_coords)}, {max(z_coords)}]")
    print(f"  Y Voxel Range: [{min(y_coords)}, {max(y_coords)}]")
    print(f"  X Voxel Range: [{min(x_coords)}, {max(x_coords)}]")

    assert 0 <= min(z_coords) <= max(z_coords) < 64
    assert 0 <= min(y_coords) <= max(y_coords) < 256
    assert 0 <= min(x_coords) <= max(x_coords) < 256

    # 5. Coordinate Transformation & Roundtrip Verification
    print(f"\n[5/8] Testing Coordinate System Conversions:")
    sample_voxel = (30, 100, 150)
    sample_phys = voxel_to_physical(sample_voxel)
    sample_roundtrip = physical_to_voxel(sample_phys)

    print(f"  Voxel {sample_voxel} -> Physical {sample_phys} um")
    print(f"  Physical {sample_phys} um -> Voxel Roundtrip {sample_roundtrip}")

    assert sample_phys == (30 * 1.625, 100 * 0.40625, 150 * 0.40625)
    assert sample_roundtrip == sample_voxel

    # 6. Edge Integrity Verification
    print(f"\n[6/8] Testing Edge Integrity:")
    all_node_ids = set(geff_loader.node_ids)
    invalid_edges = 0
    non_forward_temporal_edges = 0

    for u, v in geff_loader.directed_edges:
        u_id, v_id = int(u), int(v)
        if u_id not in all_node_ids or v_id not in all_node_ids:
            invalid_edges += 1
            continue

        n_u = geff_loader.get_node_by_id(u_id)
        n_v = geff_loader.get_node_by_id(v_id)

        if n_v.frame <= n_u.frame:
            non_forward_temporal_edges += 1

    print(f"  Invalid Edge Node References: {invalid_edges}")
    print(f"  Non-forward Temporal Edges: {non_forward_temporal_edges}")

    assert invalid_edges == 0, "Found edges referring to non-existent node IDs!"
    assert non_forward_temporal_edges == 0, "Found edges pointing backward in time!"

    # 7. NetworkX Temporal Graph Representation
    print(f"\n[7/8] Testing NetworkX Temporal Graph Conversion:")
    nx_graph = geff_loader.to_networkx_graph()
    print(f"  NetworkX Graph Nodes: {nx_graph.number_of_nodes()}")
    print(f"  NetworkX Graph Edges: {nx_graph.number_of_edges()}")

    sample_node_id = int(geff_loader.node_ids[0])
    node_attrs = nx_graph.nodes[sample_node_id]
    print(f"  Sample Node {sample_node_id} Attributes: {node_attrs}")

    assert nx_graph.number_of_nodes() == 861
    assert nx_graph.number_of_edges() == 845
    assert "voxel_pos" in node_attrs and "physical_pos" in node_attrs

    # 8. Frame Node Distribution Verification
    print(f"\n[8/8] Testing Node Distribution Across Frames:")
    nodes_f0 = geff_loader.get_nodes_by_frame(0)
    nodes_f50 = geff_loader.get_nodes_by_frame(50)
    nodes_f99 = geff_loader.get_nodes_by_frame(99)

    print(f"  Frame 0 Cell Count: {len(nodes_f0)}")
    print(f"  Frame 50 Cell Count: {len(nodes_f50)}")
    print(f"  Frame 99 Cell Count: {len(nodes_f99)}")

    assert len(nodes_f0) == 6
    assert len(nodes_f50) == 8
    assert len(nodes_f99) == 7

    print("\n==================================================")
    print("   ALL FOUNDATION VALIDATION TESTS PASSED 100%!   ")
    print("==================================================")


if __name__ == "__main__":
    run_validation()
