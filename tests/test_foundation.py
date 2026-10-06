"""
Pytest unit tests for the BioHub Cell Tracking foundation modules.
"""

import sys
from pathlib import Path

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import networkx as nx

from src.config import DatasetConfig, SpatialResolution
from src.data import (
    CoordinateTransformer,
    voxel_to_physical,
    physical_to_voxel,
    CellNode,
    ZarrLoader,
    GeffLoader,
)


@pytest.fixture
def dataset_config():
    return DatasetConfig.default_local_sample()


@pytest.fixture
def spatial_resolution():
    return SpatialResolution()


@pytest.fixture
def coordinate_transformer(spatial_resolution):
    return CoordinateTransformer(spatial_resolution)


def test_spatial_resolution(spatial_resolution):
    assert spatial_resolution.z_spacing == 1.625
    assert spatial_resolution.y_spacing == 0.40625
    assert spatial_resolution.x_spacing == 0.40625
    assert spatial_resolution.anisotropy_ratio_z_to_xy == 4.0
    assert spatial_resolution.scale_tuple_zyx == (1.625, 0.40625, 0.40625)


def test_coordinate_conversion(coordinate_transformer):
    voxel = (10, 20, 30)
    phys = coordinate_transformer.voxel_to_physical(voxel)
    assert phys == (16.25, 8.125, 12.1875)

    rt_voxel = coordinate_transformer.physical_to_voxel(phys)
    assert rt_voxel == voxel


def test_cell_node_creation(coordinate_transformer):
    node = CellNode.create(
        node_id=101,
        frame=5,
        voxel_pos=(10, 20, 30),
        transformer=coordinate_transformer
    )
    assert node.node_id == 101
    assert node.frame == 5
    assert node.voxel_pos == (10, 20, 30)
    assert node.physical_pos == (16.25, 8.125, 12.1875)


def test_zarr_loader(dataset_config):
    loader = ZarrLoader(dataset_config.zarr_path)
    assert loader.shape == (100, 64, 256, 256)
    assert str(loader.dtype) == "uint16"

    # Test single frame lazy slice
    frame_0 = loader.get_frame(0)
    assert frame_0.shape == (64, 256, 256)
    assert frame_0.dtype == np.uint16

    # Test crop slice
    crop = loader.get_crop(0, (0, 5), (10, 20), (30, 40))
    assert crop.shape == (5, 10, 10)


def test_geff_loader(dataset_config):
    loader = GeffLoader(dataset_config.geff_path)
    assert loader.num_nodes == 861
    assert loader.num_edges == 845
    assert loader.temporal_range == (0, 99)

    # Test frame node lookup
    nodes_f0 = loader.get_nodes_by_frame(0)
    assert len(nodes_f0) == 6
    for node in nodes_f0:
        assert node.frame == 0
        assert 0 <= node.voxel_pos[0] < 64
        assert 0 <= node.voxel_pos[1] < 256
        assert 0 <= node.voxel_pos[2] < 256


def test_geff_to_networkx(dataset_config):
    loader = GeffLoader(dataset_config.geff_path)
    graph = loader.to_networkx_graph()
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 861
    assert graph.number_of_edges() == 845

    # Check sample node attributes
    sample_id = int(loader.node_ids[0])
    assert "frame" in graph.nodes[sample_id]
    assert "voxel_pos" in graph.nodes[sample_id]
    assert "physical_pos" in graph.nodes[sample_id]
