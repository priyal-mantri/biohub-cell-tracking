"""
PyTorch Unit Tests for the 3D Cell-Center Detector Stage.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import torch

from src.config import DatasetConfig, SpatialResolution
from src.data import ZarrLoader, GeffLoader, CellNode, CoordinateTransformer
from src.detection import (
    UNet3D,
    TargetGenerator,
    GaussianHeatmapConfig,
    Frame3DDataset,
    PeakExtractor,
    PeakConfig,
    DetectionEvaluator,
    DetectionCandidate,
)


@pytest.fixture
def dataset_config():
    return DatasetConfig.default_local_sample()


@pytest.fixture
def target_generator():
    return TargetGenerator(config=GaussianHeatmapConfig(sigma_physical_um=2.0))


@pytest.fixture
def sample_nodes():
    transformer = CoordinateTransformer()
    # Create two test cell nodes
    n1 = CellNode.create(node_id=1, frame=0, voxel_pos=(10, 50, 60), transformer=transformer)
    n2 = CellNode.create(node_id=2, frame=0, voxel_pos=(30, 100, 120), transformer=transformer)
    return [n1, n2]


def test_target_generation_anisotropic_geometry(target_generator, sample_nodes):
    grid_shape = (64, 128, 128)
    heatmap = target_generator.generate_heatmap_for_frame(sample_nodes, grid_shape)

    assert heatmap.shape == grid_shape
    assert heatmap.dtype == np.float32

    # Peak value at node centers should be ~1.0
    val_n1 = heatmap[10, 50, 60]
    val_n2 = heatmap[30, 100, 120]
    assert np.isclose(val_n1, 1.0, atol=1e-3)
    assert np.isclose(val_n2, 1.0, atol=1e-3)

    # Background should be 0.0
    assert heatmap[0, 0, 0] == 0.0


def test_pytorch_dataset_output(dataset_config):
    zarr_loader = ZarrLoader(dataset_config.zarr_path)
    geff_loader = GeffLoader(dataset_config.geff_path)

    crop_bounds = ((0, 32), (0, 64), (0, 64))
    dataset = Frame3DDataset(
        zarr_loader=zarr_loader,
        geff_loader=geff_loader,
        frame_indices=[0, 1],
        crop_bounds=crop_bounds
    )

    assert len(dataset) == 2
    img_tensor, target_tensor, frame_t = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(target_tensor, torch.Tensor)
    assert img_tensor.shape == (1, 32, 64, 64)
    assert target_tensor.shape == (1, 32, 64, 64)
    assert img_tensor.dtype == torch.float32
    assert target_tensor.dtype == torch.float32
    assert frame_t == 0


def test_unet3d_forward_pass_and_shape():
    model = UNet3D(in_channels=1, out_channels=1, base_features=8)
    # Test tensor of shape (Batch=1, Channels=1, Z=16, Y=32, X=32)
    dummy_input = torch.randn(1, 1, 16, 32, 32)
    output = model(dummy_input)

    assert output.shape == (1, 1, 16, 32, 32)
    assert output.dtype == torch.float32
    # Sigmoid output must be in [0, 1]
    assert (output >= 0.0).all() and (output <= 1.0).all()


def test_peak_extraction_and_anisotropic_nms():
    extractor = PeakExtractor(config=PeakConfig(min_confidence=0.5, min_physical_distance_um=3.0))

    # Create dummy 3D heatmap with two distinct peaks
    heatmap = np.zeros((32, 64, 64), dtype=np.float32)
    heatmap[10, 20, 20] = 0.9  # Peak 1
    heatmap[25, 40, 40] = 0.8  # Peak 2

    # Add a nearby lower peak that should be suppressed by NMS
    heatmap[10, 21, 21] = 0.7

    candidates = extractor.extract_peaks_from_heatmap(heatmap, frame_idx=0)

    # Should detect 2 peaks after NMS
    assert len(candidates) == 2
    assert np.isclose(candidates[0].confidence, 0.9)
    assert candidates[0].voxel_pos == (10, 20, 20)
    assert np.isclose(candidates[1].confidence, 0.8)
    assert candidates[1].voxel_pos == (25, 40, 40)


def test_voxel_to_physical_conversion_in_candidates():
    transformer = CoordinateTransformer()
    cand = DetectionCandidate(
        candidate_id=1,
        frame=0,
        voxel_pos=(10, 20, 30),
        physical_pos=transformer.voxel_to_physical((10, 20, 30)),
        confidence=0.95
    )

    assert cand.physical_pos == (16.25, 8.125, 12.1875)
    node = cand.to_cell_node()
    assert isinstance(node, CellNode)
    assert node.physical_pos == (16.25, 8.125, 12.1875)


def test_detection_evaluator_matching():
    evaluator = DetectionEvaluator(max_matching_distance_um=5.0)
    transformer = CoordinateTransformer()

    # Ground truth nodes
    gt1 = CellNode.create(1, 0, (10, 20, 20), transformer)
    gt2 = CellNode.create(2, 0, (30, 40, 40), transformer)
    gt_nodes = [gt1, gt2]

    # Predictions: 1 exact match, 1 close match, 1 false positive
    pred1 = DetectionCandidate(101, 0, (10, 20, 20), transformer.voxel_to_physical((10, 20, 20)), 0.9)
    pred2 = DetectionCandidate(102, 0, (30, 41, 40), transformer.voxel_to_physical((30, 41, 40)), 0.85)
    pred3 = DetectionCandidate(103, 0, (50, 50, 50), transformer.voxel_to_physical((50, 50, 50)), 0.7)

    metrics = evaluator.evaluate_frame(gt_nodes, [pred1, pred2, pred3])

    assert metrics.num_gt_cells == 2
    assert metrics.num_pred_candidates == 3
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert np.isclose(metrics.recall, 1.0)
    assert np.isclose(metrics.precision, 2.0 / 3.0)


def test_tiny_end_to_end_detector_pass(dataset_config):
    zarr_loader = ZarrLoader(dataset_config.zarr_path)
    geff_loader = GeffLoader(dataset_config.geff_path)

    crop_bounds = ((0, 16), (0, 32), (0, 32))
    dataset = Frame3DDataset(
        zarr_loader=zarr_loader,
        geff_loader=geff_loader,
        frame_indices=[0],
        crop_bounds=crop_bounds
    )

    model = UNet3D(in_channels=1, out_channels=1, base_features=8)
    model.eval()

    img_tensor, target_tensor, frame_t = dataset[0]
    with torch.no_grad():
        pred_tensor = model(img_tensor.unsqueeze(0)).squeeze(0).squeeze(0)

    assert pred_tensor.shape == (16, 32, 32)
    heatmap_np = pred_tensor.numpy()

    extractor = PeakExtractor(config=PeakConfig(min_confidence=0.01))
    candidates = extractor.extract_peaks_from_heatmap(heatmap_np, frame_t)
    assert isinstance(candidates, list)
