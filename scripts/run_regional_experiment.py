#!/usr/bin/env python3
"""
Controlled Regional Processing Experiment & Full-Frame vs. Regional Comparison Script.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DatasetConfig
from src.data import ZarrLoader, GeffLoader, CoordinateTransformer
from src.detection import (
    UNet3D,
    TargetGenerator,
    GaussianHeatmapConfig,
    Frame3DDataset,
    PeakExtractor,
    PeakConfig,
    DetectionEvaluator,
    DetectorTrainer,
    DetectorTrainingConfig,
)
from src.regional import (
    RegionPartitioner3D,
    PartitionConfig,
    CandidateReconciler,
    ReconciliationConfig,
    RegionalInferenceEngine,
    RegionalBoundaryAnalyzer,
)


class UNetDetectorAdapter:
    """Adapter wrapping trained PyTorch UNet3D into BaseDetector interface."""

    def __init__(self, model: UNet3D, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict_heatmap(self, volume_3d: np.ndarray) -> np.ndarray:
        tensor_in = torch.from_numpy(volume_3d).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_tensor = self.model(tensor_in).squeeze(0).squeeze(0)
        return pred_tensor.cpu().numpy()

    def detect_candidates(self, volume_3d: np.ndarray, frame_idx: int, transformer=None):
        heatmap = self.predict_heatmap(volume_3d)
        extractor = PeakExtractor(resolution=transformer.resolution if transformer else None)
        return extractor.extract_peaks_from_heatmap(heatmap, frame_idx)


def run_experiment():
    print("==================================================", flush=True)
    print(" BioHub Cell Tracking - Regional Processing Expt ", flush=True)
    print("==================================================", flush=True)

    # 1. Dataset & Loaders setup
    config = DatasetConfig.default_local_sample()
    zarr_loader = ZarrLoader(config.zarr_path)
    geff_loader = GeffLoader(config.geff_path)

    print(f"\n[1/6] Initializing Loaders:", flush=True)
    print(f"  Microscopy Volume Shape: {zarr_loader.shape}, dtype: {zarr_loader.dtype}", flush=True)
    print(f"  Ground-Truth Nodes: {geff_loader.num_nodes}, Edges: {geff_loader.num_edges}", flush=True)

    target_gen = TargetGenerator(config=GaussianHeatmapConfig(sigma_physical_um=2.0))
    # Spatial crop (64, 192, 192) for fast training
    crop_bounds = ((0, 64), (16, 208), (16, 208))
    dataset = Frame3DDataset(
        zarr_loader=zarr_loader,
        geff_loader=geff_loader,
        target_generator=target_gen,
        crop_bounds=crop_bounds
    )

    # 2. Train detector briefly on frames [0, 1, 2] to get an active trained model
    print(f"\n[2/6] Preparing 3D UNet Detector:", flush=True)
    model = UNet3D(in_channels=1, out_channels=1, base_features=8)
    trainer = DetectorTrainer(model=model, config=DetectorTrainingConfig(epochs=12, learning_rate=5e-3))
    trainer.run_controlled_overfit_test(full_dataset=dataset, overfit_frames=[0, 1, 2], epochs=12, eval_interval=4)

    detector_adapter = UNetDetectorAdapter(model=model)

    # 3. Regional Partitioning Setup
    volume_shape = zarr_loader.shape[1:]  # (64, 256, 256)
    partition_config = PartitionConfig(core_shape=(32, 128, 128), guard_size=(8, 32, 32))
    partitioner = RegionPartitioner3D(volume_shape=volume_shape, config=partition_config)
    regions = partitioner.generate_regions()

    print(f"\n[3/6] Regional Partitioning Configuration:", flush=True)
    print(f"  Volume Shape: {volume_shape}", flush=True)
    print(f"  Core Shape:   {partition_config.core_shape}", flush=True)
    print(f"  Guard Size:   {partition_config.guard_size}", flush=True)
    print(f"  Total Generated Regions: {len(regions)}", flush=True)

    # Verify partitioner guarantees
    assert partitioner.verify_complete_core_coverage(regions), "Core coverage incomplete!"
    assert partitioner.verify_no_overlapping_cores(regions), "Found overlapping cores!"
    print("  [OK] Complete core coverage and zero core overlap verified 100%.", flush=True)

    # 4. Run MODE A (Full-Frame Detection) vs MODE B (Regional Detection)
    frame_idx = 0
    gt_nodes = geff_loader.get_nodes_by_frame(frame_idx)
    raw_img = zarr_loader.get_frame(frame_idx)
    norm_img = Frame3DDataset(zarr_loader, geff_loader)._normalize_image(raw_img)

    print(f"\n[4/6] Running Detector Comparisons on Frame {frame_idx}:", flush=True)

    # MODE A: Full-frame detector inference
    ff_heatmap = detector_adapter.predict_heatmap(norm_img)
    peak_config = PeakConfig(min_confidence=0.01, min_physical_distance_um=3.0)
    peak_extractor = PeakExtractor(config=peak_config)
    fullframe_candidates = peak_extractor.extract_peaks_from_heatmap(ff_heatmap, frame_idx)

    print(f"  MODE A (Full-Frame): {len(fullframe_candidates)} candidates extracted.", flush=True)

    # MODE B: Regional detector inference with guards + reconciliation
    reconcil_config = ReconciliationConfig(reconciliation_distance_um=4.0)
    regional_engine = RegionalInferenceEngine(
        detector=detector_adapter,
        partition_config=partition_config,
        peak_config=peak_config,
        reconciliation_config=reconcil_config
    )

    regional_candidates, regional_meta = regional_engine.detect_frame_regional(
        zarr_loader=zarr_loader,
        frame_idx=frame_idx,
        filter_core_only=True
    )

    print(f"  MODE B (Regional):", flush=True)
    print(f"    Raw Peaks Extracted across all regions: {regional_meta['total_raw_peaks_extracted']}", flush=True)
    print(f"    Raw Core Candidates:  {regional_meta['num_raw_core_candidates']}", flush=True)
    print(f"    Raw Guard Candidates: {regional_meta['num_raw_guard_candidates']}", flush=True)
    print(f"    Authoritative Core Candidates (Before Reconciliation): {regional_meta['num_authoritative_before_reconciliation']}", flush=True)
    print(f"    Final Reconciled Candidates: {regional_meta['num_authoritative_after_reconciliation']}", flush=True)
    print(f"    Multi-Region Duplicates Reconciled: {regional_meta['num_multi_region_duplicates_reconciled']}", flush=True)

    # 5. Boundary Diagnostics & Comparison
    print(f"\n[5/6] Boundary Diagnostics:", flush=True)
    analyzer = RegionalBoundaryAnalyzer()
    diag = analyzer.analyze_frame(
        frame_idx=frame_idx,
        gt_nodes=gt_nodes,
        fullframe_candidates=fullframe_candidates,
        regional_candidates=regional_candidates,
        regional_metadata=regional_meta,
        partitioner=partitioner
    )

    print(diag.summary_string(), flush=True)

    # 6. Verification Assertions
    print(f"\n[6/6] Architectural Assertions:", flush=True)
    assert len(regions) == 8, f"Expected 8 regions for (64,256,256) volume, got {len(regions)}"
    assert regional_meta['num_raw_core_candidates'] == regional_meta['num_authoritative_before_reconciliation']
    assert len(regional_candidates) > 0, "No regional candidates produced!"

    print("\n==================================================", flush=True)
    print(" REGIONAL PROCESSING EXPERIMENT PASSED 100%! ", flush=True)
    print("==================================================", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    run_experiment()
