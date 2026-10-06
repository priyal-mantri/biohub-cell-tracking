#!/usr/bin/env python3
"""
Controlled Overfit Experiment and Verification Script for 3D Cell-Center Detector Stage.
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
from src.data import ZarrLoader, GeffLoader
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


def run_experiment():
    print("==================================================")
    print(" BioHub Cell Tracking - 3D Detector Experiment ")
    print("==================================================")
    sys.stdout.flush()

    # 1. Dataset setup
    config = DatasetConfig.default_local_sample()
    zarr_loader = ZarrLoader(config.zarr_path)
    geff_loader = GeffLoader(config.geff_path)

    print(f"\n[1/5] Initializing Loaders & Datasets:")
    print(f"  Microscopy Volume Shape: {zarr_loader.shape}, dtype: {zarr_loader.dtype}")
    print(f"  Ground-Truth Nodes: {geff_loader.num_nodes}, Edges: {geff_loader.num_edges}")
    sys.stdout.flush()

    target_gen = TargetGenerator(config=GaussianHeatmapConfig(sigma_physical_um=2.0))
    # Using spatial crop (64, 192, 192) for ultra-fast local CPU development
    crop_bounds = ((0, 64), (16, 208), (16, 208))
    dataset = Frame3DDataset(
        zarr_loader=zarr_loader,
        geff_loader=geff_loader,
        target_generator=target_gen,
        crop_bounds=crop_bounds
    )

    # 2. Model Initialization (base_features=8 for fast CPU development)
    print(f"\n[2/5] Constructing Compact 3D U-Net:")
    model = UNet3D(in_channels=1, out_channels=1, base_features=8)
    param_count = model.count_parameters()
    print(f"  Model Parameter Count: {param_count:,} parameters")
    sys.stdout.flush()

    # 3. Peak Extractor & Evaluator setup
    peak_config = PeakConfig(min_confidence=0.08, min_physical_distance_um=3.0)
    peak_extractor = PeakExtractor(config=peak_config)
    evaluator = DetectionEvaluator(max_matching_distance_um=5.0)

    # 4. Controlled Overfit Test (10 epochs, eval every 2)
    trainer_config = DetectorTrainingConfig(epochs=10, learning_rate=5e-3)
    trainer = DetectorTrainer(model=model, config=trainer_config)

    overfit_frames = [0, 1, 2]
    history = trainer.run_controlled_overfit_test(
        full_dataset=dataset,
        overfit_frames=overfit_frames,
        epochs=10,
        eval_interval=2,
        peak_extractor=peak_extractor,
        evaluator=evaluator
    )

    # 5. Verification Assertions
    initial_loss = history["loss"][0]
    final_loss = history["loss"][-1]

    print("\n[5/5] Experiment Summary & Verification:")
    print(f"  Initial Epoch Loss: {initial_loss:.6f}")
    print(f"  Final Epoch Loss:   {final_loss:.6f}")
    print(f"  Loss Reduction:     {(1 - final_loss / initial_loss) * 100:.2f}%")
    if "precision" in history and history["precision"]:
        print(f"  Final Precision:    {history['precision'][-1]:.4f}")
        print(f"  Final Recall:       {history['recall'][-1]:.4f}")
        print(f"  Final Loc Error:    {history['mean_loc_error_um'][-1]:.3f} um")
    sys.stdout.flush()

    assert final_loss < initial_loss, "Loss did not decrease during overfit test!"

    print("\n==================================================")
    print("  3D DETECTOR EXPERIMENT PASSED SUCCESSFULLY! ")
    print("==================================================")
    sys.stdout.flush()


if __name__ == "__main__":
    run_experiment()
