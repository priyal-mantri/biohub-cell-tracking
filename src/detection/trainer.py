"""
PyTorch Training and Controlled Overfit Evaluation Engine for 3D Cell Detectors.
"""

import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.detection.unet3d import UNet3D
from src.detection.dataset import Frame3DDataset
from src.detection.peaks import PeakExtractor
from src.detection.evaluation import DetectionEvaluator, DetectionMetrics


@dataclass
class DetectorTrainingConfig:
    """
    Explicit training hyperparameter configuration.
    """
    epochs: int = 6
    learning_rate: float = 5e-3
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    weight_decay: float = 1e-5


class DetectorTrainer:
    """
    Handles model optimization, training loops, and controlled overfit testing.
    """

    def __init__(
        self,
        model: UNet3D,
        config: DetectorTrainingConfig = DetectorTrainingConfig()
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0

        for images, targets, _ in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(dataloader.dataset)

    def run_controlled_overfit_test(
        self,
        full_dataset: Frame3DDataset,
        overfit_frames: List[int] = [0, 1, 2],
        epochs: int = 6,
        eval_interval: int = 2,
        peak_extractor: Optional[PeakExtractor] = None,
        evaluator: Optional[DetectionEvaluator] = None
    ) -> Dict[str, Any]:
        """
        Executes a controlled overfit test on a small subset of frames to empirically verify
        that the model learns microscopy -> heatmap center mapping.
        """
        indices = [i for i, f in enumerate(full_dataset.frame_indices) if f in overfit_frames]
        subset_dataset = Subset(full_dataset, indices)
        dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=True)

        peak_extractor = peak_extractor or PeakExtractor()
        evaluator = evaluator or DetectionEvaluator()

        history = {
            "epochs": [],
            "loss": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "mean_loc_error_um": []
        }

        print(f"\n--- Running Controlled Overfit Test ({epochs} epochs, frames {overfit_frames}) ---")
        sys.stdout.flush()

        for ep in range(1, epochs + 1):
            avg_loss = self.train_epoch(dataloader)

            # Evaluate on interval or final epoch
            should_eval = (ep == 1 or ep % eval_interval == 0 or ep == epochs)

            if should_eval:
                self.model.eval()
                all_metrics = []
                with torch.no_grad():
                    for idx in indices:
                        img_tensor, _, frame_t = full_dataset[idx]
                        img_input = img_tensor.unsqueeze(0).to(self.device)
                        pred_heatmap = self.model(img_input).squeeze(0).squeeze(0).cpu().numpy()

                        # Extract peaks
                        candidates = peak_extractor.extract_peaks_from_heatmap(pred_heatmap, frame_t)

                        # Get ground truth
                        gt_nodes = full_dataset.geff_loader.get_nodes_by_frame(frame_t)

                        # Evaluate matching
                        metrics = evaluator.evaluate_frame(gt_nodes, candidates)
                        all_metrics.append(metrics)

                avg_prec = float(np.mean([m.precision for m in all_metrics]))
                avg_rec = float(np.mean([m.recall for m in all_metrics]))
                avg_f1 = float(np.mean([m.f1_score for m in all_metrics]))
                avg_loc_err = float(np.mean([m.mean_localization_error_um for m in all_metrics]))

                history["epochs"].append(ep)
                history["loss"].append(avg_loss)
                history["precision"].append(avg_prec)
                history["recall"].append(avg_rec)
                history["f1_score"].append(avg_f1)
                history["mean_loc_error_um"].append(avg_loc_err)

                print(
                    f"Epoch {ep:2d}/{epochs:2d} | Loss: {avg_loss:.6f} | "
                    f"Prec: {avg_prec:.3f} | Rec: {avg_rec:.3f} | F1: {avg_f1:.3f} | "
                    f"Loc Error: {avg_loc_err:.2f} um"
                )
                sys.stdout.flush()
            else:
                history["epochs"].append(ep)
                history["loss"].append(avg_loss)
                print(f"Epoch {ep:2d}/{epochs:2d} | Loss: {avg_loss:.6f}")
                sys.stdout.flush()

        return history
