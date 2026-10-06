"""
Ground-Truth Detection Evaluation & Localization Error Analysis.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.data.nodes import CellNode
from src.data.coordinates import CoordinateTransformer
from src.detection.base import DetectionCandidate


@dataclass
class DetectionMetrics:
    """
    Spatial cell-center detection evaluation metrics.
    Strictly isolated from temporal tracking/association performance.
    """
    num_gt_cells: int
    num_pred_candidates: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    mean_localization_error_um: float
    median_localization_error_um: float

    def summary_string(self) -> str:
        return (
            f"GT Cells: {self.num_gt_cells} | Pred Candidates: {self.num_pred_candidates}\n"
            f"TP: {self.true_positives} | FP: {self.false_positives} | FN: {self.false_negatives}\n"
            f"Precision: {self.precision:.4f} | Recall: {self.recall:.4f} | F1-Score: {self.f1_score:.4f}\n"
            f"Mean Localization Error: {self.mean_localization_error_um:.3f} um | "
            f"Median Error: {self.median_localization_error_um:.3f} um"
        )


class DetectionEvaluator:
    """
    Evaluates cell-center detection performance against ground-truth GEFF nodes using physical distance matching.
    """

    def __init__(
        self,
        max_matching_distance_um: float = 5.0,  # Matching distance threshold in micrometers (um)
        transformer: Optional[CoordinateTransformer] = None
    ):
        self.max_matching_distance_um = max_matching_distance_um
        self.transformer = transformer or CoordinateTransformer()

    def evaluate_frame(
        self,
        gt_nodes: List[CellNode],
        pred_candidates: List[DetectionCandidate]
    ) -> DetectionMetrics:
        """
        Evaluates predictions for a single 3D frame.
        """
        n_gt = len(gt_nodes)
        n_pred = len(pred_candidates)

        if n_gt == 0 and n_pred == 0:
            return DetectionMetrics(0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 0.0, 0.0)
        if n_gt == 0:
            return DetectionMetrics(0, n_pred, 0, n_pred, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if n_pred == 0:
            return DetectionMetrics(n_gt, 0, 0, 0, n_gt, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Build distance matrix (N_pred, N_gt) in physical micrometers
        dist_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
        for i, pred in enumerate(pred_candidates):
            for j, gt in enumerate(gt_nodes):
                dist = self.transformer.euclidean_distance_physical(pred.physical_pos, gt.physical_pos)
                dist_matrix[i, j] = dist

        # Greedy bipartite matching by physical distance ascending
        pairs = []
        for i in range(n_pred):
            for j in range(n_gt):
                d = dist_matrix[i, j]
                if d <= self.max_matching_distance_um:
                    pairs.append((d, i, j))

        pairs.sort(key=lambda x: x[0])

        matched_pred = set()
        matched_gt = set()
        matched_errors = []

        for dist, i, j in pairs:
            if i not in matched_pred and j not in matched_gt:
                matched_pred.add(i)
                matched_gt.add(j)
                matched_errors.append(dist)

        tp = len(matched_pred)
        fp = n_pred - tp
        fn = n_gt - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        mean_err = float(np.mean(matched_errors)) if matched_errors else 0.0
        median_err = float(np.median(matched_errors)) if matched_errors else 0.0

        return DetectionMetrics(
            num_gt_cells=n_gt,
            num_pred_candidates=n_pred,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mean_localization_error_um=mean_err,
            median_localization_error_um=median_err
        )
