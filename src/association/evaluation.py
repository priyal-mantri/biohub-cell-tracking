"""
Diagnostic Evaluation Metrics, Geometric Baseline Comparison & Displacement-Binned Breakdown.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.association.base import GeometricBaseline


@dataclass
class EvaluationMetrics:
    """Container for candidate edge association classification metrics."""
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    positive_edge_recall: float
    negative_edge_specificity: float
    mean_prob_positive: float
    mean_prob_negative: float
    total_edges: int
    pos_count: int
    neg_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "positive_edge_recall": self.positive_edge_recall,
            "negative_edge_specificity": self.negative_edge_specificity,
            "mean_prob_positive": self.mean_prob_positive,
            "mean_prob_negative": self.mean_prob_negative,
            "total_edges": self.total_edges,
            "pos_count": self.pos_count,
            "neg_count": self.neg_count,
        }


class AssociationEvaluator:
    """
    Evaluates candidate edge probability predictions against ground-truth binary labels.
    """

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
        strict_single_class_check: bool = False
    ) -> EvaluationMetrics:
        """
        Computes ROC-AUC, PR-AUC, Precision, Recall, F1, Specificity, and Probability stats.

        :param y_true: Binary ground-truth labels (E,) float or int (0 or 1)
        :param y_prob: Predicted continuous probabilities (E,) in [0, 1]
        :param threshold: Binary classification threshold (default 0.5)
        :param strict_single_class_check: If True, raises ValueError if y_true lacks positive or negative labels.
        :return: EvaluationMetrics dataclass
        """
        y_true = np.asarray(y_true, dtype=np.float32)
        y_prob = np.asarray(y_prob, dtype=np.float32)

        total_edges = len(y_true)
        pos_mask = (y_true == 1.0)
        neg_mask = (y_true == 0.0)

        pos_count = int(np.sum(pos_mask))
        neg_count = int(np.sum(neg_mask))

        if strict_single_class_check:
            if total_edges > 0 and (pos_count == 0 or neg_count == 0):
                raise ValueError(
                    f"Single-class evaluation population is invalid for ROC-AUC evaluation. "
                    f"Found pos_count={pos_count}, neg_count={neg_count} among {total_edges} candidate edges."
                )

        if total_edges == 0:
            return EvaluationMetrics(
                roc_auc=0.0, pr_auc=0.0, precision=0.0, recall=0.0, f1=0.0,
                positive_edge_recall=0.0, negative_edge_specificity=0.0,
                mean_prob_positive=0.0, mean_prob_negative=0.0,
                total_edges=0, pos_count=0, neg_count=0
            )

        mean_prob_pos = float(np.mean(y_prob[pos_mask])) if pos_count > 0 else 0.0
        mean_prob_neg = float(np.mean(y_prob[neg_mask])) if neg_count > 0 else 0.0

        # Threshold-based metrics
        y_pred = (y_prob >= threshold).astype(np.float32)

        tp = int(np.sum((y_pred == 1.0) & (y_true == 1.0)))
        fp = int(np.sum((y_pred == 1.0) & (y_true == 0.0)))
        tn = int(np.sum((y_pred == 0.0) & (y_true == 0.0)))
        fn = int(np.sum((y_pred == 0.0) & (y_true == 1.0)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # ROC-AUC calculation
        roc_auc = AssociationEvaluator._compute_roc_auc(y_true, y_prob)

        # PR-AUC calculation
        pr_auc = AssociationEvaluator._compute_pr_auc(y_true, y_prob)

        return EvaluationMetrics(
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            precision=prec,
            recall=rec,
            f1=f1,
            positive_edge_recall=rec,
            negative_edge_specificity=spec,
            mean_prob_positive=mean_prob_pos,
            mean_prob_negative=mean_prob_neg,
            total_edges=total_edges,
            pos_count=pos_count,
            neg_count=neg_count,
        )

    @staticmethod
    def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
        """Trapezoidal integration compatible across NumPy 1.x and 2.x."""
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(y, x))
        elif hasattr(np, "trapz"):
            return float(np.trapz(y, x))
        else:
            return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))

    @staticmethod
    def _compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculates area under ROC curve using trapezoidal numerical integration."""
        pos_mask = (y_true == 1.0)
        neg_mask = (y_true == 0.0)
        n_pos = np.sum(pos_mask)
        n_neg = np.sum(neg_mask)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Sort by decreasing score
        sort_indices = np.argsort(-y_prob)
        sorted_true = y_true[sort_indices]

        tps = np.cumsum(sorted_true == 1.0)
        fps = np.cumsum(sorted_true == 0.0)

        tpr = tps / n_pos
        fpr = fps / n_neg

        # Add origin (0, 0)
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])

        return AssociationEvaluator._trapezoid(tpr, fpr)

    @staticmethod
    def _compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculates area under Precision-Recall curve using trapezoidal numerical integration."""
        n_pos = np.sum(y_true == 1.0)
        if n_pos == 0:
            return 0.0

        sort_indices = np.argsort(-y_prob)
        sorted_true = y_true[sort_indices]

        tps = np.cumsum(sorted_true == 1.0)
        fps = np.cumsum(sorted_true == 0.0)

        precision = tps / (tps + fps)
        recall = tps / n_pos

        precision = np.concatenate([[1.0], precision])
        recall = np.concatenate([[0.0], recall])

        return AssociationEvaluator._trapezoid(precision, recall)

    @staticmethod
    def evaluate_displacement_bins(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        distances_um: np.ndarray
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluates metrics broken down by physical motion displacement bins:
        - Low: d < 1.5 um
        - Medium: 1.5 um <= d <= 3.5 um
        - High: d > 3.5 um
        """
        low_mask = (distances_um < 1.5)
        med_mask = (distances_um >= 1.5) & (distances_um <= 3.5)
        high_mask = (distances_um > 3.5)

        return {
            "Low (<1.5 um)": AssociationEvaluator.compute_metrics(y_true[low_mask], y_prob[low_mask]),
            "Medium (1.5-3.5 um)": AssociationEvaluator.compute_metrics(y_true[med_mask], y_prob[med_mask]),
            "High (>3.5 um)": AssociationEvaluator.compute_metrics(y_true[high_mask], y_prob[high_mask]),
        }
