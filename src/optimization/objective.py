"""
Objective Function Construction and Log-Odds Score Transformations.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np

from src.graph.candidate_graph import TemporalCandidateGraph, CandidateEdge
from src.optimization.base import OptimizationConfig


class ObjectiveBuilder:
    """
    Constructs edge weight coefficients for the Global ILP optimization objective.

    Transforms candidate probabilities/scores into calibrated log-odds weights:
        w(p) = log((p + eps) / (1 - p + eps))

    Objective formulation (Maximization):
        Maximize  sum_{e in E} w_e * x_e
    Converted for standard minimization solvers:
        Minimize  sum_{e in E} (-w_e) * x_e
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

    def log_odds(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Log-odds transformation mapping probabilities in [0, 1] to linear evidence weights.

        :param p: Probability value or NumPy array of probabilities
        :return: Log-odds value(s) in (-inf, +inf)
        """
        eps = self.config.epsilon
        p_clipped = np.clip(p, eps, 1.0 - eps)
        return np.log((p_clipped + eps) / (1.0 - p_clipped + eps))

    def compute_edge_weights(
        self,
        edges: List[CandidateEdge],
        stgnn_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Computes edge weight array w_e for candidate edges based on configured mode.

        :param edges: List of candidate edges
        :param stgnn_scores: Optional STGNN probability array shape (E,)
        :return: NumPy array shape (E,) float64 containing edge reward weights
        """
        E = len(edges)
        if E == 0:
            return np.empty(0, dtype=np.float64)

        distances = np.array([e.euclidean_distance_phys for e in edges], dtype=np.float64)
        geom_probs = np.exp(-distances / max(1e-6, self.config.sigma_scale_um))
        geom_log_odds = self.log_odds(geom_probs)

        if stgnn_scores is not None and len(stgnn_scores) == E:
            stgnn_log_odds = self.log_odds(stgnn_scores)
        else:
            stgnn_log_odds = geom_log_odds

        mode = self.config.mode.lower()
        if mode == "stgnn_only":
            weights = stgnn_log_odds
        elif mode == "geometry_only":
            weights = geom_log_odds
        elif mode == "combined":
            weights = self.config.alpha * stgnn_log_odds + self.config.beta * geom_log_odds
        else:
            raise ValueError(f"Unknown objective mode: '{mode}'. Use 'stgnn_only', 'geometry_only', or 'combined'.")

        return weights.astype(np.float64)

    def build_minimization_vector(
        self,
        edges: List[CandidateEdge],
        stgnn_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Constructs linear objective coefficient vector c for SciPy minimization solver (c_e = -w_e).

        :return: NumPy array shape (E,) float64 (cost vector for minimization)
        """
        weights = self.compute_edge_weights(edges, stgnn_scores=stgnn_scores)
        return (-weights).astype(np.float64)
