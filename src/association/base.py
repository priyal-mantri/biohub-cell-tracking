"""
Deterministic Geometric Baseline Association Model.
"""

from typing import List, Dict, Any, Union
import numpy as np

from src.graph.candidate_graph import TemporalCandidateGraph, CandidateEdge


class GeometricBaseline:
    """
    Deterministic geometric baseline association model.

    Computes association evidence strictly from physical distance using exponential decay:
        Score(e) = exp(- d_phys(e) / sigma_scale_um)

    Guarantees:
    - Score is strictly in [0.0, 1.0].
    - Score decreases monotonically with physical distance (micrometers).
    - Fully deterministic and calibration-ready.
    """

    def __init__(self, sigma_scale_um: float = 2.0):
        self.sigma_scale_um = max(1e-6, sigma_scale_um)

    def predict_edge_scores(
        self,
        edges: Union[List[CandidateEdge], TemporalCandidateGraph]
    ) -> np.ndarray:
        """
        Computes geometric baseline scores for candidate edges.

        :param edges: List of CandidateEdge objects or TemporalCandidateGraph
        :return: NumPy array shape (E,) containing baseline scores in [0, 1]
        """
        edge_list = edges.edges if isinstance(edges, TemporalCandidateGraph) else edges
        if not edge_list:
            return np.empty(0, dtype=np.float32)

        distances = np.array([e.euclidean_distance_phys for e in edge_list], dtype=np.float32)
        scores = np.exp(-distances / self.sigma_scale_um)
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Returns baseline configuration."""
        return {
            "model_type": "GeometricBaseline",
            "sigma_scale_um": self.sigma_scale_um,
            "formula": f"exp(-d_phys / {self.sigma_scale_um})"
        }
