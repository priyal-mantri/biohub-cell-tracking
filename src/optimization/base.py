"""
Data structures, configurations, and result containers for Global ILP Optimization.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.graph.candidate_graph import CandidateEdge


@dataclass
class OptimizationConfig:
    """
    Configuration hyperparameters for ILP objective construction and solver execution.

    :param mode: Objective mode - 'stgnn_only', 'geometry_only', or 'combined'
    :param alpha: Weight for STGNN learned log-odds score in combined mode
    :param beta: Weight for geometric prior log-odds score in combined mode
    :param birth_cost: Penalty/cost for starting a new track (0.0 = no birth penalty)
    :param termination_cost: Penalty/cost for terminating a track (0.0 = no termination penalty)
    :param epsilon: Epsilon buffer for log-odds probability transformation
    :param sigma_scale_um: Distance scaling factor (um) for geometric prior score
    :param timeout_sec: Maximum solver timeout in seconds
    """
    mode: str = "stgnn_only"  # 'stgnn_only', 'geometry_only', or 'combined'
    alpha: float = 1.0
    beta: float = 1.0
    birth_cost: float = 0.0
    termination_cost: float = 0.0
    epsilon: float = 1e-5
    sigma_scale_um: float = 5.0
    timeout_sec: float = 60.0


@dataclass
class SolverResult:
    """
    Container for solver execution status and statistics.
    """
    status: str
    success: bool
    objective_value: float
    solve_time_sec: float
    num_variables: int
    num_constraints: int
    selected_edge_count: int
    termination_reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status,
            "success": self.success,
            "objective_value": self.objective_value,
            "solve_time_sec": self.solve_time_sec,
            "num_variables": self.num_variables,
            "num_constraints": self.num_constraints,
            "selected_edge_count": self.selected_edge_count,
            "termination_reason": self.termination_reason,
        }


@dataclass
class TrackFragment:
    """
    Representation of a decoded direct-continuation track fragment across consecutive frames.
    """
    fragment_id: int
    node_ids: List[int]
    frames: List[int]
    physical_positions: List[Tuple[float, float, float]]
    length_frames: int
    total_physical_distance_um: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert fragment to dictionary representation."""
        return {
            "fragment_id": self.fragment_id,
            "node_ids": self.node_ids,
            "frames": self.frames,
            "length_frames": self.length_frames,
            "total_physical_distance_um": self.total_physical_distance_um,
        }


@dataclass
class TrackingSolution:
    """
    Global ILP tracking solution containing selected/rejected edges and decoded track fragments.
    """
    selected_edges: List[CandidateEdge]
    rejected_edges: List[CandidateEdge]
    fragments: List[TrackFragment]
    solver_result: SolverResult

    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary."""
        return {
            "solver_result": self.solver_result.to_dict(),
            "selected_edge_count": len(self.selected_edges),
            "rejected_edge_count": len(self.rejected_edges),
            "num_fragments": len(self.fragments),
            "fragments": [f.to_dict() for f in self.fragments],
        }
