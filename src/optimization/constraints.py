"""
Linear Flow Constraint Construction for One-to-One Temporal Continuation.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize import LinearConstraint

from src.graph.nodes import CanonicalNode, NodeSet
from src.graph.candidate_graph import CandidateEdge


@dataclass
class ConstraintMatrix:
    """
    Container for sparse linear constraint matrix A and upper/lower bounds.
    """
    matrix: csc_matrix  # (M, E) sparse matrix
    lb: np.ndarray  # (M,) float64 lower bounds
    ub: np.ndarray  # (M,) float64 upper bounds
    constraint_names: List[str]
    num_incoming_constraints: int
    num_outgoing_constraints: int

    def to_scipy_constraint(self) -> LinearConstraint:
        """Converts matrix representation to scipy.optimize.LinearConstraint."""
        return LinearConstraint(self.matrix, self.lb, self.ub)


class ConstraintBuilder:
    """
    Constructs one-to-one temporal continuation flow constraints:
        1. Incoming Degree Constraint: sum_{e in in(v)} x_e <= 1 for all v in V
        2. Outgoing Degree Constraint: sum_{e in out(u)} x_e <= 1 for all u in V
    """

    def build_flow_constraints(
        self,
        node_set: NodeSet,
        edges: List[CandidateEdge]
    ) -> ConstraintMatrix:
        """
        Constructs linear flow constraint matrix for candidate edge variables.

        :param node_set: NodeSet containing canonical nodes
        :param edges: List of CandidateEdge objects
        :return: ConstraintMatrix object
        """
        E = len(edges)
        if E == 0:
            empty_sparse = csc_matrix((0, 0), dtype=np.float64)
            return ConstraintMatrix(
                matrix=empty_sparse,
                lb=np.empty(0, dtype=np.float64),
                ub=np.empty(0, dtype=np.float64),
                constraint_names=[],
                num_incoming_constraints=0,
                num_outgoing_constraints=0
            )

        # Map candidate edges for fast lookup
        # Outgoing edges per node ID
        node_out_edges: Dict[int, List[int]] = {}
        # Incoming edges per node ID
        node_in_edges: Dict[int, List[int]] = {}

        for edge_idx, edge in enumerate(edges):
            src = edge.source_node_id
            dst = edge.destination_node_id

            if src not in node_out_edges:
                node_out_edges[src] = []
            node_out_edges[src].append(edge_idx)

            if dst not in node_in_edges:
                node_in_edges[dst] = []
            node_in_edges[dst].append(edge_idx)

        row_indices = []
        col_indices = []
        data_values = []
        constraint_names = []

        curr_row = 0
        n_inc = 0
        n_out = 0

        # 1. Outgoing degree constraints (sum_{e in out(u)} x_e <= 1)
        for node_id, edge_indices in sorted(node_out_edges.items()):
            for e_idx in edge_indices:
                row_indices.append(curr_row)
                col_indices.append(e_idx)
                data_values.append(1.0)
            constraint_names.append(f"out_degree_node_{node_id}")
            curr_row += 1
            n_out += 1

        # 2. Incoming degree constraints (sum_{e in in(v)} x_e <= 1)
        for node_id, edge_indices in sorted(node_in_edges.items()):
            for e_idx in edge_indices:
                row_indices.append(curr_row)
                col_indices.append(e_idx)
                data_values.append(1.0)
            constraint_names.append(f"in_degree_node_{node_id}")
            curr_row += 1
            n_inc += 1

        M = curr_row
        if M > 0:
            sparse_A = csc_matrix((data_values, (row_indices, col_indices)), shape=(M, E), dtype=np.float64)
            lb = np.zeros(M, dtype=np.float64)
            ub = np.ones(M, dtype=np.float64)
        else:
            sparse_A = csc_matrix((0, E), dtype=np.float64)
            lb = np.empty(0, dtype=np.float64)
            ub = np.empty(0, dtype=np.float64)

        return ConstraintMatrix(
            matrix=sparse_A,
            lb=lb,
            ub=ub,
            constraint_names=constraint_names,
            num_incoming_constraints=n_inc,
            num_outgoing_constraints=n_out
        )
