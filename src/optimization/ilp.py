"""
Global Integer Linear Programming (ILP) Optimization Engine.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from src.graph.nodes import CanonicalNode, NodeSet
from src.graph.candidate_graph import TemporalCandidateGraph, CandidateEdge
from src.optimization.base import OptimizationConfig, SolverResult, TrackingSolution
from src.optimization.objective import ObjectiveBuilder
from src.optimization.constraints import ConstraintBuilder
from src.optimization.solver import MILPSolverWrapper
from src.optimization.diagnostics import TrackFragmentDecoder


class GlobalILPOptimizer:
    """
    Global Integer Linear Programming (ILP) Tracking Optimizer.

    Architectural Principle:
    "Local mechanisms propose; global optimization decides."

    Formulates and solves binary one-to-one temporal continuation tracking optimization
    subject to degree constraints and STGNN/geometric log-odds objective rewards.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.objective_builder = ObjectiveBuilder(config=self.config)
        self.constraint_builder = ConstraintBuilder()
        self.solver = MILPSolverWrapper(config=self.config)
        self.fragment_decoder = TrackFragmentDecoder()

    def optimize(
        self,
        node_set: NodeSet,
        graph: TemporalCandidateGraph,
        stgnn_scores: Optional[np.ndarray] = None
    ) -> TrackingSolution:
        """
        Runs global ILP optimization on candidate graph.

        :param node_set: NodeSet container
        :param graph: TemporalCandidateGraph object
        :param stgnn_scores: Optional STGNN probability score array shape (E,)
        :return: TrackingSolution object
        """
        edges = graph.edges
        E = len(edges)

        if E == 0:
            empty_result = SolverResult(
                status="optimal",
                success=True,
                objective_value=0.0,
                solve_time_sec=0.0,
                num_variables=0,
                num_constraints=0,
                selected_edge_count=0,
                termination_reason="empty_graph"
            )
            return TrackingSolution(
                selected_edges=[],
                rejected_edges=[],
                fragments=[],
                solver_result=empty_result
            )

        # 1. Build minimization cost vector c_min = -w_e
        c_min = self.objective_builder.build_minimization_vector(edges, stgnn_scores=stgnn_scores)

        # 2. Build flow constraints matrix (in_degree <= 1, out_degree <= 1)
        constraints = self.constraint_builder.build_flow_constraints(node_set, edges)

        # 3. Execute MILP solver
        x_sol, solver_result = self.solver.solve(c_min, constraints)

        # 4. Partition selected and rejected edges
        selected_edges = [edges[idx] for idx in range(E) if x_sol[idx] == 1.0]
        rejected_edges = [edges[idx] for idx in range(E) if x_sol[idx] == 0.0]

        # 5. Decode direct-continuation track fragments
        fragments = self.fragment_decoder.decode_fragments(node_set, selected_edges)

        solution = TrackingSolution(
            selected_edges=selected_edges,
            rejected_edges=rejected_edges,
            fragments=fragments,
            solver_result=solver_result
        )

        # 6. Part 17 Solver Sanity Checks
        self._verify_solution_sanity(solution, node_set, graph)

        return solution

    def _verify_solution_sanity(
        self,
        solution: TrackingSolution,
        node_set: NodeSet,
        graph: TemporalCandidateGraph
    ) -> None:
        """
        Part 17 — Verifies solution validity and throws error if any check fails.
        """
        res = solution.solver_result
        assert res.success or res.status == "optimal", f"Solver failed with status: {res.status}"
        assert np.isfinite(res.objective_value), "Objective value is non-finite!"

        # All selected edges must belong to candidate graph
        graph_edge_pairs = set((e.source_node_id, e.destination_node_id) for e in graph.edges)
        for edge in solution.selected_edges:
            pair = (edge.source_node_id, edge.destination_node_id)
            assert pair in graph_edge_pairs, f"Selected edge {pair} not found in candidate graph!"
            assert node_set.get_node(edge.source_node_id) is not None, f"Source node {edge.source_node_id} missing!"
            assert node_set.get_node(edge.destination_node_id) is not None, f"Dest node {edge.destination_node_id} missing!"

        # Degree constraints check
        inc_counts: Dict[int, int] = {}
        out_counts: Dict[int, int] = {}

        for edge in solution.selected_edges:
            src = edge.source_node_id
            dst = edge.destination_node_id
            out_counts[src] = out_counts.get(src, 0) + 1
            inc_counts[dst] = inc_counts.get(dst, 0) + 1

        for src, cnt in out_counts.items():
            assert cnt <= 1, f"Node {src} violates outgoing degree constraint: {cnt} > 1"

        for dst, cnt in inc_counts.items():
            assert cnt <= 1, f"Node {dst} violates incoming degree constraint: {cnt} > 1"
