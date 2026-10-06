"""
Unit tests for Global ILP Optimization Engine.
"""

import pytest
import numpy as np

from src.config import SpatialResolution
from src.data.coordinates import CoordinateTransformer
from src.graph import CanonicalNode, NodeSet, CandidateGraphConfig, CandidateGraphBuilder, CandidateEdge
from src.optimization import (
    OptimizationConfig,
    ObjectiveBuilder,
    ConstraintBuilder,
    MILPSolverWrapper,
    TrackFragmentDecoder,
    ILPDiagnostics,
    GlobalILPOptimizer,
)


@pytest.fixture
def transformer():
    res = SpatialResolution(z_spacing=1.625, y_spacing=0.40625, x_spacing=0.40625)
    return CoordinateTransformer(resolution=res)


# 1. Binary Variable Creation
def test_binary_variable_creation(transformer):
    c_min = np.array([-1.0, -2.0, 0.5])
    node_set = NodeSet()
    builder = ConstraintBuilder()
    constraints = builder.build_flow_constraints(node_set, [])

    solver = MILPSolverWrapper()
    x_sol, res = solver.solve(c_min, constraints)

    assert res.num_variables == 3
    assert np.all((x_sol == 0.0) | (x_sol == 1.0))


# 2. Incoming Degree <= 1
def test_incoming_degree_constraint(transformer):
    node_set = NodeSet()
    # Source nodes A (1), B (2) both pointing to destination C (3)
    u_a = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    u_b = CanonicalNode(node_id=2, frame=0, global_voxel_pos=(0, 0, 10), physical_pos=(0.0, 0.0, 10.0), confidence=1.0)
    v_c = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(0, 0, 2), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)
    node_set.add_nodes([u_a, u_b, v_c])

    e1 = CandidateEdge(source_node_id=1, destination_node_id=3, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 2), euclidean_distance_phys=2.0, voxel_displacement_vector=(0, 0, 2), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    e2 = CandidateEdge(source_node_id=2, destination_node_id=3, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, -8), euclidean_distance_phys=8.0, voxel_displacement_vector=(0, 0, -8), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)

    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)
    graph.edges = [e1, e2]

    # Scores: e1 has high score (0.9), e2 has high score (0.85)
    stgnn_scores = np.array([0.9, 0.85])

    opt = GlobalILPOptimizer(OptimizationConfig(mode="stgnn_only"))
    sol = opt.optimize(node_set, graph, stgnn_scores=stgnn_scores)

    # Must select at most 1 incoming edge to node 3
    assert len(sol.selected_edges) == 1
    assert sol.selected_edges[0].source_node_id == 1  # e1 has higher score


# 3. Outgoing Degree <= 1
def test_outgoing_degree_constraint(transformer):
    node_set = NodeSet()
    # Source A (1) pointing to destinations B (2) and C (3)
    u_a = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v_b = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 2), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)
    v_c = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(0, 0, 4), physical_pos=(0.0, 0.0, 4.0), confidence=1.0)
    node_set.add_nodes([u_a, v_b, v_c])

    e1 = CandidateEdge(source_node_id=1, destination_node_id=2, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 2), euclidean_distance_phys=2.0, voxel_displacement_vector=(0, 0, 2), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    e2 = CandidateEdge(source_node_id=1, destination_node_id=3, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 4), euclidean_distance_phys=4.0, voxel_displacement_vector=(0, 0, 4), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)

    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)
    graph.edges = [e1, e2]

    stgnn_scores = np.array([0.9, 0.8])
    opt = GlobalILPOptimizer(OptimizationConfig(mode="stgnn_only"))
    sol = opt.optimize(node_set, graph, stgnn_scores=stgnn_scores)

    # Must select at most 1 outgoing edge from node 1
    assert len(sol.selected_edges) == 1
    assert sol.selected_edges[0].destination_node_id == 2


# 4. Births Allowed
def test_births_allowed(transformer):
    node_set = NodeSet()
    # Destination node appearing at frame 1 with no incoming edges
    v_new = CanonicalNode(node_id=10, frame=1, global_voxel_pos=(50, 50, 50), physical_pos=(50.0, 50.0, 50.0), confidence=1.0)
    node_set.add_node(v_new)

    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)

    opt = GlobalILPOptimizer()
    sol = opt.optimize(node_set, graph)

    assert sol.solver_result.success
    assert len(sol.selected_edges) == 0


# 5. Terminations Allowed
def test_terminations_allowed(transformer):
    node_set = NodeSet()
    # Source node terminating at frame 0 with no outgoing candidates
    u_old = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    node_set.add_node(u_old)

    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)

    opt = GlobalILPOptimizer()
    sol = opt.optimize(node_set, graph)

    assert sol.solver_result.success
    assert len(sol.selected_edges) == 0


# 6. No Forced Predecessor
def test_no_forced_predecessor(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 100), physical_pos=(0.0, 0.0, 100.0), confidence=1.0)
    node_set.add_nodes([u, v])

    # Low score edge (0.1) -> Optimizer should NOT select it
    e = CandidateEdge(source_node_id=1, destination_node_id=2, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 100), euclidean_distance_phys=100.0, voxel_displacement_vector=(0, 0, 100), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)
    graph.edges = [e]

    opt = GlobalILPOptimizer(OptimizationConfig(mode="stgnn_only"))
    sol = opt.optimize(node_set, graph, stgnn_scores=np.array([0.1]))

    assert len(sol.selected_edges) == 0


# 7. No Forced Successor
def test_no_forced_successor(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 50), physical_pos=(0.0, 0.0, 50.0), confidence=1.0)
    node_set.add_nodes([u, v])

    e = CandidateEdge(source_node_id=1, destination_node_id=2, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 50), euclidean_distance_phys=50.0, voxel_displacement_vector=(0, 0, 50), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)
    graph.edges = [e]

    opt = GlobalILPOptimizer(OptimizationConfig(mode="stgnn_only"))
    sol = opt.optimize(node_set, graph, stgnn_scores=np.array([0.2]))

    assert len(sol.selected_edges) == 0


# 8. Objective Construction
def test_objective_construction():
    builder = ObjectiveBuilder(OptimizationConfig(mode="stgnn_only"))

    p = np.array([0.9, 0.5, 0.1])
    w = builder.log_odds(p)

    assert w[0] > 0  # Reward for p > 0.5
    assert pytest.approx(w[1], abs=1e-3) == 0.0  # Neutral for p = 0.5
    assert w[2] < 0  # Penalty for p < 0.5


# 9. Log-Odds Transformation
def test_log_odds_transformation():
    builder = ObjectiveBuilder()
    w_pos = builder.log_odds(0.95)
    w_neg = builder.log_odds(0.05)

    assert w_pos > 0.0
    assert w_neg < 0.0
    assert pytest.approx(w_pos, abs=1e-3) == -w_neg


# 10. Geometry-Only Mode
def test_geometry_only_mode(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v_near = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 2), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)
    v_far = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(0, 0, 20), physical_pos=(0.0, 0.0, 20.0), confidence=1.0)
    node_set.add_nodes([u, v_near, v_far])

    e_near = CandidateEdge(source_node_id=1, destination_node_id=2, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 2), euclidean_distance_phys=2.0, voxel_displacement_vector=(0, 0, 2), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    e_far = CandidateEdge(source_node_id=1, destination_node_id=3, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 20), euclidean_distance_phys=20.0, voxel_displacement_vector=(0, 0, 20), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)

    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)
    graph.edges = [e_near, e_far]

    opt = GlobalILPOptimizer(OptimizationConfig(mode="geometry_only"))
    sol = opt.optimize(node_set, graph)

    assert len(sol.selected_edges) == 1
    assert sol.selected_edges[0].destination_node_id == 2


# 11. STGNN-Only Mode
def test_stgnn_only_mode(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v1 = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 5), physical_pos=(0.0, 0.0, 5.0), confidence=1.0)
    v2 = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(0, 0, 5), physical_pos=(0.0, 0.0, 5.0), confidence=1.0)
    node_set.add_nodes([u, v1, v2])

    e1 = CandidateEdge(source_node_id=1, destination_node_id=2, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 5), euclidean_distance_phys=5.0, voxel_displacement_vector=(0, 0, 5), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    e2 = CandidateEdge(source_node_id=1, destination_node_id=3, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 5), euclidean_distance_phys=5.0, voxel_displacement_vector=(0, 0, 5), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)

    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)
    graph.edges = [e1, e2]

    opt = GlobalILPOptimizer(OptimizationConfig(mode="stgnn_only"))
    sol = opt.optimize(node_set, graph, stgnn_scores=np.array([0.95, 0.3]))

    assert len(sol.selected_edges) == 1
    assert sol.selected_edges[0].destination_node_id == 2


# 12. Combined Mode
def test_combined_mode(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v1 = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 2), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)
    node_set.add_nodes([u, v1])

    e1 = CandidateEdge(source_node_id=1, destination_node_id=2, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 2), euclidean_distance_phys=2.0, voxel_displacement_vector=(0, 0, 2), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)
    graph.edges = [e1]

    opt = GlobalILPOptimizer(OptimizationConfig(mode="combined", alpha=1.0, beta=1.0))
    sol = opt.optimize(node_set, graph, stgnn_scores=np.array([0.9]))

    assert len(sol.selected_edges) == 1


# 13. Solution Decoding
def test_solution_decoding(transformer):
    node_set = NodeSet()
    n0 = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    n1 = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 2), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)
    n2 = CanonicalNode(node_id=3, frame=2, global_voxel_pos=(0, 0, 4), physical_pos=(0.0, 0.0, 4.0), confidence=1.0)
    node_set.add_nodes([n0, n1, n2])

    e1 = CandidateEdge(source_node_id=1, destination_node_id=2, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 2), euclidean_distance_phys=2.0, voxel_displacement_vector=(0, 0, 2), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    e2 = CandidateEdge(source_node_id=2, destination_node_id=3, source_frame=1, destination_frame=2, frame_difference=1, physical_displacement_vector=(0, 0, 2), euclidean_distance_phys=2.0, voxel_displacement_vector=(0, 0, 2), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)

    decoder = TrackFragmentDecoder()
    fragments = decoder.decode_fragments(node_set, [e1, e2])

    assert len(fragments) == 1
    frag = fragments[0]
    assert frag.node_ids == [1, 2, 3]
    assert frag.length_frames == 3
    assert pytest.approx(frag.total_physical_distance_um, abs=1e-4) == 4.0


# 14. Deterministic Results
def test_deterministic_results(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v1 = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 2), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)
    v2 = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(0, 0, 5), physical_pos=(0.0, 0.0, 5.0), confidence=1.0)
    node_set.add_nodes([u, v1, v2])

    e1 = CandidateEdge(source_node_id=1, destination_node_id=2, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 2), euclidean_distance_phys=2.0, voxel_displacement_vector=(0, 0, 2), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    e2 = CandidateEdge(source_node_id=1, destination_node_id=3, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 5), euclidean_distance_phys=5.0, voxel_displacement_vector=(0, 0, 5), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)

    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)
    graph.edges = [e1, e2]

    opt = GlobalILPOptimizer(OptimizationConfig(mode="stgnn_only"))
    sol1 = opt.optimize(node_set, graph, stgnn_scores=np.array([0.9, 0.7]))
    sol2 = opt.optimize(node_set, graph, stgnn_scores=np.array([0.9, 0.7]))

    assert [e.destination_node_id for e in sol1.selected_edges] == [e.destination_node_id for e in sol2.selected_edges]


# 15. Conflict Resolution Synthetic Example
def test_conflict_resolution_synthetic_example(transformer):
    """
    Synthetic conflict graph (Part 10):
    Node A (1) at t=0 has candidates B (3) [score 0.9] and C (4) [score 0.85].
    Node D (2) at t=0 also has candidate B (3) [score 0.95].
    Local thresholding (>= 0.5) would pick A->B, A->C, and D->B (violating in/out degrees).
    The ILP must globally resolve the conflict to D->B and A->C.
    """
    node_set = NodeSet()
    u_a = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    u_d = CanonicalNode(node_id=2, frame=0, global_voxel_pos=(0, 0, 20), physical_pos=(0.0, 0.0, 20.0), confidence=1.0)

    v_b = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(0, 0, 2), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)
    v_c = CanonicalNode(node_id=4, frame=1, global_voxel_pos=(0, 0, 8), physical_pos=(0.0, 0.0, 8.0), confidence=1.0)
    node_set.add_nodes([u_a, u_d, v_b, v_c])

    e_ab = CandidateEdge(source_node_id=1, destination_node_id=3, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 2), euclidean_distance_phys=2.0, voxel_displacement_vector=(0, 0, 2), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    e_ac = CandidateEdge(source_node_id=1, destination_node_id=4, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, 8), euclidean_distance_phys=8.0, voxel_displacement_vector=(0, 0, 8), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    e_db = CandidateEdge(source_node_id=2, destination_node_id=3, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0, 0, -18), euclidean_distance_phys=18.0, voxel_displacement_vector=(0, 0, -18), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)

    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)
    graph.edges = [e_ab, e_ac, e_db]

    stgnn_scores = np.array([0.90, 0.85, 0.95])
    opt = GlobalILPOptimizer(OptimizationConfig(mode="stgnn_only"))
    sol = opt.optimize(node_set, graph, stgnn_scores=stgnn_scores)

    selected_pairs = set((e.source_node_id, e.destination_node_id) for e in sol.selected_edges)
    assert selected_pairs == {(2, 3), (1, 4)}  # D->B and A->C chosen globally!


# 16. Empty Graph
def test_empty_graph():
    node_set = NodeSet()
    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)

    opt = GlobalILPOptimizer()
    sol = opt.optimize(node_set, graph)

    assert sol.solver_result.success
    assert len(sol.selected_edges) == 0


# 17. Single-Node Graph
def test_single_node_graph(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    node_set.add_node(u)

    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)

    opt = GlobalILPOptimizer()
    sol = opt.optimize(node_set, graph)

    assert sol.solver_result.success
    assert len(sol.selected_edges) == 0


# 18. Disconnected Graph
def test_disconnected_graph(transformer):
    node_set = NodeSet()
    u1 = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v1 = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 2), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)

    u2 = CanonicalNode(node_id=3, frame=0, global_voxel_pos=(100, 100, 100), physical_pos=(100.0, 100.0, 100.0), confidence=1.0)
    v2 = CanonicalNode(node_id=4, frame=1, global_voxel_pos=(100, 100, 102), physical_pos=(100.0, 100.0, 102.0), confidence=1.0)

    node_set.add_nodes([u1, v1, u2, v2])

    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0))
    graph = builder.build_candidate_graph(node_set)

    opt = GlobalILPOptimizer(OptimizationConfig(mode="geometry_only"))
    sol = opt.optimize(node_set, graph)

    assert len(sol.selected_edges) == 2


# 19. Solver Timeout / Status Handling
def test_solver_timeout_handling():
    solver = MILPSolverWrapper(OptimizationConfig(timeout_sec=60.0))
    c_min = np.array([-1.0, -1.0])
    builder = ConstraintBuilder()
    constraints = builder.build_flow_constraints(NodeSet(), [])

    x_sol, res = solver.solve(c_min, constraints)
    assert res.success
    assert res.status == "optimal"


# 20. Selected-Edge Validity
def test_selected_edge_validity(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 2), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)
    node_set.add_nodes([u, v])

    builder = CandidateGraphBuilder()
    graph = builder.build_candidate_graph(node_set)

    opt = GlobalILPOptimizer(OptimizationConfig(mode="geometry_only"))
    sol = opt.optimize(node_set, graph)

    diag_eval = ILPDiagnostics()
    report = diag_eval.evaluate_solution(sol, node_set)

    assert report.constraint_violations_count == 0
    assert report.selected_edges == 1
