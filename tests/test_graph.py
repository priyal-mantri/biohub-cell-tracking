"""
Unit tests for Global Node Reconciliation, Temporal Candidate Graph, and Diagnostics.
"""

import pytest
import numpy as np

from src.data.coordinates import CoordinateTransformer
from src.data.nodes import CellNode
from src.regional.geometry import RegionalCandidate
from src.graph.nodes import CanonicalNode, NodeReconciler, NodeSet
from src.graph.candidate_graph import (
    CandidateEdge,
    TemporalCandidateGraph,
    CandidateGraphBuilder,
    CandidateGraphConfig,
)
from src.graph.features import GraphFeatureExtractor
from src.graph.diagnostics import CandidateRecallEvaluator, NodeMatchingResult, CandidateRecallMetrics


from src.config import SpatialResolution


@pytest.fixture
def transformer():
    res = SpatialResolution(z_spacing=1.625, y_spacing=0.40625, x_spacing=0.40625)
    return CoordinateTransformer(resolution=res)


# 1. Canonical Node Creation
def test_canonical_node_creation(transformer):
    v_pos = (10, 50, 60)
    p_pos = transformer.voxel_to_physical(v_pos)

    node = CanonicalNode(
        node_id=100001,
        frame=0,
        global_voxel_pos=v_pos,
        physical_pos=p_pos,
        confidence=0.95,
        contributing_regional_ids=[1, 2],
        num_regional_observations=2,
        primary_region_id=1,
    )

    assert node.node_id == 100001
    assert node.frame == 0
    assert node.confidence == 0.95
    assert node.num_regional_observations == 2
    assert node.primary_region_id == 1


# 2. Deterministic Node IDs
def test_deterministic_node_ids(transformer):
    reconciler = NodeReconciler(reconciliation_distance_um=4.0, transformer=transformer)

    cands_f0 = [
        RegionalCandidate(candidate_id=1, region_id=0, frame=0, local_voxel_pos=(5, 5, 5), global_voxel_pos=(10, 20, 30), physical_pos=(16.25, 8.125, 12.1875), confidence=0.8, is_in_core=True),
        RegionalCandidate(candidate_id=2, region_id=1, frame=0, local_voxel_pos=(5, 5, 5), global_voxel_pos=(40, 50, 60), physical_pos=(65.0, 20.3125, 24.375), confidence=0.9, is_in_core=True),
    ]

    nodes_run1 = reconciler.reconcile_frame(0, cands_f0)
    nodes_run2 = reconciler.reconcile_frame(0, cands_f0)

    ids1 = [n.node_id for n in nodes_run1]
    ids2 = [n.node_id for n in nodes_run2]

    assert ids1 == ids2
    assert ids1 == [100001, 100002]


# 3. Physical Coordinate Correctness
def test_physical_coordinate_correctness(transformer):
    c1 = CellNode.create(node_id=1, frame=0, voxel_pos=(10, 20, 30), transformer=transformer)
    node1 = CanonicalNode.from_cell_node(c1)

    expected_phys = (10 * 1.625, 20 * 0.40625, 30 * 0.40625)
    assert pytest.approx(node1.physical_pos[0], abs=1e-4) == expected_phys[0]
    assert pytest.approx(node1.physical_pos[1], abs=1e-4) == expected_phys[1]
    assert pytest.approx(node1.physical_pos[2], abs=1e-4) == expected_phys[2]


# 4. Regional Provenance Preservation
def test_regional_provenance_preservation(transformer):
    reconciler = NodeReconciler(reconciliation_distance_um=5.0, transformer=transformer)

    # Two regional candidates close to each other in physical space (< 5.0 um)
    c1 = RegionalCandidate(
        candidate_id=10, region_id=2, frame=1,
        local_voxel_pos=(5, 5, 5), global_voxel_pos=(20, 30, 40),
        physical_pos=(32.5, 12.1875, 16.25), confidence=0.7, is_in_core=True,
        provenance={"contributing_region_ids": [2], "num_contributing_regions": 1, "primary_region_id": 2}
    )
    c2 = RegionalCandidate(
        candidate_id=11, region_id=3, frame=1,
        local_voxel_pos=(5, 5, 5), global_voxel_pos=(20, 31, 40),
        physical_pos=(32.5, 12.59375, 16.25), confidence=0.9, is_in_core=True,  # dy = 0.40625 um
        provenance={"contributing_region_ids": [3], "num_contributing_regions": 1, "primary_region_id": 3}
    )

    reconciled = reconciler.reconcile_frame(1, [c1, c2])
    assert len(reconciled) == 1

    node = reconciled[0]
    assert set(node.contributing_regional_ids) == {2, 3}
    assert node.num_regional_observations == 2
    assert node.primary_region_id == 3  # candidate c2 has higher confidence (0.9)
    assert node.confidence == 0.9


# 5. Frame-Local Reconciliation
def test_frame_local_reconciliation(transformer):
    reconciler = NodeReconciler(reconciliation_distance_um=4.0, transformer=transformer)

    # Candidates at same position in frame 0 and frame 1 must NOT be merged
    c_f0 = CellNode.create(node_id=1, frame=0, voxel_pos=(10, 10, 10), transformer=transformer)
    c_f1 = CellNode.create(node_id=2, frame=1, voxel_pos=(10, 10, 10), transformer=transformer)

    nodes_f0 = reconciler.reconcile_frame(0, [c_f0])
    nodes_f1 = reconciler.reconcile_frame(1, [c_f1])

    assert len(nodes_f0) == 1
    assert len(nodes_f1) == 1
    assert nodes_f0[0].frame == 0
    assert nodes_f1[0].frame == 1
    assert nodes_f0[0].node_id != nodes_f1[0].node_id


# 6. Temporal Candidate Generation
def test_temporal_candidate_generation(transformer):
    node_set = NodeSet()

    # Frame 0 node
    u = CanonicalNode(
        node_id=100001, frame=0,
        global_voxel_pos=(10, 20, 30), physical_pos=(16.25, 8.125, 12.1875),
        confidence=0.9
    )
    # Frame 1 node (moved slightly)
    v = CanonicalNode(
        node_id=200001, frame=1,
        global_voxel_pos=(12, 22, 32), physical_pos=(19.5, 8.9375, 13.0),
        confidence=0.85
    )

    node_set.add_nodes([u, v])

    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=15.0, top_k_per_node=3))
    graph = builder.build_candidate_graph(node_set)

    assert graph.candidate_count() == 1
    edge = graph.get_edge(100001, 200001)
    assert edge is not None
    assert edge.source_node_id == 100001
    assert edge.destination_node_id == 200001
    assert edge.frame_difference == 1


# 7. Physical Displacement Calculation
def test_physical_displacement_calculation(transformer):
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(10, 10, 10), physical_pos=(16.25, 4.0625, 4.0625), confidence=1.0)
    v = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(12, 14, 16), physical_pos=(19.5, 5.6875, 6.5), confidence=1.0)

    dz, dy, dx = u.displacement_physical(v)
    assert pytest.approx(dz, abs=1e-4) == 3.25
    assert pytest.approx(dy, abs=1e-4) == 1.625
    assert pytest.approx(dx, abs=1e-4) == 2.4375

    d_phys = u.distance_physical(v)
    expected_dist = (3.25**2 + 1.625**2 + 2.4375**2) ** 0.5
    assert pytest.approx(d_phys, abs=1e-4) == expected_dist


# 8. Anisotropic Spacing Handling
def test_anisotropic_spacing_handling(transformer):
    # 1 voxel Z shift = 1.625 um
    # 1 voxel X shift = 0.40625 um
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(10, 10, 10), physical_pos=(16.25, 4.0625, 4.0625), confidence=1.0)
    v_z = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(11, 10, 10), physical_pos=(17.875, 4.0625, 4.0625), confidence=1.0)
    v_x = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(10, 10, 11), physical_pos=(16.25, 4.0625, 4.46875), confidence=1.0)

    assert pytest.approx(u.distance_physical(v_z), abs=1e-4) == 1.625
    assert pytest.approx(u.distance_physical(v_x), abs=1e-4) == 0.40625


# 9. Candidate Lookup
def test_candidate_lookup(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=101, frame=0, global_voxel_pos=(10, 10, 10), physical_pos=(10.0, 10.0, 10.0), confidence=0.9)
    v1 = CanonicalNode(node_id=201, frame=1, global_voxel_pos=(10, 10, 12), physical_pos=(10.0, 10.0, 12.0), confidence=0.8)
    v2 = CanonicalNode(node_id=202, frame=1, global_voxel_pos=(10, 10, 15), physical_pos=(10.0, 10.0, 15.0), confidence=0.7)

    node_set.add_nodes([u, v1, v2])

    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0, top_k_per_node=5))
    graph = builder.build_candidate_graph(node_set)

    out_edges = graph.get_outgoing_edges(101)
    assert len(out_edges) == 2

    in_edges_v1 = graph.get_incoming_edges(201)
    assert len(in_edges_v1) == 1

    assert graph.get_edge(101, 201) is not None
    assert graph.get_edge(101, 999) is None


# 10. Top-K Behavior
def test_top_k_behavior(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)

    v1 = CanonicalNode(node_id=10, frame=1, global_voxel_pos=(0, 0, 2), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)
    v2 = CanonicalNode(node_id=11, frame=1, global_voxel_pos=(0, 0, 4), physical_pos=(0.0, 0.0, 4.0), confidence=1.0)
    v3 = CanonicalNode(node_id=12, frame=1, global_voxel_pos=(0, 0, 6), physical_pos=(0.0, 0.0, 6.0), confidence=1.0)

    node_set.add_nodes([u, v1, v2, v3])

    # With Top-K = 2
    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0, top_k_per_node=2))
    graph = builder.build_candidate_graph(node_set)

    out_edges = graph.get_outgoing_edges(1)
    assert len(out_edges) == 2
    assert [e.destination_node_id for e in out_edges] == [10, 11]  # Closest two destinations


# 11. Distance-Gate Behavior
def test_distance_gate_behavior(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)

    v_near = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 5), physical_pos=(0.0, 0.0, 5.0), confidence=1.0)
    v_far = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(0, 0, 25), physical_pos=(0.0, 0.0, 25.0), confidence=1.0)

    node_set.add_nodes([u, v_near, v_far])

    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0))
    graph = builder.build_candidate_graph(node_set)

    assert graph.get_edge(1, 2) is not None
    assert graph.get_edge(1, 3) is None  # Distance 25.0 > 10.0 max distance gate


# 12. Zero-Candidate Handling
def test_zero_candidate_handling(transformer):
    node_set = NodeSet()
    u_isolated = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v_far = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 100), physical_pos=(0.0, 0.0, 100.0), confidence=1.0)

    node_set.add_nodes([u_isolated, v_far])

    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0))
    graph = builder.build_candidate_graph(node_set)

    out_edges = graph.get_outgoing_edges(1)
    assert len(out_edges) == 0

    diag = graph.export_diagnostics()
    assert diag["zero_candidate_pct"] == 100.0


# 13. Deterministic Graph Construction
def test_deterministic_graph_construction(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(10, 10, 10), physical_pos=(10.0, 10.0, 10.0), confidence=0.9)
    v1 = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(12, 12, 12), physical_pos=(12.0, 12.0, 12.0), confidence=0.8)
    v2 = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(14, 14, 14), physical_pos=(14.0, 14.0, 14.0), confidence=0.7)

    node_set.add_nodes([u, v1, v2])

    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0))
    g1 = builder.build_candidate_graph(node_set)
    g2 = builder.build_candidate_graph(node_set)

    assert g1.candidate_count() == g2.candidate_count()
    assert [e.destination_node_id for e in g1.edges] == [e.destination_node_id for e in g2.edges]


# 14. Feature Extraction Shape and Content
def test_feature_extraction(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(10, 20, 30), physical_pos=(16.25, 8.125, 12.1875), confidence=0.9)
    v = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(12, 22, 32), physical_pos=(19.5, 8.9375, 13.0), confidence=0.8)

    node_set.add_nodes([u, v])
    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=15.0))
    graph = builder.build_candidate_graph(node_set)

    extractor = GraphFeatureExtractor(volume_shape=(100, 64, 256, 256), transformer=transformer)
    feats = extractor.extract(graph)

    node_feats = feats["node_features"]
    edge_feats = feats["edge_features"]

    assert node_feats.features.shape == (2, 9)
    assert edge_feats.features.shape == (1, 16)
    assert edge_feats.edge_index.shape == (2, 1)


# 15. Ground-Truth Candidate Recall Calculation
def test_ground_truth_candidate_recall(transformer):
    # Create GT nodes and edges
    gt_u = CellNode.create(node_id=1, frame=0, voxel_pos=(10, 10, 10), transformer=transformer)
    gt_v = CellNode.create(node_id=2, frame=1, voxel_pos=(12, 12, 12), transformer=transformer)
    gt_nodes = [gt_u, gt_v]
    gt_edges = [(1, 2)]

    # Create canonical nodes exactly matching GT nodes
    can_u = CanonicalNode.from_cell_node(gt_u, canonical_id=100001)
    can_v = CanonicalNode.from_cell_node(gt_v, canonical_id=200001)

    node_set = NodeSet([can_u, can_v])
    evaluator = CandidateRecallEvaluator(transformer=transformer)

    # 1. Match canonical nodes to GEFF GT
    matching_result = evaluator.match_nodes_to_geff(node_set, gt_nodes, match_threshold_um=5.0)
    assert matching_result.matched_geff_count == 2
    assert matching_result.matching_success_rate == 100.0

    # 2. Build candidate graph with sufficient distance
    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0))
    graph = builder.build_candidate_graph(node_set)

    # 3. Evaluate recall
    recall_metrics = evaluator.evaluate_recall(graph, gt_edges, gt_nodes, matching_result)
    assert recall_metrics.num_gt_edges == 1
    assert recall_metrics.gt_edges_recovered == 1
    assert recall_metrics.candidate_recall == 1.0
    assert recall_metrics.num_missing_gt_edges == 0
