"""
Unit tests for STGNN Temporal Candidate Edge Association Module.
"""

import pytest
import numpy as np
import torch

from src.config import SpatialResolution
from src.data.coordinates import CoordinateTransformer
from src.data.nodes import CellNode
from src.graph import (
    CanonicalNode,
    NodeReconciler,
    NodeSet,
    CandidateGraphConfig,
    CandidateGraphBuilder,
    TemporalCandidateGraph,
    GraphFeatureExtractor,
    CandidateRecallEvaluator,
)
from src.association import (
    CandidateEdgeLabeler,
    TemporalSplitter,
    GeometricBaseline,
    STGNNModel,
    AssociationDataset,
    AssociationEvaluator,
    STGNNTrainer,
    STGNNTrainerConfig,
)


@pytest.fixture
def transformer():
    res = SpatialResolution(z_spacing=1.625, y_spacing=0.40625, x_spacing=0.40625)
    return CoordinateTransformer(resolution=res)


@pytest.fixture
def synthetic_graph_and_dataset(transformer):
    node_set = NodeSet()

    # Frame 0 nodes
    u1 = CanonicalNode(node_id=100001, frame=0, global_voxel_pos=(10, 10, 10), physical_pos=(16.25, 4.0625, 4.0625), confidence=0.9)
    u2 = CanonicalNode(node_id=100002, frame=0, global_voxel_pos=(30, 30, 30), physical_pos=(48.75, 12.1875, 12.1875), confidence=0.85)

    # Frame 1 nodes
    v1 = CanonicalNode(node_id=200001, frame=1, global_voxel_pos=(11, 10, 10), physical_pos=(17.875, 4.0625, 4.0625), confidence=0.88)
    v2 = CanonicalNode(node_id=200002, frame=1, global_voxel_pos=(31, 30, 30), physical_pos=(50.375, 12.1875, 12.1875), confidence=0.82)
    v_far = CanonicalNode(node_id=200003, frame=1, global_voxel_pos=(10, 20, 20), physical_pos=(16.25, 8.125, 8.125), confidence=0.5)

    node_set.add_nodes([u1, u2, v1, v2, v_far])

    cfg = CandidateGraphConfig(max_physical_distance_um=15.0, top_k_per_node=3)
    builder = CandidateGraphBuilder(config=cfg)
    graph = builder.build_candidate_graph(node_set)

    # Simulated GT nodes matching u1->v1 and u2->v2
    gt_nodes = [
        CellNode.create(node_id=1, frame=0, voxel_pos=(10, 10, 10), transformer=transformer),
        CellNode.create(node_id=2, frame=0, voxel_pos=(30, 30, 30), transformer=transformer),
        CellNode.create(node_id=3, frame=1, voxel_pos=(11, 10, 10), transformer=transformer),
        CellNode.create(node_id=4, frame=1, voxel_pos=(31, 30, 30), transformer=transformer),
    ]
    gt_edges = [(1, 3), (2, 4)]

    evaluator = CandidateRecallEvaluator(transformer=transformer)
    matching_res = evaluator.match_nodes_to_geff(node_set, gt_nodes, match_threshold_um=5.0)

    dataset = AssociationDataset(
        graph=graph,
        feature_extractor=GraphFeatureExtractor(transformer=transformer),
        labeler=CandidateEdgeLabeler(),
        geff_edges=gt_edges,
        matching_result=matching_res
    )

    return graph, dataset, matching_res


# 1. Model Initialization
def test_model_initialization():
    model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=32, num_layers=2)
    assert model.node_in_dim == 9
    assert model.edge_in_dim == 16
    assert model.hidden_dim == 32
    assert len(model.gnn_layers) == 2


# 2. Forward Pass
def test_forward_pass(synthetic_graph_and_dataset):
    _, dataset, _ = synthetic_graph_and_dataset
    model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=32)

    logits = model(dataset.node_feats_tensor, dataset.edge_index_tensor, dataset.edge_feats_tensor)
    assert logits is not None
    assert isinstance(logits, torch.Tensor)


# 3. Output Shape
def test_output_shape(synthetic_graph_and_dataset):
    graph, dataset, _ = synthetic_graph_and_dataset
    model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=32)

    logits = model(dataset.node_feats_tensor, dataset.edge_index_tensor, dataset.edge_feats_tensor)
    assert logits.shape == (graph.candidate_count(), 1)


# 4. Finite Output
def test_finite_output(synthetic_graph_and_dataset):
    _, dataset, _ = synthetic_graph_and_dataset
    model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=32)

    logits = model(dataset.node_feats_tensor, dataset.edge_index_tensor, dataset.edge_feats_tensor)
    probs = torch.sigmoid(logits)

    assert torch.all(torch.isfinite(logits))
    assert torch.all(torch.isfinite(probs))
    assert torch.all((probs >= 0.0) & (probs <= 1.0))


# 5. Graph with Multiple Candidates
def test_graph_multiple_candidates(synthetic_graph_and_dataset):
    graph, dataset, _ = synthetic_graph_and_dataset
    out_u1 = graph.get_outgoing_edges(100001)
    assert len(out_u1) > 1  # Multiple candidate destinations for u1


# 6. Graph with One Candidate
def test_graph_one_candidate(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 5), physical_pos=(0.0, 0.0, 5.0), confidence=1.0)
    node_set.add_nodes([u, v])

    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0))
    graph = builder.build_candidate_graph(node_set)

    assert graph.candidate_count() == 1


# 7. Graph with Zero Outgoing Candidates
def test_graph_zero_candidates(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    node_set.add_node(u)

    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0))
    graph = builder.build_candidate_graph(node_set)

    assert graph.candidate_count() == 0
    dataset = AssociationDataset(graph=graph)
    model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=32)

    logits = model(dataset.node_feats_tensor, dataset.edge_index_tensor, dataset.edge_feats_tensor)
    assert logits.shape == (0, 1)


# 8. Feature Normalization
def test_feature_normalization(synthetic_graph_and_dataset):
    _, dataset, _ = synthetic_graph_and_dataset
    node_feats = dataset.node_features.features

    # Physical positions normalized relative to volume dimensions [0, 1]
    assert np.all(node_feats[:, :3] >= 0.0) and np.all(node_feats[:, :3] <= 1.0)


# 9. Label Generation
def test_label_generation(synthetic_graph_and_dataset):
    _, dataset, _ = synthetic_graph_and_dataset
    labels = dataset.labels

    assert len(labels) == dataset.graph.candidate_count()
    assert np.sum(labels == 1.0) == 2  # 2 true GT edges
    assert np.sum(labels == 0.0) == len(labels) - 2


# 10. Temporal Split Correctness
def test_temporal_split_correctness(synthetic_graph_and_dataset):
    graph, dataset, _ = synthetic_graph_and_dataset
    splitter = TemporalSplitter(train_range=(0, 0), val_range=(1, 1), test_range=(2, 2))
    split = splitter.split_graph_edges(graph)

    # Frame 0 edges should be in train
    assert np.sum(split.train_mask) == graph.candidate_count()
    assert np.sum(split.val_mask) == 0
    assert np.sum(split.test_mask) == 0
    assert not np.any(split.train_mask & split.val_mask)


# 11. No GT Labels Entering Features
def test_no_gt_labels_in_features(synthetic_graph_and_dataset):
    _, dataset, _ = synthetic_graph_and_dataset
    node_f_names = dataset.node_features.feature_names
    edge_f_names = dataset.edge_features.feature_names

    forbidden_terms = ["gt", "label", "track_id", "target"]
    for name in node_f_names + edge_f_names:
        for term in forbidden_terms:
            assert term not in name.lower()


# 12. Loss Computation
def test_loss_computation(synthetic_graph_and_dataset):
    _, dataset, _ = synthetic_graph_and_dataset
    model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=32)

    logits = model(dataset.node_feats_tensor, dataset.edge_index_tensor, dataset.edge_feats_tensor).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, dataset.labels_tensor)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


# 13. Gradient Flow
def test_gradient_flow(synthetic_graph_and_dataset):
    _, dataset, _ = synthetic_graph_and_dataset
    model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=32)

    logits = model(dataset.node_feats_tensor, dataset.edge_index_tensor, dataset.edge_feats_tensor).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, dataset.labels_tensor)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert param.grad.abs().sum().item() > 0.0


# 14. Parameter Update
def test_parameter_update(synthetic_graph_and_dataset):
    _, dataset, _ = synthetic_graph_and_dataset
    model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    param_before = [p.clone() for p in model.parameters()]

    logits = model(dataset.node_feats_tensor, dataset.edge_index_tensor, dataset.edge_feats_tensor).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, dataset.labels_tensor)
    loss.backward()
    optimizer.step()

    param_after = [p for p in model.parameters()]
    changed = any(not torch.equal(b, a) for b, a in zip(param_before, param_after))

    assert changed


# 15. Controlled-Overfit Pipeline
def test_controlled_overfit_pipeline(synthetic_graph_and_dataset):
    _, dataset, _ = synthetic_graph_and_dataset
    model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=32)
    trainer = STGNNTrainer(model=model, config=STGNNTrainerConfig(epochs=40))

    mask = np.ones(dataset.graph.candidate_count(), dtype=bool)
    result = trainer.run_controlled_overfit_test(dataset, train_mask=mask, epochs=40)

    assert result.overfit_success
    assert result.final_loss < result.initial_loss


# 16. Deterministic Inference with Fixed Seed
def test_deterministic_inference(synthetic_graph_and_dataset):
    _, dataset, _ = synthetic_graph_and_dataset

    torch.manual_seed(42)
    m1 = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=32)
    probs1 = m1.predict_probabilities(dataset.node_feats_tensor, dataset.edge_index_tensor, dataset.edge_feats_tensor)

    torch.manual_seed(42)
    m2 = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=32)
    probs2 = m2.predict_probabilities(dataset.node_feats_tensor, dataset.edge_index_tensor, dataset.edge_feats_tensor)

    assert torch.equal(probs1, probs2)


# 17. Train split contains positives and negatives
def test_train_split_contains_pos_and_neg(transformer):
    node_set = NodeSet()
    # 2 source nodes at t=0, 3 dest nodes at t=1
    u1 = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    u2 = CanonicalNode(node_id=2, frame=0, global_voxel_pos=(0, 0, 20), physical_pos=(0.0, 0.0, 20.0), confidence=1.0)
    v1 = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(0, 0, 2), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)
    v2 = CanonicalNode(node_id=4, frame=1, global_voxel_pos=(0, 0, 18), physical_pos=(0.0, 0.0, 18.0), confidence=1.0)
    v_neg = CanonicalNode(node_id=5, frame=1, global_voxel_pos=(0, 0, 8), physical_pos=(0.0, 0.0, 8.0), confidence=1.0)
    node_set.add_nodes([u1, u2, v1, v2, v_neg])

    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=15.0))
    graph = builder.build_candidate_graph(node_set)

    gt_nodes = [CellNode.create(node_id=n.node_id, frame=n.frame, voxel_pos=(int(n.global_voxel_pos[0]), int(n.global_voxel_pos[1]), int(n.global_voxel_pos[2])), transformer=transformer) for n in [u1, u2, v1, v2, v_neg]]
    gt_edges = [(1, 3), (2, 4)]

    evaluator = CandidateRecallEvaluator(transformer=transformer)
    matching_res = evaluator.match_nodes_to_geff(node_set, gt_nodes, match_threshold_um=5.0)

    dataset = AssociationDataset(graph=graph, labeler=CandidateEdgeLabeler(), geff_edges=gt_edges, matching_result=matching_res)
    splitter = TemporalSplitter(train_range=(0, 0), val_range=(1, 1), test_range=(2, 2))
    split = splitter.split_graph_edges(graph)

    train_labels = dataset.labels[split.train_mask]
    assert np.sum(train_labels == 1.0) > 0
    assert np.sum(train_labels == 0.0) > 0


# 18. Validation split contains positives and negatives
def test_val_split_contains_pos_and_neg(transformer):
    node_set = NodeSet()
    u1 = CanonicalNode(node_id=1, frame=70, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v1 = CanonicalNode(node_id=2, frame=71, global_voxel_pos=(0, 0, 5), physical_pos=(0.0, 0.0, 2.03125), confidence=1.0)
    v2_neg = CanonicalNode(node_id=3, frame=71, global_voxel_pos=(0, 0, 15), physical_pos=(0.0, 0.0, 6.09375), confidence=1.0)
    node_set.add_nodes([u1, v1, v2_neg])

    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0))
    graph = builder.build_candidate_graph(node_set)

    gt_nodes = [
        CellNode(node_id=1, frame=70, voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0)),
        CellNode(node_id=2, frame=71, voxel_pos=(0, 0, 5), physical_pos=(0.0, 0.0, 2.03125)),
        CellNode(node_id=3, frame=71, voxel_pos=(0, 0, 15), physical_pos=(0.0, 0.0, 6.09375)),
    ]
    gt_edges = [(1, 2)]

    matching_res = CandidateRecallEvaluator(transformer=transformer).match_nodes_to_geff(node_set, gt_nodes, match_threshold_um=5.0)
    dataset = AssociationDataset(graph=graph, labeler=CandidateEdgeLabeler(), geff_edges=gt_edges, matching_result=matching_res)

    splitter = TemporalSplitter(train_range=(0, 69), val_range=(70, 84), test_range=(85, 99))
    split = splitter.split_graph_edges(graph)

    val_labels = dataset.labels[split.val_mask]
    assert np.sum(val_labels == 1.0) > 0
    assert np.sum(val_labels == 0.0) > 0


# 19. Test split contains positives and negatives
def test_test_split_contains_pos_and_neg(transformer):
    node_set = NodeSet()
    u1 = CanonicalNode(node_id=1, frame=85, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v1 = CanonicalNode(node_id=2, frame=86, global_voxel_pos=(0, 0, 5), physical_pos=(0.0, 0.0, 2.03125), confidence=1.0)
    v2_neg = CanonicalNode(node_id=3, frame=86, global_voxel_pos=(0, 0, 15), physical_pos=(0.0, 0.0, 6.09375), confidence=1.0)
    node_set.add_nodes([u1, v1, v2_neg])

    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0))
    graph = builder.build_candidate_graph(node_set)

    gt_nodes = [
        CellNode(node_id=1, frame=85, voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0)),
        CellNode(node_id=2, frame=86, voxel_pos=(0, 0, 5), physical_pos=(0.0, 0.0, 2.03125)),
        CellNode(node_id=3, frame=86, voxel_pos=(0, 0, 15), physical_pos=(0.0, 0.0, 6.09375)),
    ]
    gt_edges = [(1, 2)]

    matching_res = CandidateRecallEvaluator(transformer=transformer).match_nodes_to_geff(node_set, gt_nodes, match_threshold_um=5.0)
    dataset = AssociationDataset(graph=graph, labeler=CandidateEdgeLabeler(), geff_edges=gt_edges, matching_result=matching_res)

    splitter = TemporalSplitter(train_range=(0, 69), val_range=(70, 84), test_range=(85, 99))
    split = splitter.split_graph_edges(graph)

    test_labels = dataset.labels[split.test_mask]
    assert np.sum(test_labels == 1.0) > 0
    assert np.sum(test_labels == 0.0) > 0


# 20. Single-class evaluation raises error
def test_single_class_evaluation_raises_error():
    y_true_all_pos = np.ones(10, dtype=np.float32)
    y_prob = np.random.rand(10).astype(np.float32)

    with pytest.raises(ValueError, match="Single-class evaluation population is invalid"):
        AssociationEvaluator.compute_metrics(y_true_all_pos, y_prob, strict_single_class_check=True)


# 21. GT edges not used during candidate generation
def test_gt_edges_not_used_during_candidate_generation(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v_near = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 5), physical_pos=(0.0, 0.0, 5.0), confidence=1.0)
    v_far = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(0, 0, 50), physical_pos=(0.0, 0.0, 50.0), confidence=1.0)
    node_set.add_nodes([u, v_near, v_far])

    # Build candidate graph WITHOUT passing any GT edge info
    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0))
    graph = builder.build_candidate_graph(node_set)

    # Edge (1, 3) is far and must be excluded regardless of GT
    assert graph.get_edge(1, 3) is None
    assert graph.get_edge(1, 2) is not None


# 22. Hard negatives retained
def test_hard_negatives_retained(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v_gt = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 5), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)
    v_hard_neg = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(0, 0, 10), physical_pos=(0.0, 0.0, 4.0), confidence=1.0)
    node_set.add_nodes([u, v_gt, v_hard_neg])

    builder = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0))
    graph = builder.build_candidate_graph(node_set)

    gt_nodes = [
        CellNode(node_id=1, frame=0, voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0)),
        CellNode(node_id=2, frame=1, voxel_pos=(0, 0, 5), physical_pos=(0.0, 0.0, 2.0)),
        CellNode(node_id=3, frame=1, voxel_pos=(0, 0, 10), physical_pos=(0.0, 0.0, 4.0)),
    ]
    gt_edges = [(1, 2)]

    matching_res = CandidateRecallEvaluator(transformer=transformer).match_nodes_to_geff(node_set, gt_nodes, match_threshold_um=5.0)
    labeler = CandidateEdgeLabeler()
    labels, stats = labeler.create_labels(graph, gt_edges, matching_res)

    assert stats.negative_edges == 1
    assert stats.neg_edge_distances_um[0] == 4.0  # Hard negative retained at 4.0 um


# 23. Labels assigned only after candidate generation
def test_labels_assigned_after_candidate_generation(transformer):
    node_set = NodeSet()
    u = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    v = CanonicalNode(node_id=2, frame=1, global_voxel_pos=(0, 0, 3), physical_pos=(0.0, 0.0, 3.0), confidence=1.0)
    node_set.add_nodes([u, v])

    # 1. Candidate graph built first independently
    graph = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=10.0)).build_candidate_graph(node_set)
    num_edges_before = graph.candidate_count()

    # 2. Labeler creates label array for pre-existing graph edges
    gt_nodes = [CellNode(node_id=n.node_id, frame=n.frame, voxel_pos=(int(n.global_voxel_pos[0]), int(n.global_voxel_pos[1]), int(n.global_voxel_pos[2])), physical_pos=n.physical_pos) for n in [u, v]]
    matching_res = CandidateRecallEvaluator(transformer=transformer).match_nodes_to_geff(node_set, gt_nodes, match_threshold_um=5.0)

    labels, _ = CandidateEdgeLabeler().create_labels(graph, [(1, 2)], matching_res)

    # 3. Candidate graph edge structure is strictly unchanged
    assert graph.candidate_count() == num_edges_before
    assert len(labels) == num_edges_before

