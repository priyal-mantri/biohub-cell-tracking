#!/usr/bin/env python3
"""
Global ILP Optimization Experiment & Diagnostic Evaluation Script.
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DatasetConfig
from src.data import ZarrLoader, GeffLoader, CoordinateTransformer
from src.regional import RegionalCandidate
from src.graph import (
    CanonicalNode,
    NodeReconciler,
    NodeSet,
    CandidateEdge,
    CandidateGraphConfig,
    CandidateGraphBuilder,
    TemporalCandidateGraph,
    GraphFeatureExtractor,
    CandidateRecallEvaluator,
)
from src.association import (
    CandidateEdgeLabeler,
    TemporalSplitter,
    STGNNModel,
    AssociationDataset,
    STGNNTrainer,
    STGNNTrainerConfig,
)
from src.optimization import (
    OptimizationConfig,
    ObjectiveBuilder,
    ConstraintBuilder,
    MILPSolverWrapper,
    TrackFragmentDecoder,
    ILPDiagnostics,
    GlobalILPOptimizer,
)


def run_ilp_experiment():
    print("==================================================================", flush=True)
    print(" BioHub Cell Tracking - Global ILP Optimization Experiment ", flush=True)
    print("==================================================================", flush=True)

    # 1. Dataset & Loaders setup
    dataset_config = DatasetConfig.default_local_sample()
    zarr_loader = ZarrLoader(dataset_config.zarr_path)
    geff_loader = GeffLoader(dataset_config.geff_path)
    transformer = CoordinateTransformer()

    print(f"\n[1/7] Loading Dataset & Reconciling Canonical Nodes:", flush=True)
    print(f"  Volume Shape: {zarr_loader.shape}, dtype: {zarr_loader.dtype}", flush=True)
    print(f"  GEFF Ground-Truth Nodes: {geff_loader.num_nodes}, Edges: {geff_loader.num_edges}", flush=True)

    node_reconciler = NodeReconciler(reconciliation_distance_um=4.0, transformer=transformer)
    node_set = NodeSet()

    all_gt_nodes = geff_loader.get_all_nodes()
    min_f, max_f = geff_loader.temporal_range

    for f in range(min_f, max_f + 1):
        frame_gt_nodes = geff_loader.get_nodes_by_frame(f)
        reg_cands = []
        for g_node in frame_gt_nodes:
            reg_cand = RegionalCandidate(
                candidate_id=g_node.node_id,
                region_id=0,
                frame=f,
                local_voxel_pos=g_node.voxel_pos,
                global_voxel_pos=g_node.voxel_pos,
                physical_pos=g_node.physical_pos,
                confidence=1.0,
                is_in_core=True,
                provenance={"contributing_region_ids": [0], "num_contributing_regions": 1, "primary_region_id": 0}
            )
            reg_cands.append(reg_cand)

        canonical_frame_nodes = node_reconciler.reconcile_frame(f, reg_cands)
        node_set.add_nodes(canonical_frame_nodes)

    print(f"  Canonical Nodes Reconciled: {node_set.num_nodes()} across {node_set.num_frames()} frames", flush=True)

    # 2. Build Production Candidate Graph & Association Population
    print(f"\n[2/7] Constructing Candidate Graph & STGNN Evidence Scores:", flush=True)
    graph_config = CandidateGraphConfig(max_physical_distance_um=20.0, top_k_per_node=5)
    graph = CandidateGraphBuilder(config=graph_config).build_candidate_graph(node_set)

    evaluator = CandidateRecallEvaluator(transformer=transformer)
    matching_result = evaluator.match_nodes_to_geff(node_set, all_gt_nodes, match_threshold_um=5.0)
    geff_all_edges = [(int(u), int(v)) for u, v in geff_loader.directed_edges]

    # Prepare dataset and train STGNN evidence provider model
    labeler = CandidateEdgeLabeler()
    assoc_dataset = AssociationDataset(
        graph=graph,
        feature_extractor=GraphFeatureExtractor(transformer=transformer),
        labeler=labeler,
        geff_edges=geff_all_edges,
        matching_result=matching_result
    )

    splitter = TemporalSplitter(train_range=(0, 69), val_range=(70, 84), test_range=(85, 99))
    split = splitter.split_graph_edges(graph)

    stgnn_model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=64, num_layers=2)
    trainer = STGNNTrainer(model=stgnn_model, config=STGNNTrainerConfig(epochs=40, learning_rate=2e-3))
    trainer.train(assoc_dataset, split.train_mask, split.val_mask, epochs=40)

    # Predict STGNN association scores for all candidate edges
    stgnn_model.eval()
    with torch.no_grad():
        stgnn_logits = stgnn_model(
            assoc_dataset.node_feats_tensor,
            assoc_dataset.edge_index_tensor,
            assoc_dataset.edge_feats_tensor
        ).squeeze(-1)
        stgnn_scores = torch.sigmoid(stgnn_logits).cpu().numpy()

    print(f"  Candidate Edges: {graph.candidate_count()}", flush=True)
    print(f"  STGNN Evidence Scores: Mean={np.mean(stgnn_scores):.3f}, Min={np.min(stgnn_scores):.3f}, Max={np.max(stgnn_scores):.3f}", flush=True)

    # 3. Synthetic Conflict Resolution Demonstration (Part 10)
    print(f"\n[3/7] Part 10 — Synthetic Conflict Resolution Demonstration:", flush=True)
    conflict_nodes = NodeSet()
    u_a = CanonicalNode(node_id=1, frame=0, global_voxel_pos=(0, 0, 0), physical_pos=(0.0, 0.0, 0.0), confidence=1.0)
    u_d = CanonicalNode(node_id=2, frame=0, global_voxel_pos=(0, 0, 20), physical_pos=(0.0, 0.0, 20.0), confidence=1.0)
    v_b = CanonicalNode(node_id=3, frame=1, global_voxel_pos=(0, 0, 2), physical_pos=(0.0, 0.0, 2.0), confidence=1.0)
    v_c = CanonicalNode(node_id=4, frame=1, global_voxel_pos=(0, 0, 8), physical_pos=(0.0, 0.0, 8.0), confidence=1.0)
    conflict_nodes.add_nodes([u_a, u_d, v_b, v_c])

    conflict_graph = CandidateGraphBuilder(CandidateGraphConfig(max_physical_distance_um=25.0)).build_candidate_graph(conflict_nodes)

    e_ab = CandidateEdge(source_node_id=1, destination_node_id=3, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0.0, 0.0, 2.0), euclidean_distance_phys=2.0, voxel_displacement_vector=(0, 0, 2), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    e_ac = CandidateEdge(source_node_id=1, destination_node_id=4, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0.0, 0.0, 8.0), euclidean_distance_phys=8.0, voxel_displacement_vector=(0, 0, 8), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)
    e_db = CandidateEdge(source_node_id=2, destination_node_id=3, source_frame=0, destination_frame=1, frame_difference=1, physical_displacement_vector=(0.0, 0.0, -18.0), euclidean_distance_phys=18.0, voxel_displacement_vector=(0, 0, -18), source_confidence=1.0, destination_confidence=1.0, pair_confidence=1.0)

    conflict_graph.edges = [e_ab, e_ac, e_db]

    # Scores: A->B (0.90), A->C (0.85), D->B (0.95)
    # Local thresholding (>=0.5) would pick ALL THREE edges (violating in/out degree constraints).
    conflict_scores = np.array([0.90, 0.85, 0.95])
    conflict_opt = GlobalILPOptimizer(OptimizationConfig(mode="stgnn_only"))
    conflict_sol = conflict_opt.optimize(conflict_nodes, conflict_graph, stgnn_scores=conflict_scores)

    selected_pairs = set((e.source_node_id, e.destination_node_id) for e in conflict_sol.selected_edges)
    print(f"  Local Thresholding Picked 3 Edges: (1->3), (1->4), (2->3) -> [INCOMING/OUTGOING CONFLICT!]", flush=True)
    print(f"  Global ILP Selected Edges: {selected_pairs} -> [GLOBALLY OPTIMAL FEASIBLE FLOW!]", flush=True)
    assert selected_pairs == {(2, 3), (1, 4)}, "Conflict resolution test failed!"
    print("  [OK] Synthetic conflict resolution verified.", flush=True)

    # 4. Run Optimization Modes (Part 12 & Part 16)
    print(f"\n[4/7] Running Global ILP Optimization Across Objective Modes:", flush=True)

    modes = [
        ("STGNN-Only ILP", OptimizationConfig(mode="stgnn_only", sigma_scale_um=5.0)),
        ("Geometry-Only ILP", OptimizationConfig(mode="geometry_only", sigma_scale_um=5.0)),
        ("Combined ILP", OptimizationConfig(mode="combined", alpha=1.0, beta=1.0, sigma_scale_um=5.0)),
    ]

    ilp_diagnostics = ILPDiagnostics()
    mode_results = []
    mode_reports = {}

    for name, cfg in modes:
        optimizer = GlobalILPOptimizer(config=cfg)
        solution = optimizer.optimize(node_set, graph, stgnn_scores=stgnn_scores)

        report = ilp_diagnostics.evaluate_solution(solution, node_set, geff_edges=geff_all_edges, matching_result=matching_result)
        mode_reports[name] = report

        mode_results.append({
            "mode": name,
            "solution": solution,
            "report": report
        })

    # 5. Print Comparison Tables
    print(f"\n[5/7] Global ILP Objective Mode Comparison Tables:\n", flush=True)

    print("--- Table 1: Global ILP Optimization & Ground-Truth Performance ---", flush=True)
    print("| Method             | Variables | Constraints | Selected Edges | Solve Time (s) | Precision | Recall | F1 Score | Violations |", flush=True)
    print("|--------------------|-----------|-------------|----------------|----------------|-----------|--------|----------|------------|", flush=True)
    for res in mode_results:
        n = res["mode"]
        sol = res["solution"]
        rep = res["report"]
        s_res = sol.solver_result
        print(f"| {n:<18} | {s_res.num_variables:<9} | {s_res.num_constraints:<11} | {rep.selected_edges:<14} | {s_res.solve_time_sec:<14.4f} | {rep.edge_precision:<9.4f} | {rep.edge_recall:<6.4f} | {rep.edge_f1:<8.4f} | {rep.constraint_violations_count:<10} |", flush=True)

    print("\n--- Table 2: Direct-Continuation Track Fragment & Node Degree Diagnostics ---", flush=True)
    print("| Method             | Fragments | Mean Length | Max Length | Isolated Nodes | Pred Only | Succ Only | Both | Neither |", flush=True)
    print("|--------------------|-----------|-------------|------------|----------------|-----------|-----------|------|---------|", flush=True)
    for res in mode_results:
        n = res["mode"]
        rep = res["report"]
        print(f"| {n:<18} | {rep.num_track_fragments:<9} | {rep.fragment_lengths['mean']:<11.2f} | {rep.longest_fragment_len:<10} | {rep.isolated_nodes_count:<14} | {rep.nodes_with_predecessor_only:<9} | {rep.nodes_with_successor_only:<9} | {rep.nodes_with_both:<4} | {rep.nodes_with_neither:<7} |", flush=True)

    # 6. Sanity Checks Verification (Part 17)
    print(f"\n[6/7] Part 17 — Solver Sanity & Constraint Verification:", flush=True)
    for res in mode_results:
        rep = res["report"]
        assert rep.constraint_violations_count == 0, f"Constraint violation in {res['mode']}!"
        assert rep.selected_edges > 0, f"No selected edges in {res['mode']}!"

    print("  [OK] Zero constraint violations verified across all optimization modes.", flush=True)

    # 7. Save Machine-Readable Results (Part 18)
    print(f"\n[7/7] Saving Results Artifact:", flush=True)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / "ilp_experiment_results.json"

    save_payload = {
        "candidate_graph_stats": {
            "num_nodes": node_set.num_nodes(),
            "num_frames": node_set.num_frames(),
            "num_candidate_edges": graph.candidate_count(),
        },
        "synthetic_conflict_test": {
            "success": True,
            "selected_pairs": [list(p) for p in selected_pairs]
        },
        "mode_results": {
            res["mode"]: {
                "solver_result": res["solution"].solver_result.to_dict(),
                "diagnostics": res["report"].to_dict(),
            }
            for res in mode_results
        }
    }

    with open(out_file, "w") as f:
        json.dump(save_payload, f, indent=2)

    print(f"  Machine-readable ILP results saved to: {out_file}", flush=True)

    print("\n==================================================================", flush=True)
    print(" IMPORTANT ARCHITECTURAL PRINCIPLE:")
    print(" 'The STGNN supplies local association evidence;")
    print("  the ILP performs globally constrained selection.'")
    print("==================================================================", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    run_ilp_experiment()
