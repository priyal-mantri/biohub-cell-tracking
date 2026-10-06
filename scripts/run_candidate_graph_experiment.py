#!/usr/bin/env python3
"""
Candidate Graph Construction, Diagnostic Recall Evaluation & Multi-Configuration Experiment.
"""

import sys
import json
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DatasetConfig
from src.data import ZarrLoader, GeffLoader, CoordinateTransformer, CellNode
from src.regional import BoundingBox3D, Region3D, RegionalCandidate
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


def run_candidate_graph_experiment():
    print("==================================================================", flush=True)
    print(" BioHub Cell Tracking - Temporal Candidate Graph & Recall Diagnostics ", flush=True)
    print("==================================================================", flush=True)

    # 1. Dataset & Loaders setup
    dataset_config = DatasetConfig.default_local_sample()
    zarr_loader = ZarrLoader(dataset_config.zarr_path)
    geff_loader = GeffLoader(dataset_config.geff_path)
    transformer = CoordinateTransformer()

    print(f"\n[1/6] Loading Dataset:", flush=True)
    print(f"  Microscopy Volume Shape: {zarr_loader.shape}, dtype: {zarr_loader.dtype}", flush=True)
    print(f"  GEFF Ground-Truth Nodes: {geff_loader.num_nodes}, Edges: {geff_loader.num_edges}", flush=True)
    print(f"  Temporal Frame Range:    {geff_loader.temporal_range}", flush=True)

    # 2. Global Node Reconciliation across all 100 frames
    print(f"\n[2/6] Global Node Reconciliation across per-frame cell detections:", flush=True)
    node_reconciler = NodeReconciler(reconciliation_distance_um=4.0, transformer=transformer)
    node_set = NodeSet()

    all_gt_nodes = geff_loader.get_all_nodes()
    min_f, max_f = geff_loader.temporal_range

    # Convert per-frame detections/nodes into canonical nodes through NodeReconciler
    for f in range(min_f, max_f + 1):
        frame_gt_nodes = geff_loader.get_nodes_by_frame(f)
        # Create regional candidate representations to test regional reconciliation pipeline
        reg_cands = []
        for idx, g_node in enumerate(frame_gt_nodes):
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

    print(f"  Total Canonical Nodes Reconciled: {node_set.num_nodes()} across {node_set.num_frames()} frames.", flush=True)

    # 3. Part 11: Data Integrity Check (Node Matching against GEFF)
    print(f"\n[3/6] Part 11 — Data Integrity Check (Node Matching against GEFF):", flush=True)
    evaluator = CandidateRecallEvaluator(transformer=transformer)

    matching_result = evaluator.match_nodes_to_geff(
        node_set=node_set,
        geff_nodes=all_gt_nodes,
        match_threshold_um=5.0
    )

    print(f"  GEFF Ground-Truth Node Count: {matching_result.geff_node_count}", flush=True)
    print(f"  Canonical Detection Node Count: {matching_result.canonical_node_count}", flush=True)
    print(f"  Matched GEFF Nodes:            {matching_result.matched_geff_count}", flush=True)
    print(f"  Unmatched GEFF Nodes:          {matching_result.unmatched_geff_count}", flush=True)
    print(f"  Node Matching Success Rate:    {matching_result.matching_success_rate:.2f}%", flush=True)

    if matching_result.match_distances_um:
        print(f"  Mean Node Match Distance:      {np.mean(matching_result.match_distances_um):.3f} um", flush=True)

    assert matching_result.matching_success_rate == 100.0, "Node matching failed! Cannot proceed to recall evaluation."
    print("  [OK] 100% Node matching success rate verified.", flush=True)

    geff_all_edges = [(int(u), int(v)) for u, v in geff_loader.directed_edges]
    print(f"  GEFF Temporal Edges to Evaluate: {len(geff_all_edges)}", flush=True)

    # 4. Multi-Configuration Candidate Graph Experiment
    print(f"\n[4/6] Part 6 — Running Multi-Configuration Candidate Graph Experiment:", flush=True)

    test_configs = [
        ("d_max=10um, top_k=3", CandidateGraphConfig(max_physical_distance_um=10.0, top_k_per_node=3)),
        ("d_max=15um, top_k=5", CandidateGraphConfig(max_physical_distance_um=15.0, top_k_per_node=5)),
        ("d_max=20um, top_k=5", CandidateGraphConfig(max_physical_distance_um=20.0, top_k_per_node=5)),
        ("d_max=20um, top_k=10", CandidateGraphConfig(max_physical_distance_um=20.0, top_k_per_node=10)),
        ("d_max=30um, top_k=None", CandidateGraphConfig(max_physical_distance_um=30.0, top_k_per_node=None)),
    ]

    experiment_results = []
    summary_table_rows = []

    for name, cfg in test_configs:
        builder = CandidateGraphBuilder(config=cfg)
        graph = builder.build_candidate_graph(node_set)

        recall_metrics = evaluator.evaluate_recall(
            graph=graph,
            geff_edges=geff_all_edges,
            geff_nodes=all_gt_nodes,
            matching_result=matching_result
        )

        row = {
            "configuration": name,
            "candidate_edges": recall_metrics.num_candidate_edges,
            "mean_candidates_per_node": recall_metrics.candidate_graph_density,
            "candidate_recall": recall_metrics.candidate_recall,
            "evaluable_recall": recall_metrics.evaluable_candidate_recall,
            "zero_candidate_pct": recall_metrics.zero_candidate_pct,
            "recovered_gt_edges": recall_metrics.gt_edges_recovered,
            "missing_gt_edges": recall_metrics.num_missing_gt_edges,
            "missing_reasons": recall_metrics.missing_reasons,
        }
        experiment_results.append(row)
        summary_table_rows.append(row)

    # 5. Print Compact Comparison Table
    print(f"\n[5/6] Candidate Graph Configuration Comparison Table:\n", flush=True)
    print(f"| Configuration           | Candidate Edges | Mean Candidates/Node | Candidate Recall | Zero-Candidate % |", flush=True)
    print(f"|-------------------------|-----------------|----------------------|------------------|------------------|", flush=True)
    for r in summary_table_rows:
        print(f"| {r['configuration']:<23} | {r['candidate_edges']:<15} | {r['mean_candidates_per_node']:<20.2f} | {r['candidate_recall']*100.0:<15.2f}% | {r['zero_candidate_pct']:<15.2f}% |", flush=True)

    # 6. Detailed GT Missing Edge Diagnostics & Feature Extraction Verification
    print(f"\n[6/6] Detailed Diagnostics & Feature Interface Verification:", flush=True)

    sample_cfg = CandidateGraphConfig(max_physical_distance_um=20.0, top_k_per_node=5)
    sample_graph = CandidateGraphBuilder(config=sample_cfg).build_candidate_graph(node_set)

    feature_extractor = GraphFeatureExtractor(volume_shape=zarr_loader.shape, transformer=transformer)
    extracted_feats = feature_extractor.extract(sample_graph)

    node_f = extracted_feats["node_features"]
    edge_f = extracted_feats["edge_features"]

    print(f"  Extracted Node Feature Matrix Shape: {node_f.features.shape}", flush=True)
    print(f"  Extracted Edge Feature Matrix Shape: {edge_f.features.shape}", flush=True)
    print(f"  Edge Index Shape:                   {edge_f.edge_index.shape}", flush=True)

    # Ground-truth edge physical distance stats
    sample_recall = evaluator.evaluate_recall(sample_graph, geff_all_edges, all_gt_nodes, matching_result)
    gt_dists = sample_recall.gt_edge_distance_dist
    print(f"  GEFF Ground-Truth Edge Distance Stats (um):", flush=True)
    print(f"    Min: {gt_dists['min']:.3f}, Median: {gt_dists['median']:.3f}, Max: {gt_dists['max']:.3f}, Mean: {gt_dists['mean']:.3f}", flush=True)
    print(f"  Missing GT Edge Breakdown (for d_max=20um, top_k=5):", flush=True)
    for reason, count in sample_recall.missing_reasons.items():
        print(f"    - {reason}: {count}", flush=True)

    # Save results to JSON artifact
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / "candidate_graph_experiment_results.json"

    save_payload = {
        "matching_result": {
            "geff_node_count": matching_result.geff_node_count,
            "canonical_node_count": matching_result.canonical_node_count,
            "matched_geff_count": matching_result.matched_geff_count,
            "unmatched_geff_count": matching_result.unmatched_geff_count,
            "matching_success_rate": matching_result.matching_success_rate,
        },
        "gt_edge_distance_dist": gt_dists,
        "configurations": experiment_results
    }

    with open(out_file, "w") as f:
        json.dump(save_payload, f, indent=2)

    print(f"  Machine-readable results saved to: {out_file}", flush=True)

    print("\n==================================================================", flush=True)
    print(" IMPORTANT DESIGN PRINCIPLE:")
    print(" 'Candidate graph generation proposes plausible temporal associations;")
    print("  it does not decide biological identity.'")
    print("==================================================================", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    run_candidate_graph_experiment()
