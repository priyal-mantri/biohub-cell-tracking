#!/usr/bin/env python3
"""
STGNN Temporal Candidate Edge Association Experiment (v2 Corrected Evaluation Population).
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


def run_stgnn_experiment():
    print("==================================================================", flush=True)
    print(" BioHub Cell Tracking - STGNN Temporal Association Experiment v2 ", flush=True)
    print("==================================================================", flush=True)

    # 1. Dataset & Loaders setup
    dataset_config = DatasetConfig.default_local_sample()
    zarr_loader = ZarrLoader(dataset_config.zarr_path)
    geff_loader = GeffLoader(dataset_config.geff_path)
    transformer = CoordinateTransformer()

    print(f"\n[1/8] Loading Dataset & Reconciling Canonical Nodes:", flush=True)
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

    print(f"  Canonical Nodes: {node_set.num_nodes()} across {node_set.num_frames()} frames", flush=True)

    # 2. Candidate Population Construction (Separation of Production Graph & Association Population)
    print(f"\n[2/8] Constructing Association Evaluation Population (Hard Negatives):", flush=True)
    print("  Distinction Documented:")
    print("    A. Production candidate graph: High-recall candidate generation (d_max=20um, top_k=5).")
    print("    B. Association evaluation population: Broader candidate population (d_max=35um, top_k=15)")
    print("       containing realistic false candidates from real next-frame cells to evaluate learned scoring.")

    eval_graph_config = CandidateGraphConfig(max_physical_distance_um=35.0, top_k_per_node=15)
    eval_graph = CandidateGraphBuilder(config=eval_graph_config).build_candidate_graph(node_set)

    evaluator = CandidateRecallEvaluator(transformer=transformer)
    matching_result = evaluator.match_nodes_to_geff(node_set, all_gt_nodes, match_threshold_um=5.0)
    geff_all_edges = [(int(u), int(v)) for u, v in geff_loader.directed_edges]

    # Verify GT edges are retained in candidate graph
    recall_check = evaluator.evaluate_recall(eval_graph, geff_all_edges, all_gt_nodes, matching_result)
    print(f"  GT Edges Recovered: {recall_check.gt_edges_recovered} / {recall_check.num_gt_edges} (Recall: {recall_check.candidate_recall*100.0:.2f}%)", flush=True)
    assert recall_check.candidate_recall == 1.0, "GT edge recall must be 100% in candidate population!"

    # 3. Label Generation & Temporal Data Splitting (Parts 3, 4, 7)
    print(f"\n[3/8] Generating Candidate Edge Labels & Mandatory Dataset Sanity Checks:", flush=True)
    labeler = CandidateEdgeLabeler()
    dataset = AssociationDataset(
        graph=eval_graph,
        feature_extractor=GraphFeatureExtractor(transformer=transformer),
        labeler=labeler,
        geff_edges=geff_all_edges,
        matching_result=matching_result
    )

    stats = dataset.label_stats
    print(f"  Total Candidate Population Edges: {stats.total_candidate_edges}", flush=True)
    print(f"  Positive GT Edges:                {stats.positive_edges} ({stats.positive_rate_pct:.2f}%)", flush=True)
    print(f"  Negative Candidate Edges:         {stats.negative_edges} ({100.0 - stats.positive_rate_pct:.2f}%)", flush=True)

    splitter = TemporalSplitter(train_range=(0, 69), val_range=(70, 84), test_range=(85, 99))
    split = splitter.split_graph_edges(eval_graph)

    tr_mask = split.train_mask
    val_mask = split.val_mask
    te_mask = split.test_mask

    tr_labels = dataset.labels[tr_mask]
    val_labels = dataset.labels[val_mask]
    te_labels = dataset.labels[te_mask]

    def get_split_stats(labels_arr):
        tot = len(labels_arr)
        pos = int(np.sum(labels_arr == 1.0))
        neg = int(np.sum(labels_arr == 0.0))
        p_pct = 100.0 * pos / tot if tot > 0 else 0.0
        n_pct = 100.0 * neg / tot if tot > 0 else 0.0
        return tot, pos, neg, p_pct, n_pct

    tr_tot, tr_pos, tr_neg, tr_ppct, tr_npct = get_split_stats(tr_labels)
    val_tot, val_pos, val_neg, val_ppct, val_npct = get_split_stats(val_labels)
    te_tot, te_pos, te_neg, te_ppct, te_npct = get_split_stats(te_labels)

    print("\n  Dataset Split Class Distribution Table:", flush=True)
    print("  -------------------------------------------------------------------------", flush=True)
    print(f"  Train (Frames 0-69):   Total={tr_tot:<5} | Pos={tr_pos:<4} ({tr_ppct:.1f}%) | Neg={tr_neg:<5} ({tr_npct:.1f}%)", flush=True)
    print(f"  Val   (Frames 70-84):  Total={val_tot:<5} | Pos={val_pos:<4} ({val_ppct:.1f}%) | Neg={val_neg:<5} ({val_npct:.1f}%)", flush=True)
    print(f"  Test  (Frames 85-99):  Total={te_tot:<5} | Pos={te_pos:<4} ({te_ppct:.1f}%) | Neg={te_neg:<5} ({te_npct:.1f}%)", flush=True)
    print("  -------------------------------------------------------------------------", flush=True)

    # Mandatory Dataset Sanity Check (Part 7)
    assert tr_pos > 0 and tr_neg > 0, "Train split must contain both positive and negative examples!"
    assert val_pos > 0 and val_neg > 0, "Validation split must contain both positive and negative examples!"
    assert te_pos > 0 and te_neg > 0, "Test split must contain both positive and negative examples!"
    print("  [OK] All splits (Train, Val, Test) contain both positive and negative examples.", flush=True)

    # 4. Hard-Negative Diagnostics & Sample Inspections (Parts 4, 8, 10)
    print(f"\n[4/8] Hard-Negative Distance Diagnostics & Label Sample Audit:", flush=True)

    pos_dists = np.array(stats.pos_edge_distances_um, dtype=np.float32)
    neg_dists = np.array(stats.neg_edge_distances_um, dtype=np.float32)

    print(f"  Positive Edge Physical Distances (um): Min={np.min(pos_dists):.3f}, Med={np.median(pos_dists):.3f}, Max={np.max(pos_dists):.3f}", flush=True)
    print(f"  Negative Edge Physical Distances (um): Min={np.min(neg_dists):.3f}, Med={np.median(neg_dists):.3f}, Max={np.max(neg_dists):.3f}", flush=True)

    bins = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 35.0)]
    print("\n  Candidate Distance Range Breakdown Table:", flush=True)
    print("  | Distance Range | Positive Edges | Negative Edges |", flush=True)
    print("  |----------------|----------------|----------------|", flush=True)
    for b_min, b_max in bins:
        p_c = int(np.sum((pos_dists >= b_min) & (pos_dists < b_max)))
        n_c = int(np.sum((neg_dists >= b_min) & (neg_dists < b_max)))
        print(f"  | {b_min:2.0f}-{b_max:<2.0f} um        | {p_c:<14} | {n_c:<14} |", flush=True)

    # Audit concrete edge examples
    print("\n  Concrete Candidate Label Audit Samples:", flush=True)
    pos_sample = [e for e in eval_graph.edges if dataset.labels[eval_graph.edges.index(e)] == 1.0][0]
    neg_sample = [e for e in eval_graph.edges if dataset.labels[eval_graph.edges.index(e)] == 0.0][0]

    u_pos_node = eval_graph.node_set.get_node(pos_sample.source_node_id)
    v_pos_node = eval_graph.node_set.get_node(pos_sample.destination_node_id)

    u_neg_node = eval_graph.node_set.get_node(neg_sample.source_node_id)
    v_neg_node = eval_graph.node_set.get_node(neg_sample.destination_node_id)

    print(f"    Positive Edge Sample: Src Node {pos_sample.source_node_id} (f={pos_sample.source_frame}) -> Dst Node {pos_sample.destination_node_id} (f={pos_sample.destination_frame})", flush=True)
    print(f"      Src Phys Pos: {u_pos_node.physical_pos} | Dst Phys Pos: {v_pos_node.physical_pos} | Phys Dist: {pos_sample.euclidean_distance_phys:.3f} um | GT Label: 1", flush=True)

    print(f"    Negative Edge Sample: Src Node {neg_sample.source_node_id} (f={neg_sample.source_frame}) -> Dst Node {neg_sample.destination_node_id} (f={neg_sample.destination_frame})", flush=True)
    print(f"      Src Phys Pos: {u_neg_node.physical_pos} | Dst Phys Pos: {v_neg_node.physical_pos} | Phys Dist: {neg_sample.euclidean_distance_phys:.3f} um | GT Label: 0", flush=True)

    # 5. Pipeline Integrity Checks & Controlled Overfit Test (Parts 12 & 14)
    print(f"\n[5/8] Part 12 — Pipeline Integrity Audit & Part 13 Controlled Overfit Test:", flush=True)
    stgnn_init_model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=64)
    trainer = STGNNTrainer(model=stgnn_init_model, config=STGNNTrainerConfig(epochs=40))

    checks = trainer.run_pipeline_integrity_checks(dataset, tr_mask, val_mask, te_mask)
    all_passed = all(checks.values())
    for check_name, passed in checks.items():
        print(f"  - {check_name:<40}: {'[PASS]' if passed else '[FAIL]'}", flush=True)
    assert all_passed, "Pipeline integrity check failed!"

    overfit_result = trainer.run_controlled_overfit_test(dataset, train_mask=tr_mask[:100], epochs=40)
    print(f"  Overfit Test Loss: {overfit_result.initial_loss:.4f} -> {overfit_result.final_loss:.4f} | Status: {'[SUCCESS]' if overfit_result.overfit_success else '[FAILED]'}", flush=True)

    # 6. Model Training on Balanced Class Weighting (Part 11)
    print(f"\n[6/8] Training Graph-Aware STGNN Model on Train Split:", flush=True)
    pos_weight_val = float(tr_neg / tr_pos) if tr_pos > 0 else 1.0
    print(f"  Using BCE Loss Pos Weight: {pos_weight_val:.2f} (derived from raw ratio {tr_neg}/{tr_pos})", flush=True)

    stgnn_model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=64, num_layers=2, dropout=0.1)
    main_trainer = STGNNTrainer(
        model=stgnn_model,
        config=STGNNTrainerConfig(epochs=50, learning_rate=2e-3, pos_weight=pos_weight_val)
    )
    history = main_trainer.train(dataset, tr_mask, val_mask, epochs=50)

    print(f"  Final Train Loss: {history['train_loss'][-1]:.4f} | Best Val Loss: {min(history['val_loss']):.4f} | Best Val ROC-AUC: {max(history['val_roc_auc']):.4f}", flush=True)

    # 7. Model Comparisons & Ablation on Held-out Test Set (Parts 12 & 13)
    print(f"\n[7/8] Held-Out Test Set Evaluation & Ablation Comparisons (Frames 85-99):", flush=True)

    test_tensors = dataset.get_split_tensors(te_mask)
    stgnn_model.eval()
    with torch.no_grad():
        test_logits = stgnn_model(test_tensors["node_feats"], test_tensors["edge_index"], test_tensors["edge_feats"]).squeeze(-1)
        stgnn_test_probs = torch.sigmoid(test_logits).cpu().numpy()

    stgnn_test_metrics = AssociationEvaluator.compute_metrics(te_labels, stgnn_test_probs, strict_single_class_check=True)

    # Geometric Baseline
    test_edges = [eval_graph.edges[idx] for idx in np.where(te_mask)[0]]
    geom_baseline = GeometricBaseline(sigma_scale_um=2.0)
    geom_probs = geom_baseline.predict_edge_scores(test_edges)
    geom_test_metrics = AssociationEvaluator.compute_metrics(te_labels, geom_probs, strict_single_class_check=True)

    # Geometry-Only STGNN Ablation
    geom_only_model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=64, num_layers=2, geometry_only=True)
    geom_trainer = STGNNTrainer(
        model=geom_only_model,
        config=STGNNTrainerConfig(epochs=50, learning_rate=2e-3, pos_weight=pos_weight_val)
    )
    geom_trainer.train(dataset, tr_mask, val_mask, epochs=50)

    geom_only_model.eval()
    with torch.no_grad():
        geom_only_test_probs = torch.sigmoid(geom_only_model(test_tensors["node_feats"], test_tensors["edge_index"], test_tensors["edge_feats"]).squeeze(-1)).cpu().numpy()
    geom_only_test_metrics = AssociationEvaluator.compute_metrics(te_labels, geom_only_test_probs, strict_single_class_check=True)

    print("\n--- Corrected Model Baseline Comparison Table ---", flush=True)
    print("| Method              | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | Specificity |", flush=True)
    print("|---------------------|---------|--------|-----------|--------|----------|-------------|", flush=True)
    print(f"| Geometric Baseline  | {geom_test_metrics.roc_auc:<7.4f} | {geom_test_metrics.pr_auc:<6.4f} | {geom_test_metrics.precision:<9.4f} | {geom_test_metrics.recall:<6.4f} | {geom_test_metrics.f1:<8.4f} | {geom_test_metrics.negative_edge_specificity:<11.4f} |", flush=True)
    print(f"| Geometry-Only STGNN | {geom_only_test_metrics.roc_auc:<7.4f} | {geom_only_test_metrics.pr_auc:<6.4f} | {geom_only_test_metrics.precision:<9.4f} | {geom_only_test_metrics.recall:<6.4f} | {geom_only_test_metrics.f1:<8.4f} | {geom_only_test_metrics.negative_edge_specificity:<11.4f} |", flush=True)
    print(f"| Graph-Aware STGNN   | {stgnn_test_metrics.roc_auc:<7.4f} | {stgnn_test_metrics.pr_auc:<6.4f} | {stgnn_test_metrics.precision:<9.4f} | {stgnn_test_metrics.recall:<6.4f} | {stgnn_test_metrics.f1:<8.4f} | {stgnn_test_metrics.negative_edge_specificity:<11.4f} |", flush=True)

    # Confusion matrix stats on test set
    y_pred_test = (stgnn_test_probs >= 0.5).astype(np.float32)
    tp = int(np.sum((y_pred_test == 1.0) & (te_labels == 1.0)))
    fp = int(np.sum((y_pred_test == 1.0) & (te_labels == 0.0)))
    tn = int(np.sum((y_pred_test == 0.0) & (te_labels == 0.0)))
    fn = int(np.sum((y_pred_test == 0.0) & (te_labels == 1.0)))

    print("\n  Graph-Aware STGNN Test Set Confusion Matrix:", flush=True)
    print(f"    TP={tp:<4} | FP={fp:<4}", flush=True)
    print(f"    FN={fn:<4} | TN={tn:<4}", flush=True)

    # 8. Save Results (Part 16)
    print(f"\n[8/8] Saving Separated Experiment Results Artifact v2:", flush=True)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / "stgnn_experiment_results_v2.json"

    save_payload = {
        "candidate_population": {
            "max_physical_distance_um": eval_graph_config.max_physical_distance_um,
            "top_k_per_node": eval_graph_config.top_k_per_node,
            "gt_edge_recall": recall_check.candidate_recall,
        },
        "class_distributions": {
            "total_population": stats.total_candidate_edges,
            "total_positives": stats.positive_edges,
            "total_negatives": stats.negative_edges,
            "train_split": {"total": tr_tot, "pos": tr_pos, "neg": tr_neg},
            "val_split": {"total": val_tot, "pos": val_pos, "neg": val_neg},
            "test_split": {"total": te_tot, "pos": te_pos, "neg": te_neg},
        },
        "hard_negative_stats": {
            "pos_min_med_max": [float(np.min(pos_dists)), float(np.median(pos_dists)), float(np.max(pos_dists))],
            "neg_min_med_max": [float(np.min(neg_dists)), float(np.median(neg_dists)), float(np.max(neg_dists))],
        },
        "overfit_test": {
            "initial_loss": overfit_result.initial_loss,
            "final_loss": overfit_result.final_loss,
            "overfit_success": overfit_result.overfit_success,
        },
        "geometric_baseline_test_metrics": geom_test_metrics.to_dict(),
        "geometry_only_stgnn_test_metrics": geom_only_test_metrics.to_dict(),
        "graph_aware_stgnn_test_metrics": stgnn_test_metrics.to_dict(),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }

    with open(out_file, "w") as f:
        json.dump(save_payload, f, indent=2)

    print(f"  Machine-readable corrected results saved to: {out_file}", flush=True)

    print("\n==================================================================", flush=True)
    print(" IMPORTANT ARCHITECTURAL PRINCIPLE:")
    print(" 'Local learned models provide association evidence;")
    print("  they do not make the final global tracking decision.'")
    print("==================================================================", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    run_stgnn_experiment()
