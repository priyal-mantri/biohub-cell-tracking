#!/usr/bin/env python3
"""
Clean, Reproducible STGNN Checkpoint Creation & Verification Script.
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import torch

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
    AssociationEvaluator,
)


def create_and_verify_checkpoint():
    print("==================================================================", flush=True)
    print(" STGNN Checkpoint Creation & Reload Verification ", flush=True)
    print("==================================================================", flush=True)

    # Set deterministic seeds
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Dataset & Graph Setup
    dataset_config = DatasetConfig.default_local_sample()
    geff_loader = GeffLoader(dataset_config.geff_path)
    transformer = CoordinateTransformer()

    node_reconciler = NodeReconciler(reconciliation_distance_um=4.0, transformer=transformer)
    node_set = NodeSet()

    all_gt_nodes = geff_loader.get_all_nodes()
    min_f, max_f = geff_loader.temporal_range

    for f in range(min_f, max_f + 1):
        frame_gt_nodes = geff_loader.get_nodes_by_frame(f)
        reg_cands = [
            RegionalCandidate(
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
            for g_node in frame_gt_nodes
        ]
        canonical_frame_nodes = node_reconciler.reconcile_frame(f, reg_cands)
        node_set.add_nodes(canonical_frame_nodes)

    # Candidate population: 35 um, K=15
    graph_config = CandidateGraphConfig(max_physical_distance_um=35.0, top_k_per_node=15)
    graph = CandidateGraphBuilder(config=graph_config).build_candidate_graph(node_set)
    print(f"  Candidate Graph: {node_set.num_nodes()} nodes, {graph.candidate_count()} candidate edges")

    evaluator = CandidateRecallEvaluator(transformer=transformer)
    matching_result = evaluator.match_nodes_to_geff(node_set, all_gt_nodes, match_threshold_um=5.0)
    geff_all_edges = [(int(u), int(v)) for u, v in geff_loader.directed_edges]

    # Temporal Split: Train 0-69, Val 70-84, Test 85-99
    splitter = TemporalSplitter(train_range=(0, 69), val_range=(70, 84), test_range=(85, 99))
    split = splitter.split_graph_edges(graph)

    print(f"  Split Edges — Train (0-69): {int(split.train_mask.sum())}, Val (70-84): {int(split.val_mask.sum())}, Test (85-99): {int(split.test_mask.sum())}")

    # 2. Association Dataset & Model Initialization
    labeler = CandidateEdgeLabeler()
    feature_extractor = GraphFeatureExtractor(transformer=transformer)
    dataset = AssociationDataset(
        graph=graph,
        feature_extractor=feature_extractor,
        labeler=labeler,
        geff_edges=geff_all_edges,
        matching_result=matching_result
    )

    model_config_dict = {
        "node_in_dim": 9,
        "edge_in_dim": 16,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1
    }

    stgnn_model = STGNNModel(**model_config_dict)

    trainer_config = STGNNTrainerConfig(
        epochs=40,
        learning_rate=2e-3,
        weight_decay=1e-4,
        seed=seed
    )

    trainer = STGNNTrainer(model=stgnn_model, config=trainer_config, device="cpu")

    # 3. Train Model (Frames 0-69 for updates, 70-84 for val checkpoint selection)
    print("\n[1/3] Training STGNN (Train: 0-69, Val selection: 70-84)...", flush=True)
    history = trainer.train(dataset, split.train_mask, split.val_mask, epochs=40)

    best_val_loss = min(history["val_loss"])
    selected_epoch = history["val_loss"].index(best_val_loss) + 1
    train_loss_at_best = history["train_loss"][selected_epoch - 1]

    print(f"  Selected Epoch: {selected_epoch} / 40")
    print(f"  Train Loss: {train_loss_at_best:.4f}, Val Loss: {best_val_loss:.4f}")

    # Get predictions from in-memory trained model
    stgnn_model.eval()
    with torch.no_grad():
        orig_logits = stgnn_model(
            dataset.node_feats_tensor,
            dataset.edge_index_tensor,
            dataset.edge_feats_tensor
        ).squeeze(-1)
        orig_probs = torch.sigmoid(orig_logits).numpy()

    # 4. Save Structured Checkpoint File
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    checkpoint_path = models_dir / "stgnn_checkpoint.pt"

    checkpoint_payload = {
        "model_state_dict": stgnn_model.state_dict(),
        "model_config": model_config_dict,
        "trainer_config": {
            "epochs": trainer_config.epochs,
            "learning_rate": trainer_config.learning_rate,
            "weight_decay": trainer_config.weight_decay,
            "seed": trainer_config.seed
        },
        "candidate_config": {
            "max_physical_distance_um": graph_config.max_physical_distance_um,
            "top_k_per_node": graph_config.top_k_per_node
        },
        "temporal_split": {
            "train_range": [0, 69],
            "val_range": [70, 84],
            "test_range": [85, 99]
        },
        "training_history": history,
        "selected_epoch": selected_epoch,
        "best_val_loss": best_val_loss,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_size_mb = checkpoint_path.stat().st_size / 1e6

    print(f"\n[2/3] Saved STGNN Checkpoint to: {checkpoint_path}")
    print(f"  Checkpoint File Size: {checkpoint_size_mb:.3f} MB ({checkpoint_path.stat().st_size} bytes)")

    # 5. Reload Verification
    print("\n[3/3] Performing Reload Verification:", flush=True)
    loaded_payload = torch.load(checkpoint_path, map_location="cpu")
    reloaded_model = STGNNModel(**loaded_payload["model_config"])
    reloaded_model.load_state_dict(loaded_payload["model_state_dict"])
    reloaded_model.eval()

    with torch.no_grad():
        reloaded_logits = reloaded_model(
            dataset.node_feats_tensor,
            dataset.edge_index_tensor,
            dataset.edge_feats_tensor
        ).squeeze(-1)
        reloaded_probs = torch.sigmoid(reloaded_logits).numpy()

    max_prob_diff = float(np.max(np.abs(orig_probs - reloaded_probs)))
    print(f"  Maximum Absolute Prediction Difference (In-Memory vs Reloaded): {max_prob_diff:.8e}")

    assert max_prob_diff < 1e-6, f"Reload verification failed! Difference {max_prob_diff} exceeds 1e-6 tolerance."
    print("  [VERIFIED] Reloaded model predictions match in-memory model within 1e-6 numerical tolerance.")

    print("\n==================================================================")
    print(" CHECKPOINT CREATION & VERIFICATION COMPLETE")
    print("==================================================================")


if __name__ == "__main__":
    create_and_verify_checkpoint()
