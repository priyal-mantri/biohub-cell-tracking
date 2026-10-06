#!/usr/bin/env python3
"""
Standalone Kaggle Notebook Execution Engine for BioHub Cell Tracking Architecture.

Dataset-general, GPU-accelerated, memory-disciplined, sample-wise tracking pipeline.
"""

import sys
import os
import time
import json
import gc
import traceback
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SpatialResolution, DatasetConfig
from src.data import ZarrLoader, GeffLoader, CoordinateTransformer
from src.regional import RegionalCandidate, RegionalInferenceEngine, PartitionConfig
from src.detection import UNet3D, PeakConfig, HeuristicIntensityDetector
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
    AssociationEvaluator,
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


def log_step(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_gpu_environment_info() -> Dict[str, Any]:
    """
    1. Environment Check & GPU Diagnostics.
    """
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (CUDA Unavailable)"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda_available else 0.0

    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "vram_gb": round(vram_gb, 2),
        "target_device": "cuda" if cuda_available else "cpu"
    }
    return info


def discover_dataset_samples(input_dir: Path) -> List[Tuple[str, Path, Optional[Path]]]:
    """
    4. Dataset Discovery: Locates all .zarr stores in input directory.
    Matches corresponding .geff ground truth stores if available.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Search for all .zarr directories
    zarr_paths = sorted(list(input_dir.glob("*.zarr")) + list(input_dir.glob("**/*.zarr")))
    # Remove duplicates
    zarr_paths = sorted(list(set(zarr_paths)))

    samples = []
    for z_path in zarr_paths:
        sample_id = z_path.stem
        # Check if corresponding .geff directory exists
        geff_candidate = z_path.parent / f"{sample_id}.geff"
        geff_path = geff_candidate if geff_candidate.exists() else None
        samples.append((sample_id, z_path, geff_path))

    return samples


def run_sample_pipeline(
    sample_id: str,
    zarr_path: Path,
    geff_path: Optional[Path],
    device: torch.device,
    stgnn_model: STGNNModel,
    ilp_optimizer: GlobalILPOptimizer,
    transformer: CoordinateTransformer
) -> Dict[str, Any]:
    """
    Executes sample-wise cell detection, graph building, STGNN scoring, and ILP optimization.
    """
    log_step(f"--- Processing Sample: {sample_id} ---")
    start_time = time.time()
    mem_start_mb = torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0

    # 1. Load Zarr Store
    zarr_loader = ZarrLoader(zarr_path)
    shape_tzyx = zarr_loader.shape
    log_step(f"  Volume Shape: {shape_tzyx}, dtype: {zarr_loader.dtype}")

    # Load GEFF GT if available
    geff_loader = GeffLoader(geff_path) if geff_path is not None and geff_path.exists() else None
    geff_all_edges = [(int(u), int(v)) for u, v in geff_loader.directed_edges] if geff_loader else None
    all_gt_nodes = geff_loader.get_all_nodes() if geff_loader else None

    # 2. Regional Cell Detection & Global Node Reconciliation across frames
    node_reconciler = NodeReconciler(reconciliation_distance_um=4.0, transformer=transformer)
    node_set = NodeSet()

    regional_engine = None
    if geff_loader is None:
        heuristic_det = HeuristicIntensityDetector(min_confidence=0.1, min_physical_distance_um=3.0)
        regional_engine = RegionalInferenceEngine(detector=heuristic_det, transformer=transformer)

    T = shape_tzyx[0]
    total_regional_candidates = 0

    for f in range(T):
        if geff_loader:
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
        else:
            # Full regional detector fallback using intensity peak extractor on 3D volume crop
            reg_cands, _ = regional_engine.detect_frame_regional(zarr_loader, f, filter_core_only=True)

        total_regional_candidates += len(reg_cands)
        canonical_frame_nodes = node_reconciler.reconcile_frame(f, reg_cands)
        node_set.add_nodes(canonical_frame_nodes)

    num_canonical_nodes = node_set.num_nodes()
    log_step(f"  Canonical Nodes Reconciled: {num_canonical_nodes} across {T} frames")

    # 3. Temporal Candidate Graph Construction (t -> t+1)
    graph_config = CandidateGraphConfig(max_physical_distance_um=20.0, top_k_per_node=5)
    graph = CandidateGraphBuilder(config=graph_config).build_candidate_graph(node_set)
    num_candidate_edges = graph.candidate_count()
    log_step(f"  Candidate Edges: {num_candidate_edges}")

    matching_result = None
    if geff_loader and all_gt_nodes:
        evaluator = CandidateRecallEvaluator(transformer=transformer)
        matching_result = evaluator.match_nodes_to_geff(node_set, all_gt_nodes, match_threshold_um=5.0)

    # 4. Association Dataset & STGNN Scoring
    feature_extractor = GraphFeatureExtractor(transformer=transformer)
    assoc_dataset = AssociationDataset(graph=graph, feature_extractor=feature_extractor)
    node_feats_tensor = assoc_dataset.node_feats_tensor.to(device)
    edge_index_tensor = assoc_dataset.edge_index_tensor.to(device)
    edge_feats_tensor = assoc_dataset.edge_feats_tensor.to(device)

    stgnn_model.eval()
    with torch.no_grad():
        logits = stgnn_model(node_feats_tensor, edge_index_tensor, edge_feats_tensor).squeeze(-1)
        stgnn_scores = torch.sigmoid(logits).cpu().numpy()

    if len(stgnn_scores) > 0:
        log_step(f"  STGNN Evidence Scores: Mean={np.mean(stgnn_scores):.3f}, Min={np.min(stgnn_scores):.3f}, Max={np.max(stgnn_scores):.3f}")
    else:
        log_step("  STGNN Evidence Scores: N/A (0 candidate edges)")

    # 5. Global ILP Optimization
    solution = ilp_optimizer.optimize(node_set, graph, stgnn_scores=stgnn_scores)
    solver_result = solution.solver_result

    # 6. Diagnostics & Evaluation
    diagnostics = ILPDiagnostics()
    report = diagnostics.evaluate_solution(solution, node_set, geff_edges=geff_all_edges, matching_result=matching_result)

    elapsed_time = time.time() - start_time
    mem_peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0

    # Cleanup memory
    del zarr_loader, node_feats_tensor, edge_index_tensor, edge_feats_tensor, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    log_step(f"  Selected Edges: {report.selected_edges}, Fragments: {report.num_track_fragments}, Solve Time: {solver_result.solve_time_sec:.3f}s")
    log_step(f"  Precision: {report.edge_precision:.4f}, Recall: {report.edge_recall:.4f}, F1: {report.edge_f1:.4f}, Violations: {report.constraint_violations_count}")
    log_step(f"--- Completed Sample: {sample_id} in {elapsed_time:.2f}s ---\n")

    metrics = {
        "sample_id": sample_id,
        "volume_shape": list(shape_tzyx),
        "num_canonical_nodes": num_canonical_nodes,
        "num_candidate_edges": num_candidate_edges,
        "selected_edges": report.selected_edges,
        "num_track_fragments": report.num_track_fragments,
        "longest_fragment_len": report.longest_fragment_len,
        "solve_time_sec": solver_result.solve_time_sec,
        "elapsed_time_sec": elapsed_time,
        "peak_vram_mb": mem_peak_mb,
        "precision": report.edge_precision,
        "recall": report.edge_recall,
        "f1": report.edge_f1,
        "constraint_violations": report.constraint_violations_count,
        "solver_status": solver_result.status,
    }

    return metrics, node_set, solution.selected_edges


def run_kaggle_pipeline(
    input_dir: Path,
    output_dir: Path,
    max_samples: Optional[int] = None
):
    """
    Main orchestration loop for full Kaggle dataset execution.
    """
    log_step("==================================================================")
    log_step(" BioHub Cell Tracking — Kaggle Full Dataset Pipeline Execution ")
    log_step("==================================================================")

    # 1. Environment Audit
    env_info = get_gpu_environment_info()
    log_step(f"Environment: Python {env_info['python_version']}, PyTorch {env_info['torch_version']}")
    log_step(f"CUDA Available: {env_info['cuda_available']}, Target Device: {env_info['target_device']} ({env_info['device_name']}, {env_info['vram_gb']} GB VRAM)")

    if not env_info["cuda_available"]:
        print("\n[WARNING] CUDA GPU is unavailable! PyTorch operations will fall back to CPU.", flush=True)

    device = torch.device(env_info["target_device"])

    # 2. Discover Samples
    log_step(f"Discovering dataset samples in: {input_dir}")
    samples = discover_dataset_samples(input_dir)
    log_step(f"Discovered {len(samples)} sample Zarr store(s).")

    if len(samples) == 0:
        raise FileNotFoundError(f"No .zarr dataset stores found in {input_dir}")

    if max_samples is not None:
        samples = samples[:max_samples]
        log_step(f"Running in limited sample mode: processing first {len(samples)} sample(s).")

    # 3. STGNN Checkpoint Loading & Initialization
    transformer = CoordinateTransformer()

    checkpoint_candidates = [
        PROJECT_ROOT / "models" / "stgnn_checkpoint.pt",
        Path("./models/stgnn_checkpoint.pt"),
        Path("./stgnn_checkpoint.pt"),
        Path("/kaggle/input/biohub-stgnn-checkpoint/stgnn_checkpoint.pt"),
        Path("/kaggle/input/stgnn-checkpoint/stgnn_checkpoint.pt"),
        Path("/kaggle/working/models/stgnn_checkpoint.pt"),
    ]

    checkpoint_path = None
    for cand in checkpoint_candidates:
        if cand.exists():
            checkpoint_path = cand
            break

    if checkpoint_path is not None:
        log_step(f"Loading verified STGNN model checkpoint from: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=device)
        stgnn_model = STGNNModel(**payload["model_config"]).to(device)
        stgnn_model.load_state_dict(payload["model_state_dict"])
        stgnn_model.eval()
        log_step(f"  [OK] Successfully loaded STGNN checkpoint (Selected Epoch: {payload.get('selected_epoch')}, Val Loss: {payload.get('best_val_loss', 0.0):.4f})")
    else:
        sample_0_id, sample_0_zarr, sample_0_geff = samples[0]
        if sample_0_geff and sample_0_geff.exists():
            log_step(f"Checkpoint not found on disk. Reproducibly fitting STGNN evidence model on training sample {sample_0_id} (Frames 0-69)...")
            stgnn_model = STGNNModel(node_in_dim=9, edge_in_dim=16, hidden_dim=64, num_layers=2).to(device)
            z_loader0 = ZarrLoader(sample_0_zarr)
            g_loader0 = GeffLoader(sample_0_geff)
            n_reconciler0 = NodeReconciler(reconciliation_distance_um=4.0, transformer=transformer)
            n_set0 = NodeSet()

            for f in range(z_loader0.shape[0]):
                f_gt = g_loader0.get_nodes_by_frame(f)
                r_cands = [
                    RegionalCandidate(
                        candidate_id=gn.node_id, region_id=0, frame=f, local_voxel_pos=gn.voxel_pos, global_voxel_pos=gn.voxel_pos, physical_pos=gn.physical_pos, confidence=1.0, is_in_core=True, provenance={"contributing_region_ids": [0], "num_contributing_regions": 1, "primary_region_id": 0}
                    ) for gn in f_gt
                ]
                n_set0.add_nodes(n_reconciler0.reconcile_frame(f, r_cands))

            g_config0 = CandidateGraphConfig(max_physical_distance_um=35.0, top_k_per_node=15)
            graph0 = CandidateGraphBuilder(config=g_config0).build_candidate_graph(n_set0)
            matching_res0 = CandidateRecallEvaluator(transformer=transformer).match_nodes_to_geff(n_set0, g_loader0.get_all_nodes(), match_threshold_um=5.0)
            geff_edges0 = [(int(u), int(v)) for u, v in g_loader0.directed_edges]

            dataset0 = AssociationDataset(graph=graph0, feature_extractor=GraphFeatureExtractor(transformer=transformer), labeler=CandidateEdgeLabeler(), geff_edges=geff_edges0, matching_result=matching_res0)
            split0 = TemporalSplitter(train_range=(0, 69), val_range=(70, 84), test_range=(85, 99)).split_graph_edges(graph0)

            trainer = STGNNTrainer(model=stgnn_model, config=STGNNTrainerConfig(epochs=40, learning_rate=2e-3, seed=42), device=env_info["target_device"])
            trainer.train(dataset0, split0.train_mask, split0.val_mask, epochs=40)
            log_step("  [OK] STGNN model fitting complete.")
        else:
            raise FileNotFoundError(
                "CRITICAL ERROR: Verified STGNN checkpoint file 'models/stgnn_checkpoint.pt' is missing and no training GEFF annotations are available! "
                "Pipeline execution ABORTED to prevent random weight inference."
            )

    ilp_optimizer = GlobalILPOptimizer(config=OptimizationConfig(mode="combined", alpha=1.0, beta=1.0, sigma_scale_um=5.0))

    # 4. Sequential Sample Execution Loop
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_results = []
    sample_predictions = []
    failures = []

    pipeline_start = time.time()

    for idx, (s_id, s_zarr, s_geff) in enumerate(samples, 1):
        log_step(f"Progress: [{idx}/{len(samples)}] {s_id}")
        try:
            res, n_set, sel_edges = run_sample_pipeline(
                sample_id=s_id,
                zarr_path=s_zarr,
                geff_path=s_geff,
                device=device,
                stgnn_model=stgnn_model,
                ilp_optimizer=ilp_optimizer,
                transformer=transformer
            )
            sample_results.append(res)
            sample_predictions.append((s_id, n_set, sel_edges))
        except Exception as e:
            err_msg = str(e)
            tb_str = traceback.format_exc()
            log_step(f"[ERROR] Sample {s_id} failed: {err_msg}")
            failures.append({
                "sample_id": s_id,
                "zarr_path": str(s_zarr),
                "error": err_msg,
                "traceback": tb_str
            })

    total_pipeline_time = time.time() - pipeline_start

    # 5. Build and Write Official submission.csv
    import csv
    sample_sub_candidates = [
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/sample_submission.csv"),
        Path("/kaggle/input/biohub-cell-tracking-during-development/sample_submission.csv"),
        Path("/kaggle/input/sample_submission.csv"),
        Path("./results/sample_submission.csv"),
    ]
    target_test_datasets = None
    for cand in sample_sub_candidates:
        if cand.exists():
            with open(cand, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                target_test_datasets = set(row["dataset"] for row in reader)
            log_step(f"Filtered submission.csv datasets using {cand}: {sorted(list(target_test_datasets))}")
            break

    submission_rows = []
    row_id = 0

    for s_id, n_set, sel_edges in sample_predictions:
        if target_test_datasets is not None and s_id not in target_test_datasets:
            continue

        # Sort canonical nodes by frame, then node_id
        nodes = sorted(n_set.all_nodes(), key=lambda n: (n.frame, n.node_id))
        for n in nodes:
            submission_rows.append({
                "id": row_id,
                "dataset": s_id,
                "row_type": "node",
                "node_id": n.node_id,
                "t": int(n.frame),
                "z": int(round(n.global_voxel_pos[0])),
                "y": int(round(n.global_voxel_pos[1])),
                "x": int(round(n.global_voxel_pos[2])),
                "source_id": -1,
                "target_id": -1,
            })
            row_id += 1

        # Sort selected edges by source_id, then target_id
        edges = sorted(sel_edges, key=lambda e: (e.source_node_id, e.destination_node_id))
        for e in edges:
            submission_rows.append({
                "id": row_id,
                "dataset": s_id,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": e.source_node_id,
                "target_id": e.destination_node_id,
            })
            row_id += 1

    fieldnames = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]

    sub_path_working = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else output_dir / "submission.csv"
    with open(sub_path_working, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(submission_rows)

    sub_path_root = Path("./submission.csv").resolve()
    if sub_path_working.resolve() != sub_path_root:
        import shutil
        shutil.copy(sub_path_working, sub_path_root)

    log_step(f"Generated official submission.csv at: {sub_path_working} ({len(submission_rows)} rows)")

    # 6. Save Outputs & Summary
    summary_payload = {
        "run_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": env_info,
            "total_samples_discovered": len(samples),
            "samples_processed": len(sample_results),
            "failures_count": len(failures),
            "total_pipeline_time_sec": round(total_pipeline_time, 2),
            "submission_rows_count": len(submission_rows),
        },
        "sample_results": sample_results,
        "failures": failures
    }

    results_file = output_dir / "sample_results.json"
    summary_file = output_dir / "run_metadata.json"
    failures_file = output_dir / "failures.json"

    with open(results_file, "w") as f:
        json.dump(sample_results, f, indent=2)

    with open(summary_file, "w") as f:
        json.dump(summary_payload["run_metadata"], f, indent=2)

    with open(failures_file, "w") as f:
        json.dump(failures, f, indent=2)

    log_step("==================================================================")
    log_step(f" PIPELINE SUMMARY: Processed {len(sample_results)}/{len(samples)} samples in {total_pipeline_time:.2f}s")
    log_step(f" Submission CSV: {sub_path_working} ({len(submission_rows)} rows)")
    log_step("==================================================================")


if __name__ == "__main__":
    default_input = PROJECT_ROOT / "train"
    default_output = PROJECT_ROOT / "results" / "kaggle_run"
    run_kaggle_pipeline(input_dir=default_input, output_dir=default_output)

