# Experimental Benchmarks & Results Report

> [!IMPORTANT]
> **Internal Engineering & Validation Notice**  
> All tracking precision, recall, F1 scores, processing node counts, and solver execution times reported below are **internal engineering and validation results** obtained on local development samples and temporal evaluation splits.  
> **An official Kaggle competition leaderboard score was NOT obtained.**  
> The internal tracking F1 of **96.40%** is explicitly labeled as:  
> **"Internal edge-level tracking evaluation: 96.40% F1"**.

---

## Stage-by-Stage Implementation & Experimental Summary

The framework was implemented, benchmarked, and validated across 7 systematic architectural stages:

```
Stage 1 — Foundation & Spatial Coordinate Transformation
Stage 2 — Volumetric Detector (3D UNet + NMS Peak Extraction)
Stage 3 — Regional Processing (Core + Guard Partitioning)
Stage 4 — Global Node Reconciliation + Temporal Candidate Graph
Stage 5 — STGNN Temporal Association Model
Stage 6 — Global ILP Optimization Engine (SciPy HiGHS)
Stage 6.1 — Zero-Leakage Held-Out Temporal Evaluation
Full-Dataset Engineering Pipeline Benchmark
```

---

## Stage 1 — Foundation & Spatial Coordinate Transformation
- **Goal**: Establish deterministic coordinate transformations between 3D anisotropic voxel space and physical space ($\mu\text{m}$).
- **Voxel Scale**: $Z = 1.625\,\mu\text{m}$, $Y = 0.40625\,\mu\text{m}$, $X = 0.40625\,\mu\text{m}$ (Anisotropy ratio = $4.0\times$).
- **Validation**: Verified bidirectional coordinate round-tripping ($[z, y, x] \leftrightarrow [x, y, z]_{\mu\text{m}}$) with zero floating-point error drift across 100 frames.

---

## Stage 2 — Volumetric Detector
- **Goal**: Detect 3D cell centers from raw volumetric intensity stores.
- **Model**: `UNet3D` receiving $64 \times 256 \times 256$ input tensors.
- **Target Map**: Anisotropic 3D Gaussian heatmaps ($\sigma_{xy}=2.0, \sigma_z=1.0$).
- **Peak Extraction**: 3D local max pooling NMS with physical coordinate extraction.
- **Validation**: Verified voxel-to-physical coordinate conversion and peak extraction on synthetic 3D volumes.

---

## Stage 3 — Regional Processing & Guard-Band Partitioning
- **Goal**: Enable scalable memory-bounded processing of large 3D volumes via spatial partitioning.
- **Partition Scheme**: 3D spatial grids with overlapping guard bands.
- **Validation**: Confirmed complete authoritative core coverage without spatial gaps or duplicate candidate counting across region boundaries.

---

## Stage 4 — Global Node Reconciliation & Temporal Candidate Graph
- **Goal**: Reconcile multi-region spatial candidate detections and construct anisotropic temporal candidate graphs for $t \to t+1$ transitions.
- **Node Reconciliation**: Detections within $4.0\,\mu\text{m}$ physical distance are merged into single canonical nodes.
- **Candidate Graph Configuration**:
  - Maximum Physical Search Radius ($R_{\text{max}}$): $35.0\,\mu\text{m}$
  - Top-$K$ Candidates per Node: $15$
- **Local Sample Benchmark**:
  - Canonical Nodes: 861 (across 100 frames)
  - Candidate Edges: 880
  - Ground-Truth Grounded Edge Recall: **100.00%** (845 / 845 GT edges captured in candidate graph).

---

## Stage 5 — STGNN Association Model
- **Goal**: Learn spatiotemporal edge association probabilities using graph neural message passing.
- **Architecture**:
  - Node Input Dim: 9
  - Edge Input Dim: 16
  - Hidden Dim: 64
  - Layers: 2
  - Dropout: 0.1
- **Training Hyperparameters**:
  - Optimizer: `AdamW`
  - Learning Rate: $0.002$ ($2\cdot 10^{-3}$)
  - Weight Decay: $0.0001$ ($10^{-4}$)
  - Epochs: 40
  - Seed: 42
- **Temporal Split Isolation**:
  - Train Split (Frames 0–69): 576 candidate edges (Model fitting)
  - Val Split (Frames 70–84): 187 candidate edges (Validation checkpoint selection)
  - Test Split (Frames 85–99): 117 candidate edges (Zero-leakage held-out evaluation)

---

## Stage 6 — Global ILP Optimization Engine
- **Goal**: Enforce global flow conservation ($\sum e_{\text{in}} \le 1, \sum e_{\text{out}} \le 1$) via SciPy HiGHS binary MILP solver.
- **Local Sample Audit Results**:
  - Number of Edge Variables: 880
  - Total Constraints: 1,695
  - Constraint Matrix Sparsity: 1,760 non-zero entries (nnz)
  - Constraint Violations: **0** across all solver modes.

---

## Stage 6.1 — Zero-Leakage Held-Out Temporal Evaluation

Strict held-out evaluation was conducted on frames 85–99 (117 candidate edges across 127 canonical nodes). Test set ground-truth labels were strictly isolated and never accessed during model training or checkpoint selection.

### Comparison Across ILP Objective Modes (Frames 85–99)

| Objective Mode | Candidate Edges | Selected Edges | Solve Time (s) | Precision | Recall | F1 Score | Constraint Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Geometry-Only ILP** | 117 | 115 | 0.0121 s | **1.0000** | 0.9829 | 0.9914 | **0** |
| **STGNN-Only ILP** | 117 | 117 | 0.0098 s | **1.0000** | **1.0000** | **1.0000** | **0** |
| **Combined ILP** | 117 | 117 | 0.0095 s | **1.0000** | **1.0000** | **1.0000** | **0** |

### Track Fragment & Topology Diagnostics (Frames 85–99)
| Objective Mode | Fragments | Mean Length | Max Length | Isolated Nodes | Pred Only | Succ Only | Both | Neither |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Geometry-Only ILP** | 12 | 10.58 | 15 | 0 | 12 | 12 | 103 | 0 |
| **STGNN-Only ILP** | 10 | 12.70 | 15 | 0 | 10 | 10 | 107 | 0 |
| **Combined ILP** | 10 | 12.70 | 15 | 0 | 10 | 10 | 107 | 0 |

---

## Full-Dataset Engineering Pipeline Results

When evaluated across the complete full-scale engineering test benchmark, the pipeline achieved the following operational metrics:

> [!NOTE]
> **Internal edge-level tracking evaluation: 96.40% F1**

### Engineering Pipeline Metrics
- **Canonical Nodes Processed**: 196,392
- **Candidate Edges Generated**: 481,334
- **Selected Associations**: 173,040
- **Track Fragments Assembled**: 16,344
- **Volumetric Dataset Stores Processed**: 203
- **Processing Failures**: **0**
- **ILP Constraint Violations**: **0**

### Tracking Edge Accuracy (Internal Validation)
- **Mean Tracking Edge Precision**: **97.89%**
- **Mean Tracking Edge Recall**: **95.08%**
- **Mean Tracking Edge F1 Score**: **96.40%**

### Resource & Computational Performance
- **Peak VRAM Usage**: ~785 MB / 15.64 GB (Highly memory-efficient)
- **Mean ILP Solve Time per Volume**: ~0.345 s
- **Maximum Observed ILP Solve Time**: 55.38 s
