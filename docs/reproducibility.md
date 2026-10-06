# Reproducibility & Local Execution Guide

This document provides step-by-step instructions for installing dependencies, acquiring the dataset from Kaggle, configuring dataset paths, running the test suite, executing pipeline stages, and verifying model checkpoints.

---

## 1. System Requirements & Dependency Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- PyTorch >= 2.0 (CPU or CUDA GPU)
- SciPy >= 1.10 (provides HiGHS solver via `scipy.optimize.milp`)
- Zarr >= 3.0 (for reading 3D+t Zarr volumetric stores)
- NetworkX >= 3.0

### Option A: Standard Pip Installation
```bash
# Clone repository
git clone https://github.com/your-username/biohub-cell-tracking.git
cd biohub-cell-tracking

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Option B: Conda / Mamba Environment
```bash
conda env create -f environment.yml
conda activate biohub-cell-tracking
pip install -e .
```

---

## 2. Dataset Acquisition & Path Configuration

### Official Dataset Access
The full dataset is **not** distributed with this repository due to storage size and Kaggle competition access policies.

- **Official Kaggle Competition Page**:  
  [BioHub Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)

### Downloading & Setting Up Data
1. Download the competition data using the Kaggle API or web interface:
   ```bash
   kaggle competitions download -c biohub-cell-tracking-during-development
   ```
2. Uncompress the downloaded Zarr stores (`.zarr`) and GEFF ground-truth graph directories (`.geff`) into your preferred local directory (e.g., `./train/` or `/data/biohub/train/`).

### Local Development Sample
During framework development and testing, the following sample volume was utilized:
- **Zarr Volume**: `6bba_05b6850b.zarr` (Shape: `[100, 64, 256, 256]`, dtype: `uint16`)
- **GEFF Ground-Truth Graph**: `6bba_05b6850b.geff` (Contains node positions and directed edge continuations)

### Configuring Local Paths
Configure dataset paths in `src/config.py` or pass `DatasetConfig` programmatically:
```python
from pathlib import Path
from src.config import DatasetConfig, SpatialResolution

# Custom dataset path configuration
config = DatasetConfig(
    zarr_path=Path("/path/to/your/train/6bba_05b6850b.zarr"),
    geff_path=Path("/path/to/your/train/6bba_05b6850b.geff"),
    resolution=SpatialResolution(z_spacing=1.625, y_spacing=0.40625, x_spacing=0.40625)
)
```

---

## 3. Running the Unit Test Suite

The test suite validates every layer of the architecture: foundation data loading, 3D UNet detection, regional partitioning, node reconciliation, candidate graph construction, STGNN message passing, and ILP optimization flow conservation.

Run the complete test suite with `pytest`:
```bash
pytest -v
```
*(All 83 unit tests should pass in under 10 seconds).*

---

## 4. Running Pipeline Stage Verification Scripts

Each architectural stage includes a standalone, reproducible verification script located in `scripts/`:

### Stage 1 — Foundation & Data Parsing
```bash
python3 scripts/validate_foundation.py
```
*Validates spatial coordinate transformations, Zarr chunking, and GEFF graph parsing.*

### Stage 2 — Volumetric Detector
```bash
python3 scripts/run_detector_experiment.py
```
*Tests 3D UNet heatmap target generation, forward pass, and 3D NMS peak extraction.*

### Stage 3 — Regional Processing & Partitioning
```bash
python3 scripts/run_regional_experiment.py
```
*Validates core and guard band volume partitioning, spatial clipping, and local-to-global coordinate mapping.*

### Stage 4 — Candidate Graph Construction
```bash
python3 scripts/run_candidate_graph_experiment.py
```
*Constructs the anisotropic temporal candidate graph ($35\,\mu\text{m}$ search radius, Top-$K=15$) and measures ground-truth edge recall.*

### Stage 5 — STGNN Association Model
```bash
python3 scripts/run_stgnn_experiment.py
```
*Trains the STGNN association model across temporal splits (Train: frames 0–69, Val: 70–84, Test: 85–99) and outputs classification metrics.*

### Stage 6 — Global ILP Optimization
```bash
python3 scripts/run_ilp_experiment.py
```
*Runs SciPy HiGHS binary MILP solver across Geometry-Only, STGNN-Only, and Combined objective modes.*

### Stage 6.1 — Held-Out Temporal Evaluation
```bash
python3 scripts/run_stage_6_1_heldout_experiment.py
```
*Executes zero-leakage held-out temporal evaluation on unseen test transitions (frames 85–99).*

---

## 5. STGNN Checkpoint Verification

The pre-trained STGNN model checkpoint is saved at:
`models/stgnn_checkpoint.pt` (File size: ~397 KB).

To verify checkpoint creation, reloading, and numerical consistency:
```bash
python3 scripts/create_stgnn_checkpoint.py
```
This script trains the model deterministically (`seed=42`), saves the structured checkpoint, reloads it, and asserts that prediction outputs match within $10^{-6}$ absolute tolerance.
