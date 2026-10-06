"""
STGNN Association Model Trainer, Pipeline Integrity Auditor & Controlled Overfit Tester.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from src.association.stgnn import STGNNModel
from src.association.dataset import AssociationDataset
from src.association.evaluation import AssociationEvaluator, EvaluationMetrics


@dataclass
class STGNNTrainerConfig:
    """
    Configuration parameters for STGNN model training.
    """
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    pos_weight: Optional[float] = None  # Class weight for BCE loss
    seed: int = 42


@dataclass
class OverfitTestResult:
    """Diagnostic outcome of Part 13 Controlled Overfit Test."""
    initial_loss: float
    final_loss: float
    initial_mean_pos_prob: float
    final_mean_pos_prob: float
    initial_mean_neg_prob: float
    final_mean_neg_prob: float
    initial_roc_auc: float
    final_roc_auc: float
    overfit_success: bool


class STGNNTrainer:
    """
    Trainer for STGNN association model with pipeline integrity checks and controlled overfit testing.
    """

    def __init__(
        self,
        model: STGNNModel,
        config: Optional[STGNNTrainerConfig] = None,
        device: str = "cpu"
    ):
        self.model = model
        self.config = config or STGNNTrainerConfig()
        self.device = torch.device(device)
        self.model.to(self.device)

        # Set deterministic seeds
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def run_pipeline_integrity_checks(
        self,
        dataset: AssociationDataset,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray
    ) -> Dict[str, bool]:
        """
        Part 12 — Verifies all 12 mandatory pipeline integrity conditions.
        """
        checks = {}
        tensors = dataset.get_split_tensors(train_mask, device=str(self.device))

        node_feats = tensors["node_feats"]
        edge_index = tensors["edge_index"]
        edge_feats = tensors["edge_feats"]
        labels = tensors["labels"]

        E = edge_index.size(1)
        N = node_feats.size(0)

        # 1. Non-empty graph
        checks["1_non_empty_graph"] = bool(N > 0 and E > 0)
        # 2. Non-zero edge count
        checks["2_non_zero_edge_count"] = bool(E > 0)

        # 3 & 4. Positive and Negative labels present across ALL splits (Train, Val, Test)
        val_tensors = dataset.get_split_tensors(val_mask, device=str(self.device))
        test_tensors = dataset.get_split_tensors(test_mask, device=str(self.device))

        tr_pos = int(torch.sum(labels == 1.0).item())
        tr_neg = int(torch.sum(labels == 0.0).item())

        val_pos = int(torch.sum(val_tensors["labels"] == 1.0).item())
        val_neg = int(torch.sum(val_tensors["labels"] == 0.0).item())

        test_pos = int(torch.sum(test_tensors["labels"] == 1.0).item())
        test_neg = int(torch.sum(test_tensors["labels"] == 0.0).item())

        checks["3_positive_labels_present_all_splits"] = bool(tr_pos > 0 and val_pos > 0 and test_pos > 0)
        checks["4_negative_labels_present_all_splits"] = bool(tr_neg > 0 and val_neg > 0 and test_neg > 0)

        # 5. Finite node features
        checks["5_finite_node_features"] = bool(torch.all(torch.isfinite(node_feats)).item())

        # 6. Finite edge features
        checks["6_finite_edge_features"] = bool(torch.all(torch.isfinite(edge_feats)).item())

        # 7. Correct source/destination indices
        if E > 0:
            valid_src = bool((edge_index[0] >= 0).all() and (edge_index[0] < N).all())
            valid_dst = bool((edge_index[1] >= 0).all() and (edge_index[1] < N).all())
            checks["7_correct_node_indices"] = bool(valid_src and valid_dst)
        else:
            checks["7_correct_node_indices"] = True

        # 8. Correct label alignment
        checks["8_correct_label_alignment"] = bool(labels.size(0) == E)

        # 9. Correct train/val/test split disjointness
        no_overlap = bool(
            not np.any(train_mask & val_mask) and
            not np.any(train_mask & test_mask) and
            not np.any(val_mask & test_mask)
        )
        checks["9_correct_split_disjointness"] = no_overlap

        # 10. Model output shape matching candidate edge count
        self.model.eval()
        with torch.no_grad():
            logits = self.model(node_feats, edge_index, edge_feats)
        checks["10_output_shape_matches_edges"] = bool(logits.shape == (E, 1))

        # 11. Gradients reaching model
        self.model.train()
        logits_tr = self.model(node_feats, edge_index, edge_feats)
        loss = F.binary_cross_entropy_with_logits(logits_tr.squeeze(-1), labels)
        self.model.zero_grad()
        loss.backward()

        grads_present = True
        for param in self.model.parameters():
            if param.requires_grad:
                if param.grad is None or float(param.grad.abs().sum()) == 0.0:
                    grads_present = False
                    break
        checks["11_gradients_reaching_model"] = grads_present

        # 12. Parameters actually changing during optimizer step
        param_before = [p.clone() for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(self.model.parameters(), lr=1e-2)
        optimizer.step()
        param_after = [p for p in self.model.parameters() if p.requires_grad]

        params_changed = any(
            not torch.equal(b, a) for b, a in zip(param_before, param_after)
        )
        checks["12_parameters_actually_changing"] = params_changed

        return checks

    def run_controlled_overfit_test(
        self,
        dataset: AssociationDataset,
        train_mask: np.ndarray,
        epochs: int = 50
    ) -> OverfitTestResult:
        """
        Part 13 — Controlled Overfit Test: Trains STGNN on small fixed candidate graph
        to verify that loss decreases and positive/negative separation increases.
        """
        tensors = dataset.get_split_tensors(train_mask, device=str(self.device))
        node_feats = tensors["node_feats"]
        edge_index = tensors["edge_index"]
        edge_feats = tensors["edge_feats"]
        labels = tensors["labels"]

        model = STGNNModel(
            node_in_dim=node_feats.size(1),
            edge_in_dim=edge_feats.size(1),
            hidden_dim=32,
            num_layers=2,
            dropout=0.0
        ).to(self.device)

        optimizer = AdamW(model.parameters(), lr=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()

        # Initial stats
        model.eval()
        with torch.no_grad():
            init_logits = model(node_feats, edge_index, edge_feats).squeeze(-1)
            init_loss = float(loss_fn(init_logits, labels).item())
            init_probs = torch.sigmoid(init_logits).cpu().numpy()
            init_metrics = AssociationEvaluator.compute_metrics(labels.cpu().numpy(), init_probs)

        # Train loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = model(node_feats, edge_index, edge_feats).squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        # Final stats
        model.eval()
        with torch.no_grad():
            final_logits = model(node_feats, edge_index, edge_feats).squeeze(-1)
            final_loss = float(loss_fn(final_logits, labels).item())
            final_probs = torch.sigmoid(final_logits).cpu().numpy()
            final_metrics = AssociationEvaluator.compute_metrics(labels.cpu().numpy(), final_probs)

        success = bool(final_loss < init_loss and final_metrics.roc_auc >= init_metrics.roc_auc)

        return OverfitTestResult(
            initial_loss=init_loss,
            final_loss=final_loss,
            initial_mean_pos_prob=init_metrics.mean_prob_positive,
            final_mean_pos_prob=final_metrics.mean_prob_positive,
            initial_mean_neg_prob=init_metrics.mean_prob_negative,
            final_mean_neg_prob=final_metrics.mean_prob_negative,
            initial_roc_auc=init_metrics.roc_auc,
            final_roc_auc=final_metrics.roc_auc,
            overfit_success=success
        )

    def train(
        self,
        dataset: AssociationDataset,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Trains STGNN model using train split and evaluates validation split per epoch.
        """
        n_epochs = epochs if epochs is not None else self.config.epochs
        train_tensors = dataset.get_split_tensors(train_mask, device=str(self.device))
        val_tensors = dataset.get_split_tensors(val_mask, device=str(self.device))

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        pos_weight_tensor = None
        if self.config.pos_weight is not None:
            pos_weight_tensor = torch.tensor([self.config.pos_weight], device=self.device)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_roc_auc": [],
            "val_f1": []
        }

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(1, n_epochs + 1):
            # Training step
            self.model.train()
            optimizer.zero_grad()

            tr_logits = self.model(
                train_tensors["node_feats"],
                train_tensors["edge_index"],
                train_tensors["edge_feats"]
            ).squeeze(-1)

            tr_loss = loss_fn(tr_logits, train_tensors["labels"])
            tr_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(
                    val_tensors["node_feats"],
                    val_tensors["edge_index"],
                    val_tensors["edge_feats"]
                ).squeeze(-1)

                val_loss = float(loss_fn(val_logits, val_tensors["labels"]).item())
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_metrics = AssociationEvaluator.compute_metrics(val_tensors["labels"].cpu().numpy(), val_probs)

            history["train_loss"].append(float(tr_loss.item()))
            history["val_loss"].append(val_loss)
            history["val_roc_auc"].append(val_metrics.roc_auc)
            history["val_f1"].append(val_metrics.f1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return history
