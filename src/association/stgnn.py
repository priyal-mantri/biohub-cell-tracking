"""
Spatio-Temporal Graph Neural Network (STGNN) Candidate Edge Association Model.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class STGNNMessagePassingLayer(nn.Module):
    """
    Edge-aware graph message-passing layer.

    Computes directional messages between connected nodes, aggregates incoming/outgoing
    neighbor context, and updates node & edge representations simultaneously.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message MLP: [h_src, h_dst, e_attr] -> message
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Node update MLP: [h_node, msg_in, msg_out] -> h_node_updated
        self.node_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Edge update MLP: [e_attr, h_src, h_dst, message] -> e_attr_updated
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        node_feats: torch.Tensor,  # (N, D_hidden)
        edge_index: torch.Tensor,  # (2, E)
        edge_feats: torch.Tensor   # (E, D_hidden)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for message passing layer.
        """
        N = node_feats.size(0)
        E = edge_index.size(1)

        if E == 0:
            return node_feats, edge_feats

        src_idx = edge_index[0]  # (E,)
        dst_idx = edge_index[1]  # (E,)

        h_src = node_feats[src_idx]  # (E, D_hidden)
        h_dst = node_feats[dst_idx]  # (E, D_hidden)

        # 1. Compute messages for each edge
        msg_in_concat = torch.cat([h_src, h_dst, edge_feats], dim=-1)  # (E, 3*D_hidden)
        messages = self.message_mlp(msg_in_concat)  # (E, D_hidden)

        # 2. Aggregate messages at target nodes (incoming) and source nodes (outgoing)
        msg_in_agg = torch.zeros(N, self.hidden_dim, device=node_feats.device)
        msg_out_agg = torch.zeros(N, self.hidden_dim, device=node_feats.device)
        count_in = torch.zeros(N, 1, device=node_feats.device)
        count_out = torch.zeros(N, 1, device=node_feats.device)

        msg_in_agg.index_add_(0, dst_idx, messages)
        count_in.index_add_(0, dst_idx, torch.ones(E, 1, device=node_feats.device))

        msg_out_agg.index_add_(0, src_idx, messages)
        count_out.index_add_(0, src_idx, torch.ones(E, 1, device=node_feats.device))

        msg_in_mean = msg_in_agg / torch.clamp(count_in, min=1.0)
        msg_out_mean = msg_out_agg / torch.clamp(count_out, min=1.0)

        # 3. Update node representations
        node_update_concat = torch.cat([node_feats, msg_in_mean, msg_out_mean], dim=-1)
        node_feats_updated = node_feats + self.node_update_mlp(node_update_concat)

        # 4. Update edge representations using updated node states
        h_src_up = node_feats_updated[src_idx]
        h_dst_up = node_feats_updated[dst_idx]
        edge_update_concat = torch.cat([edge_feats, h_src_up, h_dst_up, messages], dim=-1)
        edge_feats_updated = edge_feats + self.edge_update_mlp(edge_update_concat)

        return node_feats_updated, edge_feats_updated


class STGNNModel(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for scoring temporal candidate edge associations.
    """

    def __init__(
        self,
        node_in_dim: int = 9,
        edge_in_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        geometry_only: bool = False
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.geometry_only = geometry_only

        # Node & Edge Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Message Passing Layers
        self.gnn_layers = nn.ModuleList([
            STGNNMessagePassingLayer(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final Edge Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        node_feats: torch.Tensor,  # (N, D_node)
        edge_index: torch.Tensor,  # (2, E)
        edge_feats: torch.Tensor   # (E, D_edge)
    ) -> torch.Tensor:
        """
        Forward pass producing raw uncalibrated logits for each candidate edge.

        :return: Tensor of shape (E, 1) raw edge logits
        """
        E = edge_index.size(1)
        if E == 0:
            return torch.empty((0, 1), device=node_feats.device)

        # 1. Encode raw inputs
        h_node = self.node_encoder(node_feats)  # (N, D_hidden)
        h_edge = self.edge_encoder(edge_feats)  # (E, D_hidden)

        # 2. Apply GNN message passing unless geometry_only ablation is active
        if not self.geometry_only:
            for gnn_layer in self.gnn_layers:
                h_node, h_edge = gnn_layer(h_node, edge_index, h_edge)

        # 3. Predict candidate edge logits
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_src = h_node[src_idx]
        h_dst = h_node[dst_idx]

        combined_edge_repr = torch.cat([h_src, h_dst, h_edge], dim=-1)  # (E, 3*D_hidden)
        logits = self.classifier(combined_edge_repr)  # (E, 1)

        return logits

    def predict_probabilities(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Produces calibrated sigmoid probabilities P(edge represents true temporal continuation) in [0, 1].
        """
        with torch.no_grad():
            logits = self.forward(node_feats, edge_index, edge_feats)
            return torch.sigmoid(logits).squeeze(-1)
