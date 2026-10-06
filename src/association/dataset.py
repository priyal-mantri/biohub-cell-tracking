"""
Graph Association Dataset and Feature/Tensor Preparation.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch

from src.graph.candidate_graph import TemporalCandidateGraph, CandidateEdge
from src.graph.features import GraphFeatureExtractor, NodeFeatures, EdgeFeatures
from src.graph.diagnostics import NodeMatchingResult
from src.association.labels import CandidateEdgeLabeler, LabelStats, TemporalSplitter, TemporalSplit


class AssociationDataset:
    """
    Constructs PyTorch tensor representations from TemporalCandidateGraph for model training and evaluation.
    """

    def __init__(
        self,
        graph: TemporalCandidateGraph,
        feature_extractor: Optional[GraphFeatureExtractor] = None,
        labeler: Optional[CandidateEdgeLabeler] = None,
        geff_edges: Optional[List[Tuple[int, int]]] = None,
        matching_result: Optional[NodeMatchingResult] = None
    ):
        self.graph = graph
        self.feature_extractor = feature_extractor or GraphFeatureExtractor()
        self.labeler = labeler or CandidateEdgeLabeler()

        # Extract graph node & edge features
        feats = self.feature_extractor.extract(graph)
        self.node_features: NodeFeatures = feats["node_features"]
        self.edge_features: EdgeFeatures = feats["edge_features"]

        # Generate ground-truth labels if GEFF data provided
        if geff_edges is not None and matching_result is not None:
            self.labels, self.label_stats = self.labeler.create_labels(graph, geff_edges, matching_result)
        else:
            self.labels = np.zeros(graph.candidate_count(), dtype=np.float32)
            self.label_stats = LabelStats(
                total_candidate_edges=graph.candidate_count(),
                positive_edges=0,
                negative_edges=graph.candidate_count(),
                positive_rate_pct=0.0,
                negatives_per_positive=0.0,
                pos_edge_distances_um=[],
                neg_edge_distances_um=[]
            )

        # Convert to PyTorch Tensors
        self.node_feats_tensor = torch.from_numpy(self.node_features.features).float()
        self.edge_index_tensor = torch.from_numpy(self.edge_features.edge_index).long()
        self.edge_feats_tensor = torch.from_numpy(self.edge_features.features).float()
        self.labels_tensor = torch.from_numpy(self.labels).float()

    def get_split_tensors(
        self,
        split_mask: np.ndarray,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts subgraph tensor dict corresponding to candidate edge split mask.

        :param split_mask: Boolean numpy array of shape (E,) selecting edges
        :param device: PyTorch device ('cpu' or 'cuda')
        :return: Dict containing PyTorch tensors
        """
        mask_idx = np.where(split_mask)[0]
        dev = torch.device(device)

        if len(mask_idx) == 0:
            return {
                "node_feats": self.node_feats_tensor.to(dev),
                "edge_index": torch.empty((2, 0), dtype=torch.long, device=dev),
                "edge_feats": torch.empty((0, self.edge_features.features.shape[1]), dtype=torch.float32, device=dev),
                "labels": torch.empty(0, dtype=torch.float32, device=dev),
                "edge_mask": torch.empty(0, dtype=torch.bool, device=dev),
            }

        edge_index_sub = self.edge_index_tensor[:, mask_idx].to(dev)
        edge_feats_sub = self.edge_feats_tensor[mask_idx].to(dev)
        labels_sub = self.labels_tensor[mask_idx].to(dev)

        return {
            "node_feats": self.node_feats_tensor.to(dev),
            "edge_index": edge_index_sub,
            "edge_feats": edge_feats_sub,
            "labels": labels_sub,
            "edge_mask": torch.from_numpy(split_mask).to(dev),
        }
