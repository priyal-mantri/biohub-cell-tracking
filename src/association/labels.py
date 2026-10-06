"""
Deterministic Ground-Truth Candidate Edge Label Generation and Temporal Data Splitting.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np

from src.graph.nodes import CanonicalNode, NodeSet
from src.graph.candidate_graph import TemporalCandidateGraph, CandidateEdge
from src.graph.diagnostics import NodeMatchingResult


@dataclass
class LabelStats:
    """Statistics describing generated candidate edge ground-truth labels."""
    total_candidate_edges: int
    positive_edges: int
    negative_edges: int
    positive_rate_pct: float
    negatives_per_positive: float
    pos_edge_distances_um: List[float]
    neg_edge_distances_um: List[float]


class CandidateEdgeLabeler:
    """
    Generates binary ground-truth labels for candidate edges AFTER candidate graph construction.

    Label = 1 if the candidate edge corresponds to a true GEFF temporal edge.
    Label = 0 if the candidate edge is not a true GEFF temporal edge.
    """

    def create_labels(
        self,
        graph: TemporalCandidateGraph,
        geff_edges: List[Tuple[int, int]],
        matching_result: NodeMatchingResult
    ) -> Tuple[np.ndarray, LabelStats]:
        """
        Creates ground-truth labels array for all candidate edges in the graph.

        :param graph: TemporalCandidateGraph instance
        :param geff_edges: List of directed (source_geff_id, dest_geff_id) ground-truth edges
        :param matching_result: NodeMatchingResult mapping GEFF IDs -> Canonical Node IDs
        :return: Tuple of (labels_array shape (E,) float32, LabelStats)
        """
        geff_to_can = matching_result.geff_to_canonical_map

        # Map GEFF GT edge pairs to canonical node ID pairs
        true_canonical_edge_pairs: Set[Tuple[int, int]] = set()
        for u_gt, v_gt in geff_edges:
            u_can = geff_to_can.get(int(u_gt))
            v_can = geff_to_can.get(int(v_gt))
            if u_can is not None and v_can is not None:
                true_canonical_edge_pairs.add((u_can, v_can))

        labels = []
        pos_distances = []
        neg_distances = []

        for edge in graph.edges:
            pair = (edge.source_node_id, edge.destination_node_id)
            if pair in true_canonical_edge_pairs:
                labels.append(1.0)
                pos_distances.append(edge.euclidean_distance_phys)
            else:
                labels.append(0.0)
                neg_distances.append(edge.euclidean_distance_phys)

        labels_arr = np.array(labels, dtype=np.float32)
        n_total = len(labels)
        n_pos = int(np.sum(labels_arr == 1.0))
        n_neg = int(np.sum(labels_arr == 0.0))

        pos_rate = float(100.0 * n_pos / n_total) if n_total > 0 else 0.0
        neg_per_pos = float(n_neg / n_pos) if n_pos > 0 else 0.0

        stats = LabelStats(
            total_candidate_edges=n_total,
            positive_edges=n_pos,
            negative_edges=n_neg,
            positive_rate_pct=pos_rate,
            negatives_per_positive=neg_per_pos,
            pos_edge_distances_um=pos_distances,
            neg_edge_distances_um=neg_distances,
        )

        return labels_arr, stats


@dataclass
class TemporalSplit:
    """Container holding mask indices for Train, Validation, and Test candidate edges."""
    train_mask: np.ndarray  # boolean array (E,)
    val_mask: np.ndarray  # boolean array (E,)
    test_mask: np.ndarray  # boolean array (E,)
    train_frames: Tuple[int, int]
    val_frames: Tuple[int, int]
    test_frames: Tuple[int, int]


class TemporalSplitter:
    """
    Splits candidate edges into Train, Validation, and Test sets based strictly on source frame t.

    Guarantees:
    - Zero data leakage between temporal splits (edges belong to split of source frame t).
    """

    def __init__(
        self,
        train_range: Tuple[int, int] = (0, 69),
        val_range: Tuple[int, int] = (70, 84),
        test_range: Tuple[int, int] = (85, 99)
    ):
        self.train_range = train_range
        self.val_range = val_range
        self.test_range = test_range

    def split_graph_edges(self, graph: TemporalCandidateGraph) -> TemporalSplit:
        """
        Creates boolean masks partitioning graph.edges into train, val, and test.

        :param graph: TemporalCandidateGraph instance
        :return: TemporalSplit dataclass
        """
        n_edges = graph.candidate_count()
        train_mask = np.zeros(n_edges, dtype=bool)
        val_mask = np.zeros(n_edges, dtype=bool)
        test_mask = np.zeros(n_edges, dtype=bool)

        for idx, edge in enumerate(graph.edges):
            t = edge.source_frame
            if self.train_range[0] <= t <= self.train_range[1]:
                train_mask[idx] = True
            elif self.val_range[0] <= t <= self.val_range[1]:
                val_mask[idx] = True
            elif self.test_range[0] <= t <= self.test_range[1]:
                test_mask[idx] = True

        return TemporalSplit(
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            train_frames=self.train_range,
            val_frames=self.val_range,
            test_frames=self.test_range,
        )
