"""
Candidate Graph Diagnostics and GEFF Ground-Truth Candidate Recall Evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np

from src.graph.nodes import CanonicalNode, NodeSet
from src.graph.candidate_graph import TemporalCandidateGraph, CandidateEdge
from src.data.nodes import CellNode
from src.data.coordinates import CoordinateTransformer


@dataclass
class NodeMatchingResult:
    """
    Diagnostic results from matching canonical detection nodes to GEFF ground-truth nodes.
    """
    geff_node_count: int
    canonical_node_count: int
    per_frame_geff_counts: Dict[int, int]
    per_frame_canonical_counts: Dict[int, int]
    matched_geff_count: int
    matching_success_rate: float  # (matched_geff_count / geff_node_count) * 100.0
    unmatched_geff_count: int
    geff_to_canonical_map: Dict[int, int]  # geff_node_id -> canonical_node_id
    canonical_to_geff_map: Dict[int, int]  # canonical_node_id -> geff_node_id
    match_distances_um: List[float]


@dataclass
class CandidateRecallMetrics:
    """
    Diagnostic recall metrics evaluating candidate graph coverage against ground-truth edges.
    """
    num_gt_edges: int
    num_evaluable_gt_edges: int  # GT edges where both endpoints matched canonical nodes
    num_candidate_edges: int
    candidate_graph_density: float  # Average candidates per source node
    gt_edges_recovered: int
    candidate_recall: float  # gt_edges_recovered / num_gt_edges
    evaluable_candidate_recall: float  # gt_edges_recovered / num_evaluable_gt_edges
    num_missing_gt_edges: int
    missing_reasons: Dict[str, int]  # Reason counts for missing GT edges
    gt_edge_distance_dist: Dict[str, float]  # min, median, max, mean physical distance
    candidate_count_per_source: Dict[str, float]  # min, median, max, mean candidate count
    zero_candidate_pct: float
    one_candidate_pct: float
    multi_candidate_pct: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "num_gt_edges": self.num_gt_edges,
            "num_evaluable_gt_edges": self.num_evaluable_gt_edges,
            "num_candidate_edges": self.num_candidate_edges,
            "candidate_graph_density": self.candidate_graph_density,
            "gt_edges_recovered": self.gt_edges_recovered,
            "candidate_recall": self.candidate_recall,
            "evaluable_candidate_recall": self.evaluable_candidate_recall,
            "num_missing_gt_edges": self.num_missing_gt_edges,
            "missing_reasons": self.missing_reasons,
            "gt_edge_distance_dist": self.gt_edge_distance_dist,
            "candidate_count_per_source": self.candidate_count_per_source,
            "zero_candidate_pct": self.zero_candidate_pct,
            "one_candidate_pct": self.one_candidate_pct,
            "multi_candidate_pct": self.multi_candidate_pct,
        }


class CandidateRecallEvaluator:
    """
    Evaluator performing Part 11 Node Matching Check and Part 4 Candidate Recall Diagnostics.
    """

    def __init__(self, transformer: Optional[CoordinateTransformer] = None):
        self.transformer = transformer or CoordinateTransformer()

    def match_nodes_to_geff(
        self,
        node_set: NodeSet,
        geff_nodes: List[CellNode],
        match_threshold_um: float = 5.0
    ) -> NodeMatchingResult:
        """
        Part 11 Data Integrity Check:
        Matches CanonicalNode set to GEFF GT nodes per frame using greedy nearest physical distance.

        :param node_set: NodeSet containing canonical detection nodes
        :param geff_nodes: List of ground-truth CellNode objects from GEFF
        :param match_threshold_um: Maximum physical distance for valid matching (default 5.0 um)
        :return: NodeMatchingResult dataclass with matching metrics and mappings
        """
        geff_nodes_by_frame: Dict[int, List[CellNode]] = {}
        for g_node in geff_nodes:
            if g_node.frame not in geff_nodes_by_frame:
                geff_nodes_by_frame[g_node.frame] = []
            geff_nodes_by_frame[g_node.frame].append(g_node)

        geff_to_canonical_map: Dict[int, int] = {}
        canonical_to_geff_map: Dict[int, int] = {}
        match_distances: List[float] = []

        per_frame_geff_counts: Dict[int, int] = {}
        per_frame_canonical_counts: Dict[int, int] = {}

        all_frames = sorted(list(set(geff_nodes_by_frame.keys()) | set(node_set.frames())))

        for f in all_frames:
            f_geff = geff_nodes_by_frame.get(f, [])
            f_can = node_set.get_nodes_by_frame(f)

            per_frame_geff_counts[f] = len(f_geff)
            per_frame_canonical_counts[f] = len(f_can)

            if not f_geff or not f_can:
                continue

            # Compute physical pairwise distance matrix (n_geff, n_can)
            n_g = len(f_geff)
            n_c = len(f_can)
            dist_mat = np.zeros((n_g, n_c), dtype=np.float32)

            for i, g in enumerate(f_geff):
                for j, c in enumerate(f_can):
                    d = self.transformer.euclidean_distance_physical(g.physical_pos, c.physical_pos)
                    dist_mat[i, j] = d

            # Greedy distance matching within match_threshold_um
            matched_g_indices: Set[int] = set()
            matched_c_indices: Set[int] = set()

            # Flatten and sort candidate pairs by physical distance
            flat_pairs = []
            for i in range(n_g):
                for j in range(n_c):
                    if dist_mat[i, j] <= match_threshold_um:
                        flat_pairs.append((dist_mat[i, j], i, j))
            flat_pairs.sort(key=lambda item: item[0])

            for dist, i, j in flat_pairs:
                if i not in matched_g_indices and j not in matched_c_indices:
                    matched_g_indices.add(i)
                    matched_c_indices.add(j)

                    g_id = f_geff[i].node_id
                    c_id = f_can[j].node_id

                    geff_to_canonical_map[g_id] = c_id
                    canonical_to_geff_map[c_id] = g_id
                    match_distances.append(float(dist))

        matched_geff_count = len(geff_to_canonical_map)
        geff_node_count = len(geff_nodes)
        canonical_node_count = node_set.num_nodes()
        unmatched_geff_count = geff_node_count - matched_geff_count
        success_rate = float(100.0 * matched_geff_count / geff_node_count) if geff_node_count > 0 else 0.0

        return NodeMatchingResult(
            geff_node_count=geff_node_count,
            canonical_node_count=canonical_node_count,
            per_frame_geff_counts=per_frame_geff_counts,
            per_frame_canonical_counts=per_frame_canonical_counts,
            matched_geff_count=matched_geff_count,
            matching_success_rate=success_rate,
            unmatched_geff_count=unmatched_geff_count,
            geff_to_canonical_map=geff_to_canonical_map,
            canonical_to_geff_map=canonical_to_geff_map,
            match_distances_um=match_distances
        )

    def evaluate_recall(
        self,
        graph: TemporalCandidateGraph,
        geff_edges: List[Tuple[int, int]],
        geff_nodes: List[CellNode],
        matching_result: NodeMatchingResult
    ) -> CandidateRecallMetrics:
        """
        Part 4 Candidate Recall Diagnostic:
        Evaluates ground-truth edge recovery by candidate graph and diagnoses missing edges.

        :param graph: TemporalCandidateGraph instance
        :param geff_edges: List of directed (source_geff_id, dest_geff_id) GT edges
        :param geff_nodes: List of ground-truth CellNode objects
        :param matching_result: Pre-computed NodeMatchingResult
        :return: CandidateRecallMetrics dataclass
        """
        geff_node_map = {g.node_id: g for g in geff_nodes}
        gt_map = matching_result.geff_to_canonical_map

        num_gt = len(geff_edges)
        num_evaluable = 0
        recovered = 0

        missing_reasons = {
            "source_unmatched": 0,
            "destination_unmatched": 0,
            "distance_gate_rejected": 0,
            "top_k_truncated": 0,
            "other": 0,
        }

        gt_distances: List[float] = []

        for u_gt, v_gt in geff_edges:
            u_node = geff_node_map.get(u_gt)
            v_node = geff_node_map.get(v_gt)

            if u_node is not None and v_node is not None:
                d_phys = u_node.distance_physical(v_node)
                gt_distances.append(d_phys)

            u_can = gt_map.get(u_gt)
            v_can = gt_map.get(v_gt)

            if u_can is None or v_can is None:
                if u_can is None and v_can is None:
                    missing_reasons["source_unmatched"] += 1
                elif u_can is None:
                    missing_reasons["source_unmatched"] += 1
                else:
                    missing_reasons["destination_unmatched"] += 1
                continue

            num_evaluable += 1

            # Check if edge exists in candidate graph
            candidate_edge = graph.get_edge(u_can, v_can)
            if candidate_edge is not None:
                recovered += 1
            else:
                # Diagnose missing edge reason
                can_u_node = graph.node_set.get_node(u_can)
                can_v_node = graph.node_set.get_node(v_can)

                if can_u_node is not None and can_v_node is not None:
                    d_phys_can = can_u_node.distance_physical(can_v_node)
                    max_dist = graph.config.max_physical_distance_um

                    if d_phys_can > max_dist:
                        missing_reasons["distance_gate_rejected"] += 1
                    else:
                        # Edge passed distance gate but was truncated by top-K
                        missing_reasons["top_k_truncated"] += 1
                else:
                    missing_reasons["other"] += 1

        num_missing = num_gt - recovered
        recall = float(recovered / num_gt) if num_gt > 0 else 0.0
        evaluable_recall = float(recovered / num_evaluable) if num_evaluable > 0 else 0.0

        # Diagnostics stats for GT distances
        if gt_distances:
            dist_arr = np.array(gt_distances, dtype=np.float32)
            gt_dist_stats = {
                "min": float(np.min(dist_arr)),
                "median": float(np.median(dist_arr)),
                "max": float(np.max(dist_arr)),
                "mean": float(np.mean(dist_arr)),
            }
        else:
            gt_dist_stats = {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}

        # Candidate counts per source node
        diag = graph.export_diagnostics()
        cand_stats = {
            "min": float(diag.get("min_candidates_per_node", 0)),
            "median": float(diag.get("median_candidates_per_node", 0)),
            "max": float(diag.get("max_candidates_per_node", 0)),
            "mean": float(diag.get("mean_candidates_per_node", 0)),
        }

        return CandidateRecallMetrics(
            num_gt_edges=num_gt,
            num_evaluable_gt_edges=num_evaluable,
            num_candidate_edges=graph.candidate_count(),
            candidate_graph_density=float(diag.get("mean_candidates_per_node", 0.0)),
            gt_edges_recovered=recovered,
            candidate_recall=recall,
            evaluable_candidate_recall=evaluable_recall,
            num_missing_gt_edges=num_missing,
            missing_reasons=missing_reasons,
            gt_edge_distance_dist=gt_dist_stats,
            candidate_count_per_source=cand_stats,
            zero_candidate_pct=float(diag.get("zero_candidate_pct", 0.0)),
            one_candidate_pct=float(diag.get("one_candidate_pct", 0.0)),
            multi_candidate_pct=float(diag.get("multi_candidate_pct", 0.0)),
        )
