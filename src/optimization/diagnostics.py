"""
Track Fragment Decoding, Optimization Diagnostics, and Ground-Truth Evaluation.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np

from src.graph.nodes import CanonicalNode, NodeSet
from src.graph.candidate_graph import CandidateEdge, TemporalCandidateGraph
from src.graph.diagnostics import NodeMatchingResult
from src.optimization.base import TrackFragment, TrackingSolution


class TrackFragmentDecoder:
    """
    Decodes direct-continuation track fragments from selected ILP candidate edges.

    A direct-continuation track fragment is a linear chain of consecutive nodes:
        v_1 -> v_2 -> ... -> v_k
    connected by selected candidate edges across consecutive frames.
    """

    def decode_fragments(
        self,
        node_set: NodeSet,
        selected_edges: List[CandidateEdge]
    ) -> List[TrackFragment]:
        """
        Decodes direct-continuation track fragments.

        :param node_set: NodeSet container
        :param selected_edges: List of selected CandidateEdge objects (x_e = 1)
        :return: List of TrackFragment objects
        """
        if not selected_edges:
            return []

        # Build adjacency graph for selected edges
        # outgoing[src] = dst
        outgoing: Dict[int, int] = {}
        # incoming[dst] = src
        incoming: Dict[int, int] = {}
        # edge_map[(src, dst)] = CandidateEdge
        edge_map: Dict[Tuple[int, int], CandidateEdge] = {}

        for edge in selected_edges:
            src = edge.source_node_id
            dst = edge.destination_node_id
            outgoing[src] = dst
            incoming[dst] = src
            edge_map[(src, dst)] = edge

        # Identify starting nodes for track fragments
        # A starting node has an outgoing selected edge but NO incoming selected edge
        all_src_nodes = set(outgoing.keys())
        all_dst_nodes = set(incoming.keys())

        start_node_ids = sorted(list(all_src_nodes - all_dst_nodes))

        # Also catch any isolated cycles if they exist (unlikely in temporal DAG)
        visited_nodes: Set[int] = set()
        fragments: List[TrackFragment] = []
        frag_id = 1

        for start_id in start_node_ids:
            if start_id in visited_nodes:
                continue

            curr = start_id
            chain = [curr]
            visited_nodes.add(curr)
            total_dist = 0.0

            while curr in outgoing:
                nxt = outgoing[curr]
                if nxt in visited_nodes:
                    break  # Cycle detected or already visited

                edge = edge_map.get((curr, nxt))
                if edge:
                    total_dist += edge.euclidean_distance_phys

                chain.append(nxt)
                visited_nodes.add(nxt)
                curr = nxt

            # Retrieve frame and position info for fragment nodes
            frames = []
            phys_positions = []
            for nid in chain:
                n = node_set.get_node(nid)
                if n:
                    frames.append(n.frame)
                    phys_positions.append(n.physical_pos)
                else:
                    frames.append(0)
                    phys_positions.append((0.0, 0.0, 0.0))

            fragment = TrackFragment(
                fragment_id=frag_id,
                node_ids=chain,
                frames=frames,
                physical_positions=phys_positions,
                length_frames=len(chain),
                total_physical_distance_um=float(total_dist)
            )
            fragments.append(fragment)
            frag_id += 1

        return fragments


@dataclass
class ILPDiagnosticReport:
    """
    Comprehensive diagnostic report evaluating optimization results and ground-truth alignment.
    """
    total_nodes: int
    total_candidate_edges: int
    selected_edges: int
    rejected_edges: int
    num_track_fragments: int
    fragment_lengths: Dict[str, float]  # min, median, max, mean
    longest_fragment_len: int
    isolated_nodes_count: int
    nodes_with_predecessor_only: int
    nodes_with_successor_only: int
    nodes_with_both: int
    nodes_with_neither: int
    constraint_violations_count: int
    # Ground truth metrics
    gt_edges_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    edge_precision: float
    edge_recall: float
    edge_f1: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        return {
            "total_nodes": self.total_nodes,
            "total_candidate_edges": self.total_candidate_edges,
            "selected_edges": self.selected_edges,
            "rejected_edges": self.rejected_edges,
            "num_track_fragments": self.num_track_fragments,
            "fragment_lengths": self.fragment_lengths,
            "longest_fragment_len": self.longest_fragment_len,
            "isolated_nodes_count": self.isolated_nodes_count,
            "nodes_with_predecessor_only": self.nodes_with_predecessor_only,
            "nodes_with_successor_only": self.nodes_with_successor_only,
            "nodes_with_both": self.nodes_with_both,
            "nodes_with_neither": self.nodes_with_neither,
            "constraint_violations_count": self.constraint_violations_count,
            "gt_edges_count": self.gt_edges_count,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "edge_precision": self.edge_precision,
            "edge_recall": self.edge_recall,
            "edge_f1": self.edge_f1,
        }


class ILPDiagnostics:
    """
    Evaluator performing Part 13 Track Fragment Diagnostics and Part 14 Ground-Truth Diagnostics.
    """

    def evaluate_solution(
        self,
        solution: TrackingSolution,
        node_set: NodeSet,
        geff_edges: Optional[List[Tuple[int, int]]] = None,
        matching_result: Optional[NodeMatchingResult] = None
    ) -> ILPDiagnosticReport:
        """
        Evaluates tracking solution metrics and verifies zero constraint violations.

        :param solution: TrackingSolution object
        :param node_set: NodeSet container
        :param geff_edges: Optional list of directed (u_geff, v_geff) GT edges
        :param matching_result: Optional NodeMatchingResult for node ID mapping
        :return: ILPDiagnosticReport dataclass
        """
        selected = solution.selected_edges
        rejected = solution.rejected_edges
        all_nodes = node_set.all_nodes()
        total_nodes = len(all_nodes)
        total_cand_edges = len(selected) + len(rejected)

        # 1. Degree counts and constraint violation check
        inc_degree: Dict[int, int] = {}
        out_degree: Dict[int, int] = {}

        for edge in selected:
            src = edge.source_node_id
            dst = edge.destination_node_id

            out_degree[src] = out_degree.get(src, 0) + 1
            inc_degree[dst] = inc_degree.get(dst, 0) + 1

        # Check constraint violations (>1 incoming or >1 outgoing)
        violations = 0
        for n in all_nodes:
            nid = n.node_id
            if inc_degree.get(nid, 0) > 1:
                violations += (inc_degree[nid] - 1)
            if out_degree.get(nid, 0) > 1:
                violations += (out_degree[nid] - 1)

        # Node degree categories
        n_pred_only = 0
        n_succ_only = 0
        n_both = 0
        n_neither = 0

        for n in all_nodes:
            nid = n.node_id
            has_inc = inc_degree.get(nid, 0) > 0
            has_out = out_degree.get(nid, 0) > 0

            if has_inc and has_out:
                n_both += 1
            elif has_inc and not has_out:
                n_pred_only += 1
            elif not has_inc and has_out:
                n_succ_only += 1
            else:
                n_neither += 1

        # Fragment length stats
        frags = solution.fragments
        n_frags = len(frags)

        if n_frags > 0:
            lengths = [f.length_frames for f in frags]
            frag_stats = {
                "min": float(np.min(lengths)),
                "median": float(np.median(lengths)),
                "max": float(np.max(lengths)),
                "mean": float(np.mean(lengths)),
            }
            longest_len = int(np.max(lengths))
        else:
            frag_stats = {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
            longest_len = 0

        # 2. Ground-truth edge evaluation
        tp = fp = fn = 0
        prec = rec = f1 = 0.0
        gt_count = 0

        if geff_edges is not None and matching_result is not None:
            gt_map = matching_result.geff_to_canonical_map
            true_can_pairs: Set[Tuple[int, int]] = set()

            node_ids_in_set = set(n.node_id for n in all_nodes)
            for u_gt, v_gt in geff_edges:
                u_can = gt_map.get(int(u_gt))
                v_can = gt_map.get(int(v_gt))
                if u_can is not None and v_can is not None:
                    if u_can in node_ids_in_set and v_can in node_ids_in_set:
                        true_can_pairs.add((u_can, v_can))

            gt_count = len(true_can_pairs)
            selected_pairs = set((e.source_node_id, e.destination_node_id) for e in selected)

            tp = len(selected_pairs & true_can_pairs)
            fp = len(selected_pairs - true_can_pairs)
            fn = len(true_can_pairs - selected_pairs)

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        return ILPDiagnosticReport(
            total_nodes=total_nodes,
            total_candidate_edges=total_cand_edges,
            selected_edges=len(selected),
            rejected_edges=len(rejected),
            num_track_fragments=n_frags,
            fragment_lengths=frag_stats,
            longest_fragment_len=longest_len,
            isolated_nodes_count=n_neither,
            nodes_with_predecessor_only=n_pred_only,
            nodes_with_successor_only=n_succ_only,
            nodes_with_both=n_both,
            nodes_with_neither=n_neither,
            constraint_violations_count=violations,
            gt_edges_count=gt_count,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            edge_precision=prec,
            edge_recall=rec,
            edge_f1=f1
        )
