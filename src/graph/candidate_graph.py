"""
Temporal Candidate Graph Construction and Representation.

Connects detections across consecutive frames (t -> t+1) using configurable physical distance gating,
adaptive displacement limits, and top-K candidate limits.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.graph.nodes import CanonicalNode, NodeSet


@dataclass
class CandidateEdge:
    """
    Representation of a temporal candidate edge between consecutive frames.
    """
    source_node_id: int
    destination_node_id: int
    source_frame: int
    destination_frame: int
    frame_difference: int
    physical_displacement_vector: Tuple[float, float, float]  # (dz_phys, dy_phys, dx_phys)
    euclidean_distance_phys: float
    voxel_displacement_vector: Tuple[float, float, float]  # (dz_vox, dy_vox, dx_vox)
    source_confidence: float
    destination_confidence: float
    pair_confidence: float  # source_confidence * destination_confidence
    local_density: Optional[float] = None
    source_region_id: Optional[int] = None
    destination_region_id: Optional[int] = None
    gating_reason: str = "distance_gate"
    passed_distance_gate: bool = True
    rank_among_destinations: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            "source_node_id": self.source_node_id,
            "destination_node_id": self.destination_node_id,
            "source_frame": self.source_frame,
            "destination_frame": self.destination_frame,
            "frame_difference": self.frame_difference,
            "dz_phys": self.physical_displacement_vector[0],
            "dy_phys": self.physical_displacement_vector[1],
            "dx_phys": self.physical_displacement_vector[2],
            "euclidean_distance_phys": self.euclidean_distance_phys,
            "dz_vox": self.voxel_displacement_vector[0],
            "dy_vox": self.voxel_displacement_vector[1],
            "dx_vox": self.voxel_displacement_vector[2],
            "source_confidence": self.source_confidence,
            "destination_confidence": self.destination_confidence,
            "pair_confidence": self.pair_confidence,
            "local_density": self.local_density,
            "source_region_id": self.source_region_id,
            "destination_region_id": self.destination_region_id,
            "gating_reason": self.gating_reason,
            "passed_distance_gate": self.passed_distance_gate,
            "rank_among_destinations": self.rank_among_destinations,
        }


@dataclass
class CandidateGraphConfig:
    """
    Configuration parameters for temporal candidate graph generation.
    """
    max_physical_distance_um: float = 20.0  # Max physical distance threshold in micrometers
    top_k_per_node: Optional[int] = 5  # Max candidate destinations per source node (None = no limit)
    max_frame_gap: int = 1  # Maximum temporal frame difference (1 = consecutive frames)
    adaptive_velocity_factor: Optional[float] = None  # Optional adaptive displacement multiplier


class TemporalCandidateGraph:
    """
    Typed native graph representation storing canonical nodes and temporal candidate edges.
    Supports outgoing/incoming candidate queries, edge lookup, diagnostics, and tensor exports.
    """

    def __init__(
        self,
        node_set: NodeSet,
        edges: List[CandidateEdge],
        config: CandidateGraphConfig
    ):
        self.node_set = node_set
        self.edges = edges
        self.config = config

        self._outgoing_edges: Dict[int, List[CandidateEdge]] = {}
        self._incoming_edges: Dict[int, List[CandidateEdge]] = {}
        self._edge_by_pair: Dict[Tuple[int, int], CandidateEdge] = {}

        self._build_index()

    def _build_index(self) -> None:
        """Build fast lookup indexes for edges."""
        for edge in self.edges:
            src = edge.source_node_id
            dst = edge.destination_node_id

            if src not in self._outgoing_edges:
                self._outgoing_edges[src] = []
            self._outgoing_edges[src].append(edge)

            if dst not in self._incoming_edges:
                self._incoming_edges[dst] = []
            self._incoming_edges[dst].append(edge)

            self._edge_by_pair[(src, dst)] = edge

    def get_nodes_by_frame(self, frame: int) -> List[CanonicalNode]:
        """Retrieve all canonical nodes active at specified frame."""
        return self.node_set.get_nodes_by_frame(frame)

    def get_outgoing_edges(self, node_id: int) -> List[CandidateEdge]:
        """Retrieve all outgoing candidate edges originating from node_id."""
        return self._outgoing_edges.get(node_id, [])

    def get_incoming_edges(self, node_id: int) -> List[CandidateEdge]:
        """Retrieve all incoming candidate edges terminating at node_id."""
        return self._incoming_edges.get(node_id, [])

    def get_edge(self, source_id: int, destination_id: int) -> Optional[CandidateEdge]:
        """Retrieve candidate edge between (source_id, destination_id)."""
        return self._edge_by_pair.get((source_id, destination_id))

    def candidate_count(self) -> int:
        """Total number of candidate edges in graph."""
        return len(self.edges)

    def num_nodes(self) -> int:
        """Total number of nodes in graph."""
        return self.node_set.num_nodes()

    def export_diagnostics(self) -> Dict[str, Any]:
        """
        Calculates and exports diagnostic statistics about the candidate graph.
        """
        all_nodes = self.node_set.all_nodes()
        total_nodes = len(all_nodes)
        if total_nodes == 0:
            return {
                "total_nodes": 0,
                "total_candidate_edges": 0,
                "mean_candidates_per_node": 0.0,
                "zero_candidate_pct": 0.0,
                "one_candidate_pct": 0.0,
                "multi_candidate_pct": 0.0,
            }

        # Calculate candidate counts for source nodes (excluding last frame nodes)
        max_frame = max(self.node_set.frames()) if self.node_set.frames() else 0
        source_nodes = [n for n in all_nodes if n.frame < max_frame]
        n_sources = len(source_nodes)

        cand_counts = [len(self.get_outgoing_edges(n.node_id)) for n in source_nodes]
        counts_arr = np.array(cand_counts, dtype=np.int32) if cand_counts else np.array([0])

        zero_cnt = int(np.sum(counts_arr == 0))
        one_cnt = int(np.sum(counts_arr == 1))
        multi_cnt = int(np.sum(counts_arr > 1))

        return {
            "total_nodes": total_nodes,
            "source_nodes_evaluated": n_sources,
            "total_candidate_edges": len(self.edges),
            "mean_candidates_per_node": float(np.mean(counts_arr)) if n_sources > 0 else 0.0,
            "min_candidates_per_node": int(np.min(counts_arr)) if n_sources > 0 else 0,
            "median_candidates_per_node": float(np.median(counts_arr)) if n_sources > 0 else 0.0,
            "max_candidates_per_node": int(np.max(counts_arr)) if n_sources > 0 else 0,
            "zero_candidate_pct": float(100.0 * zero_cnt / n_sources) if n_sources > 0 else 0.0,
            "one_candidate_pct": float(100.0 * one_cnt / n_sources) if n_sources > 0 else 0.0,
            "multi_candidate_pct": float(100.0 * multi_cnt / n_sources) if n_sources > 0 else 0.0,
        }

    def to_tensor_dict(self) -> Dict[str, Any]:
        """
        Exports graph data structures to NumPy arrays suitable for future conversion into PyTorch / PyG tensors.
        """
        node_ids = []
        node_coords_phys = []
        node_coords_vox = []
        node_confidences = []
        node_frames = []

        node_id_to_idx = {}
        for idx, node in enumerate(self.node_set.all_nodes()):
            node_id_to_idx[node.node_id] = idx
            node_ids.append(node.node_id)
            node_coords_phys.append(node.physical_pos)
            node_coords_vox.append(node.global_voxel_pos)
            node_confidences.append(node.confidence)
            node_frames.append(node.frame)

        edge_index = []
        edge_attr = []
        for edge in self.edges:
            src_idx = node_id_to_idx[edge.source_node_id]
            dst_idx = node_id_to_idx[edge.destination_node_id]
            edge_index.append([src_idx, dst_idx])
            edge_attr.append([
                edge.physical_displacement_vector[0],
                edge.physical_displacement_vector[1],
                edge.physical_displacement_vector[2],
                edge.euclidean_distance_phys,
                edge.source_confidence,
                edge.destination_confidence,
                edge.pair_confidence,
            ])

        return {
            "node_ids": np.array(node_ids, dtype=np.int64),
            "node_coords_phys": np.array(node_coords_phys, dtype=np.float32),
            "node_coords_vox": np.array(node_coords_vox, dtype=np.float32),
            "node_confidences": np.array(node_confidences, dtype=np.float32),
            "node_frames": np.array(node_frames, dtype=np.int64),
            "edge_index": np.array(edge_index, dtype=np.int64).T if edge_index else np.empty((2, 0), dtype=np.int64),
            "edge_attr": np.array(edge_attr, dtype=np.float32) if edge_attr else np.empty((0, 7), dtype=np.float32),
        }


class CandidateGraphBuilder:
    """
    Constructs a TemporalCandidateGraph connecting detections across consecutive frames.
    """

    def __init__(self, config: Optional[CandidateGraphConfig] = None):
        self.config = config or CandidateGraphConfig()

    def build_candidate_graph(self, node_set: NodeSet) -> TemporalCandidateGraph:
        """
        Builds candidate graph from given Canonical NodeSet.

        :param node_set: NodeSet container with frame-indexed CanonicalNode instances
        :return: TemporalCandidateGraph instance
        """
        edges: List[CandidateEdge] = []
        frames = node_set.frames()

        for t in frames:
            source_nodes = node_set.get_nodes_by_frame(t)
            if not source_nodes:
                continue

            # Target destination frames (t + 1 up to t + max_frame_gap)
            dest_frames = [t + gap for gap in range(1, self.config.max_frame_gap + 1)]

            for u in source_nodes:
                candidate_dests: List[Tuple[CanonicalNode, float, int]] = []

                for target_t in dest_frames:
                    dest_nodes = node_set.get_nodes_by_frame(target_t)
                    frame_gap = target_t - t

                    for v in dest_nodes:
                        dist_phys = u.distance_physical(v)

                        # Effective distance limit
                        max_dist = self.config.max_physical_distance_um
                        if self.config.adaptive_velocity_factor is not None:
                            max_dist *= self.config.adaptive_velocity_factor

                        if dist_phys <= max_dist:
                            candidate_dests.append((v, dist_phys, frame_gap))

                # Sort candidate destinations by physical distance ascending
                candidate_dests.sort(key=lambda item: item[1])

                # Apply top-K per node limit if specified
                retained_candidates = candidate_dests
                if self.config.top_k_per_node is not None and self.config.top_k_per_node > 0:
                    retained_candidates = candidate_dests[:self.config.top_k_per_node]

                # Create CandidateEdge for each retained destination
                for rank, (v, dist_phys, frame_gap) in enumerate(retained_candidates, start=1):
                    phys_disp = u.displacement_physical(v)
                    vox_disp = u.displacement_voxel(v)
                    pair_conf = float(u.confidence * v.confidence)

                    edge = CandidateEdge(
                        source_node_id=u.node_id,
                        destination_node_id=v.node_id,
                        source_frame=u.frame,
                        destination_frame=v.frame,
                        frame_difference=frame_gap,
                        physical_displacement_vector=phys_disp,
                        euclidean_distance_phys=float(dist_phys),
                        voxel_displacement_vector=vox_disp,
                        source_confidence=float(u.confidence),
                        destination_confidence=float(v.confidence),
                        pair_confidence=pair_conf,
                        source_region_id=u.primary_region_id,
                        destination_region_id=v.primary_region_id,
                        gating_reason="physical_distance_and_top_k",
                        passed_distance_gate=True,
                        rank_among_destinations=rank,
                    )
                    edges.append(edge)

        return TemporalCandidateGraph(
            node_set=node_set,
            edges=edges,
            config=self.config
        )
