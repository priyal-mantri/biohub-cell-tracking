"""
Canonical Node Representation and Deterministic Frame-Local Node Reconciliation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union, Set
import numpy as np

from src.data.coordinates import CoordinateTransformer
from src.data.nodes import CellNode
from src.regional.geometry import RegionalCandidate


@dataclass
class CanonicalNode:
    """
    Canonical global graph node representation for a cell detection at a specific frame.

    Contains stable node ID, spatiotemporal coordinates (voxel & physical),
    detector confidence, regional provenance, and diagnostic metadata.
    """
    node_id: int
    frame: int  # Time index t (0-indexed)
    global_voxel_pos: Tuple[float, float, float]  # (z_voxel, y_voxel, x_voxel)
    physical_pos: Tuple[float, float, float]  # (z_phys, y_phys, x_phys) in micrometers
    confidence: float
    detector_features: Optional[Any] = None
    contributing_regional_ids: List[int] = field(default_factory=list)
    num_regional_observations: int = 1
    primary_region_id: int = 0
    diagnostic_metadata: Dict[str, Any] = field(default_factory=dict)

    def distance_physical(self, other: "CanonicalNode") -> float:
        """Compute Euclidean distance to another node in physical space (micrometers)."""
        dz = self.physical_pos[0] - other.physical_pos[0]
        dy = self.physical_pos[1] - other.physical_pos[1]
        dx = self.physical_pos[2] - other.physical_pos[2]
        return float(np.sqrt(dz * dz + dy * dy + dx * dx))

    def displacement_physical(self, other: "CanonicalNode") -> Tuple[float, float, float]:
        """Compute displacement vector from self to other in physical space (micrometers)."""
        dz = float(other.physical_pos[0] - self.physical_pos[0])
        dy = float(other.physical_pos[1] - self.physical_pos[1])
        dx = float(other.physical_pos[2] - self.physical_pos[2])
        return (dz, dy, dx)

    def displacement_voxel(self, other: "CanonicalNode") -> Tuple[float, float, float]:
        """Compute displacement vector from self to other in voxel space."""
        dz = float(other.global_voxel_pos[0] - self.global_voxel_pos[0])
        dy = float(other.global_voxel_pos[1] - self.global_voxel_pos[1])
        dx = float(other.global_voxel_pos[2] - self.global_voxel_pos[2])
        return (dz, dy, dx)

    def to_dict(self) -> Dict[str, Any]:
        """Returns node attributes as dictionary."""
        return {
            "node_id": self.node_id,
            "frame": self.frame,
            "z_voxel": self.global_voxel_pos[0],
            "y_voxel": self.global_voxel_pos[1],
            "x_voxel": self.global_voxel_pos[2],
            "z_phys": self.physical_pos[0],
            "y_phys": self.physical_pos[1],
            "x_phys": self.physical_pos[2],
            "confidence": self.confidence,
            "contributing_regional_ids": list(self.contributing_regional_ids),
            "num_regional_observations": self.num_regional_observations,
            "primary_region_id": self.primary_region_id,
            "diagnostic_metadata": self.diagnostic_metadata,
        }

    @classmethod
    def from_regional_candidate(
        cls,
        candidate: RegionalCandidate,
        canonical_id: int
    ) -> "CanonicalNode":
        """Factory method creating CanonicalNode from RegionalCandidate."""
        prov = candidate.provenance or {}
        contrib_ids = prov.get("contributing_region_ids", [candidate.region_id])
        num_obs = prov.get("num_contributing_regions", 1)
        primary_reg = prov.get("primary_region_id", candidate.region_id)

        v_pos = (
            float(candidate.global_voxel_pos[0]),
            float(candidate.global_voxel_pos[1]),
            float(candidate.global_voxel_pos[2]),
        )
        p_pos = (
            float(candidate.physical_pos[0]),
            float(candidate.physical_pos[1]),
            float(candidate.physical_pos[2]),
        )

        return cls(
            node_id=canonical_id,
            frame=candidate.frame,
            global_voxel_pos=v_pos,
            physical_pos=p_pos,
            confidence=float(candidate.confidence),
            contributing_regional_ids=sorted(list(set(contrib_ids))),
            num_regional_observations=num_obs,
            primary_region_id=primary_reg,
            diagnostic_metadata={
                "is_in_core": candidate.is_in_core,
                "original_candidate_id": candidate.candidate_id,
                "provenance": prov,
            }
        )

    @classmethod
    def from_cell_node(
        cls,
        cell_node: CellNode,
        canonical_id: Optional[int] = None
    ) -> "CanonicalNode":
        """Factory method creating CanonicalNode from foundational CellNode (e.g., GT node)."""
        nid = canonical_id if canonical_id is not None else cell_node.node_id
        v_pos = (
            float(cell_node.voxel_pos[0]),
            float(cell_node.voxel_pos[1]),
            float(cell_node.voxel_pos[2]),
        )
        p_pos = (
            float(cell_node.physical_pos[0]),
            float(cell_node.physical_pos[1]),
            float(cell_node.physical_pos[2]),
        )
        return cls(
            node_id=nid,
            frame=cell_node.frame,
            global_voxel_pos=v_pos,
            physical_pos=p_pos,
            confidence=float(cell_node.attributes.get("confidence", 1.0)),
            contributing_regional_ids=[0],
            num_regional_observations=1,
            primary_region_id=0,
            diagnostic_metadata=dict(cell_node.attributes)
        )


class NodeReconciler:
    """
    Frame-local deterministic node reconciliation across regional predictions.

    Groups detections within physical distance threshold, computes canonical positions,
    preserves complete regional provenance, and generates stable deterministic node IDs.
    """

    def __init__(
        self,
        reconciliation_distance_um: float = 4.0,
        transformer: Optional[CoordinateTransformer] = None
    ):
        self.reconciliation_distance_um = reconciliation_distance_um
        self.transformer = transformer or CoordinateTransformer()
        self.regional_to_canonical_map: Dict[Tuple[int, int], int] = {}  # (frame, candidate_id) -> canonical_id

    def reconcile_frame(
        self,
        frame: int,
        candidates: List[Union[RegionalCandidate, CanonicalNode, CellNode]]
    ) -> List[CanonicalNode]:
        """
        Reconciles candidates within a single frame into canonical global nodes.

        :param frame: Frame index t
        :param candidates: List of regional candidates or nodes active at frame t
        :return: List of canonical nodes for frame t
        """
        if not candidates:
            return []

        # Convert inputs to standard format if needed
        parsed_cands = []
        for idx, cand in enumerate(candidates):
            if isinstance(cand, CanonicalNode):
                parsed_cands.append(cand)
            elif isinstance(cand, RegionalCandidate):
                parsed_cands.append(cand)
            elif isinstance(cand, CellNode):
                parsed_cands.append(CanonicalNode.from_cell_node(cand, canonical_id=cand.node_id))
            else:
                raise TypeError(f"Unsupported candidate type: {type(cand)}")

        n_cand = len(parsed_cands)

        # Compute pairwise physical distance matrix
        phys_positions = [c.physical_pos for c in parsed_cands]
        dist_matrix = np.zeros((n_cand, n_cand), dtype=np.float32)
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                d = self.transformer.euclidean_distance_physical(phys_positions[i], phys_positions[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Connected component clustering
        visited = np.zeros(n_cand, dtype=bool)
        canonical_nodes: List[CanonicalNode] = []
        local_idx = 1

        for i in range(n_cand):
            if visited[i]:
                continue

            cluster_indices = [i]
            visited[i] = True
            queue = [i]

            while queue:
                curr = queue.pop(0)
                for neighbor in range(n_cand):
                    if not visited[neighbor] and dist_matrix[curr, neighbor] <= self.reconciliation_distance_um:
                        visited[neighbor] = True
                        cluster_indices.append(neighbor)
                        queue.append(neighbor)

            cluster_cands = [parsed_cands[idx] for idx in cluster_indices]

            # Generate deterministic canonical node ID: (frame + 1) * 100000 + local_idx
            canonical_id = (frame + 1) * 100000 + local_idx
            local_idx += 1

            # Primary candidate selected by max confidence
            primary = max(cluster_cands, key=lambda c: c.confidence)

            # Aggregate regional provenance
            all_contrib_regions: Set[int] = set()
            all_regional_candidate_ids = []
            for c in cluster_cands:
                if isinstance(c, RegionalCandidate):
                    prov = c.provenance or {}
                    contribs = prov.get("contributing_region_ids", [c.region_id])
                    all_contrib_regions.update(contribs)
                    all_regional_candidate_ids.append(c.candidate_id)
                    self.regional_to_canonical_map[(frame, c.candidate_id)] = canonical_id
                elif isinstance(c, CanonicalNode):
                    all_contrib_regions.update(c.contributing_regional_ids)
                    if "original_candidate_id" in c.diagnostic_metadata:
                        cid = c.diagnostic_metadata["original_candidate_id"]
                        self.regional_to_canonical_map[(frame, cid)] = canonical_id

            if not all_contrib_regions:
                all_contrib_regions = {primary.primary_region_id if isinstance(primary, CanonicalNode) else primary.region_id}

            # Position computation: weighted position average by confidence
            conf_weights = np.array([c.confidence for c in cluster_cands], dtype=np.float64)
            weight_sum = conf_weights.sum()
            if weight_sum > 0:
                conf_weights /= weight_sum
            else:
                conf_weights = np.ones(len(cluster_cands)) / len(cluster_cands)

            v_positions = np.array([c.global_voxel_pos if isinstance(c, CanonicalNode) else c.global_voxel_pos for c in cluster_cands], dtype=np.float64)
            p_positions = np.array([c.physical_pos for c in cluster_cands], dtype=np.float64)

            avg_v_pos = tuple(np.sum(v_positions * conf_weights[:, None], axis=0))
            avg_p_pos = tuple(np.sum(p_positions * conf_weights[:, None], axis=0))

            max_confidence = float(np.max([c.confidence for c in cluster_cands]))

            canonical_node = CanonicalNode(
                node_id=canonical_id,
                frame=frame,
                global_voxel_pos=(float(avg_v_pos[0]), float(avg_v_pos[1]), float(avg_v_pos[2])),
                physical_pos=(float(avg_p_pos[0]), float(avg_p_pos[1]), float(avg_p_pos[2])),
                confidence=max_confidence,
                contributing_regional_ids=sorted(list(all_contrib_regions)),
                num_regional_observations=len(all_contrib_regions),
                primary_region_id=primary.primary_region_id if isinstance(primary, CanonicalNode) else primary.region_id,
                diagnostic_metadata={
                    "reconciled_cluster_size": len(cluster_cands),
                    "original_candidate_ids": all_regional_candidate_ids,
                    "reconciliation_distance_um": self.reconciliation_distance_um,
                }
            )

            canonical_nodes.append(canonical_node)

        return canonical_nodes


class NodeSet:
    """
    Container for canonical nodes indexed by frame and node ID.
    """

    def __init__(self, nodes: Optional[List[CanonicalNode]] = None):
        self._nodes_by_frame: Dict[int, List[CanonicalNode]] = {}
        self._nodes_by_id: Dict[int, CanonicalNode] = {}

        if nodes:
            self.add_nodes(nodes)

    def add_node(self, node: CanonicalNode) -> None:
        """Add a single canonical node."""
        if node.frame not in self._nodes_by_frame:
            self._nodes_by_frame[node.frame] = []
        self._nodes_by_frame[node.frame].append(node)
        self._nodes_by_id[node.node_id] = node

    def add_nodes(self, nodes: List[CanonicalNode]) -> None:
        """Add multiple canonical nodes."""
        for n in nodes:
            self.add_node(n)

    def get_nodes_by_frame(self, frame: int) -> List[CanonicalNode]:
        """Retrieve all nodes active at specified frame."""
        return self._nodes_by_frame.get(frame, [])

    def get_node(self, node_id: int) -> Optional[CanonicalNode]:
        """Retrieve node by node ID."""
        return self._nodes_by_id.get(node_id)

    def all_nodes(self) -> List[CanonicalNode]:
        """Retrieve flat list of all nodes."""
        return list(self._nodes_by_id.values())

    def num_nodes(self) -> int:
        """Total number of nodes."""
        return len(self._nodes_by_id)

    def frames(self) -> List[int]:
        """List of active frame indices in sorted order."""
        return sorted(list(self._nodes_by_frame.keys()))

    def num_frames(self) -> int:
        """Number of frames."""
        return len(self._nodes_by_frame)

    def search_nearby_physical(
        self,
        physical_pos: Tuple[float, float, float],
        frame: int,
        max_dist_um: float,
        transformer: Optional[CoordinateTransformer] = None
    ) -> List[Tuple[CanonicalNode, float]]:
        """
        Searches for nodes in a given frame within physical distance threshold.

        :return: List of (CanonicalNode, physical_distance) tuples sorted by distance
        """
        trans = transformer or CoordinateTransformer()
        frame_nodes = self.get_nodes_by_frame(frame)
        results = []

        for node in frame_nodes:
            d = trans.euclidean_distance_physical(physical_pos, node.physical_pos)
            if d <= max_dist_um:
                results.append((node, d))

        results.sort(key=lambda item: item[1])
        return results
