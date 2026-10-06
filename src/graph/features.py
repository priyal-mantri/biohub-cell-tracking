"""
Deterministic Node and Edge Feature Extraction for Graph Models.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.graph.nodes import CanonicalNode
from src.graph.candidate_graph import TemporalCandidateGraph, CandidateEdge
from src.data.coordinates import CoordinateTransformer


@dataclass
class NodeFeatures:
    """
    Container for extracted node feature matrices.
    """
    node_ids: np.ndarray  # (N,) int64
    features: np.ndarray  # (N, D_node) float32
    feature_names: List[str]  # List of feature names
    frames: np.ndarray  # (N,) int64


@dataclass
class EdgeFeatures:
    """
    Container for extracted edge feature matrices.
    """
    edge_index: np.ndarray  # (2, E) int64 (source_idx, dest_idx)
    edge_ids: List[Tuple[int, int]]  # List of (source_node_id, dest_node_id)
    features: np.ndarray  # (E, D_edge) float32
    feature_names: List[str]  # List of edge feature names


class GraphFeatureExtractor:
    """
    Deterministic feature extractor for node and edge representations in candidate graphs.
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int, int] = (100, 64, 256, 256),
        transformer: Optional[CoordinateTransformer] = None,
        density_radius_um: float = 15.0
    ):
        self.volume_shape = volume_shape  # (T, Z, Y, X)
        self.transformer = transformer or CoordinateTransformer()
        self.density_radius_um = density_radius_um

        # Calculate max physical bounds
        self.max_t = max(1, volume_shape[0] - 1)
        self.max_z_vox = volume_shape[1]
        self.max_y_vox = volume_shape[2]
        self.max_x_vox = volume_shape[3]

        self.max_z_phys = self.max_z_vox * self.transformer.resolution.z_spacing
        self.max_y_phys = self.max_y_vox * self.transformer.resolution.y_spacing
        self.max_x_phys = self.max_x_vox * self.transformer.resolution.x_spacing

    def extract_node_features(self, graph: TemporalCandidateGraph) -> NodeFeatures:
        """
        Extracts normalized node feature matrix for all nodes in the candidate graph.
        """
        all_nodes = graph.node_set.all_nodes()
        n_nodes = len(all_nodes)
        if n_nodes == 0:
            return NodeFeatures(
                node_ids=np.empty(0, dtype=np.int64),
                features=np.empty((0, 8), dtype=np.float32),
                feature_names=[
                    "norm_z_phys", "norm_y_phys", "norm_x_phys",
                    "norm_z_vox", "norm_y_vox", "norm_x_vox",
                    "confidence", "norm_frame", "local_density_count"
                ],
                frames=np.empty(0, dtype=np.int64)
            )

        # Precompute per-node local density counts within same frame
        node_density_map: Dict[int, float] = {}
        for node in all_nodes:
            nearby = graph.node_set.search_nearby_physical(
                node.physical_pos,
                node.frame,
                self.density_radius_um,
                transformer=self.transformer
            )
            # count excludes self
            node_density_map[node.node_id] = float(max(0, len(nearby) - 1))

        node_ids = []
        frames = []
        feature_rows = []

        feature_names = [
            "norm_z_phys", "norm_y_phys", "norm_x_phys",
            "norm_z_vox", "norm_y_vox", "norm_x_vox",
            "confidence", "norm_frame", "local_density_count"
        ]

        for node in all_nodes:
            node_ids.append(node.node_id)
            frames.append(node.frame)

            zp_norm = float(node.physical_pos[0] / self.max_z_phys) if self.max_z_phys > 0 else 0.0
            yp_norm = float(node.physical_pos[1] / self.max_y_phys) if self.max_y_phys > 0 else 0.0
            xp_norm = float(node.physical_pos[2] / self.max_x_phys) if self.max_x_phys > 0 else 0.0

            zv_norm = float(node.global_voxel_pos[0] / self.max_z_vox) if self.max_z_vox > 0 else 0.0
            yv_norm = float(node.global_voxel_pos[1] / self.max_y_vox) if self.max_y_vox > 0 else 0.0
            xv_norm = float(node.global_voxel_pos[2] / self.max_x_vox) if self.max_x_vox > 0 else 0.0

            conf = float(node.confidence)
            t_norm = float(node.frame / self.max_t) if self.max_t > 0 else 0.0
            density = node_density_map[node.node_id]

            row = [
                zp_norm, yp_norm, xp_norm,
                zv_norm, yv_norm, xv_norm,
                conf, t_norm, density
            ]
            feature_rows.append(row)

        return NodeFeatures(
            node_ids=np.array(node_ids, dtype=np.int64),
            features=np.array(feature_rows, dtype=np.float32),
            feature_names=feature_names,
            frames=np.array(frames, dtype=np.int64)
        )

    def extract_edge_features(self, graph: TemporalCandidateGraph) -> EdgeFeatures:
        """
        Extracts edge features and edge_index for all candidate edges in graph.
        """
        all_nodes = graph.node_set.all_nodes()
        node_id_to_idx = {n.node_id: idx for idx, n in enumerate(all_nodes)}

        # Precompute node density for density difference feature
        node_density_map: Dict[int, float] = {}
        for node in all_nodes:
            nearby = graph.node_set.search_nearby_physical(
                node.physical_pos,
                node.frame,
                self.density_radius_um,
                transformer=self.transformer
            )
            node_density_map[node.node_id] = float(max(0, len(nearby) - 1))

        feature_names = [
            "dz_phys", "dy_phys", "dx_phys",
            "abs_dz_phys", "abs_dy_phys", "abs_dx_phys",
            "euclidean_dist_phys",
            "dz_vox", "dy_vox", "dx_vox",
            "source_conf", "dest_conf", "pair_conf",
            "density_diff", "frame_gap", "rank_among_dests"
        ]

        edges = graph.edges
        n_edges = len(edges)
        if n_edges == 0:
            return EdgeFeatures(
                edge_index=np.empty((2, 0), dtype=np.int64),
                edge_ids=[],
                features=np.empty((0, len(feature_names)), dtype=np.float32),
                feature_names=feature_names
            )

        edge_index_src = []
        edge_index_dst = []
        edge_ids = []
        feature_rows = []

        for edge in edges:
            src_idx = node_id_to_idx[edge.source_node_id]
            dst_idx = node_id_to_idx[edge.destination_node_id]

            edge_index_src.append(src_idx)
            edge_index_dst.append(dst_idx)
            edge_ids.append((edge.source_node_id, edge.destination_node_id))

            dz_p, dy_p, dx_p = edge.physical_displacement_vector
            abs_dz_p, abs_dy_p, abs_dx_p = abs(dz_p), abs(dy_p), abs(dx_p)
            dist_p = edge.euclidean_distance_phys

            dz_v, dy_v, dx_v = edge.voxel_displacement_vector

            src_c = edge.source_confidence
            dst_c = edge.destination_confidence
            pair_c = edge.pair_confidence

            src_dens = node_density_map.get(edge.source_node_id, 0.0)
            dst_dens = node_density_map.get(edge.destination_node_id, 0.0)
            dens_diff = dst_dens - src_dens

            gap = float(edge.frame_difference)
            rank = float(edge.rank_among_destinations)

            row = [
                dz_p, dy_p, dx_p,
                abs_dz_p, abs_dy_p, abs_dx_p,
                dist_p,
                dz_v, dy_v, dx_v,
                src_c, dst_c, pair_c,
                dens_diff, gap, rank
            ]
            feature_rows.append(row)

        edge_index = np.array([edge_index_src, edge_index_dst], dtype=np.int64)

        return EdgeFeatures(
            edge_index=edge_index,
            edge_ids=edge_ids,
            features=np.array(feature_rows, dtype=np.float32),
            feature_names=feature_names
        )

    def extract(self, graph: TemporalCandidateGraph) -> Dict[str, Any]:
        """Extract both node and edge feature containers."""
        node_feats = self.extract_node_features(graph)
        edge_feats = self.extract_edge_features(graph)
        return {
            "node_features": node_feats,
            "edge_features": edge_feats
        }
