"""
Spatial Overlap Duplicate Candidate Reconciliation across Region Guard Boundaries.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.data.coordinates import CoordinateTransformer
from src.regional.geometry import RegionalCandidate


@dataclass
class ReconciliationConfig:
    """
    Configuration for spatial duplicate candidate reconciliation.

    - reconciliation_distance_um: Physical distance threshold (micrometers) within which
      predictions from neighboring guard/core overlaps are considered duplicate observations
      of the same biological cell.
    - merge_strategy: Strategy for selecting/merging primary detection ('highest_confidence' or 'core_priority').
    """
    reconciliation_distance_um: float = 4.0  # Matching distance threshold in micrometers (um)
    merge_strategy: str = "highest_confidence"


class CandidateReconciler:
    """
    Reconciles duplicate cell-center detections produced across overlapping region guard zones.

    Guarantees:
    - Merges duplicate observations of the same cell in 3D physical space.
    - Preserves complete provenance (contributing region IDs, confidence scores, multi-region flag).
    """

    def __init__(
        self,
        config: ReconciliationConfig = ReconciliationConfig(),
        transformer: Optional[CoordinateTransformer] = None
    ):
        self.config = config
        self.transformer = transformer or CoordinateTransformer()

    def reconcile_candidates(
        self,
        candidates: List[RegionalCandidate]
    ) -> List[RegionalCandidate]:
        """
        Groups and merges duplicate candidates within reconciliation_distance_um physical threshold.

        :param candidates: List of RegionalCandidate objects emitted across regions
        :return: Reconciled list of globally consistent RegionalCandidate objects with full provenance
        """
        if not candidates:
            return []

        n_cand = len(candidates)
        if n_cand == 1:
            c = candidates[0]
            c.provenance["contributing_region_ids"] = [c.region_id]
            c.provenance["contributing_confidences"] = [c.confidence]
            c.provenance["is_multi_region_detection"] = False
            c.provenance["num_contributing_regions"] = 1
            return [c]

        # Build physical distance matrix (n_cand, n_cand) in micrometers
        dist_matrix = np.zeros((n_cand, n_cand), dtype=np.float32)
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                dist = self.transformer.euclidean_distance_physical(
                    candidates[i].physical_pos,
                    candidates[j].physical_pos
                )
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # Connected component clustering for duplicate candidate groups
        visited = np.zeros(n_cand, dtype=bool)
        reconciled_results: List[RegionalCandidate] = []
        next_candidate_id = 1

        for i in range(n_cand):
            if visited[i]:
                continue

            # Find all candidates connected to candidate i within threshold distance
            cluster_indices = [i]
            visited[i] = True

            queue = [i]
            while queue:
                curr = queue.pop(0)
                for neighbor in range(n_cand):
                    if not visited[neighbor] and dist_matrix[curr, neighbor] <= self.config.reconciliation_distance_um:
                        visited[neighbor] = True
                        cluster_indices.append(neighbor)
                        queue.append(neighbor)

            cluster_candidates = [candidates[idx] for idx in cluster_indices]

            # Select primary candidate based on strategy
            if self.config.merge_strategy == "core_priority":
                # Prefer candidate that is in core; if multiple or none, pick highest confidence
                core_cands = [c for c in cluster_candidates if c.is_in_core]
                if core_cands:
                    primary = max(core_cands, key=lambda c: c.confidence)
                else:
                    primary = max(cluster_candidates, key=lambda c: c.confidence)
            else:  # highest_confidence
                primary = max(cluster_candidates, key=lambda c: c.confidence)

            # Preserve complete regional provenance
            contributing_regions = list(set(c.region_id for c in cluster_candidates))
            contributing_confidences = [c.confidence for c in cluster_candidates]
            is_multi = len(contributing_regions) > 1

            provenance = dict(primary.provenance)
            provenance["contributing_region_ids"] = sorted(contributing_regions)
            provenance["contributing_confidences"] = contributing_confidences
            provenance["is_multi_region_detection"] = is_multi
            provenance["num_contributing_regions"] = len(contributing_regions)
            provenance["primary_region_id"] = primary.region_id

            reconciled_cand = RegionalCandidate(
                candidate_id=next_candidate_id,
                region_id=primary.region_id,
                frame=primary.frame,
                local_voxel_pos=primary.local_voxel_pos,
                global_voxel_pos=primary.global_voxel_pos,
                physical_pos=primary.physical_pos,
                confidence=primary.confidence,
                is_in_core=primary.is_in_core,
                provenance=provenance
            )
            reconciled_results.append(reconciled_cand)
            next_candidate_id += 1

        return reconciled_results
