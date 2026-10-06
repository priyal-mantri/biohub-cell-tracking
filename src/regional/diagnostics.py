"""
Boundary Proximity Diagnostics & Full-Frame vs. Regional Inference Comparator.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.data.nodes import CellNode
from src.data.coordinates import CoordinateTransformer
from src.detection.base import DetectionCandidate
from src.detection.evaluation import DetectionEvaluator, DetectionMetrics
from src.regional.geometry import Region3D, RegionalCandidate
from src.regional.partitioner import RegionPartitioner3D, PartitionConfig


@dataclass
class BoundaryDiagnostics:
    """
    Diagnostic results comparing Full-Frame vs. Regional Inference and evaluating boundary performance.
    """
    frame_idx: int
    num_regions: int
    fullframe_candidate_count: int
    regional_authoritative_raw_count: int
    regional_reconciled_count: int
    multi_region_duplicate_count: int
    fullframe_metrics: DetectionMetrics
    regional_metrics: DetectionMetrics
    near_boundary_gt_count: int
    near_boundary_recall: float
    deep_core_gt_count: int
    deep_core_recall: float

    def summary_string(self) -> str:
        return (
            f"=== Boundary Diagnostics (Frame {self.frame_idx}) ===\n"
            f"Regions: {self.num_regions} | Full-Frame Candidates: {self.fullframe_candidate_count}\n"
            f"Regional Authoritative (Raw): {self.regional_authoritative_raw_count} | Reconciled: {self.regional_reconciled_count}\n"
            f"Multi-Region Duplicates Reconciled: {self.multi_region_duplicate_count}\n"
            f"--------------------------------------------------\n"
            f"Full-Frame Metrics -> Prec: {self.fullframe_metrics.precision:.4f} | Rec: {self.fullframe_metrics.recall:.4f} | F1: {self.fullframe_metrics.f1_score:.4f} | Loc Error: {self.fullframe_metrics.mean_localization_error_um:.3f} um\n"
            f"Regional   Metrics -> Prec: {self.regional_metrics.precision:.4f} | Rec: {self.regional_metrics.recall:.4f} | F1: {self.regional_metrics.f1_score:.4f} | Loc Error: {self.regional_metrics.mean_localization_error_um:.3f} um\n"
            f"--------------------------------------------------\n"
            f"Near-Boundary GT Cells (<= 5 um to core border): {self.near_boundary_gt_count} | Regional Recall: {self.near_boundary_recall:.4f}\n"
            f"Deep-Core     GT Cells (> 5 um to core border):  {self.deep_core_gt_count} | Regional Recall: {self.deep_core_recall:.4f}"
        )


class RegionalBoundaryAnalyzer:
    """
    Analyzes spatial boundary behavior and compares full-frame vs regional detection outputs.
    """

    def __init__(
        self,
        evaluator: Optional[DetectionEvaluator] = None,
        transformer: Optional[CoordinateTransformer] = None,
        boundary_threshold_um: float = 5.0
    ):
        self.evaluator = evaluator or DetectionEvaluator(max_matching_distance_um=5.0)
        self.transformer = transformer or CoordinateTransformer()
        self.boundary_threshold_um = boundary_threshold_um

    def analyze_frame(
        self,
        frame_idx: int,
        gt_nodes: List[CellNode],
        fullframe_candidates: List[DetectionCandidate],
        regional_candidates: List[RegionalCandidate],
        regional_metadata: Dict[str, Any],
        partitioner: RegionPartitioner3D
    ) -> BoundaryDiagnostics:
        """
        Analyzes full-frame vs regional detection performance and assesses boundary behavior.
        """
        # 1. Full-frame evaluation
        ff_metrics = self.evaluator.evaluate_frame(gt_nodes, fullframe_candidates)

        # 2. Regional evaluation (convert RegionalCandidate to DetectionCandidate for evaluator)
        reg_cands_as_det = [
            DetectionCandidate(
                candidate_id=rc.candidate_id,
                frame=rc.frame,
                voxel_pos=rc.global_voxel_pos,
                physical_pos=rc.physical_pos,
                confidence=rc.confidence
            )
            for rc in regional_candidates
        ]
        reg_metrics = self.evaluator.evaluate_frame(gt_nodes, reg_cands_as_det)

        # 3. Categorize GT nodes into Near-Boundary vs Deep-Core
        regions = partitioner.generate_regions()
        near_boundary_gt: List[CellNode] = []
        deep_core_gt: List[CellNode] = []

        for gt in gt_nodes:
            # Find the region core that owns this GT node
            owning_region = None
            for r in regions:
                if r.is_global_voxel_in_core(gt.voxel_pos):
                    owning_region = r
                    break

            if owning_region is not None:
                dist_to_border = owning_region.min_distance_to_core_boundary_phys(gt.physical_pos, self.transformer)
                if dist_to_border <= self.boundary_threshold_um:
                    near_boundary_gt.append(gt)
                else:
                    deep_core_gt.append(gt)
            else:
                near_boundary_gt.append(gt)

        # Evaluate regional recall separately on Near-Boundary vs Deep-Core GT nodes
        nb_metrics = self.evaluator.evaluate_frame(near_boundary_gt, reg_cands_as_det)
        dc_metrics = self.evaluator.evaluate_frame(deep_core_gt, reg_cands_as_det)

        return BoundaryDiagnostics(
            frame_idx=frame_idx,
            num_regions=regional_metadata.get("num_regions", len(regions)),
            fullframe_candidate_count=len(fullframe_candidates),
            regional_authoritative_raw_count=regional_metadata.get("num_authoritative_before_reconciliation", 0),
            regional_reconciled_count=len(regional_candidates),
            multi_region_duplicate_count=regional_metadata.get("num_multi_region_duplicates_reconciled", 0),
            fullframe_metrics=ff_metrics,
            regional_metrics=reg_metrics,
            near_boundary_gt_count=len(near_boundary_gt),
            near_boundary_recall=nb_metrics.recall,
            deep_core_gt_count=len(deep_core_gt),
            deep_core_recall=dc_metrics.recall
        )
