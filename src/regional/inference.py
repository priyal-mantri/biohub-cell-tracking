"""
Regional Inference Engine wrapping BaseDetector with Guard Zone Processing and Overlap Reconciliation.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch

from src.data.zarr_loader import ZarrLoader
from src.data.coordinates import CoordinateTransformer
from src.detection.base import BaseDetector, DetectionCandidate
from src.detection.peaks import PeakExtractor, PeakConfig
from src.regional.geometry import Region3D, RegionalCandidate
from src.regional.partitioner import RegionPartitioner3D, PartitionConfig
from src.regional.reconciliation import CandidateReconciler, ReconciliationConfig


class RegionalInferenceEngine:
    """
    Decoupled Regional Inference Engine.

    Separation of Responsibilities:
    - The detector detects cells within a given 3D input tensor.
    - The regional inference engine determines WHERE and with WHAT contextual volume (CORE + GUARD)
      the detector operates, transforms local coordinates to global physical coordinates, enforces
      authoritative CORE emission rules, and reconciles overlap duplicates.
    """

    def __init__(
        self,
        detector: BaseDetector,
        partition_config: PartitionConfig = PartitionConfig(),
        peak_config: PeakConfig = PeakConfig(),
        reconciliation_config: ReconciliationConfig = ReconciliationConfig(),
        transformer: Optional[CoordinateTransformer] = None
    ):
        self.detector = detector
        self.partition_config = partition_config
        self.peak_config = peak_config
        self.reconciliation_config = reconciliation_config
        self.transformer = transformer or CoordinateTransformer()

        self.peak_extractor = PeakExtractor(config=peak_config, resolution=self.transformer.resolution)
        self.reconciler = CandidateReconciler(config=reconciliation_config, transformer=self.transformer)

    def _normalize_crop(
        self,
        crop_3d: np.ndarray,
        norm_min: float = 0.0,
        norm_max: float = 2000.0
    ) -> np.ndarray:
        """Normalizes 3D uint16 microscopy image slice to float32 in [0, 1]."""
        clipped = np.clip(crop_3d.astype(np.float32), norm_min, norm_max)
        normalized = (clipped - norm_min) / (norm_max - norm_min + 1e-6)
        return normalized.astype(np.float32)

    def detect_frame_regional(
        self,
        zarr_loader: ZarrLoader,
        frame_idx: int,
        filter_core_only: bool = True
    ) -> Tuple[List[RegionalCandidate], Dict[str, Any]]:
        """
        Runs complete regional inference on a single 3D microscopy frame.

        :param zarr_loader: ZarrLoader instance
        :param frame_idx: Frame index t
        :param filter_core_only: If True, enforces CRITICAL RULE: only peaks located inside
               region's authoritative CORE are emitted as candidates by that region.
        :return: (reconciled_candidates, execution_metadata)
        """
        volume_shape = zarr_loader.shape[1:]  # (Z, Y, X)
        partitioner = RegionPartitioner3D(volume_shape=volume_shape, config=self.partition_config)
        regions = partitioner.generate_regions()

        raw_regional_candidates: List[RegionalCandidate] = []
        raw_core_candidates: List[RegionalCandidate] = []
        raw_guard_candidates: List[RegionalCandidate] = []

        total_peaks_extracted = 0
        candidate_counter = 1

        for region in regions:
            # 1. Lazily slice region volume (Guard volume slice)
            z_range, y_range, x_range = region.guard_bbox.slice_tuples
            raw_crop = zarr_loader.get_crop(frame_idx, z_range, y_range, x_range)
            norm_crop = self._normalize_crop(raw_crop)

            # 2. Run detector on regional volume
            heatmap_3d = self.detector.predict_heatmap(norm_crop)

            # 3. Extract local peak candidates in region
            local_peaks = self.peak_extractor.extract_peaks_from_heatmap(heatmap_3d, frame_idx)
            total_peaks_extracted += len(local_peaks)

            # 4. Transform coordinates & check CORE vs GUARD membership
            for pk in local_peaks:
                local_zyx = pk.voxel_pos
                global_zyx = region.local_to_global_voxel(local_zyx)
                global_phys = self.transformer.voxel_to_physical(global_zyx)

                is_in_core = region.is_global_voxel_in_core(global_zyx)

                provenance = {
                    "source_region_id": region.region_id,
                    "local_voxel_pos": local_zyx,
                    "global_voxel_pos": global_zyx,
                    "physical_pos": global_phys,
                    "is_in_core": is_in_core,
                    "dist_to_core_boundary_um": region.min_distance_to_core_boundary_phys(global_phys, self.transformer)
                }

                reg_cand = RegionalCandidate(
                    candidate_id=candidate_counter,
                    region_id=region.region_id,
                    frame=frame_idx,
                    local_voxel_pos=local_zyx,
                    global_voxel_pos=global_zyx,
                    physical_pos=global_phys,
                    confidence=pk.confidence,
                    is_in_core=is_in_core,
                    provenance=provenance
                )
                candidate_counter += 1

                if is_in_core:
                    raw_core_candidates.append(reg_cand)
                else:
                    raw_guard_candidates.append(reg_cand)

                if filter_core_only:
                    if is_in_core:
                        raw_regional_candidates.append(reg_cand)
                else:
                    raw_regional_candidates.append(reg_cand)

        # 5. Spatial Duplicate Reconciliation across guard boundaries
        reconciled_candidates = self.reconciler.reconcile_candidates(raw_regional_candidates)

        metadata = {
            "frame_idx": frame_idx,
            "num_regions": len(regions),
            "total_raw_peaks_extracted": total_peaks_extracted,
            "num_raw_core_candidates": len(raw_core_candidates),
            "num_raw_guard_candidates": len(raw_guard_candidates),
            "num_authoritative_before_reconciliation": len(raw_regional_candidates),
            "num_authoritative_after_reconciliation": len(reconciled_candidates),
            "num_multi_region_duplicates_reconciled": sum(
                1 for c in reconciled_candidates if c.provenance.get("is_multi_region_detection", False)
            )
        }

        return reconciled_candidates, metadata
