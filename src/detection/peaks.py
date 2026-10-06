"""
Deterministic 3D Local Maxima Extraction and Anisotropic Non-Maximum Suppression (NMS).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.ndimage import maximum_filter

from src.config import SpatialResolution
from src.data.coordinates import CoordinateTransformer
from src.detection.base import DetectionCandidate


@dataclass
class PeakConfig:
    """
    Configuration for 3D peak extraction and non-maximum suppression (NMS).
    """
    min_confidence: float = 0.2           # Heatmap response threshold for candidate peaks
    min_physical_distance_um: float = 3.0  # Minimum physical NMS radius in micrometers (um)
    max_candidates_per_frame: int = 200    # Safety cap on maximum candidates per frame


class PeakExtractor:
    """
    Extracts candidate cell centers from continuous 3D predicted heatmaps using anisotropic local maxima filtering and NMS.
    """

    def __init__(
        self,
        config: PeakConfig = PeakConfig(),
        resolution: SpatialResolution = SpatialResolution()
    ):
        self.config = config
        self.resolution = resolution
        self.transformer = CoordinateTransformer(resolution)

    def extract_peaks_from_heatmap(
        self,
        heatmap_3d: np.ndarray,
        frame_idx: int,
        id_start_offset: int = 1000
    ) -> List[DetectionCandidate]:
        """
        Extracts peak candidate nodes from a 3D heatmap array (Z, Y, X).

        :param heatmap_3d: 3D float numpy array of shape (Z, Y, X)
        :param frame_idx: Frame index t
        :param id_start_offset: Integer offset for generating candidate IDs
        :return: List of DetectionCandidate instances
        """
        if heatmap_3d.ndim != 3:
            raise ValueError(f"Expected 3D heatmap array, got shape {heatmap_3d.shape}")

        # Compute anisotropic maximum filter footprint size in voxel indices
        # Size along axis = 2 * (dist / spacing) + 1
        d_um = self.config.min_physical_distance_um
        size_z = max(1, int(round(d_um / self.resolution.z_spacing)) | 1)
        size_y = max(1, int(round(d_um / self.resolution.y_spacing)) | 1)
        size_x = max(1, int(round(d_um / self.resolution.x_spacing)) | 1)

        footprint = (size_z, size_y, size_x)

        # 1. Apply 3D local maximum filter
        local_max = maximum_filter(heatmap_3d, size=footprint, mode="constant", cval=0.0)

        # 2. Thresholding: location must equal local maximum AND exceed min_confidence
        mask = (heatmap_3d == local_max) & (heatmap_3d >= self.config.min_confidence)
        z_indices, y_indices, x_indices = np.where(mask)

        if len(z_indices) == 0:
            return []

        # 3. Gather initial candidates with confidence scores
        raw_candidates = []
        for z, y, x in zip(z_indices, y_indices, x_indices):
            vz = int(z)
            vy = int(y)
            vx = int(x)
            conf = float(heatmap_3d[vz, vy, vx])
            phys_pos = self.transformer.voxel_to_physical((vz, vy, vx))
            raw_candidates.append({
                "voxel_pos": (vz, vy, vx),
                "physical_pos": phys_pos,
                "confidence": conf
            })

        # 4. Sort candidates by confidence descending for greedy NMS
        raw_candidates.sort(key=lambda c: c["confidence"], reverse=True)

        # 5. Greedy Anisotropic Non-Maximum Suppression (NMS) in physical space (um)
        kept_candidates: List[DetectionCandidate] = []
        suppressed = np.zeros(len(raw_candidates), dtype=bool)

        for i, cand in enumerate(raw_candidates):
            if suppressed[i]:
                continue

            c_id = id_start_offset + len(kept_candidates) + 1
            det_cand = DetectionCandidate(
                candidate_id=c_id,
                frame=frame_idx,
                voxel_pos=cand["voxel_pos"],
                physical_pos=cand["physical_pos"],
                confidence=cand["confidence"]
            )
            kept_candidates.append(det_cand)

            if len(kept_candidates) >= self.config.max_candidates_per_frame:
                break

            # Suppress nearby lower-confidence peaks within physical distance
            p_i = cand["physical_pos"]
            for j in range(i + 1, len(raw_candidates)):
                if suppressed[j]:
                    continue
                p_j = raw_candidates[j]["physical_pos"]
                dist = self.transformer.euclidean_distance_physical(p_i, p_j)
                if dist < self.config.min_physical_distance_um:
                    suppressed[j] = True

        return kept_candidates
