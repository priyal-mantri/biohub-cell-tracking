"""
Abstract Base Class and data structures for 3D Cell-Center Detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

from src.data.nodes import CellNode
from src.data.coordinates import CoordinateTransformer


@dataclass
class DetectionCandidate:
    """
    Represents a detected cell-center candidate before temporal graph association.
    """
    candidate_id: int
    frame: int  # Time index t (0-indexed)
    voxel_pos: Tuple[int, int, int]  # (z, y, x) in voxel indices
    physical_pos: Tuple[float, float, float]  # (z_phys, y_phys, x_phys) in micrometers
    confidence: float  # Model output response score in [0, 1]
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_cell_node(self) -> CellNode:
        """Converts detection candidate to internal CellNode representation."""
        attrs = dict(self.attributes)
        attrs["confidence"] = self.confidence
        return CellNode(
            node_id=self.candidate_id,
            frame=self.frame,
            voxel_pos=self.voxel_pos,
            physical_pos=self.physical_pos,
            attributes=attrs
        )


class BaseDetector(ABC):
    """
    Abstract interface for 3D cell-center detectors.
    Decouples detector implementations (e.g. U-Net3D, Keypoint Net, classical peak finders)
    from downstream temporal graph and tracking modules.
    """

    @abstractmethod
    def predict_heatmap(self, volume_3d: np.ndarray) -> np.ndarray:
        """
        Predicts 3D continuous cell-center response heatmap from 3D microscopy image.

        :param volume_3d: 3D input numpy array of shape (Z, Y, X), normalized in [0, 1]
        :return: 3D heatmap array of shape (Z, Y, X) with values in [0, 1]
        """
        pass

    @abstractmethod
    def detect_candidates(
        self,
        volume_3d: np.ndarray,
        frame_idx: int,
        transformer: Optional[CoordinateTransformer] = None
    ) -> List[DetectionCandidate]:
        """
        Full detection pipeline: predicts heatmap and extracts 3D peak candidates.

        :param volume_3d: 3D input volume of shape (Z, Y, X)
        :param frame_idx: Frame/time index t
        :param transformer: CoordinateTransformer instance for voxel-to-physical conversion
        :return: List of DetectionCandidate instances
        """
        pass


class HeuristicIntensityDetector(BaseDetector):
    """
    Intensity-based 3D peak detector for volumes without ground truth annotations.
    """

    def __init__(self, min_confidence: float = 0.1, min_physical_distance_um: float = 3.0):
        from src.detection.peaks import PeakExtractor, PeakConfig
        self.peak_config = PeakConfig(min_confidence=min_confidence, min_physical_distance_um=min_physical_distance_um)
        self.peak_extractor = PeakExtractor(config=self.peak_config)

    def predict_heatmap(self, volume_3d: np.ndarray) -> np.ndarray:
        v_min, v_max = float(np.min(volume_3d)), float(np.max(volume_3d))
        if v_max > v_min:
            return (volume_3d.astype(np.float32) - v_min) / (v_max - v_min + 1e-6)
        return np.zeros_like(volume_3d, dtype=np.float32)

    def detect_candidates(
        self,
        volume_3d: np.ndarray,
        frame_idx: int,
        transformer: Optional[CoordinateTransformer] = None
    ) -> List[DetectionCandidate]:
        heatmap = self.predict_heatmap(volume_3d)
        return self.peak_extractor.extract_peaks_from_heatmap(heatmap, frame_idx)

