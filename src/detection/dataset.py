"""
PyTorch Dataset abstraction for 3D microscopy frames and target heatmaps.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.zarr_loader import ZarrLoader
from src.data.geff_loader import GeffLoader
from src.detection.targets import TargetGenerator, GaussianHeatmapConfig


class Frame3DDataset(Dataset):
    """
    PyTorch Dataset for 3D microscopy frame-by-frame cell-center detection training and evaluation.

    Features:
    - Lazy loading: reads individual 3D frames or spatial subvolume crops from ZarrLoader.
    - Intensity normalization: scales uint16 microscopy voxels to [0, 1].
    - Target generation: generates anisotropic 3D Gaussian cell-center heatmaps on-the-fly.
    - Configurable spatial subvolume cropping for fast CPU development/training.
    """

    def __init__(
        self,
        zarr_loader: ZarrLoader,
        geff_loader: GeffLoader,
        target_generator: Optional[TargetGenerator] = None,
        frame_indices: Optional[List[int]] = None,
        crop_bounds: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None,
        norm_min: float = 0.0,
        norm_max: float = 2000.0  # Clipping upper intensity threshold for normalization
    ):
        self.zarr_loader = zarr_loader
        self.geff_loader = geff_loader
        self.target_generator = target_generator or TargetGenerator()
        self.crop_bounds = crop_bounds

        if frame_indices is None:
            self.frame_indices = list(range(self.zarr_loader.shape[0]))
        else:
            self.frame_indices = frame_indices

        self.norm_min = norm_min
        self.norm_max = norm_max

    def __len__(self) -> int:
        return len(self.frame_indices)

    def _normalize_image(self, img_3d: np.ndarray) -> np.ndarray:
        """Normalizes 3D uint16 microscopy image volume to float32 in [0, 1]."""
        clipped = np.clip(img_3d.astype(np.float32), self.norm_min, self.norm_max)
        normalized = (clipped - self.norm_min) / (self.norm_max - self.norm_min + 1e-6)
        return normalized.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        frame_t = self.frame_indices[idx]

        # 1. Retrieve 3D volume or spatial crop lazily (Z, Y, X)
        if self.crop_bounds is not None:
            z_range, y_range, x_range = self.crop_bounds
            img_3d = self.zarr_loader.get_crop(frame_t, z_range, y_range, x_range)
            z_offset, y_offset, x_offset = z_range[0], y_range[0], x_range[0]
        else:
            img_3d = self.zarr_loader.get_frame(frame_t)
            z_offset, y_offset, x_offset = 0, 0, 0

        norm_img_3d = self._normalize_image(img_3d)

        # 2. Retrieve ground truth nodes for frame_t
        raw_nodes = self.geff_loader.get_nodes_by_frame(frame_t)

        # Shift node coordinates if using spatial crop
        shifted_nodes = []
        for n in raw_nodes:
            z_v, y_v, x_v = n.voxel_pos
            if self.crop_bounds is not None:
                (z_s, z_e), (y_s, y_e), (x_s, x_e) = self.crop_bounds
                if not (z_s <= z_v < z_e and y_s <= y_v < y_e and x_s <= x_v < x_e):
                    continue
                z_v -= z_s
                y_v -= y_s
                x_v -= x_s

            shifted_nodes.append(
                n.create(
                    node_id=n.node_id,
                    frame=n.frame,
                    voxel_pos=(z_v, y_v, x_v),
                    transformer=self.geff_loader.transformer,
                    attributes=n.attributes
                )
            )

        # 3. Generate 3D target heatmap (Z, Y, X)
        target_heatmap_3d = self.target_generator.generate_heatmap_for_frame(
            nodes=shifted_nodes,
            grid_shape=norm_img_3d.shape
        )

        # 4. Convert to PyTorch Tensors [Channels=1, Z, Y, X]
        image_tensor = torch.from_numpy(norm_img_3d).unsqueeze(0)  # Shape: (1, Z, Y, X)
        target_tensor = torch.from_numpy(target_heatmap_3d).unsqueeze(0)  # Shape: (1, Z, Y, X)

        return image_tensor, target_tensor, frame_t
