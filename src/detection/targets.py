"""
Deterministic 3D Target Heatmap Generation with Physical Anisotropy Handling.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from src.config import SpatialResolution
from src.data.nodes import CellNode


@dataclass
class GaussianHeatmapConfig:
    """
    Configuration for 3D Gaussian cell-center target generation.

    The Gaussian width is defined in PHYSICAL units (micrometers, um) to ensure
    consistent spatial representation regardless of voxel resolution or anisotropy.
    """
    sigma_physical_um: float = 2.0  # Gaussian standard deviation in physical space (um)
    cutoff_sigmas: float = 3.0       # Truncation radius in units of sigma
    min_value_threshold: float = 0.01 # Minimum value cutoff for target rendering


class TargetGenerator:
    """
    Generates 3D continuous Gaussian cell-center target heatmaps from ground-truth node locations.
    """

    def __init__(
        self,
        config: GaussianHeatmapConfig = GaussianHeatmapConfig(),
        resolution: SpatialResolution = SpatialResolution()
    ):
        self.config = config
        self.resolution = resolution

        # Calculate anisotropic voxel-space standard deviations
        # sigma_voxel = sigma_physical / spacing
        self.sigma_z = self.config.sigma_physical_um / self.resolution.z_spacing
        self.sigma_y = self.config.sigma_physical_um / self.resolution.y_spacing
        self.sigma_x = self.config.sigma_physical_um / self.resolution.x_spacing

    def generate_heatmap_for_frame(
        self,
        nodes: List[CellNode],
        grid_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Generates 3D heatmap array of shape (Z, Y, X) for a set of cell nodes in a frame.

        :param nodes: List of CellNode objects in the target frame
        :param grid_shape: (Z, Y, X) output volume dimensions
        :return: 3D float32 numpy array of shape (Z, Y, X) with values in [0, 1]
        """
        z_dim, y_dim, x_dim = grid_shape
        heatmap = np.zeros((z_dim, y_dim, x_dim), dtype=np.float32)

        if not nodes:
            return heatmap

        # Compute bounding radii in voxel units
        radius_z = int(np.ceil(self.config.cutoff_sigmas * self.sigma_z))
        radius_y = int(np.ceil(self.config.cutoff_sigmas * self.sigma_y))
        radius_x = int(np.ceil(self.config.cutoff_sigmas * self.sigma_x))

        for node in nodes:
            cz, cy, cx = node.voxel_pos

            # Compute local ROI bounding box
            z_min = max(0, cz - radius_z)
            z_max = min(z_dim, cz + radius_z + 1)
            y_min = max(0, cy - radius_y)
            y_max = min(y_dim, cy + radius_y + 1)
            x_min = max(0, cx - radius_x)
            x_max = min(x_dim, cx + radius_x + 1)

            if z_min >= z_max or y_min >= y_max or x_min >= x_max:
                continue

            # Create local coordinate grids
            z_grid = np.arange(z_min, z_max, dtype=np.float32)[:, None, None]
            y_grid = np.arange(y_min, y_max, dtype=np.float32)[None, :, None]
            x_grid = np.arange(x_min, x_max, dtype=np.float32)[None, None, :]

            # Anisotropic 3D Gaussian formula
            dz2 = ((z_grid - cz) / self.sigma_z) ** 2
            dy2 = ((y_grid - cy) / self.sigma_y) ** 2
            dx2 = ((x_grid - cx) / self.sigma_x) ** 2

            gaussian_roi = np.exp(-0.5 * (dz2 + dy2 + dx2)).astype(np.float32)

            # Apply elementwise maximum to merge overlapping cell targets cleanly
            heatmap[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                heatmap[z_min:z_max, y_min:y_max, x_min:x_max], gaussian_roi
            )

        return heatmap
