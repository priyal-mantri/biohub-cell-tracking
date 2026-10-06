"""
Configurable 3D Volume Region Partitioner with Authoritative Core Partitioning & Contextual Guard Expansion.
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Set
import numpy as np

from src.regional.geometry import BoundingBox3D, Region3D


@dataclass
class PartitionConfig:
    """
    Configuration for 3D Volume Partitioning.

    - core_shape: (dz, dy, dx) voxel dimensions for non-overlapping authoritative cores.
    - guard_size: (gz, gy, gx) voxel padding expansion for contextual guard volumes.
    """
    core_shape: Tuple[int, int, int] = (32, 128, 128)
    guard_size: Tuple[int, int, int] = (8, 32, 32)


class RegionPartitioner3D:
    """
    Partitions arbitrary 3D volumetric images (Z, Y, X) into regular grid regions.

    Guarantees:
    1. Complete Core Coverage: Every voxel in the volume belongs to exactly one authoritative core.
    2. No Overlapping Cores: Region cores are strictly non-overlapping.
    3. Contextual Guards: Regions extend beyond cores by guard_size, clipped at volume boundaries.
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        config: PartitionConfig = PartitionConfig()
    ):
        self.volume_shape = volume_shape
        self.config = config

    def generate_regions(self) -> List[Region3D]:
        """
        Generates 3D Region3D instances covering the full volume.
        """
        z_dim, y_dim, x_dim = self.volume_shape
        cz, cy, cx = self.config.core_shape
        gz, gy, gx = self.config.guard_size

        regions: List[Region3D] = []
        region_id = 1

        z_starts = list(range(0, z_dim, cz))
        y_starts = list(range(0, y_dim, cy))
        x_starts = list(range(0, x_dim, cx))

        for zs in z_starts:
            ze = min(z_dim, zs + cz)
            for ys in y_starts:
                ye = min(y_dim, ys + cy)
                for xs in x_starts:
                    xe = min(x_dim, xs + cx)

                    # Authoritative Core Bounding Box
                    core_bbox = BoundingBox3D(
                        z_min=zs, z_max=ze,
                        y_min=ys, y_max=ye,
                        x_min=xs, x_max=xe
                    )

                    # Contextual Guard Bounding Box (Expanded & Clipped at Boundaries)
                    guard_bbox = BoundingBox3D(
                        z_min=max(0, zs - gz),
                        z_max=min(z_dim, ze + gz),
                        y_min=max(0, ys - gy),
                        y_max=min(y_dim, ye + gy),
                        x_min=max(0, xs - gx),
                        x_max=min(x_dim, xe + gx)
                    )

                    region = Region3D(
                        region_id=region_id,
                        core_bbox=core_bbox,
                        guard_bbox=guard_bbox
                    )
                    regions.append(region)
                    region_id += 1

        return regions

    def verify_complete_core_coverage(self, regions: List[Region3D]) -> bool:
        """
        Verifies that every single voxel in the volume is covered by exactly one authoritative core.
        """
        z_dim, y_dim, x_dim = self.volume_shape
        coverage_map = np.zeros(self.volume_shape, dtype=int)

        for r in regions:
            c = r.core_bbox
            coverage_map[c.z_min:c.z_max, c.y_min:c.y_max, c.x_min:c.x_max] += 1

        # Every voxel must equal exactly 1
        return bool((coverage_map == 1).all())

    def verify_no_overlapping_cores(self, regions: List[Region3D]) -> bool:
        """
        Verifies that no two region cores overlap.
        """
        n_regions = len(regions)
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                c1 = regions[i].core_bbox
                c2 = regions[j].core_bbox

                overlap_z = max(0, min(c1.z_max, c2.z_max) - max(c1.z_min, c2.z_min))
                overlap_y = max(0, min(c1.y_max, c2.y_max) - max(c1.y_min, c2.y_min))
                overlap_x = max(0, min(c1.x_max, c2.x_max) - max(c1.x_min, c2.x_min))

                if overlap_z > 0 and overlap_y > 0 and overlap_x > 0:
                    return False
        return True
