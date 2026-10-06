"""
Geometry definitions, coordinate transformations, and candidate representations for 3D Regional Processing.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
import math
import numpy as np

from src.data.nodes import CellNode
from src.data.coordinates import CoordinateTransformer


@dataclass(frozen=True)
class BoundingBox3D:
    """3D axis-aligned bounding box defined by half-open slice ranges [min, max)."""
    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Returns (dz, dy, dx) shape tuple."""
        return (self.z_max - self.z_min, self.y_max - self.y_min, self.x_max - self.x_min)

    @property
    def slice_tuples(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Returns ((z_min, z_max), (y_min, y_max), (x_min, x_max))."""
        return ((self.z_min, self.z_max), (self.y_min, self.y_max), (self.x_min, self.x_max))

    def contains_voxel(self, z: int, y: int, x: int) -> bool:
        """Checks if voxel (z, y, x) lies inside the bounding box [min, max)."""
        return (
            self.z_min <= z < self.z_max and
            self.y_min <= y < self.y_max and
            self.x_min <= x < self.x_max
        )


@dataclass
class Region3D:
    """
    Defines a 3D spatial processing region with an authoritative CORE volume
    and an expanded contextual GUARD volume.
    """
    region_id: int
    core_bbox: BoundingBox3D
    guard_bbox: BoundingBox3D

    @property
    def global_origin(self) -> Tuple[int, int, int]:
        """Returns origin (z_min, y_min, x_min) of guard volume in global coordinates."""
        return (self.guard_bbox.z_min, self.guard_bbox.y_min, self.guard_bbox.x_min)

    def local_to_global_voxel(self, local_zyx: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Converts local region voxel coordinates to global volume voxel coordinates."""
        oz, oy, ox = self.global_origin
        lz, ly, lx = local_zyx
        return (lz + oz, ly + oy, lx + ox)

    def global_to_local_voxel(self, global_zyx: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Converts global volume voxel coordinates to local region voxel coordinates."""
        oz, oy, ox = self.global_origin
        gz, gy, gx = global_zyx
        return (gz - oz, gy - oy, gx - ox)

    def is_global_voxel_in_core(self, global_zyx: Tuple[int, int, int]) -> bool:
        """Authoritative Core Membership Check: returns True iff voxel center is inside CORE."""
        gz, gy, gx = global_zyx
        return self.core_bbox.contains_voxel(gz, gy, gx)

    def is_global_voxel_in_guard(self, global_zyx: Tuple[int, int, int]) -> bool:
        """Guard Membership Check: returns True iff voxel center is inside GUARD volume."""
        gz, gy, gx = global_zyx
        return self.guard_bbox.contains_voxel(gz, gy, gx)

    def min_distance_to_core_boundary_phys(
        self,
        global_phys_zyx: Tuple[float, float, float],
        transformer: Optional[CoordinateTransformer] = None
    ) -> float:
        """
        Computes minimum physical distance (um) from a global physical coordinate
        to the nearest 6 faces of the region's authoritative CORE bounding box.
        """
        transformer = transformer or CoordinateTransformer()
        core_min_phys = transformer.voxel_to_physical(
            (self.core_bbox.z_min, self.core_bbox.y_min, self.core_bbox.x_min)
        )
        core_max_phys = transformer.voxel_to_physical(
            (self.core_bbox.z_max, self.core_bbox.y_max, self.core_bbox.x_max)
        )

        gz_p, gy_p, gx_p = global_phys_zyx

        # Distances to core boundaries along each axis (um)
        dist_z = min(abs(gz_p - core_min_phys[0]), abs(gz_p - core_max_phys[0]))
        dist_y = min(abs(gy_p - core_min_phys[1]), abs(gy_p - core_max_phys[1]))
        dist_x = min(abs(gx_p - core_min_phys[2]), abs(gx_p - core_max_phys[2]))

        return float(min(dist_z, dist_y, dist_x))


@dataclass
class RegionalCandidate:
    """
    Represents a detection candidate emitted by regional inference, storing full
    provenance and core membership tracking.
    """
    candidate_id: int
    region_id: int
    frame: int
    local_voxel_pos: Tuple[int, int, int]
    global_voxel_pos: Tuple[int, int, int]
    physical_pos: Tuple[float, float, float]
    confidence: float
    is_in_core: bool
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_cell_node(self) -> CellNode:
        """Converts regional candidate to internal CellNode representation."""
        attrs = dict(self.provenance)
        attrs["confidence"] = self.confidence
        attrs["region_id"] = self.region_id
        attrs["is_in_core"] = self.is_in_core
        return CellNode(
            node_id=self.candidate_id,
            frame=self.frame,
            voxel_pos=self.global_voxel_pos,
            physical_pos=self.physical_pos,
            attributes=attrs
        )
