"""
Configuration specifications for spatial resolution, dataset paths, and coordinate systems.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from pathlib import Path


@dataclass(frozen=True)
class SpatialResolution:
    """
    Explicit physical voxel scale configuration (in micrometers).

    Default competition values:
    - Z spacing: 1.625 um/voxel
    - Y spacing: 0.40625 um/voxel
    - X spacing: 0.40625 um/voxel
    """
    z_spacing: float = 1.625   # micrometers per voxel along Z
    y_spacing: float = 0.40625 # micrometers per voxel along Y
    x_spacing: float = 0.40625 # micrometers per voxel along X

    @property
    def scale_tuple_zyx(self) -> Tuple[float, float, float]:
        """Returns physical scaling factors in (Z, Y, X) order."""
        return (self.z_spacing, self.y_spacing, self.x_spacing)

    @property
    def scale_tuple_xyz(self) -> Tuple[float, float, float]:
        """Returns physical scaling factors in (X, Y, Z) order."""
        return (self.x_spacing, self.y_spacing, self.z_spacing)

    @property
    def anisotropy_ratio_z_to_xy(self) -> float:
        """Ratio of Z resolution to X/Y resolution (anisotropy factor = 4.0)."""
        return self.z_spacing / self.x_spacing


@dataclass
class DatasetConfig:
    """
    Configuration for dataset paths, expected dimensions, and metadata.
    """
    zarr_path: Path
    geff_path: Path
    resolution: SpatialResolution = SpatialResolution()
    expected_shape_tzyx: Tuple[int, int, int, int] = (100, 64, 256, 256)
    expected_dtype: str = "uint16"

    @classmethod
    def default_local_sample(
        cls,
        base_dir: Optional[Path] = None,
        sample_id: str = "6bba_05b6850b"
    ) -> "DatasetConfig":
        """Returns config instance pointing to the local development sample."""
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent / "train"
        return cls(
            zarr_path=base_dir / f"{sample_id}.zarr",
            geff_path=base_dir / f"{sample_id}.geff",
            resolution=SpatialResolution(),
        )
