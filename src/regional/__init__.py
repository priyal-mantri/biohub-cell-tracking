"""
Spatial Regional Processing with Guard Zones and Overlap Reconciliation.
"""

from .geometry import BoundingBox3D, Region3D, RegionalCandidate
from .partitioner import RegionPartitioner3D, PartitionConfig
from .reconciliation import CandidateReconciler, ReconciliationConfig
from .inference import RegionalInferenceEngine
from .diagnostics import RegionalBoundaryAnalyzer, BoundaryDiagnostics

__all__ = [
    "BoundingBox3D",
    "Region3D",
    "RegionalCandidate",
    "RegionPartitioner3D",
    "PartitionConfig",
    "CandidateReconciler",
    "ReconciliationConfig",
    "RegionalInferenceEngine",
    "RegionalBoundaryAnalyzer",
    "BoundaryDiagnostics",
]
