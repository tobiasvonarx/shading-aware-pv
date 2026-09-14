from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    source_face_ids: np.ndarray | None = None

    @property
    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]


@dataclass(frozen=True)
class ContextScene:
    surface_mesh: Mesh
    surface_mesh_without_vegetation: Mesh
    grid_shape: tuple[int, int]
    grid_resolution_m: float
    source_points: np.ndarray
    source_classes: np.ndarray
    target_footprint_rings: tuple[np.ndarray, ...]
    removed_target_points: int
    center_xy: tuple[float, float]
    half_extent_m: float
    surfaces_path: Path
    lidar_sources: tuple[dict, ...]
    coverage_complete: bool


@dataclass(frozen=True)
class MeshPartition:
    base_roof: Mesh
    main_roof: Mesh
    details: Mesh
    component_count: int
    full_mesh: Mesh
    non_pv_mesh: Mesh
    pv_component_count: int


@dataclass(frozen=True)
class RoofSamples:
    points: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    face_ids: np.ndarray

    @property
    def tilt(self) -> np.ndarray:
        return np.degrees(np.arccos(np.clip(self.normals[:, 2], -1.0, 1.0)))

    @property
    def azimuth(self) -> np.ndarray:
        return np.mod(
            np.degrees(np.arctan2(self.normals[:, 0], self.normals[:, 1])),
            360.0,
        )


@dataclass(frozen=True)
class RoofScene:
    source_path: Path
    partition: MeshPartition
    samples: RoofSamples
    excluded_by_raised_detail: np.ndarray


@dataclass(frozen=True)
class IrradianceResult:
    irradiation: np.ndarray
    direct_shaded_hours: np.ndarray


@dataclass(frozen=True)
class RoofResource:
    roof_samples: RoofSamples
    receivers: RoofSamples
    use_labels: np.ndarray
    results: dict[str, IrradianceResult]


@dataclass(frozen=True)
class ModuleCell:
    module_id: str
    array_id: str
    facet_id: str
    row: int
    column: int
    orientation: str
    mask_coverage: float
    roof_corners_xyz: np.ndarray
    receiver_corners_xyz: np.ndarray

    @property
    def projected_area_m2(self) -> float:
        xy = self.receiver_corners_xyz[:, :2]
        return 0.5 * float(
            abs(
                np.dot(xy[:, 0], np.roll(xy[:, 1], 1))
                - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))
            )
        )


@dataclass(frozen=True)
class ModuleArray:
    array_id: str
    facet_id: str
    orientation: str
    module_count: int
    detected_area_m2: float
    inferred_area_m2: float
    mask_precision: float
    mask_recall: float
    boundary_evidence: float | None
    roof_tilt_deg: float
    axis_source: str
    axis_confidence: float


@dataclass(frozen=True)
class ModuleSizeCandidate:
    width_m: float
    height_m: float
    evidence_score: float
    mask_precision: float
    mask_recall: float
    boundary_evidence: float | None
    module_count: int
    nearest_standard_width_m: float
    nearest_standard_height_m: float
    standard_distance_m: float
    within_score_tie: bool
    selected: bool


@dataclass(frozen=True)
class ModuleLayout:
    module_width_m: float
    module_height_m: float
    mounting_clearance_m: float
    cells: tuple[ModuleCell, ...]
    arrays: tuple[ModuleArray, ...]
    candidates: tuple[ModuleSizeCandidate, ...]
    detected_surface_area_m2: float
    inferred_surface_area_m2: float
    projected_area_m2: float
    mask_precision: float
    mask_recall: float
    boundary_evidence: float | None
    evidence_score: float
    evidence_gap_to_best: float
    nearest_standard_width_m: float
    nearest_standard_height_m: float
    standard_distance_m: float
    score_tie_tolerance: float
    orthophoto_path: Path


@dataclass(frozen=True)
class PanelLayout:
    source_path: Path
    samples: RoofSamples
    roof_sample_indices: np.ndarray
    detected_roof_sample_indices: np.ndarray
    footprint_rings: tuple[np.ndarray, ...]
    detected_footprint_rings: tuple[np.ndarray, ...]
    feature_count: int
    footprint_area_xy_m2: float
    detected_footprint_area_xy_m2: float
    sampled_surface_area_m2: float
    non_panel_details: Mesh
    non_panel_detail_components: int
    non_panel_obstruction_area_xy_m2: float
    occluder_mesh: Mesh
    removed_detail_components: int
    modules: ModuleLayout


@dataclass(frozen=True)
class Weather:
    hourly: pd.DataFrame
    latitude: float
    longitude: float
    elevation: float
    source: str
