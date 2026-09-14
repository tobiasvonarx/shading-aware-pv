from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from .geometry import connected_face_components, mesh_footprint
from .models import Mesh, PanelLayout, RoofSamples, RoofScene
from .modules import infer_module_layout

ROOF_USE_LABELS = (
    "existing_pv",
    "unresolved_pv",
    "other_raised_detail",
    "free_roof",
)


class NoPVModulesError(ValueError):
    """Raised when Emboss produced no usable PV footprint for a roof."""


def roof_parallel_receivers(
    roof: RoofSamples,
    mounting_clearance_m: float,
) -> RoofSamples:
    if mounting_clearance_m < 0:
        raise ValueError("Panel mounting clearance cannot be negative")
    return RoofSamples(
        points=roof.points + mounting_clearance_m * roof.normals,
        normals=roof.normals,
        areas=roof.areas,
        face_ids=roof.face_ids,
    )


def _merge_meshes(*meshes: Mesh) -> Mesh:
    vertices = []
    faces = []
    offset = 0
    for mesh in meshes:
        if not len(mesh.faces):
            continue
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    if not faces:
        return Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32))
    return Mesh(np.vstack(vertices), np.vstack(faces).astype(np.int32))


def _polygon_rings(geometry: object) -> tuple[np.ndarray, ...]:
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    return tuple(
        np.asarray(ring.coords, dtype=np.float64)
        for polygon in polygons
        for ring in (polygon.exterior, *polygon.interiors)
    )


def load_non_panel_details(scene: RoofScene) -> Mesh:
    """Return semantic non-PV solids without removing any connected roof faces."""
    return scene.partition.details


def load_panel_layout(
    scene: RoofScene,
    details_path: Path,
    orthophoto_path: Path,
    mounting_clearance_m: float,
) -> PanelLayout:
    """Infer one roof-aligned module grid from the union Emboss PV mask."""
    collection = json.loads(details_path.read_text())
    features = [
        feature
        for feature in collection.get("features", [])
        if feature.get("properties", {}).get("class_label") == "pvmodule"
    ]
    if not features:
        raise NoPVModulesError(f"No pvmodule features in {details_path}")

    detected_footprint = unary_union(
        [shape(feature["geometry"]) for feature in features]
    )
    if detected_footprint.is_empty:
        raise NoPVModulesError(f"PV footprints are empty in {details_path}")
    if not detected_footprint.is_valid:
        detected_footprint = shapely.make_valid(detected_footprint)

    if (
        detected_footprint.intersection(mesh_footprint(scene.partition.main_roof)).area
        <= 1e-4
    ):
        raise NoPVModulesError("Detected PV does not overlap a usable roof facet")
    modules = infer_module_layout(
        scene.partition.main_roof,
        detected_footprint,
        orthophoto_path,
        mounting_clearance_m,
        scene.partition.non_pv_mesh,
    )
    if not modules.cells:
        raise NoPVModulesError(
            "No individual modules could be extracted from the PV mask"
        )
    inferred_roof_footprint = unary_union(
        [Polygon(cell.roof_corners_xyz[:, :2]) for cell in modules.cells]
    )
    footprint = (
        inferred_roof_footprint
        if not inferred_roof_footprint.is_empty
        else detected_footprint
    )

    roof = scene.samples
    detected_indices = np.flatnonzero(
        shapely.intersects_xy(
            detected_footprint,
            roof.points[:, 0],
            roof.points[:, 1],
        )
    )
    selected = shapely.intersects_xy(
        footprint,
        roof.points[:, 0],
        roof.points[:, 1],
    )
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise NoPVModulesError("PV footprints contain no sampled roof points")
    roof_subset = RoofSamples(
        points=roof.points[indices],
        normals=roof.normals[indices],
        areas=roof.areas[indices],
        face_ids=roof.face_ids[indices],
    )
    samples = roof_parallel_receivers(roof_subset, mounting_clearance_m)

    non_panel_details = scene.partition.details
    removed = scene.partition.pv_component_count
    non_panel_components = len(connected_face_components(non_panel_details))
    non_panel_footprint = mesh_footprint(non_panel_details)
    return PanelLayout(
        source_path=details_path,
        samples=samples,
        roof_sample_indices=indices,
        detected_roof_sample_indices=detected_indices,
        footprint_rings=_polygon_rings(footprint),
        detected_footprint_rings=_polygon_rings(detected_footprint),
        feature_count=len(features),
        footprint_area_xy_m2=float(footprint.area),
        detected_footprint_area_xy_m2=float(detected_footprint.area),
        sampled_surface_area_m2=float(samples.areas.sum()),
        non_panel_details=non_panel_details,
        non_panel_detail_components=non_panel_components,
        non_panel_obstruction_area_xy_m2=float(non_panel_footprint.area),
        occluder_mesh=scene.partition.non_pv_mesh,
        removed_detail_components=removed,
        modules=modules,
    )


def roof_use_masks(
    scene: RoofScene,
    layout: PanelLayout | None,
) -> dict[str, np.ndarray]:
    """Return mutually exclusive semantic roof-use masks for the Emboss scene."""
    existing_pv = np.zeros(len(scene.samples.points), dtype=bool)
    if layout is not None:
        existing_pv[layout.roof_sample_indices] = True
    detected_pv = np.zeros(len(scene.samples.points), dtype=bool)
    if layout is not None:
        detected_pv[layout.detected_roof_sample_indices] = True
    unresolved_pv = detected_pv & ~existing_pv
    other_raised_detail = scene.excluded_by_raised_detail & ~(
        existing_pv | unresolved_pv
    )
    free_roof = ~(existing_pv | unresolved_pv | other_raised_detail)
    return {
        "existing_pv": existing_pv,
        "unresolved_pv": unresolved_pv,
        "other_raised_detail": other_raised_detail,
        "free_roof": free_roof,
    }


def roof_use_summary(
    scene: RoofScene,
    layout: PanelLayout | None,
) -> dict[str, float]:
    masks = roof_use_masks(scene, layout)
    areas = scene.samples.areas
    return {
        "total_main_roof_m2": float(areas.sum()),
        "existing_pv_m2": float(areas[masks["existing_pv"]].sum()),
        "unresolved_pv_m2": float(areas[masks["unresolved_pv"]].sum()),
        "other_raised_detail_m2": float(areas[masks["other_raised_detail"]].sum()),
        "free_roof_m2": float(areas[masks["free_roof"]].sum()),
    }


def roof_use_labels(
    scene: RoofScene,
    layout: PanelLayout | None,
) -> np.ndarray:
    masks = roof_use_masks(scene, layout)
    labels = np.empty(len(scene.samples.points), dtype="<U24")
    for label in ROOF_USE_LABELS:
        labels[masks[label]] = label
    return labels
