from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import shapely
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from .models import Mesh, RoofResource, RoofScene
from .modules import RoofFacet, roof_facets, receiver_exclusions


@dataclass(frozen=True)
class PlacementCandidate:
    candidate_id: int
    facet_id: str
    orientation: str
    roof_corners_xyz: np.ndarray
    receiver_corners_xyz: np.ndarray
    roof_sample_indices: np.ndarray
    annual_irradiation_kwh_m2: float


@dataclass(frozen=True)
class PlacementSolution:
    panel_count: int
    candidate_ids: np.ndarray
    roof_sample_indices: np.ndarray


@dataclass(frozen=True)
class PlacementOptimization:
    roof_setback_m: float
    obstruction_setback_m: float
    module_gap_m: float
    phase_step_m: float
    module_width_m: float
    module_height_m: float
    candidates: tuple[PlacementCandidate, ...]
    conflict_pairs: np.ndarray
    maximum_count: int
    maximum_layout: PlacementSolution
    relocated_layout: PlacementSolution


def clean_slate_order(optimization: PlacementOptimization) -> np.ndarray:
    selected = optimization.maximum_layout.candidate_ids
    scores = np.asarray(
        [optimization.candidates[index].annual_irradiation_kwh_m2 for index in selected]
    )
    return selected[np.argsort(-scores)]


def write_solution_geojson(
    path: Path,
    optimization: PlacementOptimization,
    candidate_ids: np.ndarray,
    *,
    layout: str,
) -> None:
    features = []
    for position, candidate_id in enumerate(candidate_ids):
        candidate = optimization.candidates[int(candidate_id)]
        coordinates = candidate.receiver_corners_xyz.tolist()
        coordinates.append(coordinates[0])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "layout": layout,
                    "position": position,
                    "candidate_id": candidate.candidate_id,
                    "facet_id": candidate.facet_id,
                    "orientation": candidate.orientation,
                    "module_width_m": optimization.module_width_m,
                    "module_height_m": optimization.module_height_m,
                    "roof_setback_m": optimization.roof_setback_m,
                    "obstruction_setback_m": optimization.obstruction_setback_m,
                    "module_gap_m": optimization.module_gap_m,
                    "phase_step_m": optimization.phase_step_m,
                    "annual_irradiation_kwh_m2": (candidate.annual_irradiation_kwh_m2),
                },
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
            }
        )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n"
    )


def facet_exclusions(facet: RoofFacet, exclusions: Mesh | BaseGeometry) -> BaseGeometry:
    """Project only geometry above this support plane; 2D baselines are explicit."""
    if isinstance(exclusions, Mesh):
        from emboss.roof_surface import covered_footprint

        triangles = exclusions.triangles
        if exclusions.source_face_ids is not None:
            triangles = triangles[~np.isin(exclusions.source_face_ids, facet.source_face_ids)]
        return covered_footprint(facet.polygon_xy, facet.origin, facet.normal, triangles)
    return exclusions.intersection(facet.polygon_xy)


def usable_facet(
    facet: RoofFacet,
    detail_footprint: Mesh | BaseGeometry,
    roof_setback_m: float,
    obstruction_setback_m: float,
) -> BaseGeometry:
    roof_uv = shapely.make_valid(facet.xy_to_uv(facet.polygon_xy))
    usable = roof_uv.buffer(-roof_setback_m, join_style="mitre")
    if usable.is_empty:
        return usable
    details_xy = facet_exclusions(facet, detail_footprint)
    if details_xy.is_empty:
        return usable
    details_uv = shapely.make_valid(facet.xy_to_uv(details_xy))
    return shapely.make_valid(
        usable.difference(details_uv.buffer(obstruction_setback_m))
    )


def _phase_values(pitch: float, phase_step: float) -> np.ndarray:
    """Search offsets over one module pitch."""
    return np.arange(0.0, pitch - 1e-9, phase_step)


def _facet_sample_coordinates(
    facet: RoofFacet,
    resource: RoofResource,
) -> tuple[np.ndarray, np.ndarray]:
    points = resource.roof_samples.points
    selected = shapely.intersects_xy(
        facet.polygon_xy.buffer(1e-7),
        points[:, 0],
        points[:, 1],
    )
    # Follow the actual support triangles, including survey rounding, instead
    # of discarding samples by their distance from an approximate mounting plane.
    selected &= np.isin(resource.roof_samples.face_ids, facet.sample_face_ids)
    sample_ids = np.flatnonzero(selected)
    relative = points[sample_ids] - facet.origin
    uv = np.column_stack((relative @ facet.axis_u, relative @ facet.axis_v))
    return sample_ids, uv


def generate_candidates(
    scene: RoofScene,
    resource: RoofResource,
    exclusion_footprint: Mesh | BaseGeometry,
    *,
    module_width_m: float,
    module_height_m: float,
    roof_setback_m: float,
    obstruction_setback_m: float,
    module_gap_m: float,
    phase_step_m: float,
    mounting_clearance_m: float,
) -> tuple[tuple[PlacementCandidate, ...], dict[str, list[Polygon]]]:
    values = (
        module_width_m,
        module_height_m,
        phase_step_m,
        mounting_clearance_m,
    )
    if any(value <= 0 for value in values):
        raise ValueError(
            "Module dimensions, phase step, and mounting clearance must be positive"
        )
    if min(roof_setback_m, obstruction_setback_m, module_gap_m) < 0:
        raise ValueError("Placement setbacks and module gap cannot be negative")

    details = exclusion_footprint
    candidates: list[PlacementCandidate] = []
    polygons_by_facet: dict[str, list[Polygon]] = {}
    for facet in roof_facets(scene.partition.main_roof):
        usable = usable_facet(
            facet,
            details,
            roof_setback_m,
            obstruction_setback_m,
        )
        if usable.is_empty:
            continue
        allowed = usable.buffer(1e-8)
        blocked_receivers = receiver_exclusions(facet, scene.partition.non_pv_mesh, mounting_clearance_m)
        sample_ids, sample_uv = _facet_sample_coordinates(facet, resource)
        min_u, min_v, max_u, max_v = usable.bounds
        orientations = (
            ("portrait", module_width_m, module_height_m),
            ("landscape", module_height_m, module_width_m),
        )
        best_lattice: list[
            tuple[str, Polygon, np.ndarray, np.ndarray, np.ndarray, float]
        ] = []
        best_score = (-1, -math.inf)
        for orientation, size_u, size_v in orientations:
            pitch_u = size_u + module_gap_m
            pitch_v = size_v + module_gap_m
            for phase_u in _phase_values(pitch_u, phase_step_m):
                for phase_v in _phase_values(pitch_v, phase_step_m):
                    lattice = []
                    origin_u = np.arange(
                        min_u + phase_u,
                        max_u - size_u + 1e-9,
                        pitch_u,
                    )
                    origin_v = np.arange(
                        min_v + phase_v,
                        max_v - size_v + 1e-9,
                        pitch_v,
                    )
                    for min_cell_u in origin_u:
                        for min_cell_v in origin_v:
                            polygon_uv = Polygon(
                                [
                                    (min_cell_u, min_cell_v),
                                    (min_cell_u + size_u, min_cell_v),
                                    (min_cell_u + size_u, min_cell_v + size_v),
                                    (min_cell_u, min_cell_v + size_v),
                                ]
                            )
                            if not allowed.covers(polygon_uv) or blocked_receivers.intersects(polygon_uv):
                                continue
                            inside = (
                                (sample_uv[:, 0] >= min_cell_u)
                                & (sample_uv[:, 0] < min_cell_u + size_u)
                                & (sample_uv[:, 1] >= min_cell_v)
                                & (sample_uv[:, 1] < min_cell_v + size_v)
                            )
                            candidate_samples = sample_ids[inside]
                            if not len(candidate_samples):
                                continue
                            weights = resource.receivers.areas[candidate_samples]
                            irradiation = float(
                                np.average(
                                    resource.results["full"].irradiation[
                                        candidate_samples
                                    ],
                                    weights=weights,
                                )
                            )
                            corners_uv = np.asarray(polygon_uv.exterior.coords[:-1])
                            roof_corners = facet.uv_to_xyz(corners_uv)
                            receiver_corners = roof_corners + (
                                mounting_clearance_m * facet.normal[None, :]
                            )
                            lattice.append(
                                (
                                    orientation,
                                    polygon_uv,
                                    roof_corners,
                                    receiver_corners,
                                    candidate_samples,
                                    irradiation,
                                )
                            )
                    score = (
                        len(lattice),
                        sum(item[-1] for item in lattice),
                    )
                    if score > best_score:
                        best_score = score
                        best_lattice = lattice
        for (
            orientation,
            polygon_uv,
            roof_corners,
            receiver_corners,
            candidate_samples,
            irradiation,
        ) in best_lattice:
            candidates.append(
                PlacementCandidate(
                    candidate_id=len(candidates),
                    facet_id=facet.facet_id,
                    orientation=orientation,
                    roof_corners_xyz=roof_corners,
                    receiver_corners_xyz=receiver_corners,
                    roof_sample_indices=candidate_samples,
                    annual_irradiation_kwh_m2=irradiation,
                )
            )
            polygons_by_facet.setdefault(facet.facet_id, []).append(polygon_uv)
    return tuple(candidates), polygons_by_facet


def _conflict_pairs(
    candidates: tuple[PlacementCandidate, ...],
    polygons_by_facet: dict[str, list[Polygon]],
    module_gap_m: float,
) -> np.ndarray:
    candidate_ids_by_facet: dict[str, list[int]] = {}
    for candidate in candidates:
        candidate_ids_by_facet.setdefault(candidate.facet_id, []).append(
            candidate.candidate_id
        )
    pairs = []
    for facet_id, polygons in polygons_by_facet.items():
        ids = candidate_ids_by_facet[facet_id]
        footprints = [
            polygon.buffer(module_gap_m / 2.0, join_style="mitre")
            for polygon in polygons
        ]
        tree = STRtree(footprints)
        for local_left, footprint in enumerate(footprints):
            for local_right in tree.query(footprint, predicate="intersects"):
                local_right = int(local_right)
                if local_right <= local_left:
                    continue
                if footprint.intersection(footprints[local_right]).area > 1e-8:
                    pairs.append((ids[local_left], ids[local_right]))
    return (
        np.asarray(pairs, dtype=np.int32).reshape((-1, 2))
        if pairs
        else np.empty((0, 2), dtype=np.int32)
    )


def _constraint_matrix(
    candidate_count: int,
    conflict_pairs: np.ndarray,
) -> coo_array:
    rows = np.repeat(np.arange(len(conflict_pairs)), 2)
    columns = conflict_pairs.ravel()
    return coo_array(
        (np.ones(len(columns)), (rows, columns)),
        shape=(len(conflict_pairs), candidate_count),
    )


def select_greedy(
    candidates: tuple[PlacementCandidate, ...],
    conflict_pairs: np.ndarray,
    *,
    panel_count: int,
) -> PlacementSolution:
    """Rank by irradiation and accept non-conflicting panels up to the target.

    A short result explicitly indicates failure to reach the requested count.
    This adapts SunPlace's selection rule, not its complete simulation pipeline.
    """
    if panel_count < 0:
        raise ValueError("Panel count cannot be negative")
    scores = np.array([c.annual_irradiation_kwh_m2 for c in candidates])
    if not np.isfinite(scores).all():
        raise ValueError("Panel candidates contain non-finite irradiation scores")
    neighbors = [set() for _ in candidates]
    for a, b in conflict_pairs:
        neighbors[a].add(b)
        neighbors[b].add(a)
    blocked = set()
    selected = []
    for index in np.argsort(-scores, kind="stable"):
        if len(selected) == panel_count:
            break
        if index not in blocked:
            selected.append(int(index))
            blocked.update(neighbors[index])
    ids = np.array(sorted(selected), dtype=int)
    samples = (
        np.unique(np.concatenate([candidates[i].roof_sample_indices for i in ids]))
        if len(ids)
        else np.empty(0, dtype=int)
    )
    return PlacementSolution(len(ids), ids, samples)


def select_milp(
    candidates: tuple[PlacementCandidate, ...],
    conflict_pairs: np.ndarray,
    *,
    panel_count: int | None,
) -> PlacementSolution:
    candidate_count = len(candidates)
    if not candidate_count:
        if panel_count not in (None, 0):
            raise ValueError("No feasible panel candidates were generated")
        return PlacementSolution(0, np.empty(0, dtype=int), np.empty(0, dtype=int))
    scores = np.asarray(
        [candidate.annual_irradiation_kwh_m2 for candidate in candidates]
    )
    if not np.isfinite(scores).all():
        raise ValueError("Panel candidates contain non-finite irradiation scores")
    if panel_count is None:
        scale = float(scores.max())
        normalized_scores = scores / scale if scale > 0.0 else np.zeros_like(scores)
        objective = -np.ones(candidate_count) - 1e-6 * normalized_scores
    else:
        objective = -scores
    constraints: list[LinearConstraint] = []
    if len(conflict_pairs):
        constraints.append(
            LinearConstraint(
                _constraint_matrix(candidate_count, conflict_pairs),
                lb=-np.inf,
                ub=1.0,
            )
        )
    if panel_count is not None:
        constraints.append(
            LinearConstraint(
                np.ones((1, candidate_count)),
                lb=panel_count,
                ub=panel_count,
            )
        )
    result = milp(
        c=objective,
        integrality=np.ones(candidate_count),
        bounds=Bounds(0.0, 1.0),
        constraints=constraints,
        options={"time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        target = "maximum count" if panel_count is None else f"count {panel_count}"
        raise RuntimeError(
            f"Panel-placement MILP failed for {target}: {result.message}"
        )
    selected = np.flatnonzero(result.x > 0.5)
    selected_samples = [candidates[index].roof_sample_indices for index in selected]
    sample_ids = (
        np.unique(np.concatenate(selected_samples))
        if selected_samples
        else np.empty(0, dtype=np.int32)
    )
    return PlacementSolution(
        panel_count=len(selected),
        candidate_ids=selected,
        roof_sample_indices=sample_ids,
    )


def optimize_placements(
    scene: RoofScene,
    resource: RoofResource,
    exclusion_footprint: Mesh | BaseGeometry,
    *,
    current_panel_count: int,
    module_width_m: float,
    module_height_m: float,
    roof_setback_m: float,
    obstruction_setback_m: float,
    module_gap_m: float,
    phase_step_m: float,
    mounting_clearance_m: float,
) -> PlacementOptimization:
    candidates, polygons_by_facet = generate_candidates(
        scene,
        resource,
        exclusion_footprint,
        module_width_m=module_width_m,
        module_height_m=module_height_m,
        roof_setback_m=roof_setback_m,
        obstruction_setback_m=obstruction_setback_m,
        module_gap_m=module_gap_m,
        phase_step_m=phase_step_m,
        mounting_clearance_m=mounting_clearance_m,
    )
    conflicts = _conflict_pairs(candidates, polygons_by_facet, module_gap_m)
    maximum = select_milp(candidates, conflicts, panel_count=None)
    relocated_count = min(current_panel_count, maximum.panel_count)
    relocated = select_milp(candidates, conflicts, panel_count=relocated_count)
    return PlacementOptimization(
        roof_setback_m=roof_setback_m,
        obstruction_setback_m=obstruction_setback_m,
        module_gap_m=module_gap_m,
        phase_step_m=phase_step_m,
        module_width_m=module_width_m,
        module_height_m=module_height_m,
        candidates=candidates,
        conflict_pairs=conflicts,
        maximum_count=maximum.panel_count,
        maximum_layout=maximum,
        relocated_layout=relocated,
    )


def optimize_relocation(scene, optimization, resource, layout):
    """Search generated and inferred positions together at the installed count.

    Return the mixed-search proposal and the exact inferred incumbent. The
    clean-slate maximum remains the solution of the original candidate pool.
    """
    cells = layout.modules.cells
    candidates = list(optimization.candidates)
    first = len(candidates)
    points = resource.roof_samples.points
    for cell in cells:
        indices = np.flatnonzero(shapely.intersects_xy(
            Polygon(cell.roof_corners_xyz[:, :2]), points[:, 0], points[:, 1]
        ))
        irradiation = (float(np.average(resource.results["full"].irradiation[indices],
                                        weights=resource.receivers.areas[indices]))
                       if len(indices) else 0.0)
        candidates.append(PlacementCandidate(
            len(candidates), cell.facet_id, cell.orientation,
            cell.roof_corners_xyz, cell.receiver_corners_xyz, indices, irradiation,
        ))
    facets = {facet.facet_id: facet for facet in roof_facets(scene.partition.main_roof)}
    polygons = {}
    for candidate in candidates:
        polygons.setdefault(candidate.facet_id, []).append(
            facets[candidate.facet_id].xy_to_uv(Polygon(candidate.roof_corners_xyz[:, :2]))
        )
    candidates = tuple(candidates)
    incumbent = replace(
        optimization, candidates=candidates,
        conflict_pairs=_conflict_pairs(candidates, polygons, optimization.module_gap_m),
        relocated_layout=PlacementSolution(
            len(cells), np.arange(first, len(candidates), dtype=int), layout.roof_sample_indices,
        ),
    )
    proposal = replace(incumbent, relocated_layout=select_milp(
        candidates, incumbent.conflict_pairs, panel_count=len(cells),
    ))
    return proposal, incumbent


def retain_installed_layout(proposal, incumbent, installed_kwh, relocated_kwh):
    """Accept a relocation only when its full-shading modeled AC yield improves."""
    if not np.isfinite([installed_kwh, relocated_kwh]).all():
        raise ValueError("Modeled yields must be finite")
    return proposal if relocated_kwh > installed_kwh else incumbent


def _cover_height(corners: np.ndarray, triangles: np.ndarray) -> float:
    """Independently compare rectangle/triangle intersections against roof height.

    This checks original mesh triangles, not the generated usable polygons or
    candidate lattice. Linear height differences attain their extrema at the
    intersection vertices, including narrow obstructions between sample points.
    """
    anchor = corners[0]
    local = corners - anchor
    normal = np.cross(local[1], local[2])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    panel = Polygon(local[:, :2])
    maximum = 0.0
    for triangle in triangles - anchor:
        cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if abs(cross[2]) <= 1e-10:
            continue
        overlap = panel.intersection(Polygon(triangle[:, :2]))
        if overlap.area <= 1e-10:
            continue
        coordinates = shapely.get_coordinates(overlap)
        surface_z = triangle[0, 2] - ((coordinates - triangle[0, :2]) @ cross[:2]) / cross[2]
        support_z = -(coordinates @ normal[:2]) / normal[2]
        maximum = max(maximum, float(np.max(surface_z - support_z)))
    return maximum


def placement_audit(
    scene: RoofScene,
    exclusion_footprint: Mesh | BaseGeometry,
    optimization: PlacementOptimization,
    solution: PlacementSolution,
) -> dict[str, float | int | None]:
    """Audit clearances and independently test original full-mesh height order."""
    facets = {facet.facet_id: facet for facet in roof_facets(scene.partition.main_roof)}
    details = exclusion_footprint
    minimum_roof = math.inf
    minimum_obstruction = math.inf
    panels_by_facet: dict[str, list[BaseGeometry]] = {}
    invalid = 0
    covered = 0
    maximum_cover = 0.0
    for candidate_id in solution.candidate_ids:
        candidate = optimization.candidates[int(candidate_id)]
        facet = facets[candidate.facet_id]
        panel_uv = facet.xy_to_uv(Polygon(candidate.roof_corners_xyz[:, :2]))
        panels_by_facet.setdefault(candidate.facet_id, []).append(panel_uv)
        roof_uv = shapely.make_valid(facet.xy_to_uv(facet.polygon_xy))
        minimum_roof = min(
            minimum_roof,
            float(panel_uv.distance(roof_uv.boundary)),
        )
        details_xy = facet_exclusions(facet, details)
        if not details_xy.is_empty:
            details_uv = shapely.make_valid(facet.xy_to_uv(details_xy))
            minimum_obstruction = min(
                minimum_obstruction,
                float(panel_uv.distance(details_uv)),
            )
        allowed = usable_facet(
            facet,
            details,
            optimization.roof_setback_m,
            optimization.obstruction_setback_m,
        )
        cover_height = _cover_height(candidate.receiver_corners_xyz, scene.partition.non_pv_mesh.triangles)
        maximum_cover = max(maximum_cover, cover_height)
        is_covered = cover_height > 1e-5
        covered += int(is_covered)
        invalid += int(is_covered or not allowed.buffer(1e-8).covers(panel_uv))

    minimum_module_gap = min(
        (
            float(left.distance(right))
            for panels in panels_by_facet.values()
            for index, left in enumerate(panels)
            for right in panels[index + 1 :]
        ),
        default=math.inf,
    )

    selected = set(map(int, solution.candidate_ids))
    spacing_violations = sum(
        int(int(left) in selected and int(right) in selected)
        for left, right in optimization.conflict_pairs
    )
    return {
        "minimum_roof_edge_or_ridge_clearance_m": (
            None if math.isinf(minimum_roof) else minimum_roof
        ),
        "minimum_non_pv_obstruction_clearance_m": (
            None if math.isinf(minimum_obstruction) else minimum_obstruction
        ),
        "minimum_module_gap_m": (
            None if math.isinf(minimum_module_gap) else minimum_module_gap
        ),
        "containment_or_clearance_violations": invalid,
        "covered_surface_violations": covered,
        "maximum_cover_height_m": maximum_cover,
        "panel_spacing_violations": spacing_violations,
    }
