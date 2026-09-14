from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .models import (
    Mesh,
    ModuleArray,
    ModuleCell,
    ModuleLayout,
    ModuleSizeCandidate,
)

MIN_MODULE_WIDTH_M = 0.80
MAX_MODULE_WIDTH_M = 1.40
MIN_MODULE_HEIGHT_M = 1.40
MAX_MODULE_HEIGHT_M = 2.40
MODULE_SIZE_STEP_M = 0.05
FIT_RESOLUTION_M = 0.10
MIN_CELL_MASK_COVERAGE = 0.28
MASK_SCORE_WEIGHT = 0.82
BOUNDARY_SCORE_WEIGHT = 0.18
SCORE_TIE_TOLERANCE = 0.01
# Representative families rounded to the 5 cm sweep grid. Manufacturer examples
# and exact dimensions are documented in README.md.
COMMON_MODULE_SIZE_FAMILIES_M = (
    (1.00, 1.65),
    (1.00, 1.70),
    (1.05, 1.70),
    (1.15, 1.70),
    (1.20, 1.75),
    (1.00, 2.00),
    (1.15, 2.30),
)
FACET_NORMAL_TOLERANCE_DEG = 3.0
# Match millimetre-rounded roof geometry without treating sub-millimetre
# boundary residuals as missing support. This is not an installation setback.
ROOF_GEOMETRY_TOLERANCE_M = 1e-3
# Matching tolerance for installed rectangles against reconstructed support.
# This is separate from facet adjacency and new-installation clearances.
INSTALLED_SUPPORT_TOLERANCE_M = 0.01


@dataclass(frozen=True)
class RoofFacet:
    facet_id: str
    polygon_xy: BaseGeometry
    plane: tuple[float, float, float]
    origin: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    normal: np.ndarray
    axis_source: str
    axis_confidence: float
    sample_face_ids: tuple[int, ...] = ()
    source_face_ids: tuple[int, ...] = ()

    @property
    def tilt_deg(self) -> float:
        return math.degrees(math.acos(float(np.clip(self.normal[2], -1.0, 1.0))))

    def xy_to_uv(self, geometry: BaseGeometry) -> BaseGeometry:
        a, b, _ = self.plane
        u_x = self.axis_u[0] + a * self.axis_u[2]
        u_y = self.axis_u[1] + b * self.axis_u[2]
        v_x = self.axis_v[0] + a * self.axis_v[2]
        v_y = self.axis_v[1] + b * self.axis_v[2]
        local = affinity.translate(geometry, -self.origin[0], -self.origin[1])
        return affinity.affine_transform(
            local,
            [u_x, u_y, v_x, v_y, 0.0, 0.0],
        )

    def uv_to_xyz(self, uv: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(uv, dtype=np.float64).reshape((-1, 2))
        return (
            self.origin[None, :]
            + coordinates[:, :1] * self.axis_u[None, :]
            + coordinates[:, 1:] * self.axis_v[None, :]
        )


@dataclass(frozen=True)
class _Observation:
    array_id: str
    facet: RoofFacet
    geometry_uv: BaseGeometry
    u: np.ndarray
    v: np.ndarray
    mask: np.ndarray
    gradient_u: np.ndarray
    gradient_v: np.ndarray
    resolution_m: float
    allowed_uv: BaseGeometry
    blocked_uv: BaseGeometry


@dataclass(frozen=True)
class _PhaseFit:
    orientation: str
    cell_u_m: float
    cell_v_m: float
    origin_u: float
    origin_v: float
    cells: tuple[tuple[int, int, float], ...]
    true_positive_m2: float
    predicted_m2: float
    mask_m2: float
    mask_precision: float
    mask_recall: float
    mask_f1: float
    boundary_evidence: float | None
    boundary_samples: int
    score: float


@dataclass(frozen=True)
class _CandidateFit:
    width_m: float
    height_m: float
    blocks: tuple[_PhaseFit, ...]
    score: float
    precision: float
    recall: float
    boundary_evidence: float | None
    module_count: int


def _polygon_parts(geometry: BaseGeometry) -> tuple[Polygon, ...]:
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "Polygon":
        return (geometry,)  # type: ignore[return-value]
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        return tuple(
            part
            for child in geometry.geoms
            for part in _polygon_parts(child)
        )
    return ()


def _fit_plane(points: np.ndarray) -> tuple[float, float, float]:
    xyz = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    origin = xyz.mean(axis=0)
    local = xyz - origin
    design = np.column_stack((local[:, 0], local[:, 1], np.ones(len(xyz))))
    a, b, c = np.linalg.lstsq(design, local[:, 2], rcond=None)[0]
    return float(a), float(b), float(origin[2] + c - a * origin[0] - b * origin[1])


def _upward_normal(plane: tuple[float, float, float]) -> np.ndarray:
    a, b, _ = plane
    normal = np.asarray([-a, -b, 1.0], dtype=np.float64)
    return normal / np.linalg.norm(normal)


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    result = np.asarray(axis, dtype=np.float64)
    if result[0] < 0 or (abs(result[0]) < 1e-10 and result[1] < 0):
        result = -result
    return result


def _flat_roof_axis(polygon: BaseGeometry) -> tuple[np.ndarray, float]:
    weighted_cos = weighted_sin = total_length = 0.0
    for part in _polygon_parts(polygon):
        coordinates = np.asarray(part.exterior.coords, dtype=np.float64)
        edges = np.diff(coordinates[:, :2], axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        valid = lengths > 1e-8
        angles = np.arctan2(edges[valid, 1], edges[valid, 0])
        weighted_cos += float(np.sum(lengths[valid] * np.cos(4.0 * angles)))
        weighted_sin += float(np.sum(lengths[valid] * np.sin(4.0 * angles)))
        total_length += float(np.sum(lengths[valid]))
    if total_length <= 0:
        return np.asarray([1.0, 0.0, 0.0]), 0.0
    angle = 0.25 * math.atan2(weighted_sin, weighted_cos)
    confidence = math.hypot(weighted_cos, weighted_sin) / total_length
    return _canonical_axis(np.asarray([math.cos(angle), math.sin(angle), 0.0])), confidence


def _roof_frame(
    facet_id: str,
    polygon: BaseGeometry,
    plane: tuple[float, float, float],
    sample_face_ids: tuple[int, ...] = (),
    source_face_ids: tuple[int, ...] = (),
) -> RoofFacet:
    normal = _upward_normal(plane)
    tilt = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
    if tilt >= 3.0:
        axis_u = _canonical_axis(np.cross(np.asarray([0.0, 0.0, 1.0]), normal))
        axis_u /= np.linalg.norm(axis_u)
        axis_source = "roof slope"
        axis_confidence = 1.0
    else:
        axis_u, axis_confidence = _flat_roof_axis(polygon)
        axis_source = "roof boundary"
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)
    centroid = polygon.centroid
    a, b, c = plane
    origin = np.asarray(
        [centroid.x, centroid.y, a * centroid.x + b * centroid.y + c],
        dtype=np.float64,
    )
    return RoofFacet(
        facet_id=facet_id,
        polygon_xy=polygon,
        plane=plane,
        origin=origin,
        axis_u=axis_u,
        axis_v=axis_v,
        normal=normal,
        axis_source=axis_source,
        axis_confidence=axis_confidence,
        sample_face_ids=sample_face_ids,
        source_face_ids=source_face_ids,
    )


def roof_facets(mesh: Mesh) -> tuple[RoofFacet, ...]:
    triangles = mesh.triangles
    if not len(triangles):
        return ()
    raw_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(raw_normals, axis=1)
    normals = raw_normals / np.maximum(lengths[:, None], 1e-12)
    normals[normals[:, 2] < 0] *= -1.0

    parent = list(range(len(triangles)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edge_faces: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    keys = np.rint(mesh.vertices / 1e-4).astype(np.int64)
    for face_id, face in enumerate(mesh.faces):
        for start, end in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((tuple(keys[face[start]]), tuple(keys[face[end]]))))
            edge_faces.setdefault(edge, []).append(face_id)
    cosine_threshold = math.cos(math.radians(FACET_NORMAL_TOLERANCE_DEG))
    for face_ids in edge_faces.values():
        for left in face_ids:
            for right in face_ids:
                if left < right and float(np.dot(normals[left], normals[right])) >= cosine_threshold:
                    union(left, right)

    # Clipping can split one side of a shared edge without splitting the other.
    # Join overlapping 3D edge segments as well as identical endpoint pairs, so
    # these triangulation seams do not become artificial module boundaries.
    edges = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2, 3)
    edge_lines = [LineString(edge[:, :2]) for edge in edges]
    tree = shapely.STRtree(edge_lines)
    # Millimetre-rounded survey vertices can leave sub-millimetre offsets
    # when a clipped edge is reconstructed on adjacent triangle planes.
    tolerance = ROOF_GEOMETRY_TOLERANCE_M
    for edge_id, (edge, line) in enumerate(zip(edges, edge_lines, strict=True)):
        left = edge_id // 3
        direction = edge[1] - edge[0]
        length = np.linalg.norm(direction)
        if length <= tolerance:
            continue
        direction /= length
        for other_id in tree.query(line.buffer(tolerance)):
            right = int(other_id) // 3
            if right <= left or find(left) == find(right):
                continue
            if float(np.dot(normals[left], normals[right])) < cosine_threshold:
                continue
            relative = edges[other_id] - edge[0]
            along = relative @ direction
            residual = relative - along[:, None] * direction
            if np.linalg.norm(residual, axis=1).max() > tolerance:
                continue
            if min(length, along.max()) - max(0.0, along.min()) > tolerance:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for face_id in range(len(triangles)):
        groups.setdefault(find(face_id), []).append(face_id)

    raw_facets = []
    for face_ids in groups.values():
        # Resolve sub-micrometre clipping seams so internal triangulation
        # edges never become roof boundaries.
        xy_origin = triangles[np.asarray(face_ids), :, :2].min(axis=(0, 1))
        local_polygon = shapely.union_all(
            [Polygon(triangles[face_id, :, :2] - xy_origin) for face_id in face_ids],
            grid_size=1e-7,
        )
        polygon = affinity.translate(local_polygon, *xy_origin)
        if polygon.is_empty or polygon.area <= 1e-6:
            continue
        points = triangles[np.asarray(face_ids)].reshape((-1, 3))
        source_ids = () if mesh.source_face_ids is None else tuple(sorted(set(map(int, mesh.source_face_ids[face_ids]))))
        raw_facets.append((float(polygon.area), polygon, _fit_plane(points), tuple(face_ids), source_ids))
    raw_facets.sort(key=lambda item: -item[0])
    return tuple(
        _roof_frame(f"facet_{index:02d}", polygon, plane, face_ids, source_ids)
        for index, (_area, polygon, plane, face_ids, source_ids) in enumerate(raw_facets)
    )


def receiver_exclusions(facet: RoofFacet, mesh: Mesh, mounting_clearance_m: float) -> BaseGeometry:
    """Exact full-mesh intersections above the physical mounted module plane."""
    from emboss.roof_surface import covered_footprint

    offset = mounting_clearance_m * facet.normal
    footprint = affinity.translate(facet.polygon_xy, offset[0], offset[1])
    covered = covered_footprint(footprint, facet.origin + offset, facet.normal, mesh.triangles)
    return facet.xy_to_uv(affinity.translate(covered, -offset[0], -offset[1]))


def _read_orthophoto(path: Path) -> tuple[np.ndarray, object]:
    try:
        import rasterio
    except ImportError as error:  # pragma: no cover - exercised only without geo extras
        raise RuntimeError("Module inference requires the project's geo dependencies") from error
    with rasterio.open(path) as dataset:
        if dataset.count < 3:
            raise ValueError(f"Expected an RGB orthophoto: {path}")
        rgb = np.moveaxis(dataset.read((1, 2, 3)), 0, -1)
        return np.asarray(rgb, dtype=np.uint8), dataset.transform


def _sample_orthophoto(
    facet: RoofFacet,
    u: np.ndarray,
    v: np.ndarray,
    rgb: np.ndarray,
    transform: object,
) -> tuple[np.ndarray, np.ndarray]:
    grid_u, grid_v = np.meshgrid(u, v)
    xyz = facet.uv_to_xyz(np.column_stack((grid_u.ravel(), grid_v.ravel())))
    inverse = ~transform
    columns = inverse.a * xyz[:, 0] + inverse.b * xyz[:, 1] + inverse.c
    rows = inverse.d * xyz[:, 0] + inverse.e * xyz[:, 1] + inverse.f
    columns = np.floor(columns).astype(np.int64)
    rows = np.floor(rows).astype(np.int64)
    valid = (
        (rows >= 0)
        & (rows < rgb.shape[0])
        & (columns >= 0)
        & (columns < rgb.shape[1])
    )
    sampled = np.zeros((len(rows), 3), dtype=np.float64)
    sampled[valid] = rgb[rows[valid], columns[valid]]
    gray = np.dot(sampled, np.asarray([0.299, 0.587, 0.114])).reshape(grid_u.shape)
    valid_image = valid.reshape(grid_u.shape)
    return gray, valid_image


def _observation(
    array_id: str,
    facet: RoofFacet,
    geometry_uv: BaseGeometry,
    rgb: np.ndarray,
    transform: object,
    allowed_uv: BaseGeometry,
    blocked_uv: BaseGeometry,
) -> _Observation:
    resolution = FIT_RESOLUTION_M
    min_u, min_v, max_u, max_v = geometry_uv.bounds
    padding = MAX_MODULE_HEIGHT_M
    start_u = math.floor((min_u - padding) / resolution) * resolution
    stop_u = math.ceil((max_u + padding) / resolution) * resolution
    start_v = math.floor((min_v - padding) / resolution) * resolution
    stop_v = math.ceil((max_v + padding) / resolution) * resolution
    u = np.arange(start_u + resolution / 2.0, stop_u, resolution)
    v = np.arange(start_v + resolution / 2.0, stop_v, resolution)
    grid_u, grid_v = np.meshgrid(u, v)
    mask = shapely.intersects_xy(geometry_uv, grid_u, grid_v)
    gray, valid = _sample_orthophoto(facet, u, v, rgb, transform)
    gradient_v, gradient_u = np.gradient(gray, resolution, resolution)
    gradient_u = np.abs(gradient_u)
    gradient_v = np.abs(gradient_v)
    magnitude = np.hypot(gradient_u[valid], gradient_v[valid])
    scale = float(np.percentile(magnitude, 95)) if magnitude.size else 1.0
    scale = max(scale, 1.0)
    return _Observation(
        array_id=array_id,
        facet=facet,
        geometry_uv=geometry_uv,
        u=u,
        v=v,
        mask=mask,
        gradient_u=np.where(valid, np.clip(gradient_u / scale, 0.0, 1.0), 0.0),
        gradient_v=np.where(valid, np.clip(gradient_v / scale, 0.0, 1.0), 0.0),
        resolution_m=resolution,
        allowed_uv=allowed_uv,
        blocked_uv=blocked_uv,
    )


def _phase_values(
    observation: _Observation,
    pitch: float,
    *,
    along_u: bool,
) -> tuple[float, ...]:
    lower, upper = (
        (observation.geometry_uv.bounds[0], observation.geometry_uv.bounds[2])
        if along_u
        else (observation.geometry_uv.bounds[1], observation.geometry_uv.bounds[3])
    )
    coordinates = observation.u if along_u else observation.v
    gradient = observation.gradient_u if along_u else observation.gradient_v
    mask = observation.mask
    phase_count = max(1, math.ceil(pitch / observation.resolution_m))
    phases = np.linspace(0.0, pitch, phase_count, endpoint=False)
    scores = []
    grid = coordinates[None, :]
    for phase in phases:
        origin = lower - phase
        distance = np.abs(((grid - origin + pitch / 2.0) % pitch) - pitch / 2.0)
        lines = distance <= observation.resolution_m * 0.55
        selected = mask & (lines if along_u else lines.T)
        scores.append(float(gradient[selected].mean()) if selected.any() else 0.0)
    best_image_phases = phases[np.argsort(scores)[-2:]]
    align_upper = (math.ceil((upper - lower) / pitch) * pitch - (upper - lower)) % pitch
    proposals = [0.0, float(align_upper), *(float(value) for value in best_image_phases)]
    unique = []
    for phase in proposals:
        if not any(abs(phase - existing) < observation.resolution_m * 0.5 for existing in unique):
            unique.append(phase)
    return tuple(unique)


def _mask_metrics(true_positive: float, predicted: float, observed: float) -> tuple[float, float, float]:
    precision = true_positive / predicted if predicted > 0 else 0.0
    recall = true_positive / observed if observed > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _boundary_score(
    observation: _Observation,
    origin_u: float,
    origin_v: float,
    cell_u: float,
    cell_v: float,
    cells: tuple[tuple[int, int, float], ...],
) -> tuple[float | None, int]:
    accepted = {(row, column) for row, column, _coverage in cells}
    samples = []
    for row, column in accepted:
        if (row, column + 1) in accepted:
            line_u = origin_u + (column + 1) * cell_u
            column_id = int(np.argmin(np.abs(observation.u - line_u)))
            low_v = origin_v + row * cell_v
            high_v = low_v + cell_v
            rows = (observation.v >= low_v) & (observation.v <= high_v)
            samples.extend(observation.gradient_u[rows, column_id].tolist())
        if (row + 1, column) in accepted:
            line_v = origin_v + (row + 1) * cell_v
            row_id = int(np.argmin(np.abs(observation.v - line_v)))
            low_u = origin_u + column * cell_u
            high_u = low_u + cell_u
            columns = (observation.u >= low_u) & (observation.u <= high_u)
            samples.extend(observation.gradient_v[row_id, columns].tolist())
    return (float(np.mean(samples)), len(samples)) if samples else (None, 0)


def _fit_phase(
    observation: _Observation,
    cell_u: float,
    cell_v: float,
    orientation: str,
    phase_u: float,
    phase_v: float,
) -> _PhaseFit:
    min_u, min_v, _max_u, _max_v = observation.geometry_uv.bounds
    origin_u = min_u - phase_u
    origin_v = min_v - phase_v
    rows, columns = np.nonzero(observation.mask)
    mask_u = observation.u[columns]
    mask_v = observation.v[rows]
    cell_columns = np.floor((mask_u - origin_u) / cell_u).astype(np.int64)
    cell_rows = np.floor((mask_v - origin_v) / cell_v).astype(np.int64)
    pairs = np.column_stack((cell_rows, cell_columns))
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    cell_area = cell_u * cell_v
    pixel_area = observation.resolution_m**2
    coverage = counts * pixel_area / cell_area
    accepted = coverage >= MIN_CELL_MASK_COVERAGE
    # Evaluate feasibility before choosing a phase, orientation, or module
    # size. Scoring first and deleting cells afterward can discard a whole
    # observed row even when a supported alternative fits it.
    low_u = origin_u + unique[:, 1] * cell_u
    low_v = origin_v + unique[:, 0] * cell_v
    rectangles = shapely.box(low_u, low_v, low_u + cell_u, low_v + cell_v)
    accepted &= shapely.covers(observation.allowed_uv, rectangles)
    accepted &= ~shapely.relate_pattern(rectangles, observation.blocked_uv, "T********")
    cells = tuple(
        (int(row), int(column), float(min(value, 1.0)))
        for (row, column), value in zip(unique[accepted], coverage[accepted], strict=True)
    )
    true_positive = min(
        float(np.sum(counts[accepted]) * pixel_area),
        float(observation.geometry_uv.area),
    )
    predicted = len(cells) * cell_area
    observed = float(observation.geometry_uv.area)
    precision, recall, f1 = _mask_metrics(true_positive, predicted, observed)
    boundary, boundary_samples = _boundary_score(
        observation,
        origin_u,
        origin_v,
        cell_u,
        cell_v,
        cells,
    )
    score = MASK_SCORE_WEIGHT * f1 + BOUNDARY_SCORE_WEIGHT * (boundary or 0.0)
    return _PhaseFit(
        orientation=orientation,
        cell_u_m=cell_u,
        cell_v_m=cell_v,
        origin_u=origin_u,
        origin_v=origin_v,
        cells=cells,
        true_positive_m2=true_positive,
        predicted_m2=predicted,
        mask_m2=observed,
        mask_precision=precision,
        mask_recall=recall,
        mask_f1=f1,
        boundary_evidence=boundary,
        boundary_samples=boundary_samples,
        score=score,
    )


def _fit_block(observation: _Observation, width: float, height: float) -> _PhaseFit:
    fits = []
    for orientation, cell_u, cell_v in (
        ("portrait", width, height),
        ("landscape", height, width),
    ):
        phases_u = _phase_values(observation, cell_u, along_u=True)
        phases_v = _phase_values(observation, cell_v, along_u=False)
        fits.extend(
            _fit_phase(
                observation,
                cell_u,
                cell_v,
                orientation,
                phase_u,
                phase_v,
            )
            for phase_u in phases_u
            for phase_v in phases_v
        )
    return max(fits, key=lambda fit: (fit.score, fit.mask_f1, -fit.predicted_m2))


def _fit_candidate(
    observations: tuple[_Observation, ...],
    width: float,
    height: float,
) -> _CandidateFit:
    blocks = tuple(_fit_block(observation, width, height) for observation in observations)
    true_positive = sum(block.true_positive_m2 for block in blocks)
    predicted = sum(block.predicted_m2 for block in blocks)
    observed = sum(block.mask_m2 for block in blocks)
    precision, recall, f1 = _mask_metrics(true_positive, predicted, observed)
    boundary_samples = sum(block.boundary_samples for block in blocks)
    boundary = (
        sum((block.boundary_evidence or 0.0) * block.boundary_samples for block in blocks)
        / boundary_samples
        if boundary_samples
        else None
    )
    score = MASK_SCORE_WEIGHT * f1 + BOUNDARY_SCORE_WEIGHT * (boundary or 0.0)
    return _CandidateFit(
        width_m=width,
        height_m=height,
        blocks=blocks,
        score=score,
        precision=precision,
        recall=recall,
        boundary_evidence=boundary,
        module_count=sum(len(block.cells) for block in blocks),
    )


def _size_values(start: float, stop: float) -> np.ndarray:
    count = round((stop - start) / MODULE_SIZE_STEP_M) + 1
    return np.round(np.linspace(start, stop, count), 2)


def _nearest_standard_size(
    width_m: float,
    height_m: float,
) -> tuple[float, float, float]:
    width, height = min(
        COMMON_MODULE_SIZE_FAMILIES_M,
        key=lambda size: math.hypot(width_m - size[0], height_m - size[1]),
    )
    return width, height, math.hypot(width_m - width, height_m - height)


def _select_candidate(
    fits: list[_CandidateFit],
) -> tuple[_CandidateFit, list[_CandidateFit]]:
    """Use common dimensions only to break near-ties in fit evidence."""
    if not fits:
        raise ValueError("No module-size candidates were evaluated")
    by_evidence = sorted(
        fits,
        key=lambda fit: (fit.score, fit.precision, fit.recall),
        reverse=True,
    )
    best_evidence = by_evidence[0].score
    near_ties = [
        fit
        for fit in by_evidence
        if best_evidence - fit.score <= SCORE_TIE_TOLERANCE + 1e-12
    ]
    selected = min(
        near_ties,
        key=lambda fit: (
            _nearest_standard_size(fit.width_m, fit.height_m)[2],
            -fit.score,
            -fit.precision,
            -fit.recall,
        ),
    )
    return selected, [selected, *(fit for fit in by_evidence if fit is not selected)]


def infer_module_layout(
    roof_mesh: Mesh,
    detected_footprint: BaseGeometry,
    orthophoto_path: Path,
    mounting_clearance_m: float,
    occluder_mesh: Mesh,
) -> ModuleLayout:
    """Fit one physical module size to the union PV mask across all roof facets."""
    if mounting_clearance_m < 0:
        raise ValueError("Module mounting clearance cannot be negative")
    footprint = shapely.make_valid(detected_footprint)
    facets = roof_facets(roof_mesh)
    if not facets:
        raise ValueError("Cannot infer PV modules without roof facets")
    rgb, transform = _read_orthophoto(orthophoto_path)

    observations = []
    detected_surface_area = 0.0
    for facet in facets:
        detected_on_facet = footprint.intersection(facet.polygon_xy)
        detected_uv = shapely.make_valid(facet.xy_to_uv(detected_on_facet))
        detected_surface_area += float(detected_uv.area)
        allowed_uv = facet.xy_to_uv(facet.polygon_xy).buffer(INSTALLED_SUPPORT_TOLERANCE_M)
        blocked_uv = receiver_exclusions(facet, occluder_mesh, mounting_clearance_m)
        shapely.prepare(allowed_uv)
        for part in sorted(_polygon_parts(detected_uv), key=lambda item: tuple(item.bounds)):
            if part.area <= 1e-4:
                continue
            observations.append(
                _observation(
                    f"array_{len(observations):02d}",
                    facet,
                    part,
                    rgb,
                    transform,
                    allowed_uv,
                    blocked_uv,
                )
            )
    if not observations:
        raise ValueError("The union PV mask does not overlap the main roof")
    observations_tuple = tuple(observations)

    fits = [
        _fit_candidate(observations_tuple, float(width), float(height))
        for width in _size_values(MIN_MODULE_WIDTH_M, MAX_MODULE_WIDTH_M)
        for height in _size_values(MIN_MODULE_HEIGHT_M, MAX_MODULE_HEIGHT_M)
    ]
    best, fits = _select_candidate(fits)
    best_evidence_score = max(fit.score for fit in fits)
    standard_width, standard_height, standard_distance = _nearest_standard_size(
        best.width_m,
        best.height_m,
    )

    cells = []
    arrays = []
    exact_true_positive = exact_predicted = exact_observed = 0.0
    module_index = 0
    for observation, block in zip(observations_tuple, best.blocks, strict=True):
        cell_polygons_uv = []
        accepted_rows = [row for row, _column, _coverage in block.cells]
        accepted_columns = [column for _row, column, _coverage in block.cells]
        row_offset = min(accepted_rows, default=0)
        column_offset = min(accepted_columns, default=0)
        for row, column, coverage in block.cells:
            min_u = block.origin_u + column * block.cell_u_m
            min_v = block.origin_v + row * block.cell_v_m
            corners_uv = np.asarray(
                [
                    [min_u, min_v],
                    [min_u + block.cell_u_m, min_v],
                    [min_u + block.cell_u_m, min_v + block.cell_v_m],
                    [min_u, min_v + block.cell_v_m],
                ]
            )
            roof_corners_xyz = observation.facet.uv_to_xyz(corners_uv)
            receiver_corners_xyz = (
                roof_corners_xyz
                + mounting_clearance_m * observation.facet.normal[None, :]
            )
            cell_polygons_uv.append(Polygon(corners_uv))
            cells.append(
                ModuleCell(
                    module_id=f"module_{module_index:03d}",
                    array_id=observation.array_id,
                    facet_id=observation.facet.facet_id,
                    row=row - row_offset,
                    column=column - column_offset,
                    orientation=block.orientation,
                    mask_coverage=coverage,
                    roof_corners_xyz=roof_corners_xyz,
                    receiver_corners_xyz=receiver_corners_xyz,
                )
            )
            module_index += 1
        inferred_uv = unary_union(cell_polygons_uv) if cell_polygons_uv else Polygon()
        true_positive = float(inferred_uv.intersection(observation.geometry_uv).area)
        predicted = float(inferred_uv.area)
        observed = float(observation.geometry_uv.area)
        precision, recall, _f1 = _mask_metrics(true_positive, predicted, observed)
        exact_true_positive += true_positive
        exact_predicted += predicted
        exact_observed += observed
        arrays.append(
            ModuleArray(
                array_id=observation.array_id,
                facet_id=observation.facet.facet_id,
                orientation=block.orientation,
                module_count=len(cell_polygons_uv),
                detected_area_m2=observed,
                inferred_area_m2=predicted,
                mask_precision=precision,
                mask_recall=recall,
                boundary_evidence=block.boundary_evidence,
                roof_tilt_deg=observation.facet.tilt_deg,
                axis_source=observation.facet.axis_source,
                axis_confidence=observation.facet.axis_confidence,
            )
        )

    precision, recall, _f1 = _mask_metrics(
        exact_true_positive,
        exact_predicted,
        exact_observed,
    )
    projected = unary_union(
        [Polygon(cell.receiver_corners_xyz[:, :2]) for cell in cells]
    ) if cells else Polygon()
    candidates = []
    for fit in fits:
        nearest_width, nearest_height, distance = _nearest_standard_size(
            fit.width_m,
            fit.height_m,
        )
        candidates.append(
            ModuleSizeCandidate(
                width_m=fit.width_m,
                height_m=fit.height_m,
                evidence_score=fit.score,
                mask_precision=fit.precision,
                mask_recall=fit.recall,
                boundary_evidence=fit.boundary_evidence,
                module_count=fit.module_count,
                nearest_standard_width_m=nearest_width,
                nearest_standard_height_m=nearest_height,
                standard_distance_m=distance,
                within_score_tie=(
                    best_evidence_score - fit.score
                    <= SCORE_TIE_TOLERANCE + 1e-12
                ),
                selected=fit is best,
            )
        )
    return ModuleLayout(
        module_width_m=best.width_m,
        module_height_m=best.height_m,
        mounting_clearance_m=mounting_clearance_m,
        cells=tuple(cells),
        arrays=tuple(arrays),
        candidates=tuple(candidates),
        detected_surface_area_m2=detected_surface_area,
        inferred_surface_area_m2=len(cells) * best.width_m * best.height_m,
        projected_area_m2=float(projected.area),
        mask_precision=precision,
        mask_recall=recall,
        boundary_evidence=best.boundary_evidence,
        evidence_score=best.score,
        evidence_gap_to_best=best_evidence_score - best.score,
        nearest_standard_width_m=standard_width,
        nearest_standard_height_m=standard_height,
        standard_distance_m=standard_distance,
        score_tie_tolerance=SCORE_TIE_TOLERANCE,
        orthophoto_path=orthophoto_path,
    )


def write_module_layout(layout: ModuleLayout, output_dir: Path) -> None:
    features = []
    for cell in layout.cells:
        coordinates = cell.receiver_corners_xyz.tolist()
        coordinates.append(coordinates[0])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "module_id": cell.module_id,
                    "array_id": cell.array_id,
                    "facet_id": cell.facet_id,
                    "row": cell.row,
                    "column": cell.column,
                    "orientation": cell.orientation,
                    "module_width_m": layout.module_width_m,
                    "module_height_m": layout.module_height_m,
                    "mounting_clearance_m": layout.mounting_clearance_m,
                    "mask_coverage": cell.mask_coverage,
                },
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
            }
        )
    (output_dir / "panel_modules.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n"
    )
    summary = {
        "assumption": (
            "one module size per house; receiver planes are parallel to their host "
            "roof facet and offset along its normal by the assumed mounting clearance"
        ),
        "fit_configuration": {
            "module_width_range_m": [MIN_MODULE_WIDTH_M, MAX_MODULE_WIDTH_M],
            "module_height_range_m": [MIN_MODULE_HEIGHT_M, MAX_MODULE_HEIGHT_M],
            "module_size_step_m": MODULE_SIZE_STEP_M,
            "raster_resolution_m": FIT_RESOLUTION_M,
            "minimum_cell_mask_coverage": MIN_CELL_MASK_COVERAGE,
            "mask_score_weight": MASK_SCORE_WEIGHT,
            "module_boundary_score_weight": BOUNDARY_SCORE_WEIGHT,
            "score_tie_tolerance": SCORE_TIE_TOLERANCE,
            "common_module_size_families_m": COMMON_MODULE_SIZE_FAMILIES_M,
            "common_size_rule": (
                "choose the candidate nearest a common size among candidates within "
                "the score tie tolerance of the best evidence score"
            ),
            "facet_normal_tolerance_degrees": FACET_NORMAL_TOLERANCE_DEG,
            "support_matching_tolerance_m": INSTALLED_SUPPORT_TOLERANCE_M,
            "feasibility_checked_before_scoring": True,
        },
        "module_width_m": layout.module_width_m,
        "module_height_m": layout.module_height_m,
        "mounting_clearance_m": layout.mounting_clearance_m,
        "module_count": len(layout.cells),
        "detected_surface_area_m2": layout.detected_surface_area_m2,
        "inferred_surface_area_m2": layout.inferred_surface_area_m2,
        "projected_area_m2": layout.projected_area_m2,
        "mask_precision": layout.mask_precision,
        "mask_recall": layout.mask_recall,
        "boundary_evidence": layout.boundary_evidence,
        "evidence_score": layout.evidence_score,
        "evidence_gap_to_best": layout.evidence_gap_to_best,
        "nearest_standard_width_m": layout.nearest_standard_width_m,
        "nearest_standard_height_m": layout.nearest_standard_height_m,
        "standard_distance_m": layout.standard_distance_m,
        "arrays": [asdict(array) for array in layout.arrays],
    }
    (output_dir / "module_layout.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([asdict(candidate) for candidate in layout.candidates]).to_csv(
        output_dir / "module_size_candidates.csv",
        index=False,
    )
