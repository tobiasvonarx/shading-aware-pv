from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .models import Mesh, MeshPartition, RoofSamples, RoofScene


def read_ascii_ply(path: Path) -> Mesh:
    """Read the small ASCII PLY meshes emitted by Emboss."""
    with path.open() as handle:
        if handle.readline().strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        vertex_count = face_count = None
        is_ascii = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            fields = line.strip().split()
            if fields[:2] == ["format", "ascii"]:
                is_ascii = True
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            elif fields[:2] == ["element", "face"]:
                face_count = int(fields[2])
            elif fields == ["end_header"]:
                break
        if not is_ascii or vertex_count is None or face_count is None:
            raise ValueError(
                f"Only ASCII PLY meshes with vertices and faces are supported: {path}"
            )
        vertices = np.array(
            [
                [float(value) for value in handle.readline().split()[:3]]
                for _ in range(vertex_count)
            ],
            dtype=np.float64,
        )
        faces: list[tuple[int, int, int]] = []
        for _ in range(face_count):
            fields = [int(value) for value in handle.readline().split()]
            polygon = fields[1 : fields[0] + 1]
            faces.extend(
                (polygon[0], polygon[index], polygon[index + 1])
                for index in range(1, len(polygon) - 1)
            )
    return Mesh(vertices, np.asarray(faces, dtype=np.int32))


def _empty_mesh() -> Mesh:
    return Mesh(
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.int32),
        np.empty(0, dtype=np.int64),
    )


def _submesh(mesh: Mesh, face_indices: np.ndarray) -> Mesh:
    faces = mesh.faces[np.asarray(face_indices, dtype=np.int64)]
    if not len(faces):
        return _empty_mesh()
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    source_ids = np.arange(len(mesh.faces)) if mesh.source_face_ids is None else mesh.source_face_ids
    return Mesh(mesh.vertices[used], inverse.reshape(-1, 3).astype(np.int32), source_ids[face_indices])


def mesh_footprint(
    mesh: Mesh,
    face_indices: np.ndarray | None = None,
) -> BaseGeometry:
    """Return the union of non-degenerate triangle projections in map coordinates."""
    triangles = (
        mesh.triangles
        if face_indices is None
        else mesh.triangles[np.asarray(face_indices, dtype=np.int64)]
    )
    polygons = [
        polygon
        for triangle in triangles
        if (polygon := Polygon(triangle[:, :2])).area > 1e-8
    ]
    return unary_union(polygons) if polygons else Polygon()


def connected_face_components(
    mesh: Mesh,
    weld_tolerance: float = 1e-5,
) -> list[np.ndarray]:
    """Return face components after welding geometrically coincident vertices."""
    if not len(mesh.faces):
        return []
    if weld_tolerance <= 0:
        raise ValueError("weld_tolerance must be positive")

    keys = np.rint(mesh.vertices / weld_tolerance).astype(np.int64)
    welded_ids: dict[tuple[int, int, int], int] = {}
    canonical = np.empty(len(keys), dtype=np.int64)
    for index, key in enumerate(map(tuple, keys)):
        canonical[index] = welded_ids.setdefault(key, len(welded_ids))

    faces_by_vertex: dict[int, list[int]] = defaultdict(list)
    for face_index, face in enumerate(mesh.faces):
        for vertex_id in set(canonical[face]):
            faces_by_vertex[int(vertex_id)].append(face_index)

    neighbors: list[set[int]] = [set() for _ in mesh.faces]
    for incident_faces in faces_by_vertex.values():
        for face_index in incident_faces:
            neighbors[face_index].update(incident_faces)

    components: list[np.ndarray] = []
    unseen = set(range(len(mesh.faces)))
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        component = []
        while stack:
            face_index = stack.pop()
            component.append(face_index)
            adjacent = neighbors[face_index] & unseen
            unseen.difference_update(adjacent)
            stack.extend(adjacent)
        components.append(np.asarray(component, dtype=np.int32))
    return components


def partition_mesh(mesh: Mesh, *, face_labels: np.ndarray) -> MeshPartition:
    """Separate semantic shading meshes and the exposed, eligible roof surface.

    Labels come from Emboss's scaffold and fitted solids, never connectivity.
    PV is omitted from the support envelope; other detail tops occlude the roof
    but only scaffold surfaces can support new modules.
    """
    from emboss.roof_surface import exposed_roof

    labels = np.asarray(face_labels)
    if labels.shape != (len(mesh.faces),):
        raise ValueError("Every mesh triangle requires an explicit semantic label")
    non_pv_ids = np.flatnonzero(labels != "pvmodule")
    non_pv = _submesh(mesh, non_pv_ids)
    # At equal height a semantic obstruction (for example a flush window)
    # takes priority over a support surface, independent of input face order.
    support = labels[non_pv_ids] == "scaffold"
    order = np.argsort(support, kind="stable")
    exposed = exposed_roof(non_pv.vertices, non_pv.faces[order])
    eligible = support[order][exposed.source_face_ids]
    triangles = exposed.triangles[eligible]
    roof = Mesh(
        triangles.reshape((-1, 3)),
        np.arange(len(triangles) * 3, dtype=np.int32).reshape((-1, 3)),
        non_pv.source_face_ids[order][exposed.source_face_ids][eligible],
    )
    return MeshPartition(
        base_roof=_submesh(mesh, np.flatnonzero(np.isin(labels, ["scaffold", "scaffold_body"]))),
        main_roof=roof,
        details=_submesh(mesh, np.flatnonzero(~np.isin(labels, ["scaffold", "scaffold_body", "pvmodule"]))),
        component_count=len(connected_face_components(mesh)),
        full_mesh=mesh,
        non_pv_mesh=non_pv,
        pv_component_count=len(connected_face_components(_submesh(mesh, np.flatnonzero(labels == "pvmodule")))),
    )


def sample_roof(mesh: Mesh, spacing: float) -> RoofSamples:
    """Sample the upper envelope of one mesh's intrinsic main roof."""
    if spacing <= 0:
        raise ValueError("sample spacing must be positive")
    if not len(mesh.faces):
        raise ValueError("Main roof has no triangles to sample")

    triangles = mesh.triangles
    bounds_min = triangles[:, :, :2].min(axis=(0, 1))
    bounds_max = triangles[:, :, :2].max(axis=(0, 1))
    xs = np.arange(
        np.floor(bounds_min[0] / spacing) * spacing,
        bounds_max[0] + spacing / 2,
        spacing,
    )
    ys = np.arange(
        np.floor(bounds_min[1] / spacing) * spacing,
        bounds_max[1] + spacing / 2,
        spacing,
    )
    grid_x, grid_y = np.meshgrid(xs, ys)
    xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    heights = np.full(len(xy), -np.inf, dtype=np.float64)
    normals = np.zeros((len(xy), 3), dtype=np.float64)
    face_ids = np.full(len(xy), -1, dtype=np.int32)
    for face_id, triangle in enumerate(triangles):
        projected = triangle[:, :2]
        edge_1 = projected[1] - projected[0]
        edge_2 = projected[2] - projected[0]
        denominator = edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
        if abs(denominator) < 1e-10:
            continue

        relative = xy - projected[0]
        u = (
            relative[:, 0] * edge_2[1] - relative[:, 1] * edge_2[0]
        ) / denominator
        v = (
            edge_1[0] * relative[:, 1] - edge_1[1] * relative[:, 0]
        ) / denominator
        inside = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1.0 + 1e-8)
        z = (
            (1.0 - u - v) * triangle[0, 2]
            + u * triangle[1, 2]
            + v * triangle[2, 2]
        )
        use = inside & (z > heights)
        if not use.any():
            continue

        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        if normal[2] < 0:
            normal = -normal
        heights[use] = z[use]
        normals[use] = normal
        face_ids[use] = face_id

    sampled = np.isfinite(heights)
    if not sampled.any():
        raise ValueError("Main-roof sampling produced no points")
    points = np.column_stack((xy[sampled], heights[sampled]))
    sampled_normals = normals[sampled]
    areas = spacing**2 / sampled_normals[:, 2]
    return RoofSamples(points, sampled_normals, areas, face_ids[sampled])


def raised_detail_coverage(
    samples: RoofSamples,
    details: Mesh,
    clearance: float,
) -> np.ndarray:
    covered_by_detail = np.zeros(len(samples.points), dtype=bool)
    for triangle in details.triangles:
        xy = triangle[:, :2]
        edge_1 = xy[1] - xy[0]
        edge_2 = xy[2] - xy[0]
        denominator = edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
        if abs(denominator) < 1e-8:
            continue
        bounds_min, bounds_max = xy.min(axis=0), xy.max(axis=0)
        candidates = np.flatnonzero(
            (~covered_by_detail)
            & (samples.points[:, 0] >= bounds_min[0])
            & (samples.points[:, 0] <= bounds_max[0])
            & (samples.points[:, 1] >= bounds_min[1])
            & (samples.points[:, 1] <= bounds_max[1])
        )
        if not len(candidates):
            continue
        relative = samples.points[candidates, :2] - xy[0]
        u = (
            relative[:, 0] * edge_2[1] - relative[:, 1] * edge_2[0]
        ) / denominator
        v = (
            edge_1[0] * relative[:, 1] - edge_1[1] * relative[:, 0]
        ) / denominator
        inside = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1.0 + 1e-8)
        if not inside.any():
            continue
        weights = np.column_stack((1.0 - u - v, u, v))
        surface_z = weights @ triangle[:, 2]
        raised = inside & (
            surface_z > samples.points[candidates, 2] + clearance
        )
        covered_by_detail[candidates[raised]] = True
    return covered_by_detail


def load_roof_scene(
    mesh_path: Path,
    *,
    sample_spacing: float,
    detail_clearance: float,
    roof_details_path: Path,
    scaffold_path: Path,
) -> RoofScene:
    full_mesh = read_ascii_ply(mesh_path)
    from emboss.surface_semantics import classify_faces

    labels = classify_faces(full_mesh.vertices, full_mesh.faces, roof_details_path, scaffold_path)
    partition = partition_mesh(full_mesh, face_labels=labels)
    samples = sample_roof(partition.main_roof, sample_spacing)
    return RoofScene(
        source_path=mesh_path,
        partition=partition,
        samples=samples,
        excluded_by_raised_detail=raised_detail_coverage(
            samples,
            partition.details,
            detail_clearance,
        ),
    )
