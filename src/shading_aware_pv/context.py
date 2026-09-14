from __future__ import annotations

from pathlib import Path

import numpy as np
import pyogrio
import shapely
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from .models import Mesh

GROUND_CLASS = 2
VEGETATION_CLASS = 3
BUILDING_CLASS = 6
CONTEXT_CLASSES = (GROUND_CLASS, VEGETATION_CLASS, BUILDING_CLASS)
TARGET_LAYERS = ("Roof", "Floor")


def _polygon_parts(geometry) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        return [part for item in geometry.geoms for part in _polygon_parts(item)]
    return []


def load_target_footprint(
    surfaces_path: Path,
    *,
    target_building_id: int,
    center_xy: tuple[float, float],
    search_half_extent_m: float,
    buffer_m: float,
) -> BaseGeometry:
    """Return the projected target roof/floor footprint used to remove LiDAR."""
    min_x = center_xy[0] - search_half_extent_m
    min_y = center_xy[1] - search_half_extent_m
    max_x = center_xy[0] + search_half_extent_m
    max_y = center_xy[1] + search_half_extent_m
    geometries: list[BaseGeometry] = []
    for layer in TARGET_LAYERS:
        frame = pyogrio.read_dataframe(
            surfaces_path,
            layer=layer,
            bbox=(min_x, min_y, max_x, max_y),
            columns=[],
            fid_as_index=True,
        )
        if target_building_id not in frame.index:
            continue
        selected = frame.loc[[target_building_id]]
        geometries.extend(selected.geometry.tolist())
    if not geometries:
        raise ValueError(
            f"Building {target_building_id} is absent from Roof/Floor layers in "
            f"{surfaces_path}"
        )
    return shapely.union_all(geometries).buffer(buffer_m)


def _raster_max(
    points: np.ndarray,
    *,
    bounds_xy: tuple[float, float, float, float],
    shape: tuple[int, int],
    resolution_m: float,
) -> np.ndarray:
    rows, columns = shape
    values = np.full(rows * columns, -np.inf, dtype=np.float64)
    if not len(points):
        return values.reshape(shape)
    min_x, min_y, _, _ = bounds_xy
    column = np.floor((points[:, 0] - min_x) / resolution_m).astype(int)
    row = np.floor((points[:, 1] - min_y) / resolution_m).astype(int)
    column = np.clip(column, 0, columns - 1)
    row = np.clip(row, 0, rows - 1)
    np.maximum.at(values, row * columns + column, points[:, 2])
    return values.reshape(shape)


def build_context_dsm(
    points: np.ndarray,
    classifications: np.ndarray,
    *,
    bounds_xy: tuple[float, float, float, float],
    resolution_m: float,
    target_footprint: BaseGeometry,
    excluded_surface_classes: tuple[int, ...] = (),
) -> tuple[Mesh, tuple[int, int], np.ndarray]:
    """Build a continuous height-field mesh after removing the target building."""
    if resolution_m <= 0:
        raise ValueError("DSM resolution must be positive")
    points = np.asarray(points, dtype=np.float64)
    classifications = np.asarray(classifications)
    if points.shape != (len(classifications), 3):
        raise ValueError("Points must have shape (N, 3) and one class per point")

    min_x, min_y, max_x, max_y = bounds_xy
    columns = int(np.ceil((max_x - min_x) / resolution_m))
    rows = int(np.ceil((max_y - min_y) / resolution_m))
    shape = (rows, columns)

    inside_target = shapely.contains_xy(
        target_footprint,
        points[:, 0],
        points[:, 1],
    )
    retained = ~((classifications == BUILDING_CLASS) & inside_target)
    ground = _raster_max(
        points[classifications == GROUND_CLASS],
        bounds_xy=bounds_xy,
        shape=shape,
        resolution_m=resolution_m,
    )
    valid_ground = np.isfinite(ground)
    if not valid_ground.any():
        raise ValueError("Cannot build a context DSM without class-2 ground points")
    if not valid_ground.all():
        nearest = distance_transform_edt(
            ~valid_ground,
            return_distances=False,
            return_indices=True,
        )
        ground = ground[tuple(nearest)]

    surface_points = retained & ~np.isin(
        classifications,
        excluded_surface_classes,
    )
    surface = _raster_max(
        points[surface_points],
        bounds_xy=bounds_xy,
        shape=shape,
        resolution_m=resolution_m,
    )
    heights = np.where(np.isfinite(surface), np.maximum(surface, ground), ground)

    x = min_x + (np.arange(columns) + 0.5) * resolution_m
    y = min_y + (np.arange(rows) + 0.5) * resolution_m
    grid_x, grid_y = np.meshgrid(x, y)
    vertices = np.column_stack((grid_x.ravel(), grid_y.ravel(), heights.ravel()))
    indices = np.arange(rows * columns, dtype=np.int32).reshape(shape)
    lower_left = indices[:-1, :-1].ravel()
    lower_right = indices[:-1, 1:].ravel()
    upper_left = indices[1:, :-1].ravel()
    upper_right = indices[1:, 1:].ravel()
    faces = np.vstack(
        (
            np.column_stack((lower_left, lower_right, upper_right)),
            np.column_stack((lower_left, upper_right, upper_left)),
        )
    ).astype(np.int32, copy=False)
    return Mesh(vertices, faces), shape, retained


def _footprint_rings(geometry: BaseGeometry) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(polygon.exterior.coords, dtype=np.float64)[:, :2]
        for polygon in _polygon_parts(geometry)
    )
