"""Explicit reconstruction inputs, independent of experiment directory conventions."""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import laspy
import numpy as np

from .context import (
    CONTEXT_CLASSES,
    VEGETATION_CLASS,
    _footprint_rings,
    build_context_dsm,
    load_target_footprint,
)
from .models import ContextScene


@dataclass(frozen=True)
class SimulationInputs:
    mesh_path: Path
    roof_details_path: Path
    orthophoto_path: Path
    surfaces_path: Path
    building_fid: int
    cache_dir: Path
    scaffold_path: Path
    lidar_paths: tuple[Path, ...] = ()
    survey_year: int | None = None
    sources: tuple[dict, ...] = ()
    context_cache_dir: Path | None = None

    @classmethod
    def from_reconstruction(
        cls, reconstruction, cache_dir: Path, context_cache_dir: Path | None = None
    ):
        house = reconstruction.house_input
        return cls(
            reconstruction.mesh_path,
            reconstruction.roof_details_path,
            reconstruction.orthophoto_path,
            house.surfaces_path,
            house.building_fid,
            cache_dir,
            reconstruction.scaffold_path,
            house.lidar_paths,
            house.survey_year,
            house.sources,
            context_cache_dir,
        )


def load_context_scene(
    inputs: SimulationInputs,
    *,
    center_xy: tuple[float, float],
    half_extent_m: float,
    grid_resolution_m: float,
    points_supplier: Callable | None = None,
) -> ContextScene:
    if half_extent_m <= 0:
        raise ValueError("context half-extent must be positive")
    footprint = load_target_footprint(
        inputs.surfaces_path,
        target_building_id=inputs.building_fid,
        center_xy=center_xy,
        search_half_extent_m=half_extent_m,
        buffer_m=grid_resolution_m / 2.0,
    )
    cx, cy = center_xy
    bounds = (
        cx - half_extent_m,
        cy - half_extent_m,
        cx + half_extent_m,
        cy + half_extent_m,
    )
    if points_supplier is None:
        from building_data.swiss import SwissProvider

        provider = SwissProvider(
            inputs.context_cache_dir or inputs.cache_dir / "context"
        )
        pinned = inputs.sources
        if not pinned and inputs.lidar_paths:
            local_sources = []
            for path in inputs.lidar_paths:
                with laspy.open(path) as reader:
                    x, y = np.floor(reader.header.mins[:2] / 1000).astype(int) * 1000
                    if np.any(reader.header.maxs[:2] > [x + 1000.02, y + 1000.02]):
                        raise ValueError(
                            "Explicit LiDAR paths must each contain one Swiss 1 km tile; supply a context point supplier for other extents"
                        )
                local_sources.append(
                    {
                        "tile": path.stem,
                        "path": str(path),
                        "year": inputs.survey_year,
                        "bounds": [int(x), int(y), int(x + 1000), int(y + 1000)],
                    }
                )
            pinned = tuple(local_sources)

        def points_supplier(bounds, reference_year):
            return provider.points_for_bounds(
                bounds, reference_year, pinned_sources=pinned
            )

    frame, sources = points_supplier(bounds, reference_year=inputs.survey_year)
    frame = frame.loc[frame.classification.isin(CONTEXT_CLASSES)]
    checked_sources = []
    for source in sources:
        x0, y0, x1, y1 = source["bounds"]
        count = int(
            ((frame.x >= x0) & (frame.x < x1) & (frame.y >= y0) & (frame.y < y1)).sum()
        )
        if not count:
            raise ValueError(
                f"No usable context LiDAR returns in required tile {source['tile']}"
            )
        checked_sources.append({**source, "point_count": count})
    sources = checked_sources
    frame = frame.drop_duplicates(["x", "y", "z", "classification"])
    points = frame[["x", "y", "z"]].to_numpy()
    classes = frame.classification.to_numpy(dtype=np.uint8)
    mesh, shape, retained = build_context_dsm(
        points,
        classes,
        bounds_xy=bounds,
        resolution_m=grid_resolution_m,
        target_footprint=footprint,
    )
    bare, _, _ = build_context_dsm(
        points,
        classes,
        bounds_xy=bounds,
        resolution_m=grid_resolution_m,
        target_footprint=footprint,
        excluded_surface_classes=(VEGETATION_CLASS,),
    )
    return ContextScene(
        mesh,
        bare,
        shape,
        grid_resolution_m,
        points[retained],
        classes[retained],
        _footprint_rings(footprint),
        int((~retained).sum()),
        center_xy,
        half_extent_m,
        inputs.surfaces_path,
        tuple(sources),
        True,
    )
