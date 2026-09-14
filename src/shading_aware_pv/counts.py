"""Prepare an immutable saved report's panel-count variants from cached inputs."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from .geometry import read_ascii_ply
from .inputs import SimulationInputs
from .models import Weather
from .simulation import METHOD_REVISION, SimulationConfig, simulate
from .snow import SnowData, load_snow_data


def _cached_file(root: Path, path: Path) -> Path:
    try:
        parts = path.absolute().relative_to(root.absolute()).parts
    except ValueError as error:
        raise ValueError("Saved analysis input is outside the application data directory") from error
    if ".." in parts:
        raise ValueError("Cached input path contains parent traversal")
    if any(root.joinpath(*parts[:i]).is_symlink() for i in range(1, len(parts) + 1)):
        raise ValueError("Saved analysis inputs cannot be symbolic links")
    if not path.is_file():
        raise FileNotFoundError(f"Cached input is missing: {path.name}. Run a new analysis to prepare it.")
    return path


def _cached_weather(root, directory, year, *, horizon):
    name = "weather" if horizon else "weather_no_horizon"
    metadata = json.loads(_cached_file(root, directory / f"{name}_meta.json").read_text())
    if metadata.get("year") != year or metadata.get("use_horizon", True) != horizon:
        raise ValueError("Cached weather does not match this report's year or horizon setting")
    hourly = pd.read_csv(_cached_file(root, directory / f"{name}.csv"), index_col="time", parse_dates=["time"])
    return Weather(hourly, metadata["latitude"], metadata["longitude"], metadata["elevation"], metadata["source"])


def _triangle_signature(triangles):
    return sorted(tuple(sorted(map(tuple, triangle))) for triangle in np.round(triangles, 6))


def _assert_same_baseline(saved, prepared):
    """Detect changed retained inputs when no method migration is expected."""
    def compare(left, right, field):
        if isinstance(left, dict):
            if not isinstance(right, dict) or any(key not in right for key in left):
                raise ValueError(field)
            for key, value in left.items():
                compare(value, right[key], f"{field}.{key}")
        elif isinstance(left, list):
            if len(left) != len(right):
                raise ValueError(field)
            if not left:
                return
            try:
                a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
                if a.shape != b.shape or not np.allclose(a, b, rtol=1e-10, atol=1e-8):
                    raise ValueError(field)
            except (TypeError, ValueError):
                for index, (a, b) in enumerate(zip(left, right, strict=True)):
                    compare(a, b, f"{field}[{index}]")
        elif isinstance(left, (int, float)) and not isinstance(left, bool):
            if not np.isclose(left, right, rtol=1e-10, atol=1e-8):
                raise ValueError(field)
        elif left != right:
            raise ValueError(field)
    try:
        for field in ("module_dimensions_m", "scene", "designs"):
            compare(saved[field], prepared[field], field)
        for field in ("enabled", "coverage_complete", "lidar_sources"):
            if field in saved["context"]:
                compare(saved["context"][field], prepared["context"][field], f"context.{field}")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Retained inputs no longer reproduce this report. Start a new analysis before adjusting its panel count.") from error


def prepare_count_report(root: Path, job: dict, report: dict, *, progress):
    """Recompute using the retained reconstruction and observations, never refetch.

    A changed or missing reconstruction fails explicitly. The caller writes a
    separate prepared result; the original report and imagery remain untouched.
    """
    from emboss.api import Client
    from building_data.points import read_las_points

    reconstruction = Client(root).load_result(job["house_id"])
    for path in (reconstruction.mesh_path, reconstruction.roof_details_path,
                 reconstruction.scaffold_path, reconstruction.orthophoto_path):
        _cached_file(root, path)
    mesh = read_ascii_ply(reconstruction.mesh_path)
    saved = report["scene"]
    saved_triangles = np.asarray(saved["vertices"])[np.asarray(saved["faces"], dtype=int)]
    if _triangle_signature(mesh.triangles) != _triangle_signature(saved_triangles):
        raise ValueError("The building reconstruction has changed since this report. Start a new analysis to use it.")
    config = SimulationConfig(**job["config"])
    cache = root / "solar/weather" / job["house_id"] / str(config.year)
    weather = _cached_weather(root, cache, config.year, horizon=True)
    open_weather = _cached_weather(root, cache, config.year, horizon=False)
    if len(weather.hourly) != report["weather_hours"]:
        raise ValueError("Cached weather period differs from the saved report")
    saved_timestamps = next((d["hourly"]["timestamps"] for d in report["designs"].values() if "hourly" in d), None)
    if saved_timestamps is not None and not pd.DatetimeIndex(saved_timestamps).equals(weather.hourly.iloc[:config.max_hours].index.floor("h")):
        raise ValueError("Cached weather timestamps differ from the saved report")
    metadata = report["snow"]
    if metadata.get("available"):
        decade = config.year // 10 * 10
        filename = f"ogd-smn_{metadata['station_id'].lower()}_h_historical_{decade}-{decade + 9}.csv"
        station = _cached_file(root, cache / filename)
        snow = load_snow_data(weather, {**metadata, "cached_file": str(station)})
        if not snow.available:
            raise ValueError("Cached snow observations no longer match this report")
    else:
        snow = SnowData(None, None, metadata)
    sources = report["context"].get("lidar_sources", [])

    def context_points(bounds, reference_year):
        if not sources or not unary_union([box(*s["bounds"]) for s in sources]).buffer(.02).covers(box(*bounds)):
            raise ValueError("Saved LiDAR tiles do not cover the analysis extent. Start a new analysis to prepare the missing coverage.")
        frames = []
        for source in sources:
            path = _cached_file(root, Path(source["path"]))
            frame = read_las_points(path, bounds_xy=bounds)
            x0, y0, x1, y1 = source["bounds"]
            frames.append(frame.loc[(frame.x >= x0) & (frame.x < x1) & (frame.y >= y0) & (frame.y < y1)])
        return pd.concat(frames, ignore_index=True), sources

    inputs = SimulationInputs.from_reconstruction(reconstruction, cache, context_cache_dir=root / "cache")
    result = simulate(inputs, replace(config, refresh_weather=False), weather=weather,
                      open_horizon_weather=open_weather, snow=snow,
                      context_points_supplier=context_points, progress=progress)
    if report.get("provenance", {}).get("method_revision") == METHOD_REVISION:
        _assert_same_baseline(report, result)
    # Keep the user's saved request, including its historical refresh preference.
    result["config"] = job["config"]
    result["count_preparation"] = {
        "source_job_id": job["id"], "cached_inputs": True,
        "source_method_revision": report.get("provenance", {}).get("method_revision"),
        "method_migration": report.get("provenance", {}).get("method_revision") != METHOD_REVISION,
    }
    return result
