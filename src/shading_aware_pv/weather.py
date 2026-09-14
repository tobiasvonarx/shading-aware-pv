from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from pyproj import Transformer

from .models import Weather

PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_3/"
PVGIS_DATABASE = "PVGIS-SARAH3"


def roof_location(points: np.ndarray) -> tuple[float, float, float]:
    east, north, elevation = points.mean(axis=0)
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(east, north)
    return float(latitude), float(longitude), float(elevation)


def fetch_weather(
    points: np.ndarray,
    year: int,
    output_dir: Path,
    refresh: bool = False,
    *,
    use_horizon: bool = True,
) -> Weather:
    name = "weather" if use_horizon else "weather_no_horizon"
    csv_path = output_dir / f"{name}.csv"
    metadata_path = output_dir / f"{name}_meta.json"
    if csv_path.exists() and metadata_path.exists() and not refresh:
        metadata = json.loads(metadata_path.read_text())
        if (
            int(metadata.get("year", -1)) == year
            and bool(metadata.get("use_horizon", True)) == use_horizon
        ):
            hourly = pd.read_csv(csv_path, index_col="time", parse_dates=["time"])
            hourly.index = pd.DatetimeIndex(hourly.index)
            return Weather(
                hourly=hourly,
                latitude=metadata["latitude"],
                longitude=metadata["longitude"],
                elevation=metadata["elevation"],
                source=metadata["source"],
            )

    latitude, longitude, elevation = roof_location(points)
    common = {
        "latitude": latitude,
        "longitude": longitude,
        "start": year,
        "end": year,
        "raddatabase": PVGIS_DATABASE,
        "components": True,
        "usehorizon": use_horizon,
        "url": PVGIS_URL,
    }
    horizontal, _ = pvlib.iotools.get_pvgis_hourly(**common, trackingtype=0)
    tracker, _ = pvlib.iotools.get_pvgis_hourly(**common, trackingtype=2)
    if not horizontal.index.equals(tracker.index):
        raise ValueError("PVGIS fixed and tracking series have different timestamps")

    hourly = pd.DataFrame(index=horizontal.index)
    hourly.index.name = "time"
    hourly["ghi"] = (
        horizontal["poa_direct"]
        + horizontal["poa_sky_diffuse"]
        + horizontal["poa_ground_diffuse"]
    ).clip(lower=0.0)
    hourly["dhi"] = horizontal["poa_sky_diffuse"].clip(lower=0.0)
    hourly["dni"] = tracker["poa_direct"].clip(lower=0.0)
    hourly["temp_air"] = horizontal["temp_air"]
    hourly["wind_speed"] = horizontal["wind_speed"]
    solar_position = pvlib.solarposition.get_solarposition(
        hourly.index,
        latitude,
        longitude,
        altitude=elevation,
        temperature=hourly["temp_air"],
    )
    hourly["solar_zenith"] = solar_position["apparent_zenith"]
    hourly["solar_azimuth"] = solar_position["azimuth"]
    hourly["dni_extra"] = pvlib.irradiance.get_extra_radiation(hourly.index)
    hourly["airmass"] = pvlib.atmosphere.get_relative_airmass(hourly["solar_zenith"])

    output_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(csv_path)
    metadata = {
        "year": year,
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation,
        "source": (
            f"PVGIS v5.3 {PVGIS_DATABASE}"
            + ("" if use_horizon else " without terrain horizon")
        ),
        "url": PVGIS_URL,
        "use_horizon": use_horizon,
        "horizon": (
            "PVGIS terrain horizon enabled"
            if use_horizon
            else "PVGIS terrain horizon disabled"
        ),
        "dni_method": "two-axis tracker beam component",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return Weather(hourly, latitude, longitude, elevation, metadata["source"])
