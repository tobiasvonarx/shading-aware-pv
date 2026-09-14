from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from .models import ModuleArray, RoofSamples, Weather

MAX_STATION_DISTANCE_KM = 25.0
SNOWFALL_TEMPERATURE_LIMIT_C = 2.0


@dataclass(frozen=True)
class SnowData:
    snowfall_cm: pd.Series | None
    snow_depth_cm: pd.Series | None
    metadata: dict

    @property
    def available(self) -> bool:
        return self.snowfall_cm is not None and self.snow_depth_cm is not None


def automatic_num_strings(arrays: tuple[ModuleArray, ...]) -> tuple[int, str]:
    """Choose the NREL cell-substring assumption from the dominant orientation."""
    counts = {
        orientation: sum(
            array.module_count
            for array in arrays
            if array.orientation == orientation
        )
        for orientation in ("portrait", "landscape")
    }
    orientation = max(counts, key=lambda value: (counts[value], value == "portrait"))
    num_strings = 1 if orientation == "portrait" else 3
    return num_strings, f"automatic from dominant {orientation} module orientation"


def load_snow_data(
    weather: Weather,
    station_metadata: dict,
) -> SnowData:
    """Build an hourly snowfall proxy from the cached MeteoSwiss station file."""
    station = station_metadata["station_id"]
    distance_km = float(station_metadata["distance_km"])
    metadata = {
        "available": False,
        "model": "pvlib NREL snow coverage and DC loss",
        "station_id": station,
        "station_name": station_metadata["station_name"],
        "distance_km": distance_km,
        "maximum_station_distance_km": MAX_STATION_DISTANCE_KM,
        "snowfall_proxy": (
            "positive automatic snow-depth increments confirmed by precipitation "
            f"and station air temperature <= {SNOWFALL_TEMPERATURE_LIMIT_C:g} °C"
        ),
        "snow_depth_parameter": "MeteoSwiss htoauths",
        "precipitation_parameter": "MeteoSwiss rre150h0",
        "temperature_parameter": "MeteoSwiss tre200h0",
    }
    if distance_km > MAX_STATION_DISTANCE_KM:
        metadata["reason"] = "nearest supported snow station is too far from the roof"
        return SnowData(None, None, metadata)

    station_path = Path(station_metadata["cached_file"])
    station_data = pd.read_csv(station_path, sep=";", low_memory=False)
    required = {"reference_timestamp", "htoauths", "rre150h0", "tre200h0"}
    missing = required.difference(station_data.columns)
    if missing:
        metadata["reason"] = f"station file lacks {', '.join(sorted(missing))}"
        return SnowData(None, None, metadata)

    station_data.index = pd.to_datetime(
        station_data.pop("reference_timestamp"),
        format="%d.%m.%Y %H:%M",
        utc=True,
    )
    index = weather.hourly.index.floor("h")
    observed = station_data.reindex(index)
    depth = pd.to_numeric(observed["htoauths"], errors="coerce").where(
        lambda values: values >= 0
    )
    depth = depth.interpolate(limit=6, limit_direction="both")
    if depth.isna().any():
        metadata["reason"] = (
            "snow-depth observations contain a gap longer than six hours"
        )
        return SnowData(None, None, metadata)

    precipitation = pd.to_numeric(observed["rre150h0"], errors="coerce").fillna(0.0)
    temperature = pd.to_numeric(observed["tre200h0"], errors="coerce")
    if temperature.isna().any():
        metadata["reason"] = "station air-temperature observations are incomplete"
        return SnowData(None, None, metadata)

    snowfall = depth.diff().clip(lower=0.0).fillna(0.0)
    recent_precipitation = precipitation.rolling(3, min_periods=1).sum()
    snowfall = snowfall.where(
        (recent_precipitation > 0.0)
        & (temperature <= SNOWFALL_TEMPERATURE_LIMIT_C),
        0.0,
    )
    snowfall.name = "snowfall_proxy_cm"
    depth.name = "snow_depth_cm"
    metadata.update(
        {
            "available": True,
            "source": station_metadata["source"],
            "source_url": station_metadata["source_url"],
            "snowfall_event_hours": int((snowfall >= 1.0).sum()),
            "maximum_cold_weather_snow_depth_cm": float(
                depth.where(temperature <= SNOWFALL_TEMPERATURE_LIMIT_C).max()
            ),
        }
    )
    return SnowData(snowfall, depth, metadata)


def snow_loss_by_sample(
    poa_w_m2: np.ndarray,
    samples: RoofSamples,
    weather: Weather,
    snow: SnowData,
    num_strings: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return NREL snow coverage and DC-loss fractions for each roof sample."""
    shape = poa_w_m2.shape
    if shape != (len(weather.hourly), len(samples.points)):
        raise ValueError("POA shape does not match weather and roof samples")
    if num_strings < 1:
        raise ValueError("Snow num_strings must be positive")
    if not snow.available:
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    index = weather.hourly.index.floor("h")
    snowfall = snow.snowfall_cm.reindex(index)
    snow_depth = snow.snow_depth_cm.reindex(index)
    if snowfall.isna().any() or snow_depth.isna().any():
        raise ValueError("Snow observations do not cover the simulated weather hours")
    temperature = pd.Series(
        weather.hourly["temp_air"].to_numpy(),
        index=index,
    )
    coverage = np.zeros(shape, dtype=np.float32)
    loss = np.zeros(shape, dtype=np.float32)
    for face_id in np.unique(samples.face_ids):
        sample_ids = np.flatnonzero(samples.face_ids == face_id)
        weights = samples.areas[sample_ids]
        face_poa = pd.Series(
            np.average(poa_w_m2[:, sample_ids], axis=1, weights=weights),
            index=index,
        )
        face_coverage = pvlib.snow.coverage_nrel(
            snowfall,
            face_poa,
            temperature,
            surface_tilt=float(np.average(samples.tilt[sample_ids], weights=weights)),
            snow_depth=snow_depth,
            initial_coverage=float(snow_depth.iloc[0] >= 1.0),
        ).to_numpy(dtype=np.float32)
        face_loss = np.asarray(
            pvlib.snow.dc_loss_nrel(face_coverage, num_strings),
            dtype=np.float32,
        )
        coverage[:, sample_ids] = face_coverage[:, None]
        loss[:, sample_ids] = face_loss[:, None]
    return coverage, loss


def effective_snow_loss(
    poa_w_m2: np.ndarray,
    loss_by_sample: np.ndarray,
    sample_areas: np.ndarray,
) -> np.ndarray:
    """Collapse sample losses using their hourly pre-snow energy contribution."""
    weighted_poa = poa_w_m2 * sample_areas
    total = weighted_poa.sum(axis=1)
    lost = (weighted_poa * loss_by_sample).sum(axis=1)
    return np.divide(lost, total, out=np.zeros_like(total), where=total > 0.0)
