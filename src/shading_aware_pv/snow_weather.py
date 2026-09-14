from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from urllib.request import urlopen


from .models import Weather

STATIONS = (
    {
        "id": "GVE",
        "name": "Genève / Cointrin",
        "latitude": 46.247519,
        "longitude": 6.127742,
        "elevation_m": 411.0,
    },
    {
        "id": "STG",
        "name": "St. Gallen",
        "latitude": 47.425475,
        "longitude": 9.398528,
        "elevation_m": 776.0,
    },
)


def _distance_km(latitude: float, longitude: float, station: dict) -> float:
    latitude_delta = radians(station["latitude"] - latitude)
    longitude_delta = radians(station["longitude"] - longitude)
    a = sin(latitude_delta / 2) ** 2 + (
        cos(radians(latitude))
        * cos(radians(station["latitude"]))
        * sin(longitude_delta / 2) ** 2
    )
    return 2 * 6371.0 * asin(sqrt(a))






def fetch_snow_station(
    weather: Weather,
    year: int,
    output_dir: Path,
    *,
    refresh: bool = False,
) -> dict:
    station_metadata = min(
        STATIONS,
        key=lambda station: _distance_km(
            weather.latitude,
            weather.longitude,
            station,
        ),
    )
    station_id = station_metadata["id"]
    decade_start = year // 10 * 10
    decade = f"{decade_start}-{decade_start + 9}"
    filename = f"ogd-smn_{station_id.lower()}_h_historical_{decade}.csv"
    source_url = (
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/"
        f"{station_id.lower()}/{filename}"
    )
    station_path = output_dir / filename
    if refresh or not station_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        with urlopen(source_url, timeout=60) as response:
            station_path.write_bytes(response.read())

    return {
        "station_id": station_id,
        "station_name": station_metadata["name"],
        "station_latitude": station_metadata["latitude"],
        "station_longitude": station_metadata["longitude"],
        "station_elevation_m": station_metadata["elevation_m"],
        "distance_km": _distance_km(
            weather.latitude,
            weather.longitude,
            station_metadata,
        ),
        "year": year,
        "source": "MeteoSwiss SwissMetNet hourly measurements",
        "source_url": source_url,
        "cached_file": str(station_path),
        "timestamp_alignment": "PVGIS :10 timestamps floored to the UTC hour",
    }
