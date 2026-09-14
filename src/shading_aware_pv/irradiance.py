from __future__ import annotations

import numpy as np
import pvlib

from .models import RoofSamples, Weather


def facet_irradiance(samples: RoofSamples, weather: Weather) -> dict[str, np.ndarray]:
    """Calculate unshaded irradiance once per roof-face orientation."""
    hourly = weather.hourly
    count = len(hourly)
    direct = np.empty((count, len(samples.points)), dtype=np.float32)
    sky = np.empty_like(direct)
    ground = np.empty_like(direct)
    for face_id in np.unique(samples.face_ids):
        sample_indices = np.flatnonzero(samples.face_ids == face_id)
        sample_index = sample_indices[0]
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=samples.tilt[sample_index],
            surface_azimuth=samples.azimuth[sample_index],
            solar_zenith=hourly["solar_zenith"],
            solar_azimuth=hourly["solar_azimuth"],
            dni=hourly["dni"],
            ghi=hourly["ghi"],
            dhi=hourly["dhi"],
            dni_extra=hourly["dni_extra"],
            airmass=hourly["airmass"],
            albedo=0.2,
            model="perez",
        )
        # Perez has undefined intermediate terms at night. Irradiance is zero there.
        poa = poa.fillna(0.0).clip(lower=0.0)
        direct[:, sample_indices] = poa["poa_direct"].to_numpy(dtype=np.float32)[:, None]
        sky[:, sample_indices] = poa["poa_sky_diffuse"].to_numpy(dtype=np.float32)[:, None]
        ground[:, sample_indices] = poa["poa_ground_diffuse"].to_numpy(dtype=np.float32)[:, None]
    return {"direct": direct, "sky": sky, "ground": ground}


def integrate_irradiance(
    components: dict[str, np.ndarray],
    visibility: np.ndarray,
) -> np.ndarray:
    """Return interval irradiance in Wh/m² for hourly PVGIS samples."""
    return (
        components["direct"] * visibility
        + components["sky"]
        + components["ground"]
    )
