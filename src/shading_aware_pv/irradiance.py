from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pvlib

from .models import RoofSamples, Weather


def _orientation_irradiance(
    tilt: float, azimuth: float, weather: Weather
) -> dict[str, np.ndarray]:
    hourly = weather.hourly
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
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
    return {
        name: poa[column].to_numpy(dtype=np.float32)
        for name, column in (
            ("direct", "poa_direct"),
            ("sky", "poa_sky_diffuse"),
            ("ground", "poa_ground_diffuse"),
        )
    }


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
        components = _orientation_irradiance(
            samples.tilt[sample_index], samples.azimuth[sample_index], weather
        )
        direct[:, sample_indices] = components["direct"][:, None]
        sky[:, sample_indices] = components["sky"][:, None]
        ground[:, sample_indices] = components["ground"][:, None]
    return {"direct": direct, "sky": sky, "ground": ground}


def integrate_irradiance(
    components: dict[str, np.ndarray],
    visibility: np.ndarray,
) -> np.ndarray:
    """Return interval irradiance in Wh/m² for hourly PVGIS samples."""
    return components["direct"] * visibility + components["sky"] + components["ground"]


@dataclass(frozen=True)
class FaceIrradiance:
    """Hourly fields and sample membership, stored once per supporting face."""

    components: dict[str, np.ndarray]
    sample_indices: tuple[np.ndarray, ...]
    sample_count: int

    def dense(self) -> dict[str, np.ndarray]:
        """Expand with the original float32 layout for observed-snow modeling."""
        result = {}
        for name, values in self.components.items():
            expanded = np.empty((len(values), self.sample_count), dtype=np.float32)
            for face, indices in enumerate(self.sample_indices):
                expanded[:, indices] = values[:, face, None]
            result[name] = expanded
        return result

    def diffuse_mean(self, name: str, areas: np.ndarray) -> np.ndarray:
        weights = np.array([areas[indices].sum() for indices in self.sample_indices])
        return np.average(self.components[name], axis=1, weights=weights)

    def annual_totals(self, visibility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sum each sample's resource without expanding all faces at once.

        Keep float32 additions and C-order hourly reductions identical to the
        dense calculation. The temporary energy array covers only one face.
        """
        hours = len(self.components["direct"])
        if visibility.shape != (hours, self.sample_count):
            raise ValueError("Visibility shape does not match weather and roof samples")
        irradiation = np.empty(self.sample_count, dtype=np.float64)
        shaded_hours = np.empty(self.sample_count, dtype=np.int64)
        for face, indices in enumerate(self.sample_indices):
            direct = self.components["direct"][:, face]
            sky = self.components["sky"][:, face]
            ground = self.components["ground"][:, face]
            visible = visibility[:, indices]
            # A one-column array triggers NumPy's contiguous pairwise sum. Pad
            # singleton faces to preserve the original multi-sample reduction.
            width = max(2, len(indices)) if self.sample_count > 1 else 1
            energy = np.empty((hours, width), dtype=np.float32)
            np.copyto(energy, (sky + ground)[:, None])
            np.copyto(energy, ((direct + sky) + ground)[:, None], where=visible)
            irradiation[indices] = (
                energy.sum(axis=0, dtype=np.float64)[: len(indices)] / 1000.0
            )
            shaded_hours[indices] = ((direct[:, None] > 20.0) & ~visible).sum(axis=0)
        return irradiation, shaded_hours

    def shaded_means(
        self,
        visibility: np.ndarray,
        areas: np.ndarray,
        visibility_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Reduce lit/shaded area before temperature and inverter calculations.

        Form lit and shaded irradiance in float32, in the same addition order
        as integrate_irradiance. Only the subsequent float64 weighted summation
        is regrouped. Compute shaded weights directly to keep zero shade exact.
        """
        direct_field = self.components["direct"]
        hours = len(direct_field)
        if (
            visibility.ndim != 2
            or visibility.shape[0] != hours
            or (visibility_indices is None and visibility.shape[1] != self.sample_count)
        ):
            raise ValueError("Visibility shape does not match weather and roof samples")
        if visibility_indices is not None and visibility_indices.shape != (
            self.sample_count,
        ):
            raise ValueError("Visibility indices must identify every roof sample")
        total = float(areas.sum())
        if total == 0:
            raise ZeroDivisionError("Weights sum to zero, cannot normalize irradiance")
        poa = np.zeros(hours)
        direct = np.zeros(hours)
        shaded_hours = 0.0
        for face, indices in enumerate(self.sample_indices):
            weights = areas[indices]
            weight = float(weights.sum())
            columns = (
                indices if visibility_indices is None else visibility_indices[indices]
            )
            blocked = np.einsum(
                "ij,j->i",
                ~visibility[:, columns],
                weights,
                dtype=np.float64,
                optimize=False,
            )
            blocked = np.clip(blocked, 0.0, weight)
            lit_weight = weight - blocked
            beam = direct_field[:, face]
            sky = self.components["sky"][:, face]
            ground = self.components["ground"][:, face]
            lit = (beam + sky) + ground
            shaded = sky + ground
            poa += lit * lit_weight + shaded * blocked
            direct += beam * lit_weight
            shaded_hours += float(blocked[beam > 20.0].sum())
        return poa / total, direct / total, shaded_hours / total


class IrradianceCache:
    """Reuse transposition during one analysis with unchanged weather inputs.

    The simulation owns this cache; nothing survives into a later analysis.
    Strong references to weather sources prevent reuse of their object IDs.
    Geometry keys use the actual orientation, so different face IDs and layouts
    may share fields without sharing sample weights or visibility.
    """

    def __init__(self) -> None:
        self._weather_sources: dict[int, Weather] = {}
        self._orientations: dict[
            tuple[int, str, float, str, float], dict[str, np.ndarray]
        ] = {}

    def for_samples(self, samples: RoofSamples, weather: Weather) -> FaceIrradiance:
        self._weather_sources[id(weather)] = weather
        groups = tuple(
            np.flatnonzero(samples.face_ids == face)
            for face in np.unique(samples.face_ids)
        )
        tilt, azimuth = samples.tilt, samples.azimuth
        fields = []
        for indices in groups:
            first = indices[0]
            key = (
                id(weather),
                tilt.dtype.str,
                float(tilt[first]),
                azimuth.dtype.str,
                float(azimuth[first]),
            )
            if key not in self._orientations:
                self._orientations[key] = _orientation_irradiance(
                    tilt[first], azimuth[first], weather
                )
            fields.append(self._orientations[key])
        return FaceIrradiance(
            {
                name: (
                    np.column_stack([field[name] for field in fields])
                    if fields
                    else np.empty((len(weather.hourly), 0), dtype=np.float32)
                )
                for name in ("direct", "sky", "ground")
            },
            groups,
            len(samples.points),
        )
