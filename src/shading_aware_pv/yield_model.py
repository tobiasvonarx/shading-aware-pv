from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pvlib

from .irradiance import facet_irradiance, integrate_irradiance
from .models import PanelLayout, RoofSamples, Weather
from .snow import SnowData, effective_snow_loss, snow_loss_by_sample

YIELD_STATES = {
    "open_horizon": "open-horizon counterfactual",
    "unshaded": "terrain horizon only",
    "roof": "target base roof",
    "local": "target roof and superstructures",
    "context_no_vegetation": "target and non-vegetation context",
    "full": "target, context, and vegetation",
}
DEFAULT_MODULE_POWER_DENSITY_W_M2 = 200.0


@dataclass(frozen=True)
class ElectricalInputs:
    module_wattage_w: float
    dc_capacity_kwp: float
    inverter_ac_kw: float
    general_loss_percent: float
    inverter_efficiency_percent: float
    temperature_coefficient_percent_c: float
    snow_num_strings: int
    snow_num_strings_basis: str

    def metadata(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class YieldAnalysis:
    layout: PanelLayout | RoofSamples
    hourly: pd.DataFrame
    inputs: ElectricalInputs
    metrics: dict[str, dict]
    snow: SnowData


def default_electrical_inputs(
    module_count: int,
    module_area_m2: float,
    *,
    module_wattage_w: float | None = None,
    dc_capacity_kwp: float | None = None,
    inverter_ac_kw: float | None = None,
    dc_ac_ratio: float = 1.15,
    general_loss_percent: float = 0.0,
    inverter_efficiency_percent: float = 96.0,
    temperature_coefficient_percent_c: float = -0.35,
    snow_num_strings: int = 1,
    snow_num_strings_basis: str = "portrait module default",
) -> ElectricalInputs:
    if module_count <= 0 or module_area_m2 <= 0 or dc_ac_ratio <= 0:
        raise ValueError("Module count, module area, and DC/AC ratio must be positive")
    if module_wattage_w is not None and dc_capacity_kwp is not None:
        raise ValueError("Specify module wattage or total DC capacity, not both")
    wattage = (
        float(module_wattage_w)
        if module_wattage_w is not None
        else (
            float(dc_capacity_kwp) * 1000.0 / module_count
            if dc_capacity_kwp is not None
            else module_area_m2 * DEFAULT_MODULE_POWER_DENSITY_W_M2
        )
    )
    capacity = module_count * wattage / 1000.0
    inverter = (
        float(inverter_ac_kw)
        if inverter_ac_kw is not None
        else capacity / dc_ac_ratio
    )
    values = ElectricalInputs(
        module_wattage_w=wattage,
        dc_capacity_kwp=capacity,
        inverter_ac_kw=inverter,
        general_loss_percent=general_loss_percent,
        inverter_efficiency_percent=inverter_efficiency_percent,
        temperature_coefficient_percent_c=temperature_coefficient_percent_c,
        snow_num_strings=snow_num_strings,
        snow_num_strings_basis=snow_num_strings_basis,
    )
    if (
        values.module_wattage_w <= 0
        or values.dc_capacity_kwp <= 0
        or values.inverter_ac_kw <= 0
    ):
        raise ValueError("Module wattage, DC capacity, and inverter limit must be positive")
    if not 0 <= values.general_loss_percent < 100:
        raise ValueError("General loss must be in [0, 100)")
    if not 0 < values.inverter_efficiency_percent <= 100:
        raise ValueError("Inverter efficiency must be in (0, 100]")
    if values.snow_num_strings < 1:
        raise ValueError("Snow num_strings must be positive")
    return values


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(values, axis=1, weights=weights)


def modeled_energy(
    poa_w_m2: np.ndarray,
    cell_temperature_c: np.ndarray,
    inputs: ElectricalInputs,
    snow_loss_fraction: np.ndarray | float = 0.0,
) -> np.ndarray:
    dc_kw = ideal_dc_energy(poa_w_m2, cell_temperature_c, inputs)
    dc_kw *= 1.0 - np.asarray(snow_loss_fraction)
    dc_kw *= 1.0 - inputs.general_loss_percent / 100.0
    return np.minimum(
        np.maximum(dc_kw * inputs.inverter_efficiency_percent / 100.0, 0.0),
        inputs.inverter_ac_kw,
    )


def ideal_dc_energy(
    poa_w_m2: np.ndarray,
    cell_temperature_c: np.ndarray,
    inputs: ElectricalInputs,
) -> np.ndarray:
    """Hourly DC energy before snow, generic losses, and inverter effects."""
    temperature_factor = np.maximum(
        0.0,
        1.0
        + inputs.temperature_coefficient_percent_c
        / 100.0
        * (cell_temperature_c - 25.0),
    )
    return (
        inputs.dc_capacity_kwp
        * poa_w_m2
        / 1000.0
        * temperature_factor
    )


def percent_loss(value: float, baseline: float) -> float:
    return 100.0 * (1.0 - value / baseline) if baseline > 0.0 else 0.0


def percent_change(new_value: float, old_value: float) -> float:
    return 100.0 * (new_value - old_value) / old_value if old_value > 0.0 else 0.0










def analyze_yield(
    layout: PanelLayout | RoofSamples,
    weather: Weather,
    visibility: dict[str, np.ndarray],
    inputs: ElectricalInputs,
    *,
    snow: SnowData | None = None,
    state_weather: dict[str, Weather] | None = None,
) -> YieldAnalysis:
    samples = layout if isinstance(layout, RoofSamples) else layout.samples
    if snow is None:
        snow = SnowData(None, None, {"available": False, "reason": "not supplied"})
    states = tuple(visibility)
    state_weather = state_weather or {}
    weather_by_state = {
        state: state_weather.get(state, weather) for state in states
    }
    if any(
        not value.hourly.index.equals(weather.hourly.index)
        for value in weather_by_state.values()
    ):
        raise ValueError("All shading states must use the same timestamps")
    weather_sources = {id(source): source for source in weather_by_state.values()}
    component_cache = {
        key: facet_irradiance(samples, source)
        for key, source in weather_sources.items()
    }
    index = weather.hourly.index.floor("h")
    if index.has_duplicates:
        raise ValueError("Weather timestamps are not unique after hourly alignment")
    hourly = pd.DataFrame(index=index)
    hourly.index.name = "time"
    hourly["temp_air_c"] = weather.hourly["temp_air"].to_numpy()
    hourly["wind_speed_m_s"] = weather.hourly["wind_speed"].to_numpy()

    shaded_hours: dict[str, float] = {}
    for state in states:
        state_source = weather_by_state[state]
        components = component_cache[id(state_source)]
        energy = integrate_irradiance(components, visibility[state])
        poa = _weighted_mean(energy, samples.areas)
        direct = _weighted_mean(
            components["direct"] * visibility[state],
            samples.areas,
        )
        cell_temperature = pvlib.temperature.faiman(
            poa,
            state_source.hourly["temp_air"].to_numpy(),
            state_source.hourly["wind_speed"].to_numpy(),
        )
        snow_coverage, snow_loss_by_receiver = snow_loss_by_sample(
            energy,
            samples,
            state_source,
            snow,
            inputs.snow_num_strings,
        )
        snow_loss = effective_snow_loss(
            energy,
            snow_loss_by_receiver,
            samples.areas,
        )
        hourly[f"poa_{state}_w_m2"] = poa
        hourly[f"poa_direct_{state}_w_m2"] = direct
        hourly[f"cell_temperature_{state}_c"] = cell_temperature
        hourly[f"snow_coverage_{state}_fraction"] = _weighted_mean(
            snow_coverage,
            samples.areas,
        )
        hourly[f"snow_dc_loss_{state}_fraction"] = snow_loss
        hourly[f"ideal_dc_{state}_kwh"] = ideal_dc_energy(
            poa,
            np.asarray(cell_temperature),
            inputs,
        )
        hourly[f"modeled_no_snow_{state}_kwh"] = modeled_energy(
            poa,
            np.asarray(cell_temperature),
            inputs,
        )
        hourly[f"modeled_{state}_kwh"] = modeled_energy(
            poa,
            np.asarray(cell_temperature),
            inputs,
            snow_loss,
        )
        shaded_hours[state] = float(
            np.average(
                ((components["direct"] > 20.0) & ~visibility[state]).sum(axis=0),
                weights=samples.areas,
            )
        )

    default_components = component_cache[id(weather)]
    hourly["poa_sky_diffuse_w_m2"] = _weighted_mean(
        default_components["sky"],
        samples.areas,
    )
    hourly["poa_ground_diffuse_w_m2"] = _weighted_mean(
        default_components["ground"],
        samples.areas,
    )

    metrics = {
        state: {"modeled_kwh": float(hourly[f"modeled_{state}_kwh"].sum())}
        for state in states
    }
    baseline = states[0]
    baseline_direct = float(hourly[f"poa_direct_{baseline}_w_m2"].sum())
    baseline_poa = float(hourly[f"poa_{baseline}_w_m2"].sum())
    baseline_ac = float(hourly[f"modeled_{baseline}_kwh"].sum())
    previous = baseline
    for state in states:
        direct = float(hourly[f"poa_direct_{state}_w_m2"].sum())
        poa = float(hourly[f"poa_{state}_w_m2"].sum())
        ac = float(hourly[f"modeled_{state}_kwh"].sum())
        previous_direct = float(hourly[f"poa_direct_{previous}_w_m2"].sum())
        previous_poa = float(hourly[f"poa_{previous}_w_m2"].sum())
        previous_ac = float(hourly[f"modeled_{previous}_kwh"].sum())
        metrics[state].update(
            {
                "ideal_dc_kwh": float(
                    hourly[f"ideal_dc_{state}_kwh"].sum()
                ),
                "ideal_dc_specific_yield_kwh_per_kwp": float(
                    hourly[f"ideal_dc_{state}_kwh"].sum()
                    / inputs.dc_capacity_kwp
                ),
                "annual_direct_poa_kwh_m2": direct / 1000.0,
                "annual_total_poa_kwh_m2": poa / 1000.0,
                "mean_direct_shaded_hours": shaded_hours[state],
                "direct_loss_vs_baseline_percent": percent_loss(
                    direct, baseline_direct
                ),
                "total_poa_loss_vs_baseline_percent": percent_loss(
                    poa, baseline_poa
                ),
                "ac_loss_vs_baseline_percent": percent_loss(ac, baseline_ac),
                "direct_incremental_loss_percent": percent_loss(
                    direct, previous_direct
                ),
                "total_poa_incremental_loss_percent": percent_loss(
                    poa, previous_poa
                ),
                "ac_incremental_loss_percent": percent_loss(ac, previous_ac),
                "snow_loss_kwh": float(
                    hourly[f"modeled_no_snow_{state}_kwh"].sum() - ac
                ),
                "snow_loss_percent": percent_loss(
                    ac,
                    float(hourly[f"modeled_no_snow_{state}_kwh"].sum()),
                ),
                "snow_covered_hours": int(
                    (hourly[f"snow_coverage_{state}_fraction"] > 0.0).sum()
                ),
            }
        )
        previous = state
    return YieldAnalysis(
        layout=layout,
        hourly=hourly,
        inputs=inputs,
        metrics=metrics,
        snow=snow,
    )
