"""Roof resource, installed yield, same-count relocation and clean-slate design."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from collections.abc import Callable
import json

import numpy as np

from .geometry import load_roof_scene, mesh_footprint
from .clean_slate import bare_roof_placement
from .inputs import SimulationInputs, load_context_scene
from .irradiance import facet_irradiance, integrate_irradiance
from .models import IrradianceResult, RoofResource, RoofSamples, Weather
from .optimization import optimize_placements, placement_audit, select_milp, optimize_relocation, retain_installed_layout
from .panels import (
    NoPVModulesError,
    _merge_meshes,
    load_non_panel_details,
    load_panel_layout,
    roof_parallel_receivers,
    roof_use_labels,
)
from .shadow import render_context_visibilities, render_visibility
from .snow import SnowData, automatic_num_strings, load_snow_data
from .snow_weather import fetch_snow_station
from .weather import fetch_weather
from .yield_model import YIELD_STATES, analyze_yield, default_electrical_inputs

METHOD_REVISION = 8


@dataclass(frozen=True)
class SimulationConfig:
    year: int = 2023
    sample_spacing: float = 0.2
    shadow_pixel_size: float = 0.05
    context_half_extent: float = 100.0
    context_shadow_pixel_size: float = 0.25
    context_grid_resolution: float = 1.0
    depth_epsilon: float = 0.03
    detail_clearance: float = 0.15
    mounting_clearance: float = 0.20
    roof_setback: float = 0.05
    obstruction_setback: float = 0.05
    module_gap: float = 0.0
    placement_phase_step: float = 0.10
    max_hours: int | None = None
    refresh_weather: bool = False
    module_width_m: float = 1.0
    module_height_m: float = 1.7
    module_wattage_w: float | None = None
    dc_capacity_kwp: float | None = None
    inverter_ac_kw: float | None = None
    dc_ac_ratio: float = 1.15
    general_loss_percent: float = 0.0
    inverter_efficiency_percent: float = 96.0
    temperature_coefficient_percent_c: float = -0.35

    def __post_init__(self):
        for name in (
            "sample_spacing",
            "shadow_pixel_size",
            "context_shadow_pixel_size",
            "context_grid_resolution",
            "placement_phase_step",
            "module_width_m",
            "module_height_m",
            "dc_ac_ratio",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "context_half_extent",
            "depth_epsilon",
            "detail_clearance",
            "mounting_clearance",
            "roof_setback",
            "obstruction_setback",
            "module_gap",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not isinstance(self.year, int):
            raise ValueError("year must be an integer")
        if self.max_hours is not None and (
            not isinstance(self.max_hours, int) or self.max_hours < 1
        ):
            raise ValueError("max_hours must be a positive integer")
        default_electrical_inputs(
            1,
            self.module_width_m * self.module_height_m,
            module_wattage_w=self.module_wattage_w,
            dc_capacity_kwp=self.dc_capacity_kwp,
            inverter_ac_kw=self.inverter_ac_kw,
            dc_ac_ratio=self.dc_ac_ratio,
            general_loss_percent=self.general_loss_percent,
            inverter_efficiency_percent=self.inverter_efficiency_percent,
            temperature_coefficient_percent_c=self.temperature_coefficient_percent_c,
        )


def _subset(samples, indices):
    return RoofSamples(
        samples.points[indices],
        samples.normals[indices],
        samples.areas[indices],
        samples.face_ids[indices],
    )


def simulate(
    inputs: SimulationInputs,
    config: SimulationConfig = SimulationConfig(),
    *,
    weather: Weather | None = None,
    open_horizon_weather: Weather | None = None,
    snow: SnowData | None = None,
    context_points_supplier: Callable | None = None,
    progress: Callable[[str], None] = print,
    include_count_designs: bool = True,
) -> dict:
    """Run the original numerical pipeline and return serializable model output.

    Explicit Weather/SnowData and context point suppliers support offline callers.
    Context suppliers must provide complete requested tile coverage or raise.
    Weather caches are scoped to this roof and year by the caller.
    """
    inputs.cache_dir.mkdir(parents=True, exist_ok=True)
    progress("Loading and sampling reconstructed roof")
    scene = load_roof_scene(
        inputs.mesh_path,
        roof_details_path=inputs.roof_details_path,
        scaffold_path=inputs.scaffold_path,
        sample_spacing=config.sample_spacing,
        detail_clearance=config.detail_clearance,
    )
    progress("Inferring installed module dimensions and layout")
    installed_reason = None
    try:
        layout = load_panel_layout(
            scene,
            inputs.roof_details_path,
            inputs.orthophoto_path,
            config.mounting_clearance,
        )
    except NoPVModulesError as error:
        layout = None
        installed_reason = str(error)
    details = load_non_panel_details(scene)
    occluder = scene.partition.non_pv_mesh
    receivers = roof_parallel_receivers(scene.samples, config.mounting_clearance)
    context = None
    if config.context_half_extent > 0:
        progress("Loading complete neighborhood LiDAR coverage")
        context = load_context_scene(
            inputs,
            center_xy=tuple(scene.samples.points[:, :2].mean(axis=0).astype(float)),
            half_extent_m=config.context_half_extent,
            grid_resolution_m=config.context_grid_resolution,
            points_supplier=context_points_supplier,
        )
    progress("Loading PVGIS weather and MeteoSwiss snow observations")
    if weather is None:
        weather = fetch_weather(
            receivers.points,
            config.year,
            inputs.cache_dir,
            refresh=config.refresh_weather,
        )
    if open_horizon_weather is None:
        open_horizon_weather = fetch_weather(
            receivers.points,
            config.year,
            inputs.cache_dir,
            refresh=config.refresh_weather,
            use_horizon=False,
        )
    if not weather.hourly.index.equals(open_horizon_weather.hourly.index):
        raise ValueError("PVGIS horizon variants have different timestamps")
    if snow is None:
        station = fetch_snow_station(
            weather, config.year, inputs.cache_dir, refresh=config.refresh_weather
        )
        snow = load_snow_data(weather, station)
    full_hour_count = len(weather.hourly)
    if config.max_hours is not None:
        weather = replace(
            weather, hourly=weather.hourly.iloc[: config.max_hours].copy()
        )
        open_horizon_weather = replace(
            open_horizon_weather,
            hourly=open_horizon_weather.hourly.iloc[: config.max_hours].copy(),
        )
    hourly = weather.hourly
    if hourly.empty:
        raise ValueError("Weather contains no simulation hours")
    args = dict(
        points=receivers.points,
        normals=receivers.normals,
        zenith=hourly.solar_zenith.to_numpy(),
        azimuth=hourly.solar_azimuth.to_numpy(),
        dni=hourly.dni.to_numpy(),
        depth_epsilon=config.depth_epsilon,
    )
    progress("Rendering direct-beam roof and neighborhood visibility")
    context_full = np.ones((len(hourly), len(receivers.points)), dtype=bool)
    context_bare = context_full.copy()
    render_stats = None
    if context is not None:
        (context_bare, context_full), render_stats = render_context_visibilities(
            meshes=[context.surface_mesh_without_vegetation, context.surface_mesh],
            pixel_size=config.context_shadow_pixel_size,
            **args,
        )
    base, local = render_visibility(
        meshes=[scene.partition.base_roof, occluder],
        pixel_size=config.shadow_pixel_size,
        **args,
    )
    visibility = dict(
        open_horizon=np.ones_like(local),
        unshaded=np.ones_like(local),
        roof=base,
        local=local,
        context_no_vegetation=local & context_bare,
        full=local & context_full,
    )
    results = {}
    for source, states in (
        (open_horizon_weather, ("open_horizon",)),
        (weather, tuple(YIELD_STATES)[1:]),
    ):
        components = facet_irradiance(receivers, source)
        for state in states:
            energy = integrate_irradiance(components, visibility[state])
            results[state] = IrradianceResult(
                energy.sum(axis=0, dtype=np.float64) / 1000.0,
                ((components["direct"] > 20.0) & ~visibility[state]).sum(axis=0),
            )
    resource = RoofResource(
        scene.samples, receivers, roof_use_labels(scene, layout), results
    )
    count = len(layout.modules.cells) if layout else 0
    width = layout.modules.module_width_m if layout else config.module_width_m
    height = layout.modules.module_height_m if layout else config.module_height_m
    strings, strings_basis = (
        automatic_num_strings(layout.modules.arrays)
        if layout
        else (1, "portrait module default")
    )
    installed_electrical = default_electrical_inputs(
        count or 1,
        width * height,
        module_wattage_w=config.module_wattage_w,
        dc_capacity_kwp=config.dc_capacity_kwp if layout else None,
        inverter_ac_kw=config.inverter_ac_kw if layout else None,
        dc_ac_ratio=config.dc_ac_ratio,
        general_loss_percent=config.general_loss_percent,
        inverter_efficiency_percent=config.inverter_efficiency_percent,
        temperature_coefficient_percent_c=config.temperature_coefficient_percent_c,
        snow_num_strings=strings,
        snow_num_strings_basis=strings_basis,
    )

    def analyze(indices, electrical, *, include_hourly=True):
        analysis = analyze_yield(
            _subset(receivers, indices),
            weather,
            {state: (mask if layout or state != "full" else mask & base)[:, indices]
             for state, mask in visibility.items()},
            electrical,
            snow=snow if layout else None,
            state_weather={"open_horizon": open_horizon_weather},
        )
        energy_columns = [f"modeled_{state}_kwh" for state in visibility]
        monthly = analysis.hourly[energy_columns].resample("MS").sum()
        values = {
            "snow_applied": bool(layout is not None and snow.available),
            "snow_basis": (
                "observed snow model" if layout and snow.available else
                "snow-free placement estimate" if layout is None else
                "snow observations unavailable"
            ),
            "electrical": electrical.metadata(),
            "states": analysis.metrics,
            "monthly": {
                "months": monthly.index.strftime("%Y-%m").tolist(),
                "ac_kwh": {
                    state: monthly[f"modeled_{state}_kwh"].tolist()
                    for state in visibility
                },
            },
        }
        if include_hourly:
            values["hourly"] = {
                "timestamps": analysis.hourly.index.astype(str).tolist(),
                "ac_kwh": {
                    state: analysis.hourly[f"modeled_{state}_kwh"].tolist()
                    for state in visibility
                },
            }
        return values

    designs = {}
    if layout:
        designs["installed"] = {
            "panel_count": count,
            "corners": [
                cell.receiver_corners_xyz.tolist() for cell in layout.modules.cells
            ],
            **analyze(layout.roof_sample_indices, installed_electrical),
        }
    progress("Optimizing same-count relocation and clean-slate capacity")
    placement = dict(
        module_width_m=width,
        module_height_m=height,
        roof_setback_m=config.roof_setback,
        obstruction_setback_m=config.obstruction_setback,
        module_gap_m=config.module_gap,
        phase_step_m=config.placement_phase_step,
        mounting_clearance_m=config.mounting_clearance,
    )
    if layout:
        optimization = optimize_placements(
            scene, resource, scene.partition.non_pv_mesh, current_panel_count=count,
            **placement,
        )
    else:
        optimization = bare_roof_placement(
            scene, receivers, weather, base, local, context_full,
            scene.partition.non_pv_mesh, **placement,
        )
    def design_for(solution, *, include_hourly=True, electrical=None):
        design = {
            "panel_count": solution.panel_count,
            "corners": [
                optimization.candidates[i].receiver_corners_xyz.tolist()
                for i in solution.candidate_ids
            ],
            "geometry_check": placement_audit(
                scene, scene.partition.non_pv_mesh, optimization, solution
            ),
        }
        if solution.panel_count:
            electrical = electrical or default_electrical_inputs(
                solution.panel_count,
                width * height,
                module_wattage_w=installed_electrical.module_wattage_w,
                dc_ac_ratio=installed_electrical.dc_capacity_kwp
                / installed_electrical.inverter_ac_kw,
                general_loss_percent=config.general_loss_percent,
                inverter_efficiency_percent=config.inverter_efficiency_percent,
                temperature_coefficient_percent_c=config.temperature_coefficient_percent_c,
                snow_num_strings=strings,
                snow_num_strings_basis=strings_basis,
            )
            design.update(analyze(solution.roof_sample_indices, electrical, include_hourly=include_hourly))
        else:
            months = hourly.resample("MS").size().index.strftime("%Y-%m").tolist()
            design.update(
                snow_applied=False, snow_basis="No panels selected",
                electrical={**installed_electrical.metadata(), "dc_capacity_kwp": 0.0, "inverter_ac_kw": 0.0},
                states={state: {"modeled_kwh": 0.0, "ideal_dc_kwh": 0.0} for state in visibility},
                monthly={"months": months, "ac_kwh": {state: [0.0] * len(months) for state in visibility}},
            )
        return design

    designs["clean_slate"] = design_for(optimization.maximum_layout)
    designs_by_count = (_designs_by_count(
        optimization, designs["clean_slate"], design_for, progress
    ) if include_count_designs else {})
    if layout:
        optimization, incumbent = optimize_relocation(scene, optimization, resource, layout)
        proposed = design_for(optimization.relocated_layout, electrical=installed_electrical)
        selected = retain_installed_layout(
            optimization, incumbent,
            designs["installed"]["states"]["full"]["modeled_kwh"],
            proposed["states"]["full"]["modeled_kwh"],
        )
        retained = selected is incumbent
        designs["relocated"] = {
            **(designs["installed"] if retained else proposed),
            "requested_panel_count": count, "same_count_feasible": True,
            "retained_installed": retained,
            "geometry_check": placement_audit(
                scene, scene.partition.non_pv_mesh, selected, selected.relocated_layout
            ),
        }
    progress("Preparing solar results")
    mesh = scene_mesh(scene)
    return {
        "schema": "shading-aware-pv/v1",
        "provenance": {
            "method_revision": METHOD_REVISION,
            "bare_roof_placement": "four-stage capacity and fixed-count full-shading MILP",
            "bare_roof_snow": "excluded",
        },
        "config": asdict(config),
        "building_fid": inputs.building_fid,
        "hours_simulated": len(hourly),
        "weather_hours": full_hour_count,
        "partial_period": len(hourly) < full_hour_count,
        "location": {"latitude": weather.latitude, "longitude": weather.longitude},
        "weather_source": weather.source,
        "snow": snow.metadata,
        "context": {
            "enabled": context is not None,
            "coverage_complete": context.coverage_complete if context else None,
            "lidar_sources": list(context.lidar_sources) if context else [],
            "render": asdict(render_stats) if render_stats else None,
        },
        "module_dimensions_m": [width, height],
        "installed_layout_unavailable": installed_reason,
        "designs": {key: designs[key] for key in ("installed", "relocated", "clean_slate") if key in designs},
        "designs_by_count": designs_by_count,
        "scene": {
            "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(),
            "origin": mesh.vertices.mean(axis=0).tolist(),
            "samples": receivers.points.tolist(),
            "use_labels": resource.use_labels.tolist(),
            "irradiation_kwh_m2": {
                state: value.irradiation.tolist() for state, value in results.items()
            },
        },
    }


def _designs_by_count(optimization, maximum_design, design_for, progress):
    """Solve each cardinality; a smaller optimum need not belong to the maximum."""
    variants = {}
    for target_count in range(optimization.maximum_count + 1):
        progress(f"Preparing panel count {target_count} of {optimization.maximum_count}")
        if target_count == optimization.maximum_count:
            design = {key: value for key, value in maximum_design.items() if key != "hourly"}
        else:
            solution = select_milp(optimization.candidates, optimization.conflict_pairs, panel_count=target_count)
            design = design_for(solution, include_hourly=False)
        variants[str(target_count)] = design
    return variants


def scene_mesh(scene):
    return scene.partition.full_mesh


def write_result(result: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False) + "\n")
    temporary.replace(output)
    return output
