"""Numerical regressions retained from the original pipeline test suite."""
from __future__ import annotations
import io
import json
import shlex
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box
from shading_aware_pv.context import build_context_dsm
from shading_aware_pv.geometry import mesh_footprint, partition_mesh, raised_detail_coverage, sample_roof
from shading_aware_pv.irradiance import facet_irradiance
from shading_aware_pv.models import IrradianceResult, Mesh, MeshPartition, ModuleArray, ModuleCell, ModuleLayout, PanelLayout, RoofResource, RoofSamples, RoofScene, Weather
from shading_aware_pv.modules import _CandidateFit, _roof_frame, _select_candidate
from shading_aware_pv.optimization import PlacementCandidate, optimize_placements, placement_audit, select_milp
from shading_aware_pv.panels import NoPVModulesError, load_panel_layout, roof_use_summary
from shading_aware_pv.shadow import DepthRenderer
from shading_aware_pv.snow import SnowData, automatic_num_strings, load_snow_data, snow_loss_by_sample
from shading_aware_pv.yield_model import analyze_yield, default_electrical_inputs, ideal_dc_energy, modeled_energy, percent_loss
def _flat_roof(z: float = 0.0) -> Mesh:
    return Mesh(
        vertices=np.array(
            [[0, 0, z], [2, 0, z], [2, 2, z], [0, 2, z]],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )

def test_ideal_dc_comparison_excludes_user_losses_and_inverter_limit() -> None:
    inputs = default_electrical_inputs(
        10,
        2.0,
        dc_capacity_kwp=4.0,
        inverter_ac_kw=1.0,
        general_loss_percent=20.0,
        inverter_efficiency_percent=90.0,
    )

    ideal = ideal_dc_energy(np.array([1_000.0]), np.array([25.0]), inputs)
    delivered = modeled_energy(np.array([1_000.0]), np.array([25.0]), inputs)

    assert ideal.item() == pytest.approx(4.0)
    assert delivered.item() == pytest.approx(1.0)

def _scene_with_box() -> Mesh:
    vertices = np.array(
        [
            [-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0],
            [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0],
            [-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [8, 9, 10], [8, 10, 11],
            [4, 5, 9], [4, 9, 8], [5, 6, 10], [5, 10, 9],
            [6, 7, 11], [6, 11, 10], [7, 4, 8], [7, 8, 11],
        ],
        dtype=np.int32,
    )
    return Mesh(vertices, faces)

def test_roof_sampling_uses_intrinsic_mesh_height_and_area() -> None:
    samples = sample_roof(_flat_roof(), spacing=0.25)
    assert len(samples.points) == 81
    assert np.allclose(samples.points[:, 2], 0)
    assert np.allclose(samples.normals, [0, 0, 1])
    assert np.allclose(samples.areas, 0.25**2)

def test_panel_milp_respects_setback_and_exact_requested_count() -> None:
    roof = _flat_roof()
    samples = sample_roof(roof, spacing=0.25)
    empty = Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32))
    scene = RoofScene(
        source_path=Path("roof.ply"),
        partition=MeshPartition(
            base_roof=roof,
            main_roof=roof,
            details=empty,
            component_count=1,
            full_mesh=roof, non_pv_mesh=roof, pv_component_count=0,
        ),
        samples=samples,
        excluded_by_raised_detail=np.zeros(len(samples.points), dtype=bool),
    )
    irradiation = np.linspace(900.0, 1_100.0, len(samples.points))
    resource = RoofResource(
        roof_samples=samples,
        receivers=samples,
        use_labels=np.full(len(samples.points), "free_roof"),
        results={
            "full": IrradianceResult(
                irradiation=irradiation,
                direct_shaded_hours=np.zeros(len(samples.points)),
            )
        },
    )

    result = optimize_placements(
        scene,
        resource,
        mesh_footprint(empty),
        current_panel_count=2,
        module_width_m=0.8,
        module_height_m=0.8,
        roof_setback_m=0.1,
        obstruction_setback_m=0.1,
        module_gap_m=0.0,
        phase_step_m=0.4,
        mounting_clearance_m=0.2,
    )

    assert result.maximum_count == 4
    assert result.relocated_layout.panel_count == 2
    for candidate_id in result.maximum_layout.candidate_ids:
        corners = result.candidates[int(candidate_id)].roof_corners_xyz
        assert np.all(corners[:, :2] >= 0.1 - 1e-9)
        assert np.all(corners[:, :2] <= 1.9 + 1e-9)

def test_panel_milp_accepts_a_zero_irradiation_window() -> None:
    candidates = tuple(
        PlacementCandidate(
            candidate_id=index,
            facet_id="roof",
            orientation="portrait",
            roof_corners_xyz=np.zeros((4, 3)),
            receiver_corners_xyz=np.zeros((4, 3)),
            roof_sample_indices=np.array([index]),
            annual_irradiation_kwh_m2=0.0,
        )
        for index in range(2)
    )
    no_overlaps = np.empty((0, 2), dtype=np.int32)

    maximum = select_milp(candidates, no_overlaps, panel_count=None)
    exact = select_milp(candidates, no_overlaps, panel_count=1)

    assert maximum.panel_count == 2
    assert exact.panel_count == 1

def test_panel_milp_excludes_non_pv_obstruction_footprint() -> None:
    roof = _flat_roof()
    samples = sample_roof(roof, spacing=0.2)
    obstruction = Mesh(
        vertices=np.array(
            [[0.8, 0.8, 0.3], [1.2, 0.8, 0.3], [1.2, 1.2, 0.3], [0.8, 1.2, 0.3]]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    scene = RoofScene(
        source_path=Path("roof.ply"),
        partition=MeshPartition(roof, roof, obstruction, 2, roof, roof, 0),
        samples=samples,
        excluded_by_raised_detail=np.zeros(len(samples.points), dtype=bool),
    )
    resource = RoofResource(
        roof_samples=samples,
        receivers=samples,
        use_labels=np.full(len(samples.points), "free_roof"),
        results={
            "full": IrradianceResult(
                irradiation=np.ones(len(samples.points)) * 1_000.0,
                direct_shaded_hours=np.zeros(len(samples.points)),
            )
        },
    )
    result = optimize_placements(
        scene,
        resource,
        mesh_footprint(obstruction),
        current_panel_count=2,
        module_width_m=0.4,
        module_height_m=0.4,
        roof_setback_m=0.1,
        obstruction_setback_m=0.15,
        module_gap_m=0.0,
        phase_step_m=0.2,
        mounting_clearance_m=0.2,
    )

    footprint = mesh_footprint(obstruction)
    for candidate in result.candidates:
        panel = Polygon(candidate.roof_corners_xyz[:, :2])
        assert panel.distance(footprint) >= 0.15 - 1e-8
    audit = placement_audit(
        scene,
        mesh_footprint(obstruction),
        result,
        result.maximum_layout,
    )
    assert audit["containment_or_clearance_violations"] == 0
    assert audit["panel_spacing_violations"] == 0

def test_module_aligned_lattice_allows_zero_gap_neighbors() -> None:
    roof = Mesh(
        vertices=np.array(
            [[0, 0, 0], [2.1, 0, 0], [2.1, 1.7, 0], [0, 1.7, 0]],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    samples = sample_roof(roof, spacing=0.1)
    empty = Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32))
    scene = RoofScene(
        source_path=Path("roof.ply"),
        partition=MeshPartition(roof, roof, empty, 1, roof, roof, 0),
        samples=samples,
        excluded_by_raised_detail=np.zeros(len(samples.points), dtype=bool),
    )
    resource = RoofResource(
        roof_samples=samples,
        receivers=samples,
        use_labels=np.full(len(samples.points), "free_roof"),
        results={
            "full": IrradianceResult(
                irradiation=np.ones(len(samples.points)) * 1_000.0,
                direct_shaded_hours=np.zeros(len(samples.points)),
            )
        },
    )

    touching = optimize_placements(
        scene,
        resource,
        mesh_footprint(empty),
        current_panel_count=2,
        module_width_m=1.05,
        module_height_m=1.7,
        roof_setback_m=0.0,
        obstruction_setback_m=0.0,
        module_gap_m=0.0,
        phase_step_m=0.1,
        mounting_clearance_m=0.2,
    )
    separated = optimize_placements(
        scene,
        resource,
        mesh_footprint(empty),
        current_panel_count=2,
        module_width_m=1.05,
        module_height_m=1.7,
        roof_setback_m=0.0,
        obstruction_setback_m=0.0,
        module_gap_m=0.1,
        phase_step_m=0.1,
        mounting_clearance_m=0.2,
    )

    assert touching.maximum_count == 2
    assert separated.maximum_count == 1
    selected = [
        Polygon(touching.candidates[index].roof_corners_xyz[:, :2])
        for index in touching.maximum_layout.candidate_ids
    ]
    assert selected[0].distance(selected[1]) == pytest.approx(0.0)
    audit = placement_audit(scene, mesh_footprint(empty), touching, touching.maximum_layout)
    assert audit["minimum_module_gap_m"] == pytest.approx(0.0)

def test_roof_sampling_keeps_the_uppermost_overlapping_surface() -> None:
    lower = _flat_roof(0.0)
    upper = _flat_roof(1.0)
    mesh = Mesh(
        vertices=np.vstack((lower.vertices, upper.vertices)),
        faces=np.vstack((lower.faces, upper.faces + len(lower.vertices))),
    )

    samples = sample_roof(mesh, spacing=0.25)

    assert np.allclose(samples.points[:, 2], 1.0)

def test_detail_extraction_and_occupancy_ignore_host_roof() -> None:
    samples = sample_roof(_flat_roof(), spacing=0.25)
    original = _scene_with_box()
    mesh = Mesh(original.vertices + [1.0, 1.0, 0.0], original.faces)
    details = partition_mesh(mesh, face_labels=np.array(["scaffold"] * 2 + ["other"] * (len(mesh.faces) - 2))).details
    covered = raised_detail_coverage(samples, details, clearance=0.15)
    assert len(details.faces) == 10
    assert covered.any()
    assert np.all((samples.points[covered, 0] >= 0.5) & (samples.points[covered, 0] <= 1.5))
    assert np.all((samples.points[covered, 1] >= 0.5) & (samples.points[covered, 1] <= 1.5))

def test_occupancy_is_invariant_to_each_scene_height() -> None:
    lower = _scene_with_box()
    upper = Mesh(lower.vertices + [0.0, 0.0, 10.0], lower.faces)

    lower_partition = partition_mesh(lower, face_labels=np.array(["scaffold"] * 2 + ["other"] * (len(lower.faces) - 2)))
    upper_partition = partition_mesh(upper, face_labels=np.array(["scaffold"] * 2 + ["other"] * (len(upper.faces) - 2)))
    lower_samples = sample_roof(lower_partition.main_roof, spacing=0.25)
    upper_samples = sample_roof(upper_partition.main_roof, spacing=0.25)

    lower_covered = raised_detail_coverage(
        lower_samples,
        lower_partition.details,
        clearance=0.15,
    )
    upper_covered = raised_detail_coverage(
        upper_samples,
        upper_partition.details,
        clearance=0.15,
    )

    assert np.array_equal(lower_covered, upper_covered)
    assert not raised_detail_coverage(
        upper_samples,
        lower_partition.details,
        clearance=0.15,
    ).any()

def test_mesh_partition_uses_explicit_semantics_not_detail_size() -> None:
    mesh = _scene_with_box()
    partition = partition_mesh(mesh, face_labels=np.array(["scaffold"] * 2 + ["other"] * (len(mesh.faces) - 2)))

    assert partition.component_count == 2
    assert len(partition.base_roof.faces) == 2
    assert mesh_footprint(partition.main_roof).intersection(mesh_footprint(partition.details)).area == pytest.approx(0)
    assert len(partition.details.faces) == 10

def test_mesh_partition_welds_duplicate_triangle_vertices() -> None:
    mesh = Mesh(
        vertices=np.array(
            [
                [0, 0, 0], [2, 0, 0], [2, 2, 0],
                [0, 0, 0], [2, 2, 0], [0, 2, 0],
                [0.5, 0.5, 0.2], [1.0, 0.5, 0.2], [0.5, 1.0, 0.2],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32),
    )

    partition = partition_mesh(mesh, face_labels=np.array(["scaffold"] * 2 + ["other"] * (len(mesh.faces) - 2)))

    assert partition.component_count == 2
    assert len(partition.base_roof.faces) == 2
    assert len(partition.details.faces) == 1

def test_context_dsm_removes_target_building_but_keeps_vegetation() -> None:
    points = np.array(
        [
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [0.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
            [0.5, 0.5, 5.0],
            [0.5, 0.5, 7.0],
            [1.5, 1.5, 4.0],
        ]
    )
    classifications = np.array([2, 2, 2, 2, 6, 3, 6])
    footprint = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    mesh, shape, retained = build_context_dsm(
        points,
        classifications,
        bounds_xy=(0.0, 0.0, 2.0, 2.0),
        resolution_m=1.0,
        target_footprint=footprint,
    )

    heights = mesh.vertices[:, 2].reshape(shape)
    assert heights[0, 0] == 7.0
    assert heights[1, 1] == 4.0
    assert not retained[4]
    assert retained[5]
    assert retained[6]

def test_context_dsm_can_isolate_vegetation() -> None:
    points = np.array(
        [
            [0.5, 0.5, 1.0],
            [1.5, 0.5, 2.0],
            [0.5, 0.5, 7.0],
            [1.5, 0.5, 5.0],
        ]
    )
    classifications = np.array([2, 2, 3, 6])

    mesh, shape, _ = build_context_dsm(
        points,
        classifications,
        bounds_xy=(0.0, 0.0, 2.0, 1.0),
        resolution_m=1.0,
        target_footprint=Polygon(),
        excluded_surface_classes=(3,),
    )

    assert np.array_equal(mesh.vertices[:, 2].reshape(shape), [[1.0, 5.0]])

def test_context_dsm_falls_back_to_ground_in_removed_target_cell() -> None:
    points = np.array(
        [
            [0.5, 0.5, 1.0],
            [1.5, 0.5, 2.0],
            [0.5, 0.5, 6.0],
        ]
    )
    classifications = np.array([2, 2, 6])

    mesh, shape, retained = build_context_dsm(
        points,
        classifications,
        bounds_xy=(0.0, 0.0, 2.0, 1.0),
        resolution_m=1.0,
        target_footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    )

    assert np.array_equal(mesh.vertices[:, 2].reshape(shape), [[1.0, 2.0]])
    assert np.array_equal(retained, [True, True, False])

def test_nighttime_perez_values_are_zero_not_nan() -> None:
    index = pd.DatetimeIndex(["2023-01-01 00:00Z", "2023-01-01 12:00Z"])
    hourly = pd.DataFrame(
        {
            "solar_zenith": [120.0, 30.0],
            "solar_azimuth": [0.0, 180.0],
            "dni": [0.0, 600.0],
            "ghi": [0.0, 500.0],
            "dhi": [0.0, 100.0],
            "dni_extra": [1412.0, 1412.0],
            "airmass": [np.nan, 1.2],
        },
        index=index,
    )
    samples = RoofSamples(
        points=np.array([[0.0, 0.0, 0.0]]),
        normals=np.array([[0.0, -0.5, np.sqrt(0.75)]]),
        areas=np.array([1.0]),
        face_ids=np.array(["roof-1"], dtype=object),
    )
    weather = Weather(hourly, 47.0, 8.0, 400.0, "test")
    components = facet_irradiance(samples, weather)
    assert all(np.isfinite(component).all() for component in components.values())
    assert all(component[0, 0] == 0.0 for component in components.values())

def test_panel_layout_uses_semantic_footprint_and_removes_its_solid(
    tmp_path: Path,
) -> None:
    mesh = _scene_with_box()
    partition = partition_mesh(mesh, face_labels=np.array(["scaffold"] * 2 + ["pvmodule"] * (len(mesh.faces) - 2)))
    samples = sample_roof(partition.main_roof, spacing=0.25)
    scene = RoofScene(
        source_path=tmp_path / "mesh.ply",
        partition=partition,
        samples=samples,
        excluded_by_raised_detail=raised_detail_coverage(
            samples,
            partition.details,
            clearance=0.15,
        ),
    )
    details_path = tmp_path / "roof_details.geojson"
    details_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"class_label": "pvmodule"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
                                [-0.5, 0.5], [-0.5, -0.5],
                            ]],
                        },
                    }
                ],
            }
        )
    )
    orthophoto_path = tmp_path / "orthophoto.tif"
    with rasterio.open(
        orthophoto_path,
        "w",
        driver="GTiff",
        width=40,
        height=40,
        count=3,
        dtype="uint8",
        transform=from_origin(-2.0, 2.0, 0.1, 0.1),
    ) as dataset:
        dataset.write(np.full((3, 40, 40), 128, dtype=np.uint8))

    layout = load_panel_layout(scene, details_path, orthophoto_path, 0.20)

    assert layout.feature_count == 1
    assert np.isclose(layout.detected_footprint_area_xy_m2, 1.0)
    assert len(layout.modules.cells) == 1
    assert layout.removed_detail_components == 1
    assert len(layout.samples.points) > 0
    roof_points = scene.samples.points[layout.roof_sample_indices]
    receiver_offsets = layout.samples.points - roof_points
    assert np.allclose(
        receiver_offsets,
        0.20 * layout.samples.normals,
    )
    cell = layout.modules.cells[0]
    assert np.allclose(
        np.linalg.norm(
            cell.receiver_corners_xyz - cell.roof_corners_xyz,
            axis=1,
        ),
        0.20,
    )
    assert len(layout.occluder_mesh.faces) == len(partition.base_roof.faces)
    assert len(layout.non_panel_details.faces) == 0
    assert layout.non_panel_detail_components == 0
    assert layout.non_panel_obstruction_area_xy_m2 == 0.0
    roof_use = roof_use_summary(scene, layout)
    assert np.isclose(
        roof_use["total_main_roof_m2"],
        roof_use["existing_pv_m2"]
        + roof_use["unresolved_pv_m2"]
        + roof_use["other_raised_detail_m2"]
        + roof_use["free_roof_m2"],
    )

def test_panel_layout_reports_missing_pv_as_a_detection_failure(
    tmp_path: Path,
) -> None:
    scene = RoofScene(
        source_path=tmp_path / "mesh.ply",
        partition=MeshPartition(
            main_roof=_flat_roof(),
            base_roof=_flat_roof(),
            details=Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)),
            component_count=1,
            full_mesh=_flat_roof(), non_pv_mesh=_flat_roof(), pv_component_count=0,
        ),
        samples=sample_roof(_flat_roof(), spacing=0.25),
        excluded_by_raised_detail=np.zeros(64, dtype=bool),
    )
    details_path = tmp_path / "roof_details.geojson"
    details_path.write_text('{"type":"FeatureCollection","features":[]}')

    with pytest.raises(NoPVModulesError, match="No pvmodule features"):
        load_panel_layout(scene, details_path, tmp_path / "unused.tif", 0.20)

def test_module_dimensions_are_measured_in_the_tilted_roof_plane() -> None:
    roof_xy = Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])
    frame = _roof_frame("roof", roof_xy, (0.5, 0.0, 10.0))
    physical_module = np.array([[0, 0], [1, 0], [1, 2], [0, 2]], dtype=float)
    projected = Polygon(frame.uv_to_xyz(physical_module)[:, :2])
    recovered = frame.xy_to_uv(projected)

    min_u, min_v, max_u, max_v = recovered.bounds
    assert np.isclose(max_u - min_u, 1.0)
    assert np.isclose(max_v - min_v, 2.0)
    assert projected.area < np.ptp(physical_module[:, 0]) * np.ptp(physical_module[:, 1])

def test_common_module_size_breaks_only_near_ties() -> None:
    def candidate(width: float, height: float, score: float) -> _CandidateFit:
        return _CandidateFit(
            width_m=width,
            height_m=height,
            blocks=(),
            score=score,
            precision=0.9,
            recall=0.9,
            boundary_evidence=None,
            module_count=10,
        )

    nonstandard = candidate(1.25, 1.70, 0.900)
    standard_near_tie = candidate(1.05, 1.70, 0.895)
    selected, _ranked = _select_candidate([nonstandard, standard_near_tie])
    assert selected is standard_near_tie

    standard_weaker = candidate(1.05, 1.70, 0.880)
    selected, _ranked = _select_candidate([nonstandard, standard_weaker])
    assert selected is nonstandard

def test_electrical_defaults_use_inferred_module_geometry() -> None:
    inputs = default_electrical_inputs(100, 1.0)

    assert inputs.module_wattage_w == 200.0
    assert inputs.dc_capacity_kwp == 20.0
    assert np.isclose(inputs.inverter_ac_kw, 20.0 / 1.15)
    assert inputs.general_loss_percent == 0.0
    assert inputs.snow_num_strings == 1

def test_module_wattage_and_total_capacity_are_equivalent_inputs() -> None:
    by_module = default_electrical_inputs(12, 1.7, module_wattage_w=425.0)
    by_system = default_electrical_inputs(12, 1.7, dc_capacity_kwp=5.1)

    assert by_module.dc_capacity_kwp == 5.1
    assert by_system.module_wattage_w == 425.0
    with pytest.raises(ValueError, match="module wattage or total DC capacity"):
        default_electrical_inputs(
            12,
            1.7,
            module_wattage_w=425.0,
            dc_capacity_kwp=5.1,
        )

def test_snow_num_strings_follows_dominant_module_orientation() -> None:
    def module_array(orientation: str, count: int) -> ModuleArray:
        return ModuleArray(
            array_id=orientation,
            facet_id="roof",
            orientation=orientation,
            module_count=count,
            detected_area_m2=10.0,
            inferred_area_m2=10.0,
            mask_precision=1.0,
            mask_recall=1.0,
            boundary_evidence=None,
            roof_tilt_deg=30.0,
            axis_source="roof",
            axis_confidence=1.0,
        )

    assert automatic_num_strings((module_array("portrait", 8),))[0] == 1
    strings, basis = automatic_num_strings(
        (module_array("portrait", 3), module_array("landscape", 5))
    )
    assert strings == 3
    assert "landscape" in basis

def test_meteoswiss_snow_proxy_uses_confirmed_depth_increases(tmp_path: Path) -> None:
    index = pd.date_range("2023-01-01", periods=4, freq="h", tz="UTC")
    station_path = tmp_path / "station.csv"
    pd.DataFrame(
        {
            "reference_timestamp": index.strftime("%d.%m.%Y %H:%M"),
            "htoauths": [0.0, 2.0, 2.0, 1.0],
            "rre150h0": [0.0, 1.0, 0.0, 0.0],
            "tre200h0": [-1.0, -1.0, -1.0, -1.0],
        }
    ).to_csv(station_path, sep=";", index=False)
    weather = Weather(
        hourly=pd.DataFrame({"temp_air": [-1.0] * 4}, index=index),
        latitude=47.0,
        longitude=9.0,
        elevation=500.0,
        source="test",
    )
    snow = load_snow_data(
        weather,
        {
            "station_id": "TST",
            "station_name": "Test",
            "distance_km": 1.0,
            "cached_file": str(station_path),
            "source": "test station",
            "source_url": "https://example.test",
        },
    )

    assert snow.available
    assert snow.snowfall_cm.tolist() == [0.0, 2.0, 0.0, 0.0]
    assert snow.metadata["snowfall_event_hours"] == 1

def test_nrel_snow_loss_is_applied_before_inverter_clipping() -> None:
    index = pd.date_range("2023-01-01", periods=4, freq="h", tz="UTC")
    samples = RoofSamples(
        points=np.array([[0.0, 0.0, 0.0]]),
        normals=np.array([[0.0, 0.0, 1.0]]),
        areas=np.ones(1),
        face_ids=np.array([0]),
    )
    weather = Weather(
        hourly=pd.DataFrame({"temp_air": [-2.0] * 4}, index=index),
        latitude=47.0,
        longitude=9.0,
        elevation=500.0,
        source="test",
    )
    snow = SnowData(
        snowfall_cm=pd.Series([2.0, 0.0, 0.0, 0.0], index=index),
        snow_depth_cm=pd.Series([2.0, 2.0, 2.0, 0.0], index=index),
        metadata={"available": True},
    )
    poa = np.array([[0.0], [200.0], [400.0], [400.0]])
    coverage, loss = snow_loss_by_sample(poa, samples, weather, snow, 1)

    assert coverage[:, 0].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert loss[:, 0].tolist() == [1.0, 1.0, 1.0, 0.0]
    inputs = default_electrical_inputs(5, 1.0)
    energy = modeled_energy(
        poa[:, 0],
        np.full(4, 25.0),
        inputs,
        loss[:, 0],
    )
    assert energy[:3].tolist() == [0.0, 0.0, 0.0]
    assert energy[3] > 0.0

def test_zero_irradiation_has_zero_reported_loss() -> None:
    assert percent_loss(0.0, 0.0) == 0.0

def test_yield_analysis_separates_direct_and_total_poa_losses() -> None:
    index = pd.DatetimeIndex(["2023-06-01 12:00Z"])
    weather = Weather(
        hourly=pd.DataFrame(
            {
                "solar_zenith": [30.0],
                "solar_azimuth": [180.0],
                "dni": [600.0],
                "ghi": [500.0],
                "dhi": [100.0],
                "dni_extra": [1325.0],
                "airmass": [1.2],
                "temp_air": [20.0],
                "wind_speed": [2.0],
            },
            index=index,
        ),
        latitude=47.0,
        longitude=9.0,
        elevation=500.0,
        source="test",
    )
    samples = RoofSamples(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        normals=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        areas=np.ones(2),
        face_ids=np.array([0, 0]),
    )
    layout = PanelLayout(
        source_path=Path("panels.geojson"),
        samples=samples,
        roof_sample_indices=np.array([0, 1]),
        detected_roof_sample_indices=np.array([0, 1]),
        footprint_rings=(),
        detected_footprint_rings=(),
        feature_count=1,
        footprint_area_xy_m2=2.0,
        detected_footprint_area_xy_m2=2.0,
        sampled_surface_area_m2=2.0,
        non_panel_details=Mesh(
            np.empty((0, 3)),
            np.empty((0, 3), dtype=np.int32),
        ),
        non_panel_detail_components=0,
        non_panel_obstruction_area_xy_m2=0.0,
        occluder_mesh=_flat_roof(),
        removed_detail_components=1,
        modules=ModuleLayout(
            module_width_m=1.0,
            module_height_m=2.0,
            mounting_clearance_m=0.20,
            cells=(),
            arrays=(),
            candidates=(),
            detected_surface_area_m2=2.0,
            inferred_surface_area_m2=2.0,
            projected_area_m2=2.0,
            mask_precision=1.0,
            mask_recall=1.0,
            boundary_evidence=None,
            evidence_score=1.0,
            evidence_gap_to_best=0.0,
            nearest_standard_width_m=1.0,
            nearest_standard_height_m=2.0,
            standard_distance_m=0.0,
            score_tie_tolerance=0.01,
            orthophoto_path=Path("orthophoto.tif"),
        ),
    )
    visibility = {
        "unshaded": np.array([[True, True]]),
        "local": np.array([[True, False]]),
        "full": np.array([[False, False]]),
    }

    analysis = analyze_yield(
        layout,
        weather,
        visibility,
        default_electrical_inputs(2, 1.0),
    )

    local = analysis.metrics["local"]
    full = analysis.metrics["full"]
    assert np.isclose(local["direct_loss_vs_baseline_percent"], 50.0)
    assert 0 < local["total_poa_loss_vs_baseline_percent"] < 50.0
    assert np.isclose(local["mean_direct_shaded_hours"], 0.5)
    assert np.isclose(full["direct_loss_vs_baseline_percent"], 100.0)
    assert np.isclose(full["mean_direct_shaded_hours"], 1.0)

def test_greedy_exposes_count_failure_on_conflicting_candidates():
    from shading_aware_pv.optimization import select_greedy
    candidates = tuple(
        PlacementCandidate(i, 'roof', 'portrait', np.zeros((4, 3)),
                           np.zeros((4, 3)), np.array([i]), score)
        for i, score in enumerate([10., 6., 6.])
    )
    conflicts = np.array([[0, 1], [0, 2]])
    greedy = select_greedy(candidates, conflicts, panel_count=2)
    optimal = select_milp(candidates, conflicts, panel_count=2)
    assert greedy.candidate_ids.tolist() == [0]
    assert greedy.panel_count == 1
    assert optimal.candidate_ids.tolist() == [1, 2]
    no_conflicts = np.empty((0, 2), dtype=int)
    assert select_greedy(candidates, no_conflicts, panel_count=2).candidate_ids.tolist() == [0, 1]
    assert select_greedy(candidates, no_conflicts, panel_count=0).panel_count == 0
