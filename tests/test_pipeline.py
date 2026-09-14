"""Offline end-to-end tests use actual EGL rendering and placement optimization."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shading_aware_pv.inputs import SimulationInputs
from shading_aware_pv.models import Mesh, Weather
from shading_aware_pv.shadow import DepthRenderer
from shading_aware_pv.simulation import SimulationConfig, simulate, write_result
from shading_aware_pv.snow import SnowData


def test_egl_visibility_is_translation_invariant():
    points = np.array([[0.0, 0.0, 0.2], [1.5, 0.0, 0.2]])
    vertices = np.array(
        [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [-0.5, 0.5, 1.0]]
    )
    for origin in (np.zeros(3), np.array([2600000.0, 1200000.0, 450.0])):
        renderer = DepthRenderer(
            [Mesh(vertices + origin, np.array([[0, 1, 2], [0, 2, 3]]))],
            points + origin,
            pixel_size=0.02,
        )
        try:
            np.testing.assert_array_equal(
                renderer.visibility(0, np.array([0.0, 0.0, 1.0])), [False, True]
            )
        finally:
            renderer.close()


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("snow_available", [False, True])
def test_complete_pipeline_with_and_without_existing_pv(tmp_path, installed, snow_available):
    mesh = tmp_path / "mesh.ply"
    mesh.write_text("""ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
0 0 2
4 0 2
4 4 2
0 4 2
3 0 1 2
3 0 2 3
""")
    details = tmp_path / "details.geojson"
    features = []
    if installed:
        import rasterio
        from rasterio.transform import from_origin

        features = [
            {
                "type": "Feature",
                "properties": {"class_label": "pvmodule"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0.8, 0.8], [1.8, 0.8], [1.8, 1.8], [0.8, 1.8], [0.8, 0.8]]
                    ],
                },
            }
        ]
        with rasterio.open(
            tmp_path / "unused.tif",
            "w",
            driver="GTiff",
            width=40,
            height=40,
            count=3,
            dtype="uint8",
            transform=from_origin(0, 4, 0.1, 0.1),
        ) as image:
            image.write(np.full((3, 40, 40), 128, dtype=np.uint8))
    details.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    scaffold = tmp_path / "scaffold.geojson"
    scaffold.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"kind": "roof_face", "plane_coeffs": [0, 0, 2], "face_id": "roof"},
        "geometry": {"type": "Polygon", "coordinates": [[[0,0],[4,0],[4,4],[0,4],[0,0]]]}
    }]}))
    index = pd.date_range("2023-06-01 12:00", periods=2, freq="h", tz="UTC")
    hourly = pd.DataFrame(
        dict(
            ghi=[900.0, 800.0],
            dhi=[100.0, 100.0],
            dni=[800.0, 700.0],
            temp_air=[20.0, 21.0],
            wind_speed=[2.0, 2.0],
            solar_zenith=[20.0, 25.0],
            solar_azimuth=[180.0, 200.0],
            dni_extra=[1360.0, 1360.0],
            airmass=[1.1, 1.2],
        ),
        index=index,
    )
    if snow_available:
        hourly.loc[index[-1] + pd.Timedelta(hours=1)] = hourly.iloc[-1]
        index = hourly.index
        hourly["temp_air"] = -20.0
        snow = SnowData(
            pd.Series([10.0, 0.0, 0.0], index=index),
            pd.Series([10.0, 10.0, 10.0], index=index),
            {"available": True, "source": "synthetic snowfall"},
        )
    else:
        snow = SnowData(None, None, {"available": False, "reason": "fixture"})
    weather = Weather(hourly, 46.2, 6.1, 450.0, "synthetic regression fixture")
    result = simulate(
        SimulationInputs(
            mesh,
            details,
            tmp_path / "unused.tif",
            tmp_path / "unused.gpkg",
            1,
            tmp_path / "cache",
            scaffold,
        ),
        SimulationConfig(
            context_half_extent=0,
            sample_spacing=0.25,
            shadow_pixel_size=0.1,
            placement_phase_step=0.4,
            dc_capacity_kwp=4.0,
            inverter_ac_kw=3.3,
        ),
        weather=weather,
        open_horizon_weather=weather,
        snow=snow,
        progress=lambda text: None,
    )
    if installed:
        assert result["installed_layout_unavailable"] is None
        installed_design = result["designs"]["installed"]
        assert installed_design["electrical"]["dc_capacity_kwp"] == 4.0
        assert installed_design["electrical"]["inverter_ac_kw"] == 3.3
        relocated = result["designs"]["relocated"]
        assert relocated["same_count_feasible"]
        assert relocated["states"]["full"]["modeled_kwh"] >= installed_design["states"]["full"]["modeled_kwh"]
        if relocated["retained_installed"]:
            for key in ("corners", "hourly", "monthly", "states", "electrical"):
                assert relocated[key] == installed_design[key]
        assert relocated["panel_count"] == installed_design["panel_count"]
        assert (
            relocated["electrical"]["dc_capacity_kwp"]
            == installed_design["electrical"]["dc_capacity_kwp"]
        )
    else:
        assert result["installed_layout_unavailable"]
        assert "installed" not in result["designs"]
    design = result["designs"]["clean_slate"]
    if not installed:
        assert design["electrical"]["module_wattage_w"] == pytest.approx(340.0)
        assert design["electrical"]["dc_capacity_kwp"] == pytest.approx(
            design["panel_count"] * 0.34
        )
    assert design["panel_count"] > 0
    assert design["states"]["full"]["modeled_kwh"] >= 0
    if installed and snow_available:
        assert design["snow_applied"]
        assert design["states"]["full"]["snow_loss_kwh"] > 0
    else:
        assert not design["snow_applied"]
        assert design["states"]["full"]["modeled_kwh"] > 0
        assert design["states"]["full"]["snow_loss_kwh"] == 0
    if not installed:
        assert design["snow_basis"] == "snow-free placement estimate"
    assert design["geometry_check"]["containment_or_clearance_violations"] == 0
    assert set(design["states"]) == {
        "open_horizon",
        "unshaded",
        "roof",
        "local",
        "context_no_vegetation",
        "full",
    }
    variants = result["designs_by_count"]
    assert set(variants) == {str(k) for k in range(design["panel_count"] + 1)}
    assert variants[str(design["panel_count"])] == {key: value for key, value in design.items() if key != "hourly"}
    for key, variant in variants.items():
        assert variant["panel_count"] == len(variant["corners"]) == int(key)
        assert "hourly" not in variant
        for state, metric in variant["states"].items():
            assert sum(variant["monthly"]["ac_kwh"][state]) == pytest.approx(metric["modeled_kwh"])
        if int(key):
            assert variant["electrical"]["dc_capacity_kwp"] == pytest.approx(int(key) * variant["electrical"]["module_wattage_w"] / 1000)
        else:
            assert variant["electrical"]["dc_capacity_kwp"] == 0
            assert not variant["snow_applied"]
            assert all(metric["modeled_kwh"] == 0 for metric in variant["states"].values())
    output = write_result(result, tmp_path / "result.json")
    assert json.loads(output.read_text())["schema"] == "shading-aware-pv/v1"
    assert result["provenance"]["method_revision"] == 8
    assert not list(tmp_path.glob("*.html"))


@pytest.mark.parametrize(
    "options",
    [
        {"max_hours": 0},
        {"sample_spacing": 0},
        {"context_half_extent": -1},
        {"module_gap": -1},
        {"module_width_m": float("nan")},
    ],
)
def test_invalid_simulation_settings_fail_before_work(options):
    with pytest.raises(ValueError):
        SimulationConfig(**options)


def test_snow_station_acquisition_does_not_require_radiation_measurements(tmp_path):
    from shading_aware_pv.snow_weather import fetch_snow_station

    index = pd.date_range("2023-01-01", periods=1, tz="UTC")
    weather = Weather(pd.DataFrame(index=index), 46.247519, 6.127742, 411.0, "fixture")
    path = tmp_path / "ogd-smn_gve_h_historical_2020-2029.csv"
    path.write_text(
        "reference_timestamp;htoauths;rre150h0;tre200h0\n01.01.2023 00:00;1;1;0\n"
    )
    result = fetch_snow_station(weather, 2023, tmp_path)
    assert result["cached_file"] == str(path)
    assert result["station_id"] == "GVE"
    assert result["distance_km"] == pytest.approx(0.0)
    assert "components" not in result
