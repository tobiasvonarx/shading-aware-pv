"""Numerical and geometry regressions for reuse within a solar analysis."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from shading_aware_pv.geometry import partition_mesh, sample_roof
from shading_aware_pv.irradiance import (
    FaceIrradiance,
    IrradianceCache,
    facet_irradiance,
    integrate_irradiance,
)
from shading_aware_pv.models import Mesh, RoofSamples, RoofScene, Weather
from shading_aware_pv.optimization import (
    PlacementAuditor,
    PlacementCandidate,
    PlacementOptimization,
    PlacementSolution,
    placement_audit,
)
from shading_aware_pv.shadow import DepthRenderer, sun_vectors
from shading_aware_pv.snow import SnowData, effective_snow_loss, snow_loss_by_sample
from shading_aware_pv.yield_model import analyze_yield, default_electrical_inputs


@pytest.fixture
def weather():
    hours = 48
    index = pd.date_range("2023-01-01", periods=hours, freq="h", tz="UTC")
    zenith = np.tile(np.linspace(20, 120, 24), 2)
    daylight = zenith < 90
    return Weather(
        pd.DataFrame(
            {
                "solar_zenith": zenith,
                "solar_azimuth": np.linspace(0, 350, hours),
                "dni": np.where(daylight, 650.0, 0.0),
                "ghi": np.where(daylight, 450.0, 0.0),
                "dhi": np.where(daylight, 90.0, 0.0),
                "dni_extra": 1412.0,
                "airmass": np.where(daylight, 1.5, np.nan),
                "temp_air": np.linspace(-5, 10, hours),
                "wind_speed": 2.0,
            },
            index=index,
        ),
        47.0,
        8.0,
        400.0,
        "test",
    )


@pytest.fixture
def samples():
    normals = np.array([[0, 0, 1], [0, -0.5, np.sqrt(0.75)], [0.6, 0, 0.8]])
    face_ids = np.array(["a", "b", "a", "b", "c", "c", "a"])
    return RoofSamples(
        np.zeros((7, 3)),
        normals[[0, 1, 0, 1, 2, 2, 0]],
        np.array([0.01, 0.6, 0.3, 1.8, 0.05, 0.2, 4.0]),
        face_ids,
    )


@pytest.mark.parametrize("pattern", ["lit", "shaded", "mixed"])
@pytest.mark.parametrize("normal_dtype", [np.float32, np.float64])
def test_compact_fields_match_dense_weighted_reductions(
    weather, samples, pattern, normal_dtype
):
    samples = replace(samples, normals=samples.normals.astype(normal_dtype))
    dense = facet_irradiance(samples, weather)
    fields = IrradianceCache().for_samples(samples, weather)
    for name in dense:
        np.testing.assert_array_equal(fields.dense()[name], dense[name])
    visible = np.random.default_rng(83).random((48, 7)) > 0.4
    if pattern != "mixed":
        visible[:] = pattern == "lit"
    poa, direct, shaded = fields.shaded_means(visible, samples.areas)
    np.testing.assert_allclose(
        poa,
        np.average(integrate_irradiance(dense, visible), axis=1, weights=samples.areas),
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        direct,
        np.average(dense["direct"] * visible, axis=1, weights=samples.areas),
        rtol=0,
        atol=1e-10,
    )
    assert shaded == pytest.approx(
        np.average(
            ((dense["direct"] > 20) & ~visible).sum(axis=0), weights=samples.areas
        ),
        abs=1e-12,
    )
    if pattern == "lit":
        assert shaded == 0.0
    for name in ("sky", "ground"):
        np.testing.assert_allclose(
            fields.diffuse_mean(name, samples.areas),
            np.average(dense[name], axis=1, weights=samples.areas),
            rtol=0,
            atol=1e-10,
        )


def test_cache_reuses_orientations_but_separates_weather_and_changed_normals(
    monkeypatch,
    weather,
    samples,
):
    import shading_aware_pv.irradiance as module

    original = module._orientation_irradiance
    calls = []

    def count(*args):
        calls.append(args[:2])
        return original(*args)

    monkeypatch.setattr(module, "_orientation_irradiance", count)
    cache = IrradianceCache()
    cache.for_samples(samples, weather)
    assert len(calls) == 3
    subset = np.array([6, 3, 5])
    smaller = RoofSamples(
        *(
            getattr(samples, attr)[subset]
            for attr in ("points", "normals", "areas", "face_ids")
        )
    )
    cache.for_samples(smaller, weather)
    assert len(calls) == 3
    altered = replace(weather, hourly=weather.hourly.assign(dni=300.0))
    cache.for_samples(smaller, altered)
    assert len(calls) == 6
    tilted = replace(smaller, normals=np.tile([0.0, 0.0, -1.0], (3, 1)))
    cache.for_samples(tilted, weather)
    assert len(calls) == 7


@pytest.mark.parametrize("indices", [[0, 1, 2, 3, 4, 5, 6], [0, 1, 4, 6], [4], []])
@pytest.mark.parametrize("pattern", ["lit", "shaded", "mixed"])
def test_annual_face_totals_match_dense_resource(weather, samples, indices, pattern):
    selected = np.array(indices, dtype=int)
    subset = RoofSamples(
        *(
            getattr(samples, name)[selected]
            for name in ("points", "normals", "areas", "face_ids")
        )
    )
    visible = np.random.default_rng(84).random((48, len(indices))) > 0.4
    if pattern != "mixed":
        visible[:] = pattern == "lit"
    components = facet_irradiance(subset, weather)
    expected = (
        integrate_irradiance(components, visible).sum(axis=0, dtype=np.float64) / 1000.0
    )
    irradiation, shaded = (
        IrradianceCache().for_samples(subset, weather).annual_totals(visible)
    )
    np.testing.assert_array_equal(irradiation, expected)
    np.testing.assert_array_equal(
        shaded, ((components["direct"] > 20) & ~visible).sum(axis=0)
    )


@pytest.mark.parametrize("observed_snow", [False, True])
@pytest.mark.parametrize("area_dtype", [np.float32, np.float64])
def test_shared_visibility_selection_matches_explicit_slice(
    weather, samples, observed_snow, area_dtype
):
    # Reordered and repeated columns must retain sample membership and weights.
    indices = np.array([6, 3, 5, 3])
    subset = RoofSamples(
        *(
            getattr(samples, name)[indices]
            for name in ("points", "normals", "areas", "face_ids")
        )
    )
    subset = replace(subset, areas=subset.areas.astype(area_dtype))
    rng = np.random.default_rng(32)
    visibility = {
        state: rng.random((48, 7)) > 0.3 for state in ("open_horizon", "roof", "full")
    }
    other_weather = replace(
        weather, hourly=weather.hourly.assign(dni=weather.hourly.dni * 0.8)
    )
    snow = SnowData(
        pd.Series([5.0] + [0.0] * 47, index=weather.hourly.index),
        pd.Series(np.linspace(10, 0, 48), index=weather.hourly.index),
        {"available": observed_snow},
    )
    inputs = default_electrical_inputs(4, 1.7)
    kwargs = dict(snow=snow, state_weather={"open_horizon": other_weather})
    expected = analyze_yield(
        subset,
        weather,
        {state: mask[:, indices] for state, mask in visibility.items()},
        inputs,
        **kwargs,
    )
    actual = analyze_yield(
        subset, weather, visibility, inputs, visibility_indices=indices, **kwargs
    )
    pd.testing.assert_frame_equal(actual.hourly, expected.hourly, check_exact=True)
    assert actual.metrics == expected.metrics


@pytest.mark.parametrize("sample_count", [1, 7])
def test_annual_totals_keep_reduction_order_for_single_sample_faces(sample_count):
    groups = (
        (np.array([0]),)
        if sample_count == 1
        else (np.array([0, 3, 5, 6]), np.array([1]), np.array([2, 4]))
    )
    rng = np.random.default_rng(34)
    fields = FaceIrradiance(
        {
            name: (10 ** rng.uniform(-12, 4, (8760, len(groups)))).astype("f4")
            for name in ("direct", "sky", "ground")
        },
        groups,
        sample_count,
    )
    visible = rng.random((8760, sample_count)) > 0.4
    expected = (
        integrate_irradiance(fields.dense(), visible).sum(axis=0, dtype=np.float64)
        / 1000.0
    )
    actual, _ = fields.annual_totals(visible)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "cells",
    [
        [],
        [(1, 2)],
        [(2, 5), (-1, -3), (2, 5), (-1, 7), (0, 2)],
        [(0, 0), (1000000, -1000000), (0, 0)],
        [(-(2**62), 0), (2**62, 1)],
    ],
)
def test_grid_cell_counts_preserve_order_and_large_sparse_spans(cells):
    from shading_aware_pv.modules import _cell_counts

    pairs = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
    actual = _cell_counts(pairs[:, 0], pairs[:, 1])
    expected = np.unique(pairs, axis=0, return_counts=True)
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)


def test_fitting_counts_keep_phase_scores_and_feasibility(monkeypatch):
    import shading_aware_pv.modules as module

    u, v = np.arange(-0.95, 3, 0.1), np.arange(-0.95, 2, 0.1)
    rng = np.random.default_rng(91)
    mask = rng.random((len(v), len(u))) > 0.2
    observation = module._Observation(
        "test",
        None,
        box(-1, -1, 3, 2),
        u,
        v,
        mask,
        rng.random(mask.shape),
        rng.random(mask.shape),
        0.1,
        box(-1, -1, 3, 2),
        box(0.2, -0.5, 0.25, 1.5),
    )
    requests = [
        (width, height, orientation, phase_u, phase_v)
        for width, height, orientation in [
            (1.0, 1.7, "portrait"),
            (1.7, 1.0, "landscape"),
        ]
        for phase_u, phase_v in [(0.0, 0.0), (0.05, 0.02), (0.5, 0.7)]
    ]
    actual = [module._fit_phase(observation, *request) for request in requests]
    monkeypatch.setattr(
        module,
        "_cell_counts",
        lambda rows, columns: np.unique(
            np.column_stack((rows, columns)), axis=0, return_counts=True
        ),
    )
    assert actual == [module._fit_phase(observation, *request) for request in requests]


def test_snow_free_path_avoids_sample_snow_arrays(monkeypatch, weather, samples):
    import shading_aware_pv.yield_model as model

    def unexpected(*args, **kwargs):
        raise AssertionError("Unavailable snow must not allocate sample losses")

    monkeypatch.setattr(model, "snow_loss_by_sample", unexpected)
    result = analyze_yield(
        samples,
        weather,
        {"full": np.ones((48, 7), bool)},
        default_electrical_inputs(4, 1.7),
    )
    assert not result.hourly.snow_coverage_full_fraction.any()
    assert not result.hourly.snow_dc_loss_full_fraction.any()


@pytest.mark.parametrize("area_dtype", [np.float32, np.float64])
def test_no_snow_reductions_and_input_validation(weather, samples, area_dtype):
    samples = replace(samples, areas=samples.areas.astype(area_dtype))
    visible = np.random.default_rng(93).random((48, 7)) > 0.4
    inputs = default_electrical_inputs(4, 1.7)
    result = analyze_yield(samples, weather, {"full": visible}, inputs)
    energy = integrate_irradiance(facet_irradiance(samples, weather), visible)
    expected = np.average(energy, axis=1, weights=samples.areas)
    if area_dtype == np.float32:
        np.testing.assert_array_equal(result.hourly.poa_full_w_m2, expected)
    else:
        np.testing.assert_allclose(
            result.hourly.poa_full_w_m2, expected, rtol=0, atol=1e-10
        )
    assert result.hourly.snow_dc_loss_full_fraction.dtype == np.result_type(
        np.float32, area_dtype
    )
    with pytest.raises(ValueError, match="Snow num_strings must be positive"):
        analyze_yield(
            samples, weather, {"full": visible}, replace(inputs, snow_num_strings=0)
        )


def test_observed_snow_keeps_dense_calculation(weather, samples):
    snow = SnowData(
        pd.Series([5.0] + [0.0] * 47, index=weather.hourly.index),
        pd.Series(np.linspace(10, 0, 48), index=weather.hourly.index),
        {"available": True},
    )
    visible = np.random.default_rng(42).random((48, 7)) > 0.3
    electrical = default_electrical_inputs(4, 1.7)
    result = analyze_yield(samples, weather, {"full": visible}, electrical, snow=snow)
    energy = integrate_irradiance(facet_irradiance(samples, weather), visible)
    coverage, loss = snow_loss_by_sample(
        energy, samples, weather, snow, electrical.snow_num_strings
    )
    np.testing.assert_array_equal(
        result.hourly.poa_full_w_m2, np.average(energy, axis=1, weights=samples.areas)
    )
    np.testing.assert_array_equal(
        result.hourly.snow_coverage_full_fraction,
        np.average(coverage, axis=1, weights=samples.areas),
    )
    np.testing.assert_array_equal(
        result.hourly.snow_dc_loss_full_fraction,
        effective_snow_loss(energy, loss, samples.areas),
    )
    assert result.hourly.snow_coverage_full_fraction.max() > 0


def test_auditor_reuses_checks_and_handles_new_candidate_pools(monkeypatch):
    import shading_aware_pv.optimization as module

    roof = Mesh(
        np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 4.0, 0.0]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    partition = partition_mesh(roof, face_labels=np.array(["scaffold", "scaffold"]))
    samples = sample_roof(partition.main_roof, spacing=0.5)
    scene = RoofScene(
        Path("roof.ply"), partition, samples, np.zeros(len(samples.points), bool)
    )
    auditor = PlacementAuditor(scene, box(2.8, 2.8, 3.2, 3.2))
    facet_id = next(iter(auditor.facets))
    corners = np.array([[0.5, 0.5, 0], [1.5, 0.5, 0], [1.5, 2, 0], [0.5, 2, 0]])
    candidate = PlacementCandidate(
        0, facet_id, "portrait", corners, corners + [0, 0, 0.2], np.array([0]), 1000.0
    )
    solution = PlacementSolution(1, np.array([0]), np.array([0]))
    optimization = PlacementOptimization(
        0.05,
        0.05,
        0.0,
        0.1,
        1.0,
        1.5,
        (candidate,),
        np.empty((0, 2), int),
        1,
        solution,
        solution,
    )
    original = module._cover_height
    calls = []

    def count(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, "_cover_height", count)
    first = auditor.audit(optimization, solution)
    assert auditor.audit(optimization, solution) == first
    assert len(calls) == 1
    changed_corners = corners + [2, 2, 0]
    moved = replace(
        candidate,
        roof_corners_xyz=changed_corners,
        receiver_corners_xyz=changed_corners + [0, 0, 0.2],
    )
    changed = replace(optimization, candidates=(moved,))
    check = auditor.audit(changed, solution)
    assert len(calls) == 2
    assert check["containment_or_clearance_violations"] == 1
    assert check == placement_audit(scene, auditor.details, changed, solution)
    strict = replace(optimization, roof_setback_m=0.75)
    assert auditor.audit(strict, solution)["containment_or_clearance_violations"] == 1
    conflict = replace(optimization, conflict_pairs=np.array([[0, 0]]))
    assert auditor.audit(conflict, solution)["panel_spacing_violations"] == 1


@pytest.mark.parametrize("fallback", [False, True])
def test_cropped_gpu_readback_matches_full_depth_maps(monkeypatch, fallback):
    import moderngl

    try:
        probe = moderngl.create_standalone_context(backend="egl")
    except Exception as error:
        pytest.skip(f"EGL context unavailable: {error}")
    probe.release()
    rng = np.random.default_rng(1701)
    triangles = rng.uniform(-4, 4, (30, 3, 3))
    origin = np.array([2600000.0, 1200000.0, 500.0])
    mesh = Mesh(triangles.reshape(-1, 3) + origin, np.arange(90).reshape(-1, 3))
    points = rng.uniform(-5, 5, (201, 3)) + origin
    renderer = DepthRenderer([mesh], points, pixel_size=0.03)
    if not fallback and renderer._read_subimage is None:
        renderer.close()
        pytest.skip("Texture subimage extension unavailable")
    if fallback:
        renderer._read_subimage = None
    gather = renderer._sample_depth
    comparisons = []

    def checked(x, y):
        reference = np.frombuffer(
            renderer.depth_texture.read(alignment=1), dtype="f4"
        ).reshape(renderer.resolution, renderer.resolution)[y, x]
        result = gather(x, y)
        np.testing.assert_array_equal(result, reference)
        comparisons.append(1)
        return result

    monkeypatch.setattr(renderer, "_sample_depth", checked)
    try:
        for sun in sun_vectors(np.linspace(0, 89.999, 64), np.linspace(0, 720, 64)):
            renderer.visibility(0, sun)
        renderer.visibility(None, np.array([0.0, 0.0, 1.0]))
        checked(
            np.array([0, 0, renderer.resolution - 1]),
            np.array([0, 0, renderer.resolution - 1]),
        )
    finally:
        renderer.close()
    assert len(comparisons) == 66
