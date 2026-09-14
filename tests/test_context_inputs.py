from pathlib import Path
import pandas as pd
import pytest
from shapely.geometry import box

from shading_aware_pv.inputs import SimulationInputs, load_context_scene


def test_context_adapter_pins_sources_and_filters_original_classes(
    tmp_path, monkeypatch
):
    from building_data import swiss
    from shading_aware_pv import inputs as module

    source = {
        "tile": "fixture",
        "bounds": [0, 0, 1000, 1000],
        "path": "fixture.las",
        "year": 2023,
    }
    frame = pd.DataFrame(
        {
            "x": [0.5, 1.5, 0.5, 0.5, 0.5],
            "y": [0.5, 1.5, 0.5, 0.5, 0.5],
            "z": [0.0, 0.0, 3.0, 100.0, 3.0],
            "classification": [2, 2, 3, 5, 3],
        }
    )
    seen = {}

    class Provider:
        def __init__(self, cache):
            seen["cache"] = cache

        def points_for_bounds(self, bounds, reference_year, pinned_sources):
            seen.update(bounds=bounds, year=reference_year, sources=pinned_sources)
            return frame, (source,)

    monkeypatch.setattr(swiss, "SwissProvider", Provider)
    monkeypatch.setattr(
        module, "load_target_footprint", lambda *args, **kwargs: box(10, 10, 11, 11)
    )
    inputs = SimulationInputs(
        Path("mesh"),
        Path("details"),
        Path("image"),
        Path("surfaces"),
        5,
        tmp_path / "weather",
        Path("scaffold"),
        survey_year=2023,
        sources=(source,),
        context_cache_dir=tmp_path / "cache",
    )
    scene = load_context_scene(
        inputs, center_xy=(1.0, 1.0), half_extent_m=1.0, grid_resolution_m=1.0
    )
    assert seen == {
        "cache": tmp_path / "cache",
        "bounds": (0.0, 0.0, 2.0, 2.0),
        "year": 2023,
        "sources": (source,),
    }
    assert scene.source_classes.tolist() == [2, 2, 3]
    assert scene.surface_mesh.vertices[:, 2].max() == 3.0
    assert scene.surface_mesh_without_vegetation.vertices[:, 2].max() == 0.0
    assert scene.lidar_sources[0]["point_count"] == 4


def test_context_adapter_rejects_tile_without_usable_context_returns(
    tmp_path, monkeypatch
):
    from shading_aware_pv import inputs as module

    monkeypatch.setattr(
        module, "load_target_footprint", lambda *args, **kwargs: box(10, 10, 11, 11)
    )
    frame = pd.DataFrame({"x": [0.5], "y": [0.5], "z": [100.0], "classification": [5]})
    supplier = lambda bounds, reference_year: (
        frame,
        ({"tile": "empty-context", "bounds": [0, 0, 1000, 1000]},),
    )
    inputs = SimulationInputs(
        Path("mesh"), Path("details"), Path("image"), Path("surfaces"), 5, tmp_path, Path("scaffold")
    )
    with pytest.raises(ValueError, match="No usable context LiDAR returns"):
        load_context_scene(
            inputs,
            center_xy=(1.0, 1.0),
            half_extent_m=1.0,
            grid_resolution_m=1.0,
            points_supplier=supplier,
        )
