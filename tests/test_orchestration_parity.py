"""Optional full annual oracle; requires an independently copied source/data bundle.

Set SOLAR_PARITY_MANIFEST to a JSON file with reference_root and cases. Each case
must use the current fixed src/solar_poc and src/emboss code. Each case
has name, inputs (SimulationInputs fields including scaffold_path), and config
(SimulationConfig fields). Provide cached weather.csv, weather_meta.json,
weather_no_horizon.csv, weather_no_horizon_meta.json and applicable MeteoSwiss
station data in inputs.cache_dir. Context cases additionally supply context_sources
from a completed result's context.lidar_sources and the matching input source paths.
Run: SOLAR_PARITY_MANIFEST=/path/to/manifest.json uv run --locked pytest -q
     tests/test_orchestration_parity.py
The original orchestration executes in a child interpreter, isolated from unit tests.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def assert_equal_tree(tree):
    if isinstance(tree, dict):
        if "equal" in tree:
            assert tree["equal"], tree
        for value in tree.values():
            assert_equal_tree(value)


def test_original_orchestration_parity(tmp_path):
    manifest_path = os.environ.get("SOLAR_PARITY_MANIFEST")
    if not manifest_path:
        pytest.skip("Set SOLAR_PARITY_MANIFEST for the copied-source annual oracle")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("orchestration_oracle.py")),
         manifest_path, str(tmp_path)],
        check=True,
    )
    manifest = json.loads(Path(manifest_path).read_text())
    for case in manifest["cases"]:
        report = json.loads((tmp_path / case["name"] / "comparison.json").read_text())
        assert_equal_tree(report["comparisons"])
        assert_equal_tree(report["designs"])
        for design in report["designs"].values():
            if "corners_equal" in design:
                assert design["corners_equal"], case["name"]
                assert design["panel_count"] == design["new_panel_count"]
            if "ac_difference_kwh" in design:
                assert abs(design["ac_difference_kwh"]) < 1e-8
        assert report["old_study_capacity"] == report["new_capacity"]
    assert not list(tmp_path.rglob("*.html"))
