"""Solar imagery selection stays explicit, persisted and separate from numerics."""
from concurrent.futures import Future
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from shading_aware_pv import api, inputs, simulation


def test_shared_gallery_mount_and_choice_persistence(tmp_path, monkeypatch):
    app = api.create_app(tmp_path)
    monkeypatch.setattr(app.state.acquisition_store, "house", lambda _: object())
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.executor, "submit", lambda *args: Future())
        for path in ("/emboss-imagery/gallery.js", "/emboss-imagery/style.css"):
            assert client.get(path).status_code == 200
        assert client.post("/api/houses/missing/imagery").status_code == 404
        assert client.get("/api/imagery-jobs/missing").status_code == 404
        request = {"houses": ["a", "b", "c"], "image_choices": {"a": "strip-42", "b": None}}
        response = client.post("/api/solar/runs", json=request)
        assert response.status_code == 202
        run = response.json()
        assert run["image_choices"] == request["image_choices"]
        assert [j["image_choices"] for j in run["jobs"]] == [{"a": "strip-42"}, {"b": None}, {}]
        single = client.post("/api/houses/a/solar", json={"image_choices": {"a": "raw"}})
        assert single.status_code == 202
        assert single.json()["image_choices"] == {"a": "raw"}
    with TestClient(api.create_app(tmp_path)) as reloaded:
        assert reloaded.get(f"/api/solar/runs/{run['id']}").json()["image_choices"] == request["image_choices"]


@pytest.mark.parametrize("choices", [{"other": "raw"}, {"a": "  "}, {"a": 42}])
def test_invalid_choices_do_not_queue_jobs(tmp_path, monkeypatch, choices):
    app = api.create_app(tmp_path)
    monkeypatch.setattr(app.state.acquisition_store, "house", lambda _: object())
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.executor, "submit", lambda *args: pytest.fail("Invalid choice was queued"))
        for endpoint, payload in [("/api/solar/runs", {"houses": ["a"]}), ("/api/houses/a/solar", {})]:
            assert client.post(endpoint, json={**payload, "image_choices": choices}).status_code == 422
    assert not list((tmp_path / "solar/jobs").glob("*.json"))


@pytest.mark.parametrize("choices,expected", [({}, {}), ({"a": None}, {"strip_id_override": None, "reset_strip_override": True}), ({"a": "raw"}, {"strip_id_override": "raw", "reset_strip_override": False})])
def test_worker_passes_choice_without_forcing_and_keeps_result_schema(tmp_path, monkeypatch, choices, expected):
    from emboss import api as emboss_api
    calls = []
    class Client:
        def __init__(self, *args, **kwargs): pass
        def reconstruct(self, house_id, progress, **kwargs):
            calls.append((house_id, kwargs))
            return object()
    monkeypatch.setattr(emboss_api, "Client", Client)
    monkeypatch.setattr(inputs.SimulationInputs, "from_reconstruction", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(simulation, "simulate", lambda *args, **kwargs: {"scientific": "unchanged"})
    jobs = tmp_path / "solar/jobs"
    jobs.mkdir(parents=True)
    path = jobs / f"{'a' * 32}.json"
    path.write_text(json.dumps({"id": "a" * 32, "house_id": "a", "status": "queued", "image_choices": choices}))
    api._CLIENTS.clear()
    try:
        api._run_job(str(tmp_path), "a" * 32, "a", {})
        assert json.loads(path.read_text())["status"] == "complete"
        assert calls == [("a", expected)]
        assert json.loads(path.with_name(f"{'a' * 32}.result.json").read_text()) == {"scientific": "unchanged"}
    finally:
        api._CLIENTS.clear()
