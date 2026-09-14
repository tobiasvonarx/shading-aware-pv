import json
from fastapi.testclient import TestClient
from shading_aware_pv.api import create_app


def test_acquisition_and_solar_api_share_application_data(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKERS", "1")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/acquire").status_code == 200
        assert client.get("/api/houses").json() == []
        assert client.get("/api/solar/config").json()["year"] == 2023
        assert client.post("/api/houses/missing/solar", json={}).status_code == 404
        assert client.get("/api/solar/jobs/not-an-id").status_code == 404
        job_id = "f" * 32
        path = tmp_path / "solar/jobs" / f"{job_id}.json"
        path.write_text(json.dumps({"id": job_id, "status": "complete"}))
        path.with_name(f"{job_id}.result.json").write_text('{"schema":"test"}')
        assert client.get(f"/api/solar/jobs/{job_id}").json()["status"] == "complete"
        response = client.get(f"/api/solar/jobs/{job_id}/result")
        assert response.headers["content-type"] == "application/json"
        assert response.json()["schema"] == "test"


def test_batch_runs_keep_every_house_and_persist_partial_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKERS", "2")
    app = create_app(tmp_path)
    monkeypatch.setattr(app.state.acquisition_store, "house", lambda house_id: object())
    submitted = []
    from concurrent.futures import Future

    def enqueue(*args):
        submitted.append(args)
        return Future()

    with TestClient(app) as client:
        executor = app.state.executor
        monkeypatch.setattr(executor, "submit", enqueue)
        response = client.post(
            "/api/solar/runs",
            json={
                "houses": ["first", "second", "first"],
                "config": {
                    "module_wattage_w": 430,
                    "general_loss_percent": 8,
                    "dc_ac_ratio": 1.2,
                },
            },
        )
        assert response.status_code == 202
        run = response.json()
        assert run["houses"] == ["first", "second"]
        assert len(submitted) == 2
        assert len(run["jobs"]) == 2
        for index, job in enumerate(run["jobs"]):
            path = tmp_path / "solar/jobs" / f"{job['id']}.json"
            job.update(
                status="complete" if index == 0 else "failed",
                message="done" if index == 0 else "fixture failure",
            )
            path.write_text(json.dumps(job))
        status = client.get(f"/api/solar/runs/{run['id']}").json()
        assert status["status"] == "finished"
        assert status["complete_count"] == status["failed_count"] == 1
        assert status["config"]["module_wattage_w"] == 430
        assert len(client.get("/api/solar/jobs").json()) == 2
    with TestClient(create_app(tmp_path)) as reloaded:
        runs = reloaded.get("/api/solar/runs").json()
        assert runs[0]["id"] == run["id"]
        assert runs[0]["jobs"][1]["message"] == "fixture failure"


def test_worker_reuses_client_checks_reconstruction_and_passes_device(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from emboss import api as emboss_api
    from shading_aware_pv import api, inputs, simulation

    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("WORKERS", "2")
    created = []
    reconstructed = []

    class Client:
        def __init__(self, root, workers, device):
            created.append((root, workers, device))

        def reconstruct(self, house_id, progress):
            reconstructed.append(house_id)
            return object()

    monkeypatch.setattr(emboss_api, "Client", Client)
    monkeypatch.setattr(
        inputs.SimulationInputs,
        "from_reconstruction",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        simulation, "simulate", lambda *args, **kwargs: {"schema": "fixture"}
    )
    jobs = tmp_path / "solar/jobs"
    jobs.mkdir(parents=True)
    api._CLIENTS.clear()
    try:
        for number in range(2):
            job_id = str(number) * 32
            (jobs / f"{job_id}.json").write_text(
                json.dumps({"id": job_id, "house_id": "roof", "status": "queued"})
            )
            api._run_job(str(tmp_path), job_id, "roof", {})
            assert (
                json.loads((jobs / f"{job_id}.json").read_text())["status"]
                == "complete"
            )
        assert reconstructed == ["roof", "roof"]
        assert created == [(tmp_path, 2, "cpu")]
    finally:
        api._CLIENTS.clear()


def test_worker_process_failure_is_persisted(tmp_path, monkeypatch):
    from concurrent.futures import Future

    app = create_app(tmp_path)
    monkeypatch.setattr(app.state.acquisition_store, "house", lambda house_id: object())
    future = Future()
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.executor, "submit", lambda *args: future)
        run = client.post("/api/solar/runs", json={"houses": ["roof"]}).json()
        future.set_exception(RuntimeError("process terminated"))
        saved = client.get(f"/api/solar/runs/{run['id']}").json()
        assert saved["failed_count"] == 1
        assert saved["status"] == "finished"
        assert "process terminated" in saved["jobs"][0]["message"]
