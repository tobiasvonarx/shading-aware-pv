"""Removal retains every report byte and isolates runs, paths and active work."""

import json

import pytest
from fastapi.testclient import TestClient

from shading_aware_pv.api import create_app


def saved_run(root, number=1, statuses=("complete", "failed")):
    run_id = f"{number:032x}"
    job_ids = [f"{number * 10 + i:032x}" for i in range(len(statuses))]
    for i, (job_id, status) in enumerate(zip(job_ids, statuses)):
        (root / f"solar/jobs/{job_id}.json").write_text(json.dumps({
            "id": job_id, "run_id": run_id, "house_id": f"roof-{number}-{i}",
            "status": status, "message": status, "created_at": "2026-09-12T10:00:00Z",
        }))
        if status == "complete":
            (root / f"solar/jobs/{job_id}.result.json").write_text('{"energy":123.4}\n')
    (root / f"solar/runs/{run_id}.json").write_text(json.dumps({
        "id": run_id, "job_ids": job_ids, "houses": [f"roof-{number}-{i}" for i in range(len(statuses))],
        "created_at": "2026-09-12T10:00:00Z", "config": {},
    }))
    return run_id, job_ids


def test_remove_run_retains_files_and_restores_after_reload(tmp_path):
    app = create_app(tmp_path)
    run, jobs = saved_run(tmp_path)
    other, _ = saved_run(tmp_path, 2)
    before = {p: p.read_bytes() for p in (tmp_path / "solar").rglob("*.json")}
    with TestClient(app) as client:
        assert client.delete(f"/api/solar/runs/{run}").status_code == 200
        assert [r["id"] for r in client.get("/api/solar/runs").json()] == [other]
        for endpoint in [f"runs/{run}", f"jobs/{jobs[0]}", f"jobs/{jobs[0]}/result"]:
            assert client.get(f"/api/solar/{endpoint}").status_code == 404
        assert len(client.get("/api/solar/jobs").json()) == 2
        assert client.delete(f"/api/solar/runs/{run}").status_code == 404
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/solar/trash").json()[0]["id"] == run
        assert client.post(f"/api/solar/runs/{run}/restore").status_code == 200
        assert client.get(f"/api/solar/jobs/{jobs[0]}/result").json() == {"energy": 123.4}
        assert len(client.get("/api/solar/runs").json()) == 2
        assert client.get("/api/solar/trash").json() == []
        assert client.post(f"/api/solar/runs/{run}/restore").status_code == 404
    assert all(path.read_bytes() == data for path, data in before.items())


def test_remove_reports_empty_run_and_nested_undo(tmp_path):
    app = create_app(tmp_path)
    run, jobs = saved_run(tmp_path)
    with TestClient(app) as client:
        assert client.delete(f"/api/solar/jobs/{jobs[0]}").status_code == 200
        remaining = client.get(f"/api/solar/runs/{run}").json()
        assert remaining["job_ids"] == [jobs[1]]
        assert remaining["houses"] == ["roof-1-1"]
        assert remaining["complete_count"] == 0 and remaining["failed_count"] == 1
        assert client.delete(f"/api/solar/runs/{run}").status_code == 200
        assert client.post(f"/api/solar/jobs/{jobs[0]}/restore").status_code == 409
        assert client.post(f"/api/solar/runs/{run}/restore").status_code == 200
        assert client.get(f"/api/solar/jobs/{jobs[0]}").status_code == 404
        assert client.delete(f"/api/solar/jobs/{jobs[1]}").status_code == 200
        assert client.get("/api/solar/runs").json() == []
        assert client.post(f"/api/solar/jobs/{jobs[0]}/restore").json()["run_id"] == run
        assert client.get("/api/solar/runs").json()[0]["job_ids"] == [jobs[0]]


@pytest.mark.parametrize("status", ["queued", "running"])
def test_active_run_and_report_cannot_be_removed(tmp_path, status):
    app = create_app(tmp_path)
    run, jobs = saved_run(tmp_path, statuses=("complete", status))
    with TestClient(app) as client:
        assert client.delete(f"/api/solar/runs/{run}").status_code == 409
        assert client.delete(f"/api/solar/jobs/{jobs[1]}").status_code == 409
        assert client.get("/api/solar/trash").json() == []
        assert len(client.get(f"/api/solar/runs/{run}").json()["jobs"]) == 2


@pytest.mark.parametrize("target", ["run", "job", "result", "archive", "lock", "directory"])
def test_removal_rejects_symlinks_without_touching_target(tmp_path, target):
    app = create_app(tmp_path / "data")
    root = tmp_path / "data"
    run, jobs = saved_run(root)
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret":"leave untouched"}')
    paths = {
        "run": root / f"solar/runs/{run}.json",
        "job": root / f"solar/jobs/{jobs[0]}.json",
        "result": root / f"solar/jobs/{jobs[0]}.result.json",
        "archive": root / "solar/removed.json",
        "lock": root / "solar/removed.lock",
    }
    if target == "directory":
        original = root / "solar/runs"
        original.rename(root / "saved-runs")
        original.symlink_to(root / "saved-runs", target_is_directory=True)
    else:
        path = paths[target]
        path.unlink(missing_ok=True)
        path.symlink_to(outside)
    with TestClient(app) as client:
        assert client.delete(f"/api/solar/runs/{run}").status_code == 404
        if target == "result":
            assert client.get(f"/api/solar/jobs/{jobs[0]}/result").status_code == 404
    assert outside.read_text() == '{"secret":"leave untouched"}'


def test_removal_rejects_traversal_and_cross_run_references(tmp_path):
    app = create_app(tmp_path)
    run, jobs = saved_run(tmp_path)
    other, other_jobs = saved_run(tmp_path, 2)
    path = tmp_path / f"solar/runs/{run}.json"
    metadata = json.loads(path.read_text())
    metadata["job_ids"] = [other_jobs[0]]
    path.write_text(json.dumps(metadata))
    with TestClient(app) as client:
        assert client.delete(f"/api/solar/runs/{run}").status_code == 409
        for identifier in ["not-valid", "..%2foutside", "A" * 32]:
            assert client.delete(f"/api/solar/jobs/{identifier}").status_code == 404
        assert client.get(f"/api/solar/runs/{other}").status_code == 200
        assert client.get("/api/solar/trash").json() == []


def test_concurrent_removals_keep_both_undo_records(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    app = create_app(tmp_path)
    _, jobs = saved_run(tmp_path)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda job: client.delete(f"/api/solar/jobs/{job}"), jobs))
        assert all(response.status_code == 200 for response in responses)
        assert {item["id"] for item in client.get("/api/solar/trash").json()} == set(jobs)
        assert client.get("/api/solar/runs").json() == []


def test_corrupt_removal_index_is_not_overwritten(tmp_path):
    app = create_app(tmp_path)
    run, _ = saved_run(tmp_path)
    index = tmp_path / "solar/removed.json"
    index.write_text("interrupted external edit")
    with TestClient(app) as client:
        assert client.delete(f"/api/solar/runs/{run}").status_code == 409
        assert client.get("/api/solar/trash").status_code == 409
    assert index.read_text() == "interrupted external edit"
