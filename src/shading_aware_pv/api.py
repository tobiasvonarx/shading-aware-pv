"""Persisted multi-building solar jobs and an interactive local workspace."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import multiprocessing
import os
import tempfile
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field

from .simulation import METHOD_REVISION, SimulationConfig

_CLIENTS = {}


def _save(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, allow_nan=False))
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_job(data_dir: str, job_id: str, house_id: str, options: dict):
    root = Path(data_dir)
    path = root / "solar" / "jobs" / f"{job_id}.json"
    status = json.loads(path.read_text())
    status["status"] = "running"

    def progress(message):
        status["message"] = message
        _save(path, status)

    try:
        from emboss.api import Client
        from .inputs import SimulationInputs
        from .simulation import simulate, write_result

        workers = int(os.getenv("WORKERS", "2"))
        device = os.getenv("DEVICE", "auto")
        key = (str(root), workers, device)
        if key not in _CLIENTS:
            _CLIENTS[key] = Client(root, workers=workers, device=device)
        client = _CLIENTS[key]
        lock_dir = root / "solar" / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        progress("Waiting for this building’s analysis workspace")
        with FileLock(str(lock_dir / f"{house_id}.lock")):
            progress("Checking reconstruction inputs and model fingerprint")
            choices = status.get("image_choices", {})
            image_options = {}
            if house_id in choices:
                image_options = {
                    "strip_id_override": choices[house_id],
                    "reset_strip_override": choices[house_id] is None,
                }
            reconstruction = client.reconstruct(
                house_id, progress=progress, **image_options
            )
            inputs = SimulationInputs.from_reconstruction(
                reconstruction,
                root / "solar" / "weather" / house_id / str(options.get("year", 2023)),
                context_cache_dir=root / "cache",
            )
            result = simulate(inputs, SimulationConfig(**options), progress=progress)
            write_result(result, path.with_name(f"{job_id}.result.json"))
        status.update(
            status="complete",
            message="Solar analysis complete",
            result_url=f"/api/solar/jobs/{job_id}/result",
        )
    except Exception as error:
        status.update(status="failed", message=f"{type(error).__name__}: {error}")
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    _save(path, status)


def _run_count_job(data_dir: str, job_id: str, attempt_id: str):
    from .counts import prepare_count_report
    from .simulation import write_result

    root = Path(data_dir)
    status_path = root / "solar/counts" / f"{job_id}.json"
    work_lock = FileLock(str(status_path.with_suffix(".work.lock")))
    with work_lock:
        _prepare_count_job(root, job_id, attempt_id, status_path, prepare_count_report, write_result)


def _prepare_count_job(root, job_id, attempt_id, status_path, prepare_count_report, write_result):
    status = json.loads(status_path.read_text())
    if status.get("attempt_id") != attempt_id:
        return
    def progress(message):
        status.update(status="running", message=message)
        _save(status_path, status)
    try:
        job = json.loads((root / "solar/jobs" / f"{job_id}.json").read_text())
        report = json.loads((root / "solar/jobs" / f"{job_id}.result.json").read_text())
        lock_dir = root / "solar/locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        progress("Waiting for the saved building workspace")
        with FileLock(str(lock_dir / f"{job['house_id']}.lock")):
            progress("Loading saved reconstruction and cached observations")
            result = prepare_count_report(root, job, report, progress=progress)
            write_result(result, status_path.with_name(f"{job_id}.result.json"))
        status.update(status="complete", message="Panel counts ready", result_url=f"/api/solar/jobs/{job_id}/counts/result")
    except Exception as error:
        status.update(status="failed", message=f"{type(error).__name__}: {error}")
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    _save(status_path, status)


class SolarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: dict = Field(default_factory=dict)
    image_choices: dict[str, str | None] = Field(default_factory=dict)


class SolarRunRequest(SolarRequest):
    houses: list[str] = Field(min_length=1)


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    from emboss.api import Client
    from emboss.imagery_web import mount_imagery
    from building_data.api import mount_acquisition
    from building_data.store import AcquisitionStore
    from building_data.runtime import worker_count

    root = Path(data_dir or os.getenv("DATA_DIR", "data")).resolve()
    workers = worker_count()
    jobs = root / "solar" / "jobs"
    runs = root / "solar" / "runs"
    counts = root / "solar" / "counts"
    jobs.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    store = AcquisitionStore(root, workers=workers)

    @asynccontextmanager
    async def lifespan(app):
        app.state.executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
        yield
        app.state.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Shading-aware PV", lifespan=lifespan)
    app.state.count_futures = {}
    mount_acquisition(app, store)
    mount_imagery(app, Client(root, workers=workers, device=os.getenv("DEVICE", "auto")))

    def safe_path(path):
        relative = path.relative_to(root)
        if any((root.joinpath(*relative.parts[:i])).is_symlink()
               for i in range(1, len(relative.parts) + 1)):
            raise HTTPException(404, "Unknown solar job or run")
        return path

    def existing_path(directory, identifier):
        if not isinstance(identifier, str) or len(identifier) != 32 or any(
            c not in "0123456789abcdef" for c in identifier
        ):
            raise HTTPException(404, "Unknown solar job or run")
        path = safe_path(directory / f"{identifier}.json")
        if not path.is_file():
            raise HTTPException(404, "Unknown solar job or run")
        return path

    archive_path = root / "solar" / "removed.json"
    archive_lock = root / "solar" / "removed.lock"

    def archive_state():
        safe_path(archive_path)
        if not archive_path.exists():
            return {"runs": {}, "jobs": {}}
        try:
            state = json.loads(archive_path.read_text())
            if not all(isinstance(state.get(key), dict) for key in ("runs", "jobs")):
                raise ValueError("Invalid removal index")
            return state
        except (ValueError, OSError) as error:
            raise HTTPException(409, "Removed items could not be read") from error

    def archived_job(job, state):
        return job["id"] in state["jobs"] or job.get("run_id") in state["runs"]

    def read_job(job_id, *, include_removed=False, state=None):
        job = json.loads(existing_path(jobs, job_id).read_text())
        if job.get("id") != job_id:
            raise HTTPException(409, "Solar report identity is inconsistent")
        if not include_removed and archived_job(job, state or archive_state()):
            raise HTTPException(404, "Unknown solar job or run")
        return job

    def read_run(run_id):
        run = json.loads(existing_path(runs, run_id).read_text())
        if run.get("id") != run_id:
            raise HTTPException(409, "Solar run identity is inconsistent")
        members = [read_job(job_id, include_removed=True) for job_id in run["job_ids"]]
        if any(job.get("run_id") != run_id for job in members):
            raise HTTPException(409, "Solar report belongs to another run")
        return run, members

    def change_archive(kind, identifier, *, restore=False):
        safe_path(archive_lock)
        with FileLock(str(archive_lock)):
            state = archive_state()
            if kind == "runs":
                item, members = read_run(identifier)
            else:
                item = read_job(identifier, include_removed=True)
                members = [item]
                if item.get("run_id") in state["runs"]:
                    raise HTTPException(409, "Undo the run removal first")
                if item.get("run_id"):
                    _, siblings = read_run(item["run_id"])
                    if identifier not in {job["id"] for job in siblings}:
                        raise HTTPException(409, "Solar report is absent from its run")
            if any(job["status"] not in ("complete", "failed") for job in members):
                raise HTTPException(409, "Wait for the analysis to finish before removing it")
            for job in members:
                safe_path(jobs / f"{job['id']}.result.json")
                preparation = safe_path(counts / f"{job['id']}.json")
                if preparation.exists() and read_count_status(preparation)["status"] in ("queued", "running"):
                    raise HTTPException(409, "Wait for panel-count preparation to finish before removing this report")
            if restore:
                if identifier not in state[kind]:
                    raise HTTPException(404, "Unknown removed item")
                del state[kind][identifier]
            else:
                if identifier in state[kind]:
                    raise HTTPException(404, "Unknown solar job or run")
                state[kind][identifier] = {
                    "id": identifier,
                    "kind": kind,
                    "label": (f"Run from {item['created_at']}" if kind == "runs"
                              else f"Report for {item['house_id']}"),
                    "removed_at": datetime.now(timezone.utc).isoformat(),
                }
            safe_path(archive_path)
            _save(archive_path, state)
        return {"id": identifier, "kind": kind, "removed": not restore,
                "run_id": identifier if kind == "runs" else item.get("run_id")}

    def validate(houses, options):
        try:
            config = SimulationConfig(**options)
            for house_id in houses:
                store.house(house_id)
            return config
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    def validate_choices(houses, choices):
        if not set(choices).issubset(houses):
            raise HTTPException(422, "Image choices must belong to the requested buildings")
        if any(value is not None and not value.strip() for value in choices.values()):
            raise HTTPException(422, "Image choices must be a source ID or null for automatic selection")

    def prepare_job(house_id, config, run_id=None, image_choices=None):
        status = {
            "id": uuid.uuid4().hex,
            "house_id": house_id,
            "run_id": run_id,
            "image_choices": {house_id: image_choices[house_id]} if image_choices and house_id in image_choices else {},
            "status": "queued",
            "message": "Waiting for a solar worker",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
        }
        _save(jobs / f"{status['id']}.json", status)
        return status

    def submit(status):
        def failed(error):
            path = jobs / f"{status['id']}.json"
            latest = json.loads(path.read_text())
            latest.update(
                status="failed",
                message=f"Solar worker stopped: {error}",
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            _save(path, latest)

        def completed(future):
            if future.cancelled():
                failed("job was cancelled during shutdown")
            elif future.exception() is not None:
                failed(future.exception())

        try:
            future = app.state.executor.submit(
                _run_job, str(root), status["id"], status["house_id"], status["config"]
            )
            future.add_done_callback(completed)
        except Exception as error:
            failed(error)

    def run_status(path, state=None):
        state = state or archive_state()
        run, members = read_run(path.stem)
        if run["id"] in state["runs"]:
            raise HTTPException(404, "Unknown solar job or run")
        run["jobs"] = [job for job in members if not archived_job(job, state)]
        run["houses"] = list(dict.fromkeys(job["house_id"] for job in run["jobs"]))
        run["job_ids"] = [job["id"] for job in run["jobs"]]
        run["complete_count"] = sum(job["status"] == "complete" for job in run["jobs"])
        run["failed_count"] = sum(job["status"] == "failed" for job in run["jobs"])
        run["status"] = (
            "finished"
            if all(job["status"] in ("complete", "failed") for job in run["jobs"])
            else "running"
        )
        return run

    @app.get("/api/solar/config")
    def defaults():
        return asdict(SimulationConfig())

    @app.post("/api/houses/{house_id}/solar", status_code=202)
    def start(house_id: str, request: SolarRequest):
        validate_choices([house_id], request.image_choices)
        status = prepare_job(house_id, validate([house_id], request.config),
                             image_choices=request.image_choices)
        submit(status)
        return status

    @app.post("/api/solar/runs", status_code=202)
    def start_run(request: SolarRunRequest):
        houses = list(dict.fromkeys(request.houses))
        validate_choices(houses, request.image_choices)
        config = validate(houses, request.config)
        run_id = uuid.uuid4().hex
        prepared = [prepare_job(house_id, config, run_id, request.image_choices) for house_id in houses]
        run = {
            "id": run_id,
            "houses": houses,
            "image_choices": request.image_choices,
            "job_ids": [job["id"] for job in prepared],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
        }
        path = runs / f"{run_id}.json"
        _save(path, run)
        for status in prepared:
            submit(status)
        return run_status(path)

    @app.get("/api/solar/runs")
    def list_runs():
        state = archive_state()
        visible = [
            run_status(path, state)
            for path in sorted(
                runs.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if not path.is_symlink() and path.stem not in state["runs"]
        ]
        return [run for run in visible if run["jobs"]]

    @app.delete("/api/solar/runs/{run_id}")
    def remove_run(run_id: str):
        return change_archive("runs", run_id)

    @app.post("/api/solar/runs/{run_id}/restore")
    def restore_run(run_id: str):
        return change_archive("runs", run_id, restore=True)

    @app.get("/api/solar/trash")
    def removed_items():
        state = archive_state()
        return sorted([*state["runs"].values(), *state["jobs"].values()],
                      key=lambda item: item["removed_at"], reverse=True)

    @app.get("/api/solar/runs/{run_id}")
    def get_run(run_id: str):
        return run_status(existing_path(runs, run_id))

    @app.get("/api/solar/jobs")
    def list_jobs():
        state = archive_state()
        paths = [
            path
            for path in jobs.glob("*.json")
            if not path.name.endswith(".result.json") and not path.is_symlink()
        ]
        visible = [
            read_job(path.stem, include_removed=True)
            for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
        ]
        return [job for job in visible if not archived_job(job, state)]

    @app.delete("/api/solar/jobs/{job_id}")
    def remove_report(job_id: str):
        return change_archive("jobs", job_id)

    @app.post("/api/solar/jobs/{job_id}/restore")
    def restore_report(job_id: str):
        return change_archive("jobs", job_id, restore=True)

    @app.get("/api/solar/jobs/{job_id}")
    def job(job_id: str):
        return read_job(job_id)

    @app.get("/api/solar/jobs/{job_id}/result")
    def result(job_id: str):
        read_job(job_id)
        path = safe_path(jobs / f"{job_id}.result.json")
        if not path.is_file():
            raise HTTPException(404, "Solar result is not ready")
        return FileResponse(path, media_type="application/json")

    def read_count_status(path):
        status = json.loads(path.read_text())
        if status.get("id") != path.stem:
            raise HTTPException(409, "Panel-count preparation identity is inconsistent")
        if status["status"] == "complete" and status.get("method_revision") != METHOD_REVISION:
            status.update(status="failed", message="Panel counts use an older method. Prepare them again to use the current roof geometry.")
            _save(path, status)
        if status["status"] in ("queued", "running") and status["id"] not in app.state.count_futures:
            lock = FileLock(str(safe_path(path.with_suffix(".work.lock"))))
            try:
                with lock.acquire(timeout=0):
                    status.update(status="failed", message="Preparation was interrupted by a restart. Retry to prepare panel counts.")
                    _save(path, status)
            except Timeout:
                pass
        return status

    @app.post("/api/solar/jobs/{job_id}/counts", status_code=202)
    def prepare_counts(job_id: str):
        saved_job = read_job(job_id)
        if saved_job["status"] != "complete":
            raise HTTPException(409, "Wait for the analysis to finish before preparing panel counts")
        result_path = safe_path(jobs / f"{job_id}.result.json")
        if not result_path.is_file():
            raise HTTPException(404, "Solar result is not ready")
        report = json.loads(result_path.read_text())
        if report.get("designs_by_count") and report.get("provenance", {}).get("method_revision") == METHOD_REVISION:
            return {"id": job_id, "status": "complete", "message": "Panel counts ready", "result_url": saved_job["result_url"]}
        safe_path(counts).mkdir(parents=True, exist_ok=True)
        safe_path(counts / f"{job_id}.result.json")
        lock = safe_path(counts / f"{job_id}.lock")
        with FileLock(str(lock)):
            path = safe_path(counts / f"{job_id}.json")
            if path.exists():
                existing = read_count_status(path)
                if existing["status"] in ("queued", "running", "complete"):
                    return existing
            status = {"id": job_id, "attempt_id": uuid.uuid4().hex, "method_revision": METHOD_REVISION, "status": "queued", "message": "Waiting for a solar worker", "created_at": datetime.now(timezone.utc).isoformat()}
            _save(path, status)
            def completed(future):
                error = "preparation cancelled" if future.cancelled() else future.exception()
                if error is not None:
                    latest = json.loads(path.read_text())
                    if latest.get("attempt_id") != status["attempt_id"]:
                        return
                    latest.update(status="failed", message=f"Panel-count worker stopped: {error}")
                    _save(path, latest)
            try:
                future = app.state.executor.submit(_run_count_job, str(root), job_id, status["attempt_id"])
                app.state.count_futures[job_id] = future
                future.add_done_callback(completed)
            except Exception as error:
                status.update(status="failed", message=str(error))
                _save(path, status)
            return status

    @app.get("/api/solar/jobs/{job_id}/counts")
    def count_status(job_id: str):
        saved_job = read_job(job_id)
        path = safe_path(counts / f"{job_id}.json")
        if path.is_file():
            return read_count_status(path)
        result_path = safe_path(jobs / f"{job_id}.result.json")
        if result_path.is_file():
            report = json.loads(result_path.read_text())
            if report.get("designs_by_count") and report.get("provenance", {}).get("method_revision") == METHOD_REVISION:
                return {"id": job_id, "status": "complete", "message": "Panel counts ready", "result_url": saved_job["result_url"]}
        raise HTTPException(404, "Panel counts have not been prepared")

    @app.get("/api/solar/jobs/{job_id}/counts/result")
    def count_result(job_id: str):
        read_job(job_id)
        status_path = safe_path(counts / f"{job_id}.json")
        path = safe_path(counts / f"{job_id}.result.json")
        if not status_path.is_file() or read_count_status(status_path)["status"] != "complete" or not path.is_file():
            raise HTTPException(404, "Panel counts are not ready")
        return FileResponse(path, media_type="application/json")

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    return app
