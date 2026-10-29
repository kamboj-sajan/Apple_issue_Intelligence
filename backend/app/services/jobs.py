"""Minimal background-job runner for ingestion and analysis.

Each job runs the existing CLI commands (``python -m ml.cli ...``, ``python -m backend.app.cli load``)
as subprocesses from a worker thread, so the API keeps serving requests while a job runs. Status,
the current step and a rolling log tail are stored in the ``jobs`` table.

Limitations (fine at this project's scale, documented in the README):
* jobs live in the API process: a restart marks running jobs ``interrupted`` and does not resume them;
* at most one job per type runs at a time (a second request gets HTTP 409);
* with several API workers, each would run its own runner, so deploy a single worker or use a real queue.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime

from sqlalchemy import Engine, update
from sqlalchemy.orm import Session

from backend.app.core.config import REPO_ROOT
from backend.app.core.errors import APIError
from backend.app.db.models import Job

LOG_TAIL_CHARS = 6000
ANALYZE_DEFAULT = ["preprocess", "split", "embed", "clusters", "anomalies", "graph", "drift"]


def ingest_steps(mode: str, max_requests: int | None) -> list[list[str]]:
    args = ["ml.cli", mode]
    if max_requests:
        args += ["--max-requests", str(max_requests)]
    return [args]


def analyze_steps(steps: list[str] | None, load_database: bool) -> list[list[str]]:
    out = [["ml.cli", s] for s in (steps or ANALYZE_DEFAULT)]
    if load_database:
        out.append(["backend.app.cli", "load"])
    return out


class JobRunner:
    def __init__(self, engine: Engine):
        self.engine = engine
        self._lock = threading.Lock()
        self._active: dict[str, str] = {}

    def mark_interrupted(self) -> None:
        with Session(self.engine) as s:
            s.execute(
                update(Job)
                .where(Job.status.in_(["queued", "running"]))
                .values(
                    status="interrupted", finished_at=datetime.now(UTC), error="API restarted while the job was running"
                )
            )
            s.commit()

    def submit(self, job_type: str, params: dict, steps: list[list[str]]) -> Job:
        with self._lock:
            if job_type in self._active:
                raise APIError(
                    409,
                    "job_already_running",
                    f"a {job_type} job is already running",
                    {"job_id": self._active[job_type]},
                )
            job_id = str(uuid.uuid4())
            self._active[job_type] = job_id
        job = Job(
            job_id=job_id, job_type=job_type, status="queued", params=params, created_at=datetime.now(UTC), log_tail=""
        )
        with Session(self.engine, expire_on_commit=False) as s:
            s.add(job)
            s.commit()
        threading.Thread(target=self._run, args=(job_id, job_type, steps), daemon=True).start()
        return job

    def _update(self, job_id: str, **values) -> None:
        with Session(self.engine) as s:
            s.execute(update(Job).where(Job.job_id == job_id).values(**values))
            s.commit()

    def _run(self, job_id: str, job_type: str, steps: list[list[str]]) -> None:
        tail = ""
        self._update(job_id, status="running", started_at=datetime.now(UTC))
        try:
            for step in steps:
                label = " ".join(step)
                self._update(job_id, current_step=label)
                proc = subprocess.Popen(
                    [sys.executable, "-m", *step],
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                last_flush = time.monotonic()
                for line in proc.stdout:
                    tail = (tail + line)[-LOG_TAIL_CHARS:]
                    if time.monotonic() - last_flush > 3:
                        self._update(job_id, log_tail=tail)
                        last_flush = time.monotonic()
                if proc.wait() != 0:
                    raise RuntimeError(f"step '{label}' exited with code {proc.returncode}")
                tail = (tail + f"\n[step finished: {label}]\n")[-LOG_TAIL_CHARS:]
                self._update(job_id, log_tail=tail)
            self._update(job_id, status="succeeded", finished_at=datetime.now(UTC), current_step=None, log_tail=tail)
        except Exception as exc:  # recorded on the job; the API process keeps running
            self._update(job_id, status="failed", finished_at=datetime.now(UTC), error=str(exc), log_tail=tail)
        finally:
            with self._lock:
                self._active.pop(job_type, None)
