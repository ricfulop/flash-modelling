#!/usr/bin/env python3
"""Convergence-improved BigDFT follow-up queue for Sunday-critical slabs."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"

QUEUE_PATH = Path(__file__).resolve().parent / "08_single_node_bigdft_queue.py"
SPEC = importlib.util.spec_from_file_location("single_node_queue", QUEUE_PATH)
queue = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = queue
SPEC.loader.exec_module(queue)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def heartbeat(run_root: Path, message: str) -> None:
    with (run_root / "heartbeat.log").open("a") as fh:
        fh.write(f"{now()} {message}\n")


def write_followup_input(job, job_dir: Path) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    queue.surface_posinp(job.atoms, job_dir / "posinp.xyz", surface=job.surface)
    shutil.copy2(queue.find_w_psp(), job_dir / "psppar.W.yaml")
    input_yaml = {
        "outdir": "run",
        "logfile": "Yes",
        "dft": {
            "hgrids": [job.hgrid, job.hgrid, job.hgrid],
            "rmult": [job.rmult[0], job.rmult[1]],
            "ixc": "PBE",
            "inputpsiid": 0,
            "itermax": 80,
            "nrepmax": 4,
            "gnrm_cv": 1.0e-4,
            "output_denspot": 22 if job.surface else 0,
        },
        "mix": {"tel": 0.72 / 27.211386245988, "occopt": 1},
    }
    (job_dir / "input.yaml").write_text(yaml.safe_dump(input_yaml, sort_keys=False))
    (job_dir / "job.json").write_text(
        json.dumps({k: v for k, v in asdict(job).items() if k != "atoms"}, indent=2) + "\n"
    )


def followup_jobs(label: str):
    source = {job.name: job for job in queue.jobs("peer" if label == "peer" else "local")}
    wanted = [
        "01_clean_w001_small_phi",
        "02_defect_w001_small_phi",
        "03_stepped_w001_small_phi",
        "04_separated_small_EFP",
    ]
    jobs = []
    for name in wanted:
        if name in source:
            job = source[name]
            jobs.append(
                replace(
                    job,
                    name=f"{job.name}_nrep4",
                    max_seconds=max(job.max_seconds * 2, 14400),
                    notes=f"{job.notes}; convergence follow-up nrepmax=4",
                )
            )
    return jobs


def run(cmd: str, *, timeout: int | None = None, log: Path | None = None):
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as fh:
            fh.write(f"\n$ {cmd}\n")
            fh.flush()
            return subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                text=True,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
    return subprocess.run(cmd, shell=True, executable="/bin/bash", timeout=timeout)


def main() -> int:
    label = os.environ.get("ELECTRODEFECT_BIGDFT_LABEL", socket.gethostname())
    run_id = datetime.now().strftime(f"bigdft_followup_%Y%m%d_%H%M%S_{label}")
    run_root = RUNS / run_id
    log_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    activate = queue.bigdft_activation()
    jobs = followup_jobs(label)
    status = {
        "state": "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "label": label,
        "activate": activate,
        "jobs": [{"name": j.name, "notes": j.notes, "state": "pending", "elapsed_s": 0} for j in jobs],
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "follow-up queue start")
    for idx, job in enumerate(jobs):
        archive_dir = run_root / "work" / job.name
        exec_dir = Path("/tmp") / run_id / job.name
        if exec_dir.exists():
            shutil.rmtree(exec_dir)
        exec_dir.mkdir(parents=True, exist_ok=True)
        write_followup_input(job, exec_dir)
        write_followup_input(job, archive_dir)
        status["jobs"][idx].update({"state": "running", "started_at": now()})
        status["updated_at"] = now()
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"job start {job.name}")
        start = time.time()
        rc = 1
        try:
            cmd = f"source {activate} >/tmp/use-bigdft-{run_id}.out && cd {exec_dir} && bigdft -l yes"
            rc = run(cmd, timeout=job.max_seconds, log=log_dir / f"{job.name}.out").returncode
        except subprocess.TimeoutExpired:
            rc = 124
        elapsed = time.time() - start
        queue.copy_tree_contents(exec_dir, archive_dir)
        status["jobs"][idx].update(
            {
                "state": "completed" if rc == 0 else "failed",
                "returncode": rc,
                "elapsed_s": elapsed,
                "finished_at": now(),
            }
        )
        status["updated_at"] = now()
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"job finish {job.name} rc={rc} elapsed={elapsed:.0f}s")
    status["state"] = "completed"
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "follow-up queue completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
