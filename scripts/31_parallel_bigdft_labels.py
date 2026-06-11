#!/usr/bin/env python3
"""Run selected BigDFT label jobs concurrently.

This is a throughput launcher for independent, pre-specified label jobs. It
uses the same input writer as `18_phase2_bigdft_labels.py`, but executes a
filtered subset in parallel so idle CPU cores are used while the original serial
queue is still running.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
LABEL_PATH = Path(__file__).resolve().parent / "18_phase2_bigdft_labels.py"
SPEC = importlib.util.spec_from_file_location("phase2_bigdft_labels", LABEL_PATH)
labels = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = labels
SPEC.loader.exec_module(labels)


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


def run_command(cmd: str, log: Path, timeout: int) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(f"\n$ {cmd}\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            text=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    return int(proc.returncode)


def mpi_prefix(nproc: int) -> str:
    if nproc <= 1:
        return ""
    mpirun = "/usr/bin/mpirun" if Path("/usr/bin/mpirun").exists() else "mpirun"
    return (
        "OMPI_MCA_pml=ob1 "
        "OMPI_MCA_btl=self,tcp "
        "OMPI_MCA_mtl='^ofi' "
        "UCX_TLS=tcp,self "
        "FI_PROVIDER=tcp "
        "OMPI_ALLOW_RUN_AS_ROOT=1 "
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 "
        f"{mpirun} -np {int(nproc)} "
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", action="append", type=Path, required=True)
    p.add_argument("--label", default="parallel_autonomous")
    p.add_argument("--job-name", action="append", default=None, help="Exact generated job name to run; may repeat.")
    p.add_argument("--skip-job-name", action="append", default=None, help="Exact generated job name to skip; may repeat.")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--hgrid", type=float, default=0.55)
    p.add_argument("--rmult", nargs=2, type=float, default=[4.0, 7.0])
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--no-matrix-export", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    jobs = labels.load_jobs(
        args.root,
        float(args.hgrid),
        tuple(float(x) for x in args.rmult),
        int(args.max_seconds),
        matrix_export=not bool(args.no_matrix_export),
    )
    if args.job_name:
        wanted = set(args.job_name)
        jobs = [job for job in jobs if job.name in wanted]
    if args.skip_job_name:
        skipped = set(args.skip_job_name)
        jobs = [job for job in jobs if job.name not in skipped]
    if not jobs:
        raise SystemExit("no BigDFT jobs selected")

    run_root = RUNS / datetime.now().strftime(f"phase2_bigdft_parallel_%Y%m%d_%H%M%S_{args.label}")
    tmp_root = Path("/tmp") / run_root.name
    log_root = run_root / "logs"
    work_root = run_root / "work"
    run_root.mkdir(parents=True, exist_ok=True)
    activate = labels.queue.bigdft_activation()
    status_lock = threading.Lock()
    status = {
        "state": "running",
        "run_root": str(run_root),
        "started_at": now(),
        "roots": [str(root) for root in args.root],
        "activate": activate,
        "parallel": int(args.parallel),
        "nproc_per_job": int(args.nproc),
        "matrix_export": not bool(args.no_matrix_export),
        "jobs": [
            {**labels.asdict(job), "state": "pending", "archive_dir": str(work_root / job.name)}
            for job in jobs
        ],
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"parallel label start n_jobs={len(jobs)} parallel={args.parallel} nproc={args.nproc}")

    def update_job(index: int, patch: dict) -> None:
        with status_lock:
            status["jobs"][index].update(patch)
            status["updated_at"] = now()
            write_json(run_root / "status.json", status)

    def run_one(index: int, job) -> int:
        archive_dir = work_root / job.name
        exec_dir = tmp_root / job.name
        if exec_dir.exists():
            shutil.rmtree(exec_dir)
        exec_dir.mkdir(parents=True, exist_ok=True)
        labels.write_bigdft_input(job, exec_dir)
        labels.write_bigdft_input(job, archive_dir)
        update_job(index, {"state": "running", "started_at": now()})
        heartbeat(run_root, f"job start {job.name}")
        start = time.time()
        rc = 1
        try:
            cmd = f"source {activate} >/tmp/use-bigdft-{run_root.name}-{job.name}.out && cd {exec_dir} && {mpi_prefix(args.nproc)}bigdft -l yes"
            rc = run_command(cmd, log_root / f"{job.name}.out", timeout=int(job.max_seconds))
        except subprocess.TimeoutExpired:
            rc = 124
        elapsed = time.time() - start
        labels.queue.copy_tree_contents(exec_dir, archive_dir)
        scalars = labels.dft_bigdft.parse_log_scalars(archive_dir)
        state = "completed" if rc == 0 else "failed"
        update_job(
            index,
            {
                "state": state,
                "returncode": rc,
                "elapsed_s": round(elapsed, 3),
                "finished_at": now(),
                "scalars": scalars,
            },
        )
        heartbeat(run_root, f"job finish {job.name} rc={rc} elapsed={elapsed:.0f}s")
        return rc

    with ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as executor:
        futures = [executor.submit(run_one, index, job) for index, job in enumerate(jobs)]
        for future in as_completed(futures):
            future.result()

    failed = [job for job in status["jobs"] if job.get("state") != "completed"]
    status["state"] = "completed" if not failed else "completed_with_failures"
    status["finished_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"parallel label {status['state']}")
    print(run_root)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
