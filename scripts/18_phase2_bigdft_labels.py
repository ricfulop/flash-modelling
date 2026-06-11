#!/usr/bin/env python3
"""Run CPU BigDFT labels for Phase 2 MACE active-learning candidates."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

DFT_BIGDFT_PATH = Path(__file__).resolve().parents[1] / "src" / "electrodefect" / "dft_bigdft.py"
DFT_SPEC = importlib.util.spec_from_file_location("dft_bigdft_direct", DFT_BIGDFT_PATH)
dft_bigdft = importlib.util.module_from_spec(DFT_SPEC)
assert DFT_SPEC.loader is not None
DFT_SPEC.loader.exec_module(dft_bigdft)


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
QUEUE_PATH = Path(__file__).resolve().parent / "08_single_node_bigdft_queue.py"
SPEC = importlib.util.spec_from_file_location("single_node_queue", QUEUE_PATH)
queue = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = queue
SPEC.loader.exec_module(queue)


DEFAULT_ROOTS = [
    RUNS / "phase2_mlip_20260603_213335_spark-5da5",
    RUNS / "phase2_mlip_20260603_213410_spark-5da5",
]


@dataclass
class LabelJob:
    name: str
    material: str
    morphology: str
    label_kind: str
    source_atoms: str
    hgrid: float
    rmult: tuple[float, float]
    max_seconds: int
    surface: bool = True
    matrix_export: bool = True


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


def read_manifest(root: Path) -> dict:
    path = root / "label_queue_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing label queue manifest: {path}")
    return json.loads(path.read_text())


def find_psp(material: str) -> Path:
    """Find the Krack/PBE pseudopotential for a Phase 2 label material."""
    name = f"psppar.{material}.yaml"
    candidates = [
        Path(f"/home/ricfulop/Desktop/Cursor/bigdft-suite/PyBigDFT/BigDFT/Database/psppar/Krack/PBE/{name}"),
        Path(f"/home/nvidia/Cursor/bigdft-suite/PyBigDFT/BigDFT/Database/psppar/Krack/PBE/{name}"),
        Path(f"/home/ricfulop/Desktop/Cursor/.local-bigdft/envs/bigdft/lib/python3.11/site-packages/BigDFT/Database/psppar/Krack/PBE/{name}"),
        Path(f"/home/nvidia/Cursor/.local-bigdft/envs/bigdft/lib/python3.11/site-packages/BigDFT/Database/psppar/Krack/PBE/{name}"),
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    raise FileNotFoundError(f"could not find Krack/PBE pseudopotential {name}")


def load_jobs(roots: list[Path], hgrid: float, rmult: tuple[float, float],
              max_seconds: int, matrix_export: bool = True) -> list[LabelJob]:
    jobs: list[LabelJob] = []
    for root in roots:
        manifest = read_manifest(root)
        for item in manifest.get("items", []):
            source = Path(item["source_atoms"])
            if not source.exists():
                raise FileNotFoundError(f"queued atoms file is missing: {source}")
            idx = len(jobs)
            name = f"{idx:02d}_{item['material']}_{item['morphology']}_{item['label_kind']}"
            jobs.append(
                LabelJob(
                    name=name,
                    material=item["material"],
                    morphology=item["morphology"],
                    label_kind=item["label_kind"],
                    source_atoms=str(source),
                    hgrid=hgrid,
                    rmult=rmult,
                    max_seconds=max_seconds,
                    matrix_export=matrix_export,
                )
            )
    return jobs


def write_bigdft_input(job: LabelJob, job_dir: Path) -> None:
    from ase.io import read

    atoms = read(job.source_atoms)
    job_dir.mkdir(parents=True, exist_ok=True)
    queue.surface_posinp(atoms, job_dir / "posinp.xyz", surface=job.surface)
    shutil.copy2(find_psp(job.material), job_dir / f"psppar.{job.material}.yaml")
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
        "mix": {"tel": 0.72 / dft_bigdft.EV_PER_HA, "occopt": 1},
    }
    if job.matrix_export:
        input_yaml["import"] = "linear"
        input_yaml["lin_general"] = {"output_mat": 1}
    (job_dir / "input.yaml").write_text(yaml.safe_dump(input_yaml, sort_keys=False))
    (job_dir / "job.json").write_text(json.dumps(asdict(job), indent=2) + "\n")


def run(cmd: str, *, timeout: int | None, log: Path) -> subprocess.CompletedProcess:
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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", action="append", type=Path, default=None,
                   help="Phase 2 MACE run root containing label_queue_manifest.json. May repeat.")
    p.add_argument("--label", default=os.environ.get("ELECTRODEFECT_BIGDFT_LABEL", socket.gethostname()))
    p.add_argument("--hgrid", type=float, default=0.55)
    p.add_argument("--rmult", nargs=2, type=float, default=[4.0, 7.0])
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--max-jobs", type=int, default=0, help="0 means all queued labels")
    p.add_argument("--no-matrix-export", action="store_true",
                   help="Disable BigDFT linear support-function H/S matrix export.")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    roots = args.root or DEFAULT_ROOTS
    jobs = load_jobs(
        roots,
        float(args.hgrid),
        tuple(float(x) for x in args.rmult),
        int(args.max_seconds),
        matrix_export=not bool(args.no_matrix_export),
    )
    if args.max_jobs:
        jobs = jobs[: int(args.max_jobs)]
    if not jobs:
        raise SystemExit("no Phase 2 label jobs found")

    run_id = datetime.now().strftime(f"phase2_bigdft_labels_%Y%m%d_%H%M%S_{args.label}")
    run_root = RUNS / run_id
    log_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    activate = queue.bigdft_activation()
    status = {
        "state": "preparing" if args.prepare_only else "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "label": args.label,
        "activate": activate,
        "roots": [str(root) for root in roots],
        "matrix_export": not bool(args.no_matrix_export),
        "jobs": [
            {
                **asdict(job),
                "state": "pending",
                "elapsed_s": 0.0,
                "archive_dir": str(run_root / "work" / job.name),
            }
            for job in jobs
        ],
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"label queue start n_jobs={len(jobs)} prepare_only={args.prepare_only}")

    tmp_root = Path("/tmp") / run_id
    for idx, job in enumerate(jobs):
        archive_dir = run_root / "work" / job.name
        exec_dir = tmp_root / job.name
        if exec_dir.exists():
            shutil.rmtree(exec_dir)
        exec_dir.mkdir(parents=True, exist_ok=True)
        write_bigdft_input(job, exec_dir)
        write_bigdft_input(job, archive_dir)
        status["jobs"][idx]["state"] = "prepared"
        status["updated_at"] = now()
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"job prepared {job.name}")

        if args.prepare_only:
            continue

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
        scalars = dft_bigdft.parse_log_scalars(archive_dir)
        status["jobs"][idx].update(
            {
                "state": "completed" if rc == 0 else "failed",
                "returncode": rc,
                "elapsed_s": elapsed,
                "finished_at": now(),
                "scalars": scalars,
            }
        )
        status["updated_at"] = now()
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"job finish {job.name} rc={rc} elapsed={elapsed:.0f}s")
        if rc != 0 and args.stop_on_failure:
            status["state"] = "failed"
            write_json(run_root / "status.json", status)
            return rc if rc in range(1, 126) else 1

    if args.prepare_only:
        status["state"] = "prepared"
    else:
        failed = [job for job in status["jobs"] if job["state"] != "completed"]
        status["state"] = "completed" if not failed else "completed_with_failures"
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"label queue {status['state']}")
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
