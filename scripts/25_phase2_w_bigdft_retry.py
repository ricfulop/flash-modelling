#!/usr/bin/env python3
"""Retry Phase 2 W BigDFT labels with stricter SCF settings.

The first Phase 2 W morphology labels produced finite energies but did not meet
BigDFT convergence criteria. This runner reruns only the W morphology labels
with a larger SCF budget while preserving the matrix-export request.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"

LABEL_PATH = Path(__file__).resolve().parent / "18_phase2_bigdft_labels.py"
LABEL_SPEC = importlib.util.spec_from_file_location("phase2_labels", LABEL_PATH)
labels = importlib.util.module_from_spec(LABEL_SPEC)
assert LABEL_SPEC.loader is not None
sys.modules[LABEL_SPEC.name] = labels
LABEL_SPEC.loader.exec_module(labels)

QUEUE_PATH = Path(__file__).resolve().parent / "08_single_node_bigdft_queue.py"
QUEUE_SPEC = importlib.util.spec_from_file_location("single_node_queue", QUEUE_PATH)
queue = importlib.util.module_from_spec(QUEUE_SPEC)
assert QUEUE_SPEC.loader is not None
sys.modules[QUEUE_SPEC.name] = queue
QUEUE_SPEC.loader.exec_module(queue)

sys.path.insert(0, str(REPO / "src"))
from electrodefect import dft_bigdft


@dataclass(frozen=True)
class RetrySetting:
    name: str
    hgrid: float
    rmult: tuple[float, float]
    itermax: int
    nrepmax: int
    gnrm_cv: float
    max_seconds: int


DEFAULT_SETTINGS = {
    "strict_same_grid": RetrySetting(
        name="strict_same_grid",
        hgrid=0.55,
        rmult=(4.0, 7.0),
        itermax=240,
        nrepmax=10,
        gnrm_cv=1.0e-4,
        max_seconds=28800,
    ),
    "expanded_fine": RetrySetting(
        name="expanded_fine",
        hgrid=0.50,
        rmult=(4.5, 8.0),
        itermax=240,
        nrepmax=10,
        gnrm_cv=1.0e-4,
        max_seconds=36000,
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def heartbeat(run_root: Path, message: str) -> None:
    with (run_root / "heartbeat.log").open("a") as fh:
        fh.write(f"{now()} {message}\n")


def run(cmd: str, *, timeout: int | None, log: Path):
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


def terminal_state(state: str | None) -> bool:
    return state in {
        "completed",
        "completed_with_failures",
        "failed",
        "prepared",
        "time_budget_reached",
        "aborted_for_matrix_export_relaunch",
    }


def wait_for_status(path: Path, poll_s: int) -> None:
    while True:
        status = read_json(path, {})
        if terminal_state(status.get("state")):
            return
        time.sleep(max(5, int(poll_s)))


def wait_for_status_glob(pattern: str, poll_s: int) -> Path:
    """Wait for the newest matching status file to reach a terminal state."""
    while True:
        matches = sorted(RUNS.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0)
        if matches:
            latest = matches[-1]
            status = read_json(latest, {})
            if terminal_state(status.get("state")):
                return latest
        time.sleep(max(5, int(poll_s)))


def selected_settings(names: list[str]) -> list[RetrySetting]:
    if not names:
        names = ["strict_same_grid"]
    settings = []
    for name in names:
        if name not in DEFAULT_SETTINGS:
            raise ValueError(f"unknown setting {name!r}; choose one of {sorted(DEFAULT_SETTINGS)}")
        settings.append(DEFAULT_SETTINGS[name])
    return settings


def load_w_jobs(roots: list[Path], settings: list[RetrySetting], max_jobs: int = 0) -> list[tuple[object, RetrySetting]]:
    source_jobs = [
        job for job in labels.load_jobs(roots, hgrid=0.55, rmult=(4.0, 7.0), max_seconds=28800)
        if job.material == "W"
    ]
    if max_jobs:
        source_jobs = source_jobs[: int(max_jobs)]
    jobs = []
    for source in source_jobs:
        for setting in settings:
            retry = replace(
                source,
                name=f"{source.name}_{setting.name}",
                hgrid=setting.hgrid,
                rmult=setting.rmult,
                max_seconds=setting.max_seconds,
                matrix_export=True,
            )
            jobs.append((retry, setting))
    return jobs


def write_bigdft_input(job, setting: RetrySetting, job_dir: Path) -> None:
    from ase.io import read

    atoms = read(job.source_atoms)
    job_dir.mkdir(parents=True, exist_ok=True)
    queue.surface_posinp(atoms, job_dir / "posinp.xyz", surface=job.surface)
    shutil.copy2(labels.find_psp(job.material), job_dir / f"psppar.{job.material}.yaml")
    input_yaml = {
        "outdir": "run",
        "logfile": "Yes",
        "dft": {
            "hgrids": [setting.hgrid, setting.hgrid, setting.hgrid],
            "rmult": [setting.rmult[0], setting.rmult[1]],
            "ixc": "PBE",
            "inputpsiid": 0,
            "itermax": setting.itermax,
            "nrepmax": setting.nrepmax,
            "gnrm_cv": setting.gnrm_cv,
            "output_denspot": 22 if job.surface else 0,
        },
        "mix": {"tel": 0.72 / dft_bigdft.EV_PER_HA, "occopt": 1},
        "import": "linear",
        "lin_general": {"output_mat": 1},
    }
    (job_dir / "input.yaml").write_text(yaml.safe_dump(input_yaml, sort_keys=False))
    meta = {**asdict(job), "retry_setting": asdict(setting)}
    (job_dir / "job.json").write_text(json.dumps(meta, indent=2) + "\n")


def regex_last(pattern: str, text: str, cast=float):
    hits = re.findall(pattern, text, flags=re.IGNORECASE)
    if not hits:
        return None
    value = hits[-1]
    if isinstance(value, tuple):
        value = value[-1]
    try:
        return cast(str(value).strip())
    except Exception:
        return None


def matrix_artifacts(job_dir: Path) -> list[str]:
    names: list[str] = []
    for path in job_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(job_dir))
        lower = rel.lower()
        if (
            lower.endswith(".mtx")
            or lower.endswith("_hr.dat")
            or fnmatch.fnmatch(lower, "*ham*")
            or fnmatch.fnmatch(lower, "*overlap*")
        ):
            names.append(rel)
    return sorted(names)


def parse_job(job_dir: Path) -> dict:
    text = (job_dir / "log.yaml").read_text(errors="replace") if (job_dir / "log.yaml").exists() else ""
    meta = read_json(job_dir / "job.json", {})
    scalars = dft_bigdft.parse_log_scalars(job_dir)
    artifacts = matrix_artifacts(job_dir)
    return {
        "job": job_dir.name,
        "material": meta.get("material"),
        "morphology": meta.get("morphology"),
        "label_kind": meta.get("label_kind"),
        "setting": (meta.get("retry_setting") or {}).get("name"),
        "hgrid": (meta.get("retry_setting") or {}).get("hgrid"),
        "rmult": (meta.get("retry_setting") or {}).get("rmult"),
        "itermax": (meta.get("retry_setting") or {}).get("itermax"),
        "nrepmax": (meta.get("retry_setting") or {}).get("nrepmax"),
        "gnrm_cv": (meta.get("retry_setting") or {}).get("gnrm_cv"),
        "energy_Ha": scalars.get("energy_Ha"),
        "fermi_Ha": scalars.get("fermi_Ha"),
        "charge_e": scalars.get("charge_e"),
        "infocode": scalars.get("infocode"),
        "last_gnrm": regex_last(r"iter:\s*\d+,\s*FKS:\s*[+\-0-9.Ee]+,\s*gnrm:\s*([+\-0-9.Ee]+)", text),
        "last_delta_Ha": regex_last(r"iter:\s*\d+,\s*FKS:\s*[+\-0-9.Ee]+,\s*gnrm:\s*[+\-0-9.Ee]+,\s*D:\s*([+\-0-9.Ee]+)", text),
        "warning_count": len(re.findall(r"#WARNING|WARNING:", text, flags=re.IGNORECASE)),
        "has_nan": bool(re.search(r"(?<![A-Za-z])(?:NaN|nan)(?![A-Za-z])", text)),
        "matrix_artifact_count": len(artifacts),
        "matrix_artifacts": artifacts[:12],
        "path": str(job_dir),
    }


def write_rows(run_root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with (run_root / "w_retry_rows.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    strict = [
        row for row in rows
        if row.get("infocode") == 0
        and not row.get("has_nan")
        and int(row.get("warning_count") or 0) == 0
    ]
    usable = [
        row for row in rows
        if row.get("energy_Ha") is not None
        and row.get("fermi_Ha") is not None
        and not row.get("has_nan")
    ]
    return {
        "n_rows": len(rows),
        "n_usable_finite": len(usable),
        "n_strict_infocode0_no_warnings": len(strict),
        "jobs_with_matrix_artifacts": [
            row["job"] for row in rows if int(row.get("matrix_artifact_count") or 0) > 0
        ],
        "remaining_nonconverged": [
            row["job"] for row in rows if row.get("infocode") not in (0, 0.0)
        ],
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", action="append", type=Path, default=None)
    p.add_argument("--label", default=os.environ.get("ELECTRODEFECT_W_RETRY_LABEL", socket.gethostname()))
    p.add_argument("--setting", action="append", default=[], choices=sorted(DEFAULT_SETTINGS))
    p.add_argument("--max-jobs", type=int, default=0, help="Limit source W jobs before settings expansion.")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--wait-for-status", type=Path, default=None)
    p.add_argument("--wait-for-status-glob", default=None, help="RUNS-relative glob for a status.json to finish.")
    p.add_argument("--poll-seconds", type=int, default=300)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.wait_for_status:
        wait_for_status(args.wait_for_status, args.poll_seconds)
    if args.wait_for_status_glob:
        wait_for_status_glob(args.wait_for_status_glob, args.poll_seconds)

    roots = args.root or labels.DEFAULT_ROOTS
    settings = selected_settings(args.setting)
    jobs = load_w_jobs(roots, settings, max_jobs=args.max_jobs)
    if not jobs:
        raise SystemExit("no W retry jobs found")

    run_id = datetime.now().strftime(f"phase2_w_bigdft_retry_%Y%m%d_%H%M%S_{args.label}")
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
        "settings": [asdict(setting) for setting in settings],
        "matrix_export": True,
        "jobs": [
            {
                **asdict(job),
                "retry_setting": asdict(setting),
                "state": "pending",
                "elapsed_s": 0.0,
                "archive_dir": str(run_root / "work" / job.name),
            }
            for job, setting in jobs
        ],
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"W retry queue start n_jobs={len(jobs)} prepare_only={args.prepare_only}")

    tmp_root = Path("/tmp") / run_id
    for idx, (job, setting) in enumerate(jobs):
        archive_dir = run_root / "work" / job.name
        exec_dir = tmp_root / job.name
        if exec_dir.exists():
            shutil.rmtree(exec_dir)
        exec_dir.mkdir(parents=True, exist_ok=True)
        write_bigdft_input(job, setting, exec_dir)
        write_bigdft_input(job, setting, archive_dir)
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
        parsed = parse_job(archive_dir)
        status["jobs"][idx].update(
            {
                "state": "completed" if rc == 0 else "failed",
                "returncode": rc,
                "elapsed_s": elapsed,
                "finished_at": now(),
                "parsed": parsed,
            }
        )
        status["updated_at"] = now()
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"job finish {job.name} rc={rc} elapsed={elapsed:.0f}s")
        if rc != 0 and args.stop_on_failure:
            status["state"] = "failed"
            status["updated_at"] = now()
            write_json(run_root / "status.json", status)
            return rc if rc in range(1, 126) else 1

    rows = [parse_job(job_dir) for job_dir in sorted((run_root / "work").glob("*")) if job_dir.is_dir()]
    write_rows(run_root, rows)
    summary = summarize(rows)
    write_json(run_root / "w_retry_summary.json", summary)
    status["summary"] = summary
    if args.prepare_only:
        status["state"] = "prepared"
    else:
        failed = [job for job in status["jobs"] if job["state"] != "completed"]
        status["state"] = "completed" if not failed else "completed_with_failures"
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"W retry queue {status['state']}")
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
