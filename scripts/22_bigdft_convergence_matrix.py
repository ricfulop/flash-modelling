#!/usr/bin/env python3
"""Targeted BigDFT convergence matrix for Sunday-critical W slabs.

This queue is intentionally small: it tests whether the clean/defect/stepped
work-function proxies are stable across stricter BigDFT settings before they are
promoted beyond "provisional".
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
QUEUE_PATH = Path(__file__).resolve().parent / "08_single_node_bigdft_queue.py"
SPEC = importlib.util.spec_from_file_location("single_node_queue", QUEUE_PATH)
queue = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = queue
SPEC.loader.exec_module(queue)

sys.path.insert(0, str(REPO / "src"))
from electrodefect import dft_bigdft


@dataclass(frozen=True)
class ConvergenceSetting:
    name: str
    hgrid: float
    rmult: tuple[float, float]
    itermax: int
    nrepmax: int
    gnrm_cv: float
    max_seconds: int


@dataclass(frozen=True)
class ConvergenceJob:
    name: str
    role: str
    source_job: str
    setting: ConvergenceSetting
    atoms: object
    notes: str
    surface: bool = True


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
    return subprocess.run(cmd, shell=True, executable="/bin/bash", text=True, timeout=timeout)


def source_jobs() -> dict[str, object]:
    return {job.name: job for job in queue.jobs("local")}


def default_settings(source) -> list[ConvergenceSetting]:
    return [
        ConvergenceSetting(
            name="nrep8_same_grid",
            hgrid=float(source.hgrid),
            rmult=tuple(float(x) for x in source.rmult),
            itermax=160,
            nrepmax=8,
            gnrm_cv=5.0e-5,
            max_seconds=max(int(source.max_seconds) * 2, 18000),
        ),
        ConvergenceSetting(
            name="expanded_fine",
            hgrid=min(float(source.hgrid), 0.50),
            rmult=(max(float(source.rmult[0]), 4.5), max(float(source.rmult[1]), 8.0)),
            itermax=160,
            nrepmax=8,
            gnrm_cv=5.0e-5,
            max_seconds=max(int(source.max_seconds) * 2, 21600),
        ),
    ]


def make_jobs(cases: list[str]) -> list[ConvergenceJob]:
    sources = source_jobs()
    wanted = {
        "clean": "01_clean_w001_small_phi",
        "defect": "02_defect_w001_small_phi",
        "stepped": "03_stepped_w001_small_phi",
    }
    if cases:
        wanted = {role: name for role, name in wanted.items() if role in set(cases)}
    jobs: list[ConvergenceJob] = []
    for role, source_name in wanted.items():
        source = sources[source_name]
        for setting in default_settings(source):
            jobs.append(
                ConvergenceJob(
                    name=f"{role}_{setting.name}",
                    role=role,
                    source_job=source_name,
                    setting=setting,
                    atoms=source.atoms,
                    notes=f"{source.notes}; convergence matrix setting {setting.name}",
                    surface=source.surface,
                )
            )
    return jobs


def write_bigdft_input(job: ConvergenceJob, job_dir: Path) -> None:
    setting = job.setting
    job_dir.mkdir(parents=True, exist_ok=True)
    queue.surface_posinp(job.atoms, job_dir / "posinp.xyz", surface=job.surface)
    shutil.copy2(queue.find_w_psp(), job_dir / "psppar.W.yaml")
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
    }
    (job_dir / "input.yaml").write_text(yaml.safe_dump(input_yaml, sort_keys=False))
    meta = {
        "name": job.name,
        "role": job.role,
        "source_job": job.source_job,
        "notes": job.notes,
        "surface": job.surface,
        "setting": asdict(setting),
        "n_atoms": len(job.atoms),
    }
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


def count_warnings(text: str) -> int:
    return len(re.findall(r"#WARNING|WARNING:", text, flags=re.IGNORECASE))


def contains_numeric_nan(text: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])(?:NaN|nan)(?![A-Za-z])", text))


def parse_job(job_dir: Path) -> dict:
    log_path = job_dir / "log.yaml"
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    meta = read_json(job_dir / "job.json", {})
    scalars = dft_bigdft.parse_log_scalars(job_dir)
    phi = None
    try:
        phi = dft_bigdft.work_function(job_dir).get("phi_eV")
    except Exception:
        phi = None
    return {
        "job": job_dir.name,
        "role": meta.get("role"),
        "source_job": meta.get("source_job"),
        "setting": (meta.get("setting") or {}).get("name"),
        "hgrid": (meta.get("setting") or {}).get("hgrid"),
        "rmult": (meta.get("setting") or {}).get("rmult"),
        "itermax": (meta.get("setting") or {}).get("itermax"),
        "nrepmax": (meta.get("setting") or {}).get("nrepmax"),
        "gnrm_cv": (meta.get("setting") or {}).get("gnrm_cv"),
        "atoms": meta.get("n_atoms"),
        "energy_Ha": scalars.get("energy_Ha"),
        "fermi_Ha": scalars.get("fermi_Ha"),
        "charge_e": scalars.get("charge_e"),
        "infocode": scalars.get("infocode"),
        "warning_count": count_warnings(text),
        "has_nan": contains_numeric_nan(text),
        "elapsed_log_s": regex_last(r"Elapsed time \(s\)\s*:\s*([+\-0-9.Ee]+)", text),
        "phi_proxy_eV": phi,
        "path": str(job_dir),
    }


def finite_number(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def stability_summary(rows: list[dict], phi_tol_eV: float, fermi_tol_Ha: float) -> dict:
    summary = {"roles": {}, "stable_proxy_roles": [], "strict_claim_roles": []}
    for role in sorted({row.get("role") for row in rows if row.get("role")}):
        role_rows = [row for row in rows if row.get("role") == role]
        completed = [
            row for row in role_rows
            if finite_number(row.get("energy_Ha"))
            and finite_number(row.get("fermi_Ha"))
            and finite_number(row.get("phi_proxy_eV"))
            and not row.get("has_nan")
        ]
        phi_values = np.asarray([float(row["phi_proxy_eV"]) for row in completed], dtype=float)
        fermi_values = np.asarray([float(row["fermi_Ha"]) for row in completed], dtype=float)
        phi_range = float(np.max(phi_values) - np.min(phi_values)) if len(phi_values) else None
        fermi_range = float(np.max(fermi_values) - np.min(fermi_values)) if len(fermi_values) else None
        stable_proxy = bool(
            len(completed) >= 2
            and phi_range is not None
            and phi_range <= phi_tol_eV
            and fermi_range is not None
            and fermi_range <= fermi_tol_Ha
        )
        strict_claim = bool(
            stable_proxy
            and all(int(float(row.get("infocode") or 999)) == 0 for row in completed)
            and all(int(row.get("warning_count") or 0) == 0 for row in completed)
        )
        if stable_proxy:
            summary["stable_proxy_roles"].append(role)
        if strict_claim:
            summary["strict_claim_roles"].append(role)
        summary["roles"][role] = {
            "n_attempted": len(role_rows),
            "n_completed_finite": len(completed),
            "phi_range_eV": phi_range,
            "fermi_range_Ha": fermi_range,
            "stable_proxy": stable_proxy,
            "strict_claim": strict_claim,
            "settings_completed": [row.get("setting") for row in completed],
        }
    return summary


def wait_for_status(status_path: Path, poll_s: int) -> None:
    terminal = {
        "completed",
        "completed_with_failures",
        "failed",
        "prepared",
        "time_budget_reached",
        "aborted_for_matrix_export_relaunch",
    }
    while True:
        status = read_json(status_path, {})
        if status.get("state") in terminal:
            return
        time.sleep(max(5, int(poll_s)))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default=os.environ.get("ELECTRODEFECT_CONV_LABEL", socket.gethostname()))
    p.add_argument("--case", action="append", default=[], choices=["clean", "defect", "stepped"])
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--wait-for-status", type=Path, default=None)
    p.add_argument("--poll-seconds", type=int, default=300)
    p.add_argument("--phi-tol-eV", type=float, default=0.15)
    p.add_argument("--fermi-tol-Ha", type=float, default=0.002)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.wait_for_status:
        wait_for_status(args.wait_for_status, args.poll_seconds)
    run_id = datetime.now().strftime(f"bigdft_convergence_matrix_%Y%m%d_%H%M%S_{args.label}")
    run_root = RUNS / run_id
    log_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    activate = queue.bigdft_activation()
    jobs = make_jobs(args.case)
    if args.max_jobs:
        jobs = jobs[: int(args.max_jobs)]
    status = {
        "state": "prepared" if args.prepare_only else "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "label": args.label,
        "activate": activate,
        "phi_tol_eV": args.phi_tol_eV,
        "fermi_tol_Ha": args.fermi_tol_Ha,
        "jobs": [
            {
                "name": job.name,
                "role": job.role,
                "source_job": job.source_job,
                "setting": asdict(job.setting),
                "state": "pending",
                "archive_dir": str(run_root / "work" / job.name),
            }
            for job in jobs
        ],
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"convergence matrix start n_jobs={len(jobs)} prepare_only={args.prepare_only}")

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
            rc = run(cmd, timeout=job.setting.max_seconds, log=log_dir / f"{job.name}.out").returncode
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

    rows = [parse_job(job_dir) for job_dir in sorted((run_root / "work").glob("*")) if job_dir.is_dir()]
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (run_root / "convergence_matrix_rows.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = stability_summary(rows, phi_tol_eV=float(args.phi_tol_eV), fermi_tol_Ha=float(args.fermi_tol_Ha))
    write_json(run_root / "convergence_stability_summary.json", summary)
    status["stability_summary"] = summary
    if args.prepare_only:
        status["state"] = "prepared"
    else:
        failed = [job for job in status["jobs"] if job["state"] != "completed"]
        status["state"] = "completed" if not failed else "completed_with_failures"
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"convergence matrix {status['state']}")
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
