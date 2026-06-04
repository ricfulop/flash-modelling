#!/usr/bin/env python3
"""Independent single-node BigDFT queue.

Runs BigDFT without cross-node MPI. This is intended as the reliable DFT path
while GPU KPM sweeps continue and two-node MPI is debugged separately.
"""
from __future__ import annotations

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

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
SEED = 20260603


@dataclass
class Job:
    name: str
    atoms: object
    hgrid: float
    rmult: tuple[float, float]
    max_seconds: int
    notes: str
    surface: bool = True


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(cmd: str, *, timeout: int | None = None, log: Path | None = None) -> subprocess.CompletedProcess:
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
    return subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def heartbeat(run_root: Path, message: str) -> None:
    with (run_root / "heartbeat.log").open("a") as fh:
        fh.write(f"{now()} {message}\n")


def bigdft_activation() -> str:
    candidates = [
        os.environ.get("ELECTRODEFECT_BIGDFT_ACTIVATE"),
        "/home/ricfulop/Desktop/Cursor/use_bigdft.sh",
        "/home/nvidia/Cursor/use_bigdft.sh",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if Path(candidate).exists():
                return candidate
        except OSError:
            continue
    raise FileNotFoundError("Could not find use_bigdft.sh; set ELECTRODEFECT_BIGDFT_ACTIVATE")


def find_w_psp() -> Path:
    candidates = [
        Path("/home/ricfulop/Desktop/Cursor/bigdft-suite/PyBigDFT/BigDFT/Database/psppar/Krack/PBE/psppar.W.yaml"),
        Path("/home/nvidia/Cursor/bigdft-suite/PyBigDFT/BigDFT/Database/psppar/Krack/PBE/psppar.W.yaml"),
        Path("/home/ricfulop/Desktop/Cursor/.local-bigdft/envs/bigdft/lib/python3.11/site-packages/BigDFT/Database/psppar/Krack/PBE/psppar.W.yaml"),
        Path("/home/nvidia/Cursor/.local-bigdft/envs/bigdft/lib/python3.11/site-packages/BigDFT/Database/psppar/Krack/PBE/psppar.W.yaml"),
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    raise FileNotFoundError("Could not find psppar.W.yaml")


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target, symlinks=True)
        else:
            shutil.copy2(item, target)


def surface_posinp(atoms, path: Path, surface: bool = True) -> None:
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    lengths = atoms.get_cell().lengths()
    if np.any(np.asarray(lengths) <= 0):
        mins = pos.min(axis=0)
        maxs = pos.max(axis=0)
        lengths = np.maximum(maxs - mins + 12.0, 12.0)
    line2 = "surface " + " ".join(f"{x:.8f}" for x in lengths) if surface else "free"
    with path.open("w") as fh:
        fh.write(f"{len(atoms)} angstroem\n")
        fh.write(line2 + "\n")
        for sym, xyz in zip(symbols, pos):
            fh.write(f"{sym} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")


def write_bigdft_input(job: Job, job_dir: Path) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    surface_posinp(job.atoms, job_dir / "posinp.xyz", surface=job.surface)
    shutil.copy2(find_w_psp(), job_dir / "psppar.W.yaml")
    input_yaml = {
        "outdir": "run",
        "logfile": "Yes",
        "dft": {
            "hgrids": [job.hgrid, job.hgrid, job.hgrid],
            "rmult": [job.rmult[0], job.rmult[1]],
            "ixc": "PBE",
            "inputpsiid": 0,
            "output_denspot": 22 if job.surface else 0,
        },
        "mix": {"tel": 0.72 / 27.211386245988, "occopt": 1},
    }
    (job_dir / "input.yaml").write_text(yaml.safe_dump(input_yaml, sort_keys=False))
    (job_dir / "job.json").write_text(json.dumps({k: v for k, v in asdict(job).items() if k != "atoms"}, indent=2) + "\n")


def build_atoms():
    from ase import Atoms
    from ase.build import bcc100

    rng = np.random.default_rng(SEED)

    def slab(nx: int, ny: int, nz: int, vacuum: float = 14.0):
        atoms = bcc100("W", size=(nx, ny, nz), a=3.165, vacuum=0.0, orthogonal=True)
        cell = atoms.get_cell()
        cell[2, 2] += vacuum
        atoms.set_cell(cell)
        atoms.positions[:, 2] -= atoms.positions[:, 2].min()
        atoms.positions[:, 2] += 0.5 * vacuum
        return atoms

    def defected(base, frac: float = 0.10, surface_band: float = 7.0):
        atoms = base.copy()
        z = atoms.positions[:, 2]
        candidates = np.flatnonzero(z > z.max() - surface_band)
        n_def = max(1, int(frac * len(candidates)))
        vac = rng.choice(candidates, size=n_def, replace=False)
        keep = np.ones(len(atoms), dtype=bool)
        extras = []
        axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        for idx in vac:
            keep[idx] = False
            base_pos = atoms.positions[idx] + 0.20 * 3.165 * axis
            extras.extend([base_pos + 0.18 * 3.165 * axis, base_pos - 0.18 * 3.165 * axis])
        out = atoms[keep]
        out += Atoms("W" * len(extras), positions=np.asarray(extras))
        out.set_cell(base.get_cell())
        return out

    def stepped(base):
        atoms = base.copy()
        z = atoms.positions[:, 2]
        x = atoms.positions[:, 0]
        mask = ~((z > z.max() - 1.8) & (x < np.median(x)))
        out = atoms[mask]
        out.set_cell(base.get_cell())
        return out

    small = slab(3, 3, 4)
    medium = slab(4, 4, 6)
    return {
        "w_atom": Atoms("W", positions=[[0.0, 0.0, 0.0]], cell=[18, 18, 18], pbc=False),
        "clean_small": small,
        "defect_small": defected(small, frac=0.10),
        "stepped_small": stepped(small),
        "separated_small": defected(small, frac=0.12),
        "clean_medium": medium,
        "defect_medium": defected(medium, frac=0.10),
    }


def jobs(label: str) -> list[Job]:
    atoms = build_atoms()
    full = [
        Job("00_w_atom_smoke", atoms["w_atom"], 0.55, (4.0, 6.0), 900, "single-node W pseudopotential smoke", False),
        Job("01_clean_w001_small_phi", atoms["clean_small"], 0.52, (4.0, 7.0), 7200, "clean W(001) small slab"),
        Job("02_defect_w001_small_phi", atoms["defect_small"], 0.52, (4.0, 7.0), 9000, "defect W(001) small slab"),
        Job("03_stepped_w001_small_phi", atoms["stepped_small"], 0.52, (4.0, 7.0), 9000, "stepped W small slab"),
        Job("04_separated_small_EFP", atoms["separated_small"], 0.50, (4.5, 8.0), 10800, "separated FP small slab"),
        Job("05_clean_w001_medium_phi", atoms["clean_medium"], 0.48, (5.0, 9.0), 14400, "clean W medium slab"),
        Job("06_defect_w001_medium_phi", atoms["defect_medium"], 0.48, (5.0, 9.0), 18000, "defect W medium slab"),
    ]
    if label == "peer":
        # Start peer on defect-bearing cells so local and peer queues are complementary.
        return [full[0], full[2], full[4], full[6], full[3]]
    return [full[0], full[1], full[3], full[5], full[2]]


def main() -> int:
    label = os.environ.get("ELECTRODEFECT_BIGDFT_LABEL", socket.gethostname())
    run_id = datetime.now().strftime(f"single_bigdft_%Y%m%d_%H%M%S_{label}")
    run_root = RUNS / run_id
    log_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    activate = bigdft_activation()

    status = {
        "state": "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "label": label,
        "activate": activate,
        "jobs": [{"name": j.name, "notes": j.notes, "state": "pending", "elapsed_s": 0} for j in jobs(label)],
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "queue start")

    for idx, job in enumerate(jobs(label)):
        status["jobs"][idx]["state"] = "running"
        status["jobs"][idx]["started_at"] = now()
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"job start {job.name}")
        archive_dir = run_root / "work" / job.name
        exec_dir = Path("/tmp") / run_id / job.name
        if exec_dir.exists():
            shutil.rmtree(exec_dir)
        exec_dir.mkdir(parents=True, exist_ok=True)
        write_bigdft_input(job, exec_dir)
        write_bigdft_input(job, archive_dir)
        start = time.time()
        rc = 1
        try:
            cmd = f"source {activate} >/tmp/use-bigdft-{run_id}.out && cd {exec_dir} && bigdft -l yes"
            rc = run(cmd, timeout=job.max_seconds, log=log_dir / f"{job.name}.out").returncode
        except subprocess.TimeoutExpired:
            rc = 124
        elapsed = time.time() - start
        copy_tree_contents(exec_dir, archive_dir)
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
        if idx == 0 and rc != 0:
            status["state"] = "failed_smoke"
            write_json(run_root / "status.json", status)
            return 3

    status["state"] = "completed"
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "queue completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
