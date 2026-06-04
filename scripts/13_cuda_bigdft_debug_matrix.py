#!/usr/bin/env python3
"""Isolated CUDA BigDFT debug matrix.

This is not a production queue. It runs minimal N2/W smokes in a fresh run
directory to identify which `perf` option first introduces NaNs.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
BASE = Path("/home/ricfulop/Desktop/Cursor")
CUDA_ACTIVATE = Path(os.environ.get("BIGDFT_CUDA_ACTIVATE", str(BASE / "use_bigdft_cuda.sh")))
BIGDFT_SOURCE = BASE / "bigdft-suite"


@dataclass(frozen=True)
class Case:
    name: str
    system: str
    perf: dict | None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def prepare_system(case: Case, case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    if case.system == "n2":
        shutil.copy2(BIGDFT_SOURCE / "bigdft/tests/tutorials/N2/posinp.xyz", case_dir / "posinp.xyz")
        shutil.copy2(BIGDFT_SOURCE / "bigdft/tests/tutorials/N2/psppar.N", case_dir / "psppar.N")
        payload = {
            "outdir": "run",
            "logfile": "Yes",
            "dft": {
                "hgrids": [0.45, 0.45, 0.45],
                "rmult": [5.0, 8.0],
                "ixc": 1,
                "inputpsiid": 0,
                "output_denspot": 0,
            },
        }
    elif case.system == "w_atom":
        (case_dir / "posinp.xyz").write_text(
            "1 angstroem\n"
            "free\n"
            "W 0.0 0.0 0.0\n"
        )
        psp = BIGDFT_SOURCE / "PyBigDFT/BigDFT/Database/psppar/Krack/PBE/psppar.W.yaml"
        shutil.copy2(psp, case_dir / "psppar.W.yaml")
        payload = {
            "outdir": "run",
            "logfile": "Yes",
            "dft": {
                "hgrids": [0.55, 0.55, 0.55],
                "rmult": [4.0, 6.0],
                "ixc": "PBE",
                "inputpsiid": 0,
                "output_denspot": 0,
            },
        }
    else:
        raise ValueError(case.system)
    if case.perf is not None:
        payload["perf"] = case.perf
    (case_dir / "input.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


def parse_log(case_dir: Path) -> dict:
    log_path = case_dir / "log.yaml"
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    energy = None
    m = re.findall(r"Energy \(Hartree\)\s*:\s*([+\-0-9.EeNaInf]+)", text)
    if m:
        try:
            energy = float(m[-1])
        except ValueError:
            energy = math.nan
    charge = None
    m = re.findall(r"Total electronic charge\s*:\s*([+\-0-9.EeNaInf]+)", text)
    if m:
        try:
            charge = float(m[-1])
        except ValueError:
            charge = math.nan
    infocode = None
    m = re.findall(r"BigDFT infocode\s*:\s*([+\-0-9]+)", text)
    if m:
        infocode = int(m[-1])
    has_nan = bool(re.search(r"(?<![A-Za-z])(?:NaN|nan)(?![A-Za-z])", text))
    material_accel = None
    m = re.findall(r"Material acceleration\s*:\s*([^#\n]+)", text)
    if m:
        material_accel = m[-1].strip()
    gpu_accel = None
    m = re.findall(r"GPU acceleration\s*:\s*([^,\n]+)", text)
    if m:
        gpu_accel = m[-1].strip()
    return {
        "energy_Ha": energy,
        "charge_e": charge,
        "infocode": infocode,
        "has_nan": has_nan,
        "material_acceleration": material_accel,
        "gpu_acceleration_line": gpu_accel,
    }


def run_case(run_root: Path, case: Case, timeout_s: int) -> dict:
    case_dir = run_root / case.name
    prepare_system(case, case_dir)
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_LAUNCH_BLOCKING": "1",
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        }
    )
    cmd = (
        f"source {CUDA_ACTIVATE} >/tmp/use-bigdft-cuda-debug.out && "
        f"cd {case_dir} && rm -rf run debug log.yaml forces_posinp.xyz && "
        "nice -n 15 bigdft -l yes"
    )
    out_path = case_dir / "stdout.txt"
    start = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        rc = proc.returncode
        stdout = proc.stdout
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
    out_path.write_text(stdout)
    parsed = parse_log(case_dir)
    return {
        "case": case.name,
        "system": case.system,
        "perf": case.perf,
        "returncode": rc,
        "started_at": start.isoformat(timespec="seconds"),
        "finished_at": now(),
        **parsed,
        "path": str(case_dir),
    }


def main() -> int:
    run_root = RUNS / datetime.now().strftime("cuda_bigdft_debug_%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    cases = [
        Case("n2_no_perf_control", "n2", None),
        Case("n2_explicit_no_accel_no_blas", "n2", {"accel": "NO", "blas": False}),
        Case("n2_explicit_no_accel_blas", "n2", {"accel": "NO", "blas": True}),
        Case("n2_blas_only", "n2", {"blas": True}),
        Case("n2_cuda_no_blas", "n2", {"accel": "CUDAGPU", "blas": False}),
        Case("n2_cuda_blas", "n2", {"accel": "CUDAGPU", "blas": True}),
        Case("w_no_perf_control", "w_atom", None),
        Case("w_cuda_blas", "w_atom", {"accel": "CUDAGPU", "blas": True}),
    ]
    rows = []
    for case in cases:
        row = run_case(run_root, case, timeout_s=240)
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
        write_json(run_root / "results.json", {"created_at": now(), "cases": rows})
        # Stop early after the CUDAGPU tests if the known failure is reproduced.
        if case.name == "n2_cuda_blas" and row.get("has_nan"):
            continue
    write_json(run_root / "results.json", {"created_at": now(), "cases": rows})
    print(f"CUDA BigDFT debug matrix complete -> {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
