#!/usr/bin/env python3
"""Minimal Quantum ESPRESSO smoke runner for Phase 2 W/Mo primary DFT.

This is intentionally tiny: it verifies that pw.x, MPI/OpenMP runtime libraries,
and local PBE UPF pseudopotentials are usable before launching the slab-label queue.
ABINIT remains the fallback/cross-check route when a second code is needed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_QE_ROOT = Path("/home/ricfulop/Desktop/Cursor/.local-qe")
DEFAULT_TOOLCHAIN = Path("/home/ricfulop/Desktop/Cursor/.local-bigdft-opencl/envs/bigdft-opencl")
DEFAULT_NVHPC = DEFAULT_QE_ROOT / "nvhpc"
NVHPC_ARCH = "Linux_aarch64"
NVHPC_VERSION = "25.11"
NVHPC_CUDA_VERSION = "13.0"

MATERIALS = {
    "W": {
        "mass": 183.84,
        "a_ang": 3.165,
        "pseudo": "W.pbe-spn-kjpaw_psl.1.0.0.UPF",
    },
    "Mo": {
        "mass": 95.95,
        "a_ang": 3.147,
        "pseudo": "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_pw_x(explicit: str | None, *, gpu: bool) -> str:
    if explicit:
        return explicit
    if gpu:
        candidates = [
            DEFAULT_QE_ROOT / "install/qe-7.4.1-cuda/bin/pw.x",
            DEFAULT_QE_ROOT / "build/qe-7.4.1-cuda/bin/pw.x",
            DEFAULT_QE_ROOT / "build/qe-7.4.1-cuda/PW/src/pw.x",
        ]
    else:
        candidates = [
            DEFAULT_QE_ROOT / "install/qe-7.4.1-cpu/bin/pw.x",
            DEFAULT_QE_ROOT / "build/qe-7.4.1-cpu/bin/pw.x",
            DEFAULT_QE_ROOT / "build/qe-7.4.1-cpu/PW/src/pw.x",
        ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "pw.x"


def write_input(
    path: Path,
    *,
    material: str,
    pseudo_dir: Path,
    ecutwfc: float,
    ecutrho: float,
    kgrid: int,
    conv_thr: float,
    electron_maxstep: int,
) -> None:
    spec = MATERIALS[material]
    a_bohr = spec["a_ang"] / 0.529177210903
    text = f"""&CONTROL
  calculation = 'scf',
  prefix = '{material.lower()}_bulk_smoke',
  outdir = './qe_out',
  pseudo_dir = '{pseudo_dir}',
  tstress = .true.,
  tprnfor = .true.,
/
&SYSTEM
  ibrav = 1,
  celldm(1) = {a_bohr:.12f},
  nat = 2,
  ntyp = 1,
  ecutwfc = {ecutwfc:.6f},
  ecutrho = {ecutrho:.6f},
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.02,
/
&ELECTRONS
  conv_thr = {conv_thr:.3e},
  electron_maxstep = {int(electron_maxstep)},
  mixing_beta = 0.30,
/
ATOMIC_SPECIES
  {material} {spec["mass"]:.6f} {spec["pseudo"]}
ATOMIC_POSITIONS crystal
  {material} 0.0000000000 0.0000000000 0.0000000000
  {material} 0.5000000000 0.5000000000 0.5000000000
K_POINTS automatic
  {int(kgrid)} {int(kgrid)} {int(kgrid)} 0 0 0
"""
    path.write_text(text)


def parse_output(text: str) -> dict:
    energy_matches = re.findall(r"!\s+total energy\s+=\s+([-+0-9.Ee]+)\s+Ry", text)
    force_matches = re.findall(r"Total force\s+=\s+([-+0-9.Ee]+)\s+Total SCF correction", text)
    wall_matches = re.findall(r"PWSCF\s+:\s+([0-9.]+)s CPU\s+([0-9.]+)s WALL", text)
    return {
        "converged": "convergence has been achieved" in text,
        "job_done": "JOB DONE." in text,
        "total_energy_ry": float(energy_matches[-1]) if energy_matches else None,
        "total_force_ry_bohr": float(force_matches[-1]) if force_matches else None,
        "pwscf_wall_s": float(wall_matches[-1][1]) if wall_matches else None,
        "gpu_markers": [
            line.strip()
            for line in text.splitlines()
            if "gpu" in line.lower() or "cuda" in line.lower() or "cusolver" in line.lower()
        ][:40],
        "error_markers": [
            line.strip()
            for line in text.splitlines()
            if "%%%%%%" in line or "Error in routine" in line or "stopping ..." in line
        ][:20],
    }


def qe_environment(*, gpu: bool, omp_threads: int) -> dict:
    env = os.environ.copy()
    path_parts = []
    ld_parts = []
    if gpu:
        nvhpc_base = DEFAULT_NVHPC / NVHPC_ARCH / NVHPC_VERSION
        path_parts.extend([
            str(nvhpc_base / "compilers/bin"),
            str(nvhpc_base / "comm_libs/mpi/bin"),
        ])
        ld_parts.extend([
            str(nvhpc_base / "compilers/lib"),
            str(nvhpc_base / f"cuda/{NVHPC_CUDA_VERSION}/lib64"),
            str(nvhpc_base / f"math_libs/{NVHPC_CUDA_VERSION}/lib64"),
            str(nvhpc_base / "comm_libs/mpi/lib"),
        ])
        env.setdefault("NVCOMPILERS", str(DEFAULT_NVHPC))
    if DEFAULT_TOOLCHAIN.exists():
        path_parts.append(str(DEFAULT_TOOLCHAIN / "bin"))
        ld_parts.append(str(DEFAULT_TOOLCHAIN / "lib"))
    env["PATH"] = ":".join([*path_parts, env.get("PATH", "")])
    env["LD_LIBRARY_PATH"] = ":".join([*ld_parts, env.get("LD_LIBRARY_PATH", "")])
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    return env


def run_qe(args: argparse.Namespace) -> dict:
    material = args.material
    pseudo_dir = args.pseudo_dir.resolve()
    pseudo = pseudo_dir / MATERIALS[material]["pseudo"]
    if not pseudo.exists():
        raise FileNotFoundError(f"missing pseudopotential: {pseudo}")

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / f"{material.lower()}_bulk_smoke.in"
    output_path = run_dir / f"{material.lower()}_bulk_smoke.out"
    summary_path = run_dir / f"{material.lower()}_bulk_smoke_summary.json"
    write_input(
        input_path,
        material=material,
        pseudo_dir=pseudo_dir,
        ecutwfc=args.ecutwfc,
        ecutrho=args.ecutrho,
        kgrid=args.kgrid,
        conv_thr=args.conv_thr,
        electron_maxstep=args.electron_maxstep,
    )

    pw_x = find_pw_x(args.pw_x, gpu=args.gpu)
    command = [pw_x, "-in", str(input_path)]
    if args.mpi > 1:
        command = ["mpirun", "-np", str(args.mpi), *command]

    env = qe_environment(gpu=args.gpu, omp_threads=args.omp_threads)

    started = time.time()
    with output_path.open("w") as out:
        out.write(f"# started {now()}\n")
        out.write(f"# command {' '.join(shlex.quote(part) for part in command)}\n")
        out.flush()
        proc = subprocess.run(
            command,
            cwd=run_dir,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_s,
        )

    text = output_path.read_text(errors="replace")
    parsed = parse_output(text)
    summary = {
        "state": "completed" if proc.returncode == 0 and parsed["job_done"] else "failed",
        "timestamp_utc": now(),
        "material": material,
        "returncode": proc.returncode,
        "runtime_s": round(time.time() - started, 3),
        "pw_x": pw_x,
        "gpu_requested": bool(args.gpu),
        "mpi": args.mpi,
        "omp_threads": args.omp_threads,
        "input": str(input_path),
        "output": str(output_path),
        "pseudo": str(pseudo),
        **parsed,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", choices=sorted(MATERIALS), default="W")
    parser.add_argument("--run-dir", type=Path, default=REPO / "runs/qe_smoke")
    parser.add_argument("--pseudo-dir", type=Path, default=REPO / "data/qe_pseudos")
    parser.add_argument("--pw-x", default=os.environ.get("QE_PW_X"))
    parser.add_argument("--gpu", action="store_true", help="Use the CUDA/NVHPC QE build and runtime.")
    parser.add_argument("--mpi", type=int, default=1)
    parser.add_argument("--omp-threads", type=int, default=2)
    parser.add_argument("--kgrid", type=int, default=2)
    parser.add_argument("--ecutwfc", type=float, default=40.0)
    parser.add_argument("--ecutrho", type=float, default=320.0)
    parser.add_argument("--conv-thr", type=float, default=1.0e-6)
    parser.add_argument("--electron-maxstep", type=int, default=80)
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()
    summary = run_qe(args)
    return 0 if summary["state"] == "completed" and summary["converged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
