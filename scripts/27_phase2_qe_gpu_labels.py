#!/usr/bin/env python3
"""Run GPU Quantum ESPRESSO labels for Phase 2 W/Mo candidates.

This is the QE/GPU analogue of the BigDFT label queue. It intentionally keeps
the same manifest-driven job selection, but writes plane-wave slab SCF inputs
for the validated QE 7.5 + NVHPC CUDA build.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
DEFAULT_QE_ROOT = Path("/home/ricfulop/Desktop/Cursor/.local-qe")
DEFAULT_NVHPC = DEFAULT_QE_ROOT / "nvhpc"
NVHPC_ARCH = "Linux_aarch64"
NVHPC_VERSION = "25.11"
NVHPC_CUDA_VERSION = "13.0"
DEFAULT_ROOTS = [
    RUNS / "phase2_mlip_20260603_213335_spark-5da5",
    RUNS / "phase2_mlip_20260603_213410_spark-5da5",
]

MATERIALS = {
    "W": {
        "mass": 183.84,
        "pseudo": "W.pbe-spn-kjpaw_psl.1.0.0.UPF",
    },
    "Mo": {
        "mass": 95.95,
        "pseudo": "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF",
    },
}


@dataclass(frozen=True)
class LabelJob:
    name: str
    material: str
    morphology: str
    label_kind: str
    source_atoms: str
    max_seconds: int


@dataclass(frozen=True)
class QESetting:
    name: str
    ecutwfc: float
    ecutrho: float
    kgrid: tuple[int, int, int]
    conv_thr: float
    electron_maxstep: int
    diagonalization: str
    mixing_mode: str
    mixing_beta: float
    mixing_ndim: int
    smearing: str
    degauss: float
    startingpot: str
    startingwfc: str
    max_seconds: int


DEFAULT_SETTING = QESetting(
    name="gpu_pbe_paw_smoke_grid",
    ecutwfc=40.0,
    ecutrho=320.0,
    kgrid=(2, 2, 1),
    conv_thr=1.0e-6,
    electron_maxstep=120,
    diagonalization="david",
    mixing_mode="plain",
    mixing_beta=0.25,
    mixing_ndim=8,
    smearing="mv",
    degauss=0.02,
    startingpot="atomic",
    startingwfc="atomic+random",
    max_seconds=3600,
)


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


def load_jobs(roots: list[Path], max_seconds: int) -> list[LabelJob]:
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
                    max_seconds=max_seconds,
                )
            )
    return jobs


def find_pw_x(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    candidates = [
        DEFAULT_QE_ROOT / "install/qe-7.5-gpu/bin/pw.x",
        DEFAULT_QE_ROOT / "src/q-e-qe-7.5/bin/pw.x",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "pw.x"


def qe_environment(omp_threads: int) -> dict:
    env = os.environ.copy()
    nvhpc_base = DEFAULT_NVHPC / NVHPC_ARCH / NVHPC_VERSION
    path_parts = [
        str(nvhpc_base / "compilers/bin"),
        str(nvhpc_base / "comm_libs/mpi/bin"),
    ]
    ld_parts = [
        str(nvhpc_base / "compilers/lib"),
        str(nvhpc_base / f"cuda/{NVHPC_CUDA_VERSION}/lib64"),
        str(nvhpc_base / f"math_libs/{NVHPC_CUDA_VERSION}/lib64"),
        str(nvhpc_base / "math_libs/nvpl/lib"),
        str(nvhpc_base / "comm_libs/mpi/lib"),
    ]
    env["PATH"] = ":".join([*path_parts, env.get("PATH", "")])
    env["LD_LIBRARY_PATH"] = ":".join([*ld_parts, env.get("LD_LIBRARY_PATH", "")])
    env["NVCOMPILERS"] = str(DEFAULT_NVHPC)
    env["NVHPC_CUDA_HOME"] = str(nvhpc_base / f"cuda/{NVHPC_CUDA_VERSION}")
    env["NVCOMPILER_COMM_LIBS_HOME"] = str(nvhpc_base / f"comm_libs/{NVHPC_CUDA_VERSION}")
    env["OMP_NUM_THREADS"] = str(int(omp_threads))
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    return env


def format_cell(cell) -> str:
    rows = []
    for row in cell:
        rows.append("  " + " ".join(f"{float(x):.12f}" for x in row))
    return "\n".join(rows)


def write_qe_input(job, setting: QESetting, job_dir: Path, pseudo_dir: Path) -> Path:
    from ase.io import read, write

    atoms = read(job.source_atoms)
    material = job.material
    if material not in MATERIALS:
        raise ValueError(f"unsupported QE material: {material!r}")
    pseudo = pseudo_dir / MATERIALS[material]["pseudo"]
    if not pseudo.exists():
        raise FileNotFoundError(f"missing QE pseudopotential: {pseudo}")

    job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job.source_atoms, job_dir / "source_atoms.xyz")
    write(job_dir / "qe_atoms.xyz", atoms)

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    atom_lines = "\n".join(
        f"  {sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}"
        for sym, pos in zip(symbols, positions)
    )
    kx, ky, kz = setting.kgrid
    input_path = job_dir / f"{job.name}.in"
    input_path.write_text(
        f"""&CONTROL
  calculation = 'scf',
  prefix = '{job.name}',
  outdir = './qe_out',
  pseudo_dir = '{pseudo_dir.resolve()}',
  tstress = .true.,
  tprnfor = .true.,
/
&SYSTEM
  ibrav = 0,
  nat = {len(atoms)},
  ntyp = 1,
  input_dft = 'PBE',
  ecutwfc = {setting.ecutwfc:.6f},
  ecutrho = {setting.ecutrho:.6f},
  occupations = 'smearing',
  smearing = '{setting.smearing}',
  degauss = {setting.degauss:.6f},
  nosym = .true.,
  noinv = .true.,
/
&ELECTRONS
  conv_thr = {setting.conv_thr:.3e},
  electron_maxstep = {setting.electron_maxstep},
  diagonalization = '{setting.diagonalization}',
  mixing_mode = '{setting.mixing_mode}',
  mixing_beta = {setting.mixing_beta:.6f},
  mixing_ndim = {setting.mixing_ndim},
  startingpot = '{setting.startingpot}',
  startingwfc = '{setting.startingwfc}',
/
ATOMIC_SPECIES
  {material} {MATERIALS[material]["mass"]:.6f} {MATERIALS[material]["pseudo"]}
CELL_PARAMETERS angstrom
{format_cell(atoms.get_cell().array)}
ATOMIC_POSITIONS angstrom
{atom_lines}
K_POINTS automatic
  {kx} {ky} {kz} 0 0 0
"""
    )
    meta = {
        **asdict(job),
        "qe_setting": asdict(setting),
        "pseudo": str(pseudo),
        "qe_input": str(input_path),
    }
    (job_dir / "job.json").write_text(json.dumps(meta, indent=2) + "\n")
    return input_path


def parse_qe_output(path: Path) -> dict:
    text = path.read_text(errors="replace") if path.exists() else ""
    energy_matches = re.findall(r"!\s+total energy\s+=\s+([-+0-9.Ee]+)\s+Ry", text)
    force_matches = re.findall(r"Total force\s+=\s+([-+0-9.Ee]+)\s+Total SCF correction", text)
    wall_matches = re.findall(r"PWSCF\s+:\s+([0-9.]+)s CPU\s+([0-9.]+)s WALL", text)
    return {
        "converged": "convergence has been achieved" in text,
        "job_done": "JOB DONE." in text,
        "gpu_active": "GPU acceleration is ACTIVE" in text,
        "total_energy_ry": float(energy_matches[-1]) if energy_matches else None,
        "total_force_ry_bohr": float(force_matches[-1]) if force_matches else None,
        "pwscf_wall_s": float(wall_matches[-1][1]) if wall_matches else None,
        "gpu_markers": [
            line.strip()
            for line in text.splitlines()
            if "gpu" in line.lower() or "cuda" in line.lower() or "cusolver" in line.lower()
        ][:50],
        "error_markers": [
            line.strip()
            for line in text.splitlines()
            if "%%%%%%" in line or "Error in routine" in line or "stopping ..." in line
        ][:30],
    }


def summarize_gpu_monitor(path: Path) -> dict:
    utils: list[float] = []
    powers: list[float] = []
    if not path.exists():
        return {"samples": 0}
    for line in path.read_text(errors="replace").splitlines():
        if "," not in line:
            continue
        util, power = [part.strip() for part in line.split(",", 1)]
        try:
            utils.append(float(util))
            powers.append(float(power))
        except ValueError:
            continue
    return {
        "samples": len(utils),
        "max_util_pct": max(utils) if utils else None,
        "avg_util_pct": round(sum(utils) / len(utils), 3) if utils else None,
        "max_power_w": max(powers) if powers else None,
    }


def start_gpu_monitor(path: Path) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w")
    cmd = (
        "while true; do date +%s.%N; "
        "nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits; "
        "sleep 1; done"
    )
    return subprocess.Popen(["bash", "-lc", cmd], stdout=fh, stderr=subprocess.STDOUT)


def stop_gpu_monitor(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def run_job(command: list[str], cwd: Path, env: dict, output_path: Path, timeout: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as out:
        out.write(f"# started {now()}\n")
        out.write(f"# command {' '.join(shlex.quote(part) for part in command)}\n")
        out.flush()
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    return int(proc.returncode)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", action="append", type=Path, default=None)
    p.add_argument("--label", default=os.environ.get("ELECTRODEFECT_QE_LABEL", socket.gethostname()))
    p.add_argument("--material", choices=sorted(MATERIALS), default=None)
    p.add_argument("--max-jobs", type=int, default=0, help="0 means all queued labels")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--pseudo-dir", type=Path, default=REPO / "data/qe_pseudos")
    p.add_argument("--pw-x", default=os.environ.get("QE_PW_X"))
    p.add_argument("--mpi", type=int, default=1)
    p.add_argument("--omp-threads", type=int, default=1)
    p.add_argument("--ecutwfc", type=float, default=DEFAULT_SETTING.ecutwfc)
    p.add_argument("--ecutrho", type=float, default=DEFAULT_SETTING.ecutrho)
    p.add_argument("--kgrid", nargs=3, type=int, default=list(DEFAULT_SETTING.kgrid))
    p.add_argument("--conv-thr", type=float, default=DEFAULT_SETTING.conv_thr)
    p.add_argument("--electron-maxstep", type=int, default=DEFAULT_SETTING.electron_maxstep)
    p.add_argument("--diagonalization", default=DEFAULT_SETTING.diagonalization)
    p.add_argument("--mixing-mode", default=DEFAULT_SETTING.mixing_mode)
    p.add_argument("--mixing-beta", type=float, default=DEFAULT_SETTING.mixing_beta)
    p.add_argument("--mixing-ndim", type=int, default=DEFAULT_SETTING.mixing_ndim)
    p.add_argument("--smearing", default=DEFAULT_SETTING.smearing)
    p.add_argument("--degauss", type=float, default=DEFAULT_SETTING.degauss)
    p.add_argument("--startingpot", default=DEFAULT_SETTING.startingpot)
    p.add_argument("--startingwfc", default=DEFAULT_SETTING.startingwfc)
    p.add_argument("--max-seconds", type=int, default=DEFAULT_SETTING.max_seconds)
    p.add_argument("--acc-notify", action="store_true", help="Emit NVHPC CUDA kernel launch lines into QE output.")
    return p


def main() -> int:
    args = parser().parse_args()
    roots = args.root or DEFAULT_ROOTS
    jobs = load_jobs(roots, max_seconds=int(args.max_seconds))
    if args.material:
        jobs = [job for job in jobs if job.material == args.material]
    if args.max_jobs:
        jobs = jobs[: int(args.max_jobs)]
    if not jobs:
        raise SystemExit("no Phase 2 label jobs selected")

    setting = QESetting(
        name=DEFAULT_SETTING.name,
        ecutwfc=float(args.ecutwfc),
        ecutrho=float(args.ecutrho),
        kgrid=tuple(int(x) for x in args.kgrid),
        conv_thr=float(args.conv_thr),
        electron_maxstep=int(args.electron_maxstep),
        diagonalization=str(args.diagonalization),
        mixing_mode=str(args.mixing_mode),
        mixing_beta=float(args.mixing_beta),
        mixing_ndim=int(args.mixing_ndim),
        smearing=str(args.smearing),
        degauss=float(args.degauss),
        startingpot=str(args.startingpot),
        startingwfc=str(args.startingwfc),
        max_seconds=int(args.max_seconds),
    )
    run_id = datetime.now().strftime(f"phase2_qe_gpu_labels_%Y%m%d_%H%M%S_{args.label}")
    run_root = RUNS / run_id
    work_root = run_root / "work"
    log_root = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    pw_x = find_pw_x(args.pw_x)
    env = qe_environment(args.omp_threads)
    if args.acc_notify:
        env["NVCOMPILER_ACC_NOTIFY"] = "1"
    else:
        env.setdefault("NVCOMPILER_ACC_NOTIFY", "0")

    status = {
        "state": "preparing" if args.prepare_only else "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "label": args.label,
        "pw_x": pw_x,
        "qe_gpu": True,
        "roots": [str(root) for root in roots],
        "setting": asdict(setting),
        "mpi": int(args.mpi),
        "omp_threads": int(args.omp_threads),
        "jobs": [
            {
                **asdict(job),
                "state": "pending",
                "archive_dir": str(work_root / job.name),
            }
            for job in jobs
        ],
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"QE GPU label queue start n_jobs={len(jobs)} prepare_only={args.prepare_only}")

    for idx, job in enumerate(jobs):
        job_dir = work_root / job.name
        input_path = write_qe_input(job, setting, job_dir, args.pseudo_dir)
        status["jobs"][idx].update({"state": "prepared", "input": str(input_path)})
        status["updated_at"] = now()
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"job prepared {job.name}")
        if args.prepare_only:
            continue

        output_path = job_dir / f"{job.name}.out"
        monitor_path = job_dir / "gpu_monitor.log"
        command = [pw_x, "-in", str(input_path)]
        if int(args.mpi) > 1:
            command = ["mpirun", "-np", str(int(args.mpi)), *command]

        status["jobs"][idx].update({"state": "running", "started_at": now()})
        status["updated_at"] = now()
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"job start {job.name}")

        start = time.time()
        monitor = start_gpu_monitor(monitor_path)
        rc = 1
        timed_out = False
        try:
            rc = run_job(command, job_dir, env, output_path, int(args.max_seconds))
        except subprocess.TimeoutExpired:
            rc = 124
            timed_out = True
        finally:
            stop_gpu_monitor(monitor)
        elapsed = time.time() - start
        parsed = parse_qe_output(output_path)
        gpu_monitor = summarize_gpu_monitor(monitor_path)
        completed = rc == 0 and parsed["job_done"] and parsed["converged"] and parsed["gpu_active"]
        status["jobs"][idx].update(
            {
                "state": "completed" if completed else "failed",
                "returncode": rc,
                "timed_out": timed_out,
                "elapsed_s": round(elapsed, 3),
                "finished_at": now(),
                "output": str(output_path),
                "gpu_monitor": gpu_monitor,
                **parsed,
            }
        )
        status["updated_at"] = now()
        write_json(job_dir / "summary.json", status["jobs"][idx])
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"job finish {job.name} rc={rc} elapsed={elapsed:.1f}s gpu_active={parsed['gpu_active']}")
        if not completed and args.stop_on_failure:
            status["state"] = "failed"
            status["updated_at"] = now()
            write_json(run_root / "status.json", status)
            return rc if rc in range(1, 126) else 1

    if args.prepare_only:
        status["state"] = "prepared"
    else:
        failed = [job for job in status["jobs"] if job["state"] != "completed"]
        status["state"] = "completed" if not failed else "completed_with_failures"
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"QE GPU label queue {status['state']}")
    print(run_root)
    return 0 if status["state"] in {"prepared", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
