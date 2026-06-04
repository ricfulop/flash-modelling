#!/usr/bin/env python3
"""Smoke-gated overnight Tier B runner.

The runner keeps progress visible through:
  runs/<run_id>/status.json
  runs/<run_id>/heartbeat.log
  runs/<run_id>/logs/<job>.out

It uses a conservative ladder: quick W/BigDFT smoke, small W slab jobs,
then larger follow-up jobs so the machines keep working overnight.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUN_ID = datetime.now().strftime("overnight_%Y%m%d_%H%M%S")
RUN_ROOT = REPO / "runs" / RUN_ID
LOG_DIR = RUN_ROOT / "logs"
REMOTE_ROOT = f"/tmp/{RUN_ID}"
LOCAL_BIGDFT = "/home/ricfulop/Desktop/Cursor/use_bigdft.sh"
REMOTE_BIGDFT = "/home/nvidia/Cursor/use_bigdft.sh"
PSP_W = Path("/home/ricfulop/Desktop/Cursor/bigdft-suite/PyBigDFT/BigDFT/Database/psppar/Krack/PBE/psppar.W.yaml")
SEED = 20260603


@dataclass
class Job:
    name: str
    kind: str
    atoms: object | None
    hgrid: float
    rmult: tuple[float, float]
    mpi: int
    max_seconds: int
    notes: str
    surface: bool = True
    linear: bool = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(cmd: str, *, cwd: Path | None = None, timeout: int | None = None, log: Path | None = None) -> subprocess.CompletedProcess:
    if log:
        with log.open("a") as fh:
            fh.write(f"\n$ {cmd}\n")
            fh.flush()
            return subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                cwd=cwd,
                text=True,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
    return subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def write_status(payload: dict) -> None:
    payload = {**payload, "updated_at": now(), "run_id": RUN_ID}
    tmp = RUN_ROOT / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(RUN_ROOT / "status.json")


def heartbeat(message: str) -> None:
    with (RUN_ROOT / "heartbeat.log").open("a") as fh:
        fh.write(f"{now()} {message}\n")


def write_markdown_summary(status: dict) -> None:
    lines = [f"# Overnight Tier B Run {RUN_ID}", "", f"Updated: {now()}", ""]
    lines.append("## Jobs")
    for job in status["jobs"]:
        lines.append(f"- `{job['name']}`: {job['state']} ({job.get('elapsed_s', 0):.0f}s) - {job.get('notes', '')}")
    lines.append("")
    lines.append("## How To Monitor")
    lines.append(f"- `tail -f {RUN_ROOT / 'heartbeat.log'}`")
    lines.append(f"- `watch -n 60 'python -m json.tool {RUN_ROOT / 'status.json'} | sed -n \"1,120p\"'`")
    lines.append(f"- `tail -f {LOG_DIR}/<job>.out`")
    (RUN_ROOT / "summary.md").write_text("\n".join(lines) + "\n")


def surface_posinp(atoms, path: Path, surface: bool = True) -> None:
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
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
    if job.atoms is not None:
        surface_posinp(job.atoms, job_dir / "posinp.xyz", surface=job.surface)
    shutil.copy2(PSP_W, job_dir / "psppar.W.yaml")
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
        "mix": {
            "tel": 0.72 / 27.211386245988,
            "occopt": 1,
        },
    }
    if job.linear:
        input_yaml["import"] = "linear"
        input_yaml["lin_general"] = {"output_mat": 1}
    import yaml

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

    def defected(base, frac: float = 0.08, surface_band: float = 7.0):
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
        top = z > z.max() - 1.8
        left_half = x < np.median(x)
        mask = ~(top & left_half)
        out = atoms[mask]
        out.set_cell(base.get_cell())
        return out

    small = slab(3, 3, 4)
    medium = slab(4, 4, 6)
    large = slab(5, 5, 8)
    return {
        "w_atom": Atoms("W", positions=[[0.0, 0.0, 0.0]], cell=[18, 18, 18], pbc=False),
        "clean_small": small,
        "defect_small": defected(small, frac=0.10),
        "stepped_small": stepped(small),
        "recombined_small": small.copy(),
        "separated_small": defected(small, frac=0.12),
        "clean_medium": medium,
        "defect_medium": defected(medium, frac=0.10),
        "clean_large": large,
        "defect_large": defected(large, frac=0.10),
    }


def jobs() -> list[Job]:
    atoms = build_atoms()
    return [
        Job("00_w_atom_smoke", "bigdft", atoms["w_atom"], 0.55, (4.0, 6.0), 2, 900, "W pseudopotential and two-node BigDFT smoke", False),
        Job("01_clean_w001_small_phi", "bigdft", atoms["clean_small"], 0.50, (4.0, 7.0), 2, 3600, "small clean W(001) slab; work-function precursor"),
        Job("02_defect_w001_small_phi", "bigdft", atoms["defect_small"], 0.50, (4.0, 7.0), 2, 5400, "small near-surface FP W slab; delta-phi precursor"),
        Job("03_stepped_w001_small_phi", "bigdft", atoms["stepped_small"], 0.50, (4.0, 7.0), 2, 5400, "small stepped W slab; edge lowering precursor"),
        Job("04_recombined_small_EFP", "bigdft", atoms["recombined_small"], 0.48, (4.5, 8.0), 2, 5400, "small recombined reference for E_FP"),
        Job("05_separated_small_EFP", "bigdft", atoms["separated_small"], 0.48, (4.5, 8.0), 2, 7200, "small separated FP cell for E_FP"),
        Job("06_clean_w001_medium_phi", "bigdft", atoms["clean_medium"], 0.45, (5.0, 9.0), 2, 10800, "medium clean W slab"),
        Job("07_defect_w001_medium_phi", "bigdft", atoms["defect_medium"], 0.45, (5.0, 9.0), 2, 14400, "medium defect W slab"),
        Job("08_clean_w001_large_stretch", "bigdft", atoms["clean_large"], 0.43, (5.0, 10.0), 2, 21600, "large clean W stretch job"),
        Job("09_defect_w001_large_stretch", "bigdft", atoms["defect_large"], 0.43, (5.0, 10.0), 2, 28800, "large defect W stretch job"),
        Job("10_local_gpu_transport_repeat", "torch", None, 0.0, (0, 0), 1, 7200, "local GPU Tier A/KPM repeat workload"),
        Job("11_peer_gpu_transport_repeat", "torch_remote", None, 0.0, (0, 0), 1, 7200, "peer GPU Tier A/KPM repeat workload"),
    ]


def write_launch_wrapper(job_dir: Path) -> None:
    script = job_dir / "run_bigdft.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "h=$(hostname)\n"
        "if [[ \"$h\" == spark-0808* ]]; then\n"
        f"  source {REMOTE_BIGDFT} >/tmp/use-bigdft-{RUN_ID}.out\n"
        "else\n"
        f"  source {LOCAL_BIGDFT} >/tmp/use-bigdft-{RUN_ID}.out\n"
        "fi\n"
        "cd \"$(dirname \"$0\")\"\n"
        "exec bigdft -l yes\n"
    )
    os.chmod(script, 0o755)


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


def mpirun_command(exec_dir: str, nproc: int) -> str:
    return (
        "OMPI_MCA_plm_rsh_agent='ssh' "
        "OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 "
        "OMPI_MCA_pml=ob1 "
        "OMPI_MCA_btl=self,tcp "
        "OMPI_MCA_btl_tcp_if_include=enp1s0f0np0 "
        "OMPI_MCA_oob_tcp_if_include=enp1s0f0np0 "
        "OMPI_MCA_mtl='^ofi' "
        "UCX_TLS=tcp,self "
        "FI_PROVIDER=tcp "
        f"/usr/bin/mpirun --host spark-5da5,dgx-peer -np {nproc} {shlex.quote(exec_dir + '/run_bigdft.sh')}"
    )


def preflight(status: dict) -> bool:
    heartbeat("preflight start")
    peer_cuda_cmd = "cd /home/nvidia/Cursor/flash-modelling && source env/activate_dgx.sh >/tmp/activate_peer.out && python3 scripts/gpu_smoke.py"
    peer_cuda_remote = f"bash -lc {shlex.quote(peer_cuda_cmd)}"
    checks = [
        ("local_cuda", "cd /home/ricfulop/Desktop/Cursor/flash-modelling && source env/activate_dgx.sh >/tmp/activate_dgx.out && python3 scripts/gpu_smoke.py"),
        ("peer_cuda", f"ssh dgx-peer {shlex.quote(peer_cuda_remote)}"),
        ("mpi_hostname", "OMPI_MCA_plm_rsh_agent='ssh' OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 /usr/bin/mpirun --host spark-5da5,dgx-peer -np 2 hostname"),
        ("bigdft_envs", "OMPI_MCA_plm_rsh_agent='ssh' OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 /usr/bin/mpirun --host spark-5da5,dgx-peer -np 2 bash -lc 'h=$(hostname); if [ \"$h\" = \"spark-0808\" ]; then source /home/nvidia/Cursor/use_bigdft.sh >/tmp/use-bigdft-preflight.out; else source /home/ricfulop/Desktop/Cursor/use_bigdft.sh >/tmp/use-bigdft-preflight.out; fi; which bigdft; bigdft --help >/tmp/bigdft-help-preflight.out'"),
    ]
    for name, cmd in checks:
        log = LOG_DIR / f"preflight_{name}.out"
        heartbeat(f"preflight {name}")
        proc = run(cmd, timeout=240, log=log)
        status["preflight"][name] = {"returncode": proc.returncode, "log": str(log)}
        write_status(status)
        write_markdown_summary(status)
        if proc.returncode != 0:
            heartbeat(f"preflight failed {name}")
            return False
    heartbeat("preflight ok")
    return True


def run_torch_job(job: Job, status_job: dict, remote: bool = False) -> int:
    env = (
        "ELECTRODEFECT_KPM_MOMENTS=${ELECTRODEFECT_KPM_MOMENTS:-8192} "
        "ELECTRODEFECT_KPM_RANDOM_VECTORS=${ELECTRODEFECT_KPM_RANDOM_VECTORS:-64} "
        "ELECTRODEFECT_KPM_ENERGIES=${ELECTRODEFECT_KPM_ENERGIES:-4000} "
        "ELECTRODEFECT_MAX_EXACT=${ELECTRODEFECT_MAX_EXACT:-900} "
    )
    if remote:
        cmd = (
            "cd /home/nvidia/Cursor/flash-modelling && "
            "source env/activate_dgx.sh >/tmp/activate_peer.out && "
            f"{env} python3 scripts/04_transport.py"
        )
    else:
        cmd = (
            "cd /home/ricfulop/Desktop/Cursor/flash-modelling && "
            "source env/activate_dgx.sh >/tmp/activate_dgx.out && "
            f"{env} python3 scripts/04_transport.py"
        )
    if remote:
        cmd = f"ssh dgx-peer {shlex.quote(f'bash -lc {shlex.quote(cmd)}')}"
    return run(cmd, timeout=job.max_seconds, log=LOG_DIR / f"{job.name}.out").returncode


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "starting",
        "run_root": str(RUN_ROOT),
        "remote_root": REMOTE_ROOT,
        "seed": SEED,
        "preflight": {},
        "jobs": [],
    }
    for job in jobs():
        status["jobs"].append({"name": job.name, "state": "pending", "notes": job.notes, "elapsed_s": 0})
    write_status(status)
    write_markdown_summary(status)

    if not preflight(status):
        status["state"] = "failed_preflight"
        write_status(status)
        write_markdown_summary(status)
        return 2

    status["state"] = "running"
    write_status(status)

    for idx, job in enumerate(jobs()):
        status_job = status["jobs"][idx]
        status_job["state"] = "running"
        status_job["started_at"] = now()
        write_status(status)
        write_markdown_summary(status)
        heartbeat(f"job start {job.name}")
        start = time.time()
        rc = 1
        try:
            if job.kind == "bigdft":
                archive_dir = RUN_ROOT / "work" / job.name
                exec_dir = Path("/tmp") / RUN_ID / job.name
                remote_dir = f"{REMOTE_ROOT}/{job.name}"
                write_bigdft_input(job, archive_dir)
                if exec_dir.exists():
                    shutil.rmtree(exec_dir)
                exec_dir.mkdir(parents=True, exist_ok=True)
                for filename in ("posinp.xyz", "input.yaml", "psppar.W.yaml", "job.json"):
                    shutil.copy2(archive_dir / filename, exec_dir / filename)
                write_launch_wrapper(exec_dir)
                run(f"ssh dgx-peer 'rm -rf {remote_dir} && mkdir -p {remote_dir}'", timeout=120, log=LOG_DIR / f"{job.name}.sync.out")
                run(
                    f"scp {exec_dir}/posinp.xyz {exec_dir}/input.yaml {exec_dir}/psppar.W.yaml {exec_dir}/job.json {exec_dir}/run_bigdft.sh dgx-peer:{remote_dir}/",
                    timeout=120,
                    log=LOG_DIR / f"{job.name}.sync.out",
                )
                rc = run(mpirun_command(str(exec_dir), job.mpi), timeout=job.max_seconds, log=LOG_DIR / f"{job.name}.out").returncode
                copy_tree_contents(exec_dir, archive_dir)
                run(
                    f"ssh dgx-peer 'cd {remote_dir} && tar -czf /tmp/{RUN_ID}-{job.name}-peer.tgz .' && "
                    f"scp dgx-peer:/tmp/{RUN_ID}-{job.name}-peer.tgz {archive_dir}/peer.tgz",
                    timeout=180,
                    log=LOG_DIR / f"{job.name}.sync.out",
                )
            elif job.kind == "torch":
                rc = run_torch_job(job, status_job, remote=False)
            elif job.kind == "torch_remote":
                rc = run_torch_job(job, status_job, remote=True)
            else:
                rc = 99
        except subprocess.TimeoutExpired:
            rc = 124
        elapsed = time.time() - start
        status_job["elapsed_s"] = elapsed
        status_job["returncode"] = rc
        status_job["finished_at"] = now()
        status_job["state"] = "completed" if rc == 0 else "failed"
        heartbeat(f"job finish {job.name} rc={rc} elapsed={elapsed:.0f}s")
        write_status(status)
        write_markdown_summary(status)
        if job.name == "00_w_atom_smoke" and rc != 0:
            status["state"] = "failed_smoke"
            write_status(status)
            write_markdown_summary(status)
            return 3

    status["state"] = "completed"
    write_status(status)
    write_markdown_summary(status)
    heartbeat("run completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
