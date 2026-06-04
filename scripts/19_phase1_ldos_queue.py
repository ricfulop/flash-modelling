#!/usr/bin/env python3
"""Run CPU BigDFT LDOS/orbital-export jobs for Phase 1 Tier C emission.

This is intentionally gated: it writes `clean_ldos.npz` and
`recombined_ldos.npz` only when BigDFT emits orbital/wavefunction cubes that
can be reduced to the `emission.py` data contract. Otherwise it records a
blocked status and leaves Stage 03 emission as missing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
CONFIG = REPO / "config" / "w_percolation.yaml"
QUEUE_PATH = Path(__file__).resolve().parent / "08_single_node_bigdft_queue.py"
REMAINING_PATH = Path(__file__).resolve().parent / "18_phase1_remaining_queue.py"

SPEC_Q = importlib.util.spec_from_file_location("single_node_queue", QUEUE_PATH)
queue = importlib.util.module_from_spec(SPEC_Q)
assert SPEC_Q.loader is not None
sys.modules[SPEC_Q.name] = queue
SPEC_Q.loader.exec_module(queue)

SPEC_R = importlib.util.spec_from_file_location("phase1_remaining_queue", REMAINING_PATH)
remaining = importlib.util.module_from_spec(SPEC_R)
assert SPEC_R.loader is not None
sys.modules[SPEC_R.name] = remaining
SPEC_R.loader.exec_module(remaining)

sys.path.insert(0, str(REPO / "src"))
from electrodefect import dft_bigdft


@dataclass
class LDOSJob:
    name: str
    atoms: object
    hgrid: float
    rmult: tuple[float, float]
    max_seconds: int
    notes: str
    nvirt: int
    nplot: int
    Te_eV: float
    surface: bool = True


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


def wait_for_run(path: Path, poll_s: int, timeout_s: int) -> dict:
    start = time.time()
    while True:
        status_path = path / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("state") in {"completed", "failed", "failed_smoke", "blocked"}:
                return status
        if time.time() - start > timeout_s:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(poll_s)


def config_defaults() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text())
    return {
        "phi_eff": float(cfg["emission"]["phi_eff"]),
        "Te_eV": float(cfg["drive_proxy"]["Te_eV"]),
    }


def ldos_jobs(*, nvirt: int, nplot: int, Te_eV: float):
    cfgs, meta = remaining.phase1_configs()
    return [
        (
            LDOSJob(
                "06_clean_small_LDOS_orbitals",
                cfgs["clean"].copy(),
                0.55,
                (4.0, 7.0),
                7200,
                "clean W(001) small slab with orbital output for LDOS baseline",
                nvirt=nvirt,
                nplot=nplot,
                Te_eV=Te_eV,
            ),
            {"phase1_structures": meta, "ldos_role": "clean", "geometry_role": "clean"},
        ),
        (
            LDOSJob(
                "07_separated_fp_small_LDOS_orbitals",
                cfgs["separated"].copy(),
                0.55,
                (4.0, 7.0),
                7200,
                "separated Frenkel-pair W(001) slab with orbital output for recombination-source LDOS",
                nvirt=nvirt,
                nplot=nplot,
                Te_eV=Te_eV,
            ),
            {
                "phase1_structures": meta,
                "ldos_role": "recombination_source",
                "geometry_role": "separated_frenkel_pair",
                "stage03_argument": "recombined_ldos",
            },
        ),
    ]


def write_bigdft_input(job: LDOSJob, job_dir: Path, extra_meta: dict | None = None) -> None:
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
            "output_denspot": 22,
            "nplot": job.nplot,
            "nvirt": job.nvirt,
            "norbv": job.nvirt,
        },
        "output": {
            "orbitals": "ETSF",
            "outputpsiid": "orbitals",
        },
        "mix": {"tel": job.Te_eV / dft_bigdft.EV_PER_HA, "occopt": 1},
    }
    (job_dir / "input.yaml").write_text(yaml.safe_dump(input_yaml, sort_keys=False))
    meta = {
        "name": job.name,
        "hgrid": job.hgrid,
        "rmult": list(job.rmult),
        "max_seconds": job.max_seconds,
        "notes": job.notes,
        "nvirt": job.nvirt,
        "nplot": job.nplot,
        "Te_eV": job.Te_eV,
        "surface": job.surface,
    }
    if extra_meta:
        meta.update(extra_meta)
    meta["n_atoms"] = len(job.atoms)
    (job_dir / "job.json").write_text(json.dumps(meta, indent=2) + "\n")


def find_orbital_cubes(job_dir: Path) -> list[Path]:
    cubes = []
    excluded = {"electronic_density.cube", "hartree_potential.cube", "local_potential.cube", "external_potential.cube"}
    for path in job_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".cube", ".cub"}:
            if path.name.lower() not in excluded and "potential" not in path.name.lower() and "density" not in path.name.lower():
                cubes.append(path)
    return sorted(cubes)


def mapped_orbital_cubes(job_dir: Path) -> list[tuple[Path, float]]:
    energy_map = job_dir / "cubes" / "state_energies.json"
    if not energy_map.exists():
        return []
    payload = json.loads(energy_map.read_text())
    out = []
    for name, energy in payload.items():
        cube = energy_map.parent / name
        if not cube.exists():
            raise FileNotFoundError(f"mapped cube does not exist: {cube}")
        out.append((cube, float(energy)))
    return sorted(out, key=lambda item: item[1])


def find_wavefunction_files(job_dir: Path) -> list[Path]:
    candidates = []
    for path in job_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(tok in name for tok in ("wf", "wavefunction", "orbital")) and path.suffix.lower() not in {".yaml", ".json", ".txt"}:
            candidates.append(path)
        elif path.suffix.lower() in {".etsf", ".bin"}:
            candidates.append(path)
    return sorted(candidates)


def parse_orbital_eigenvalues(log_path: Path) -> list[dict]:
    """Parse the final BigDFT orbital table as band-indexed eigenvalues in eV."""
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    blocks = re.findall(r"Orbitals:\s*\[(.*?)\]\s*#\s*\d+", text, flags=re.DOTALL)
    if not blocks:
        raise ValueError(f"no BigDFT orbital eigenvalue table found in {log_path}")
    table = blocks[-1]
    rows = []
    for match in re.finditer(
        r"\{e:\s*([+\-]?\d+\.\d+(?:[Ee][+\-]?\d+)?),\s*f:\s*([+\-]?\d+\.\d+(?:[Ee][+\-]?\d+)?)\}\s*,?\s*#\s*(\d+)",
        table,
    ):
        rows.append(
            {
                "energy_eV": float(match.group(1)) * dft_bigdft.EV_PER_HA,
                "occupation": float(match.group(2)),
                "band": int(match.group(3)),
            }
        )
    if not rows:
        raise ValueError(f"failed to parse BigDFT orbital eigenvalues from {log_path}")
    return rows


def selected_emission_bands(job_dir: Path, *, phi_eff: float, min_above: int, max_bands: int) -> list[dict]:
    scalars = dft_bigdft.parse_log_scalars(job_dir)
    fermi_Ha = scalars.get("fermi_Ha")
    if fermi_Ha is None:
        raise ValueError(f"missing Fermi Energy in {job_dir}")
    E_F = float(fermi_Ha) * dft_bigdft.EV_PER_HA
    rows = parse_orbital_eigenvalues(job_dir / "log.yaml")
    threshold = E_F + phi_eff
    max_energy = max(row["energy_eV"] for row in rows)
    if max_energy < threshold:
        raise ValueError(
            f"computed eigenvalue window stops at {max_energy:.3f} eV, below emission threshold "
            f"{threshold:.3f} eV; rerun with more virtual orbitals"
        )
    above = [row for row in rows if row["energy_eV"] >= threshold]
    if len(above) < min_above:
        raise ValueError(
            f"only {len(above)} eigenstates above emission threshold {threshold:.3f} eV; "
            f"need at least {min_above} for the configured LDOS gate"
        )
    return above[:max_bands]


def choose_wavefunction_file(job_dir: Path) -> Path | None:
    wf_files = find_wavefunction_files(job_dir)
    if not wf_files:
        return None
    etsf = [p for p in wf_files if p.suffix.lower() == ".etsf"]
    if etsf:
        return max(etsf, key=lambda p: p.stat().st_size)
    return max(wf_files, key=lambda p: p.stat().st_size)


def export_wavefunctions(
    job_dir: Path,
    activate: str,
    log: Path,
    *,
    phi_eff: float,
    min_above: int,
    max_bands: int,
) -> list[tuple[Path, float]]:
    mapped = mapped_orbital_cubes(job_dir)
    if mapped:
        return mapped
    bands = selected_emission_bands(job_dir, phi_eff=phi_eff, min_above=min_above, max_bands=max_bands)
    wf_file = choose_wavefunction_file(job_dir)
    if wf_file is None:
        raise FileNotFoundError("BigDFT did not write an exportable wavefunction file")
    out_dir = job_dir / "cubes"
    out_dir.mkdir(exist_ok=True)
    exported: list[tuple[Path, float]] = []
    energy_map = {}
    for row in bands:
        band = int(row["band"])
        before = {p.resolve() for p in out_dir.glob("*.cube")}
        cmd = (
            f"source {shlex.quote(activate)} >/tmp/use-bigdft-ldos-export.out && "
            f"cd {shlex.quote(str(out_dir))} && "
            f"bigdft-tool -a export-wf {shlex.quote(str(wf_file))} --i-band={band}"
        )
        rc = run(cmd, timeout=600, log=log).returncode
        if rc != 0:
            raise RuntimeError(f"bigdft-tool export-wf failed for band {band}")
        after = [p for p in out_dir.glob("*.cube") if p.resolve() not in before]
        if not after:
            raise FileNotFoundError(f"bigdft-tool did not create a cube for band {band}")
        target = out_dir / f"band_{band:05d}.cube"
        after[0].replace(target)
        exported.append((target, float(row["energy_eV"])))
        energy_map[target.name] = float(row["energy_eV"])
    write_json(out_dir / "state_energies.json", energy_map)
    return exported


def write_ldos_npz(job_dir: Path, cubes: list[tuple[Path, float]], output_npz: Path) -> dict:
    scalars = dft_bigdft.parse_log_scalars(job_dir)
    fermi_Ha = scalars.get("fermi_Ha")
    if fermi_Ha is None:
        raise ValueError(f"missing Fermi Energy in {job_dir}")
    densities = []
    z_grid = None
    energies = []
    for cube, energy in cubes:
        origin, axes, values = dft_bigdft.read_cube(cube)
        z, density_z = dft_bigdft.cube_density_z(origin, axes, values, square_values=True)
        if z_grid is None:
            z_grid = z
        elif len(z_grid) != len(z) or not np.allclose(z_grid, z):
            raise ValueError(f"inconsistent z grid in {cube}")
        densities.append(density_z)
        energies.append(float(energy))
    z_surface = dft_bigdft.infer_surface_z(job_dir)
    E_F = float(fermi_Ha) * dft_bigdft.EV_PER_HA
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    cube_dir = job_dir / "cubes"
    cube_dir.mkdir(exist_ok=True)
    energy_map = {}
    for idx, (cube_path, energy) in enumerate(cubes, start=1):
        target = cube_dir / f"state_{idx:03d}.cube"
        if cube_path.resolve() != target.resolve():
            shutil.copy2(cube_path, target)
        energy_map[target.name] = float(energy)
    write_json(cube_dir / "state_energies.json", energy_map)
    np.savez_compressed(
        output_npz,
        energies=np.asarray(energies, dtype=float),
        density_z=np.asarray(densities, dtype=float),
        z_grid=np.asarray(z_grid, dtype=float),
        z_surface=float(z_surface),
        E_F=float(E_F),
    )
    return {
        "status": "accepted",
        "ldos_npz": str(output_npz),
        "n_orbital_cubes": len(cubes),
        "E_F_eV": E_F,
        "z_surface": float(z_surface),
    }


def postprocess_ldos(run_root: Path, activate: str, status: dict, *, phi_eff: float, min_above: int, max_bands: int) -> None:
    results = {}
    for job in status["jobs"]:
        role = "clean" if "clean" in job["name"] else "recombined"
        job_dir = run_root / "work" / job["name"]
        export_log = run_root / "logs" / f"{job['name']}.export.out"
        try:
            cubes = export_wavefunctions(
                job_dir,
                activate,
                export_log,
                phi_eff=phi_eff,
                min_above=min_above,
                max_bands=max_bands,
            )
            results[role] = write_ldos_npz(job_dir, cubes, run_root / "ldos" / f"{role}_ldos.npz")
        except Exception as exc:
            results[role] = {"status": "blocked", "reason": f"{type(exc).__name__}: {exc}", "job_dir": str(job_dir)}
    status["ldos"] = results
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"ldos postprocess {results}")


def resolve_bigdft_observables(explicit: str | None, after_run: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    if not after_run:
        return None
    status_path = Path(after_run) / "status.json"
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text())
    synth = status.get("refresh", {}).get("synthesis", {}).get("path")
    if not synth:
        return None
    return Path(synth) / "bigdft_observables.csv"


def refresh_tier_c(run_root: Path, status: dict, bigdft_observables: Path | None) -> None:
    clean = status.get("ldos", {}).get("clean", {})
    recombined = status.get("ldos", {}).get("recombined", {})
    if clean.get("status") != "accepted" or recombined.get("status") != "accepted":
        status["refresh"] = {"tier_bc": {"returncode": None, "reason": "LDOS artifacts not accepted"}}
        write_json(run_root / "status.json", status)
        return
    if bigdft_observables is None or not bigdft_observables.exists():
        status["refresh"] = {"tier_bc": {"returncode": None, "reason": "No explicit or upstream BigDFT observables CSV found"}}
        write_json(run_root / "status.json", status)
        return
    out = run_root / "tier_bc_with_ldos"
    cmd = (
        f"{shlex.quote(sys.executable)} scripts/03_bigdft_emission.py --bigdft-observables {shlex.quote(str(bigdft_observables))} "
        f"--clean-ldos {shlex.quote(clean['ldos_npz'])} --recombined-ldos {shlex.quote(recombined['ldos_npz'])} --output {shlex.quote(str(out))}"
    )
    rc = run(cmd, timeout=600, log=run_root / "logs" / "refresh_tier_bc_ldos.out").returncode
    status.setdefault("refresh", {})["tier_bc_with_ldos"] = {"returncode": rc, "path": str(out)}
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"refresh tier_bc_with_ldos rc={rc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-run", default=None, help="Optional Phase 1 EFP run root to wait for before launching LDOS.")
    parser.add_argument("--bigdft-observables", default=None, help="Explicit BigDFT observables CSV for Stage 03 refresh.")
    parser.add_argument("--phi-eff", type=float, default=None, help="Emission threshold in eV; defaults to config emission.phi_eff.")
    parser.add_argument("--Te-eV", type=float, default=None, help="Electronic temperature in eV; defaults to config drive_proxy.Te_eV.")
    parser.add_argument("--nvirt", type=int, default=64, help="Virtual orbitals requested from BigDFT for LDOS.")
    parser.add_argument("--nplot", type=int, default=64, help="Plotted orbitals requested from BigDFT for LDOS.")
    parser.add_argument("--min-above-barrier", type=int, default=4, help="Minimum parsed eigenstates required above E_F + phi_eff.")
    parser.add_argument("--max-export-bands", type=int, default=16, help="Maximum above-barrier bands exported to cubes.")
    parser.add_argument("--wait-timeout-s", type=int, default=86400)
    parser.add_argument("--poll-s", type=int, default=300)
    parser.add_argument("--label", default=os.environ.get("ELECTRODEFECT_LDOS_LABEL", socket.gethostname()))
    args = parser.parse_args()

    if args.after_run:
        wait_for_run(Path(args.after_run), args.poll_s, args.wait_timeout_s)

    defaults = config_defaults()
    phi_eff = defaults["phi_eff"] if args.phi_eff is None else args.phi_eff
    Te_eV = defaults["Te_eV"] if args.Te_eV is None else args.Te_eV
    run_id = datetime.now().strftime(f"phase1_ldos_%Y%m%d_%H%M%S_{args.label}")
    run_root = RUNS / run_id
    log_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    activate = queue.bigdft_activation()
    jobs = ldos_jobs(nvirt=args.nvirt, nplot=args.nplot, Te_eV=Te_eV)
    status = {
        "state": "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "activate": activate,
        "after_run": args.after_run,
        "ldos_gate": {
            "phi_eff_eV": phi_eff,
            "Te_eV": Te_eV,
            "nvirt": args.nvirt,
            "nplot": args.nplot,
            "min_above_barrier": args.min_above_barrier,
            "max_export_bands": args.max_export_bands,
        },
        "jobs": [
            {"name": job.name, "notes": job.notes, "state": "pending", "elapsed_s": 0, "n_atoms": len(job.atoms)}
            for job, _ in jobs
        ],
        "refresh": {},
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "Phase 1 LDOS queue start")

    for idx, (job, meta) in enumerate(jobs):
        archive_dir = run_root / "work" / job.name
        exec_dir = Path("/tmp") / run_id / job.name
        if exec_dir.exists():
            shutil.rmtree(exec_dir)
        exec_dir.mkdir(parents=True, exist_ok=True)
        write_bigdft_input(job, exec_dir, meta)
        write_bigdft_input(job, archive_dir, meta)
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

    postprocess_ldos(
        run_root,
        activate,
        status,
        phi_eff=phi_eff,
        min_above=args.min_above_barrier,
        max_bands=args.max_export_bands,
    )
    bigdft_observables = resolve_bigdft_observables(args.bigdft_observables, args.after_run)
    refresh_tier_c(run_root, status, bigdft_observables)
    status["state"] = "completed"
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "Phase 1 LDOS queue completed")
    print(f"Phase 1 LDOS queue complete -> {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
