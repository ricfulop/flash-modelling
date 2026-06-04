#!/usr/bin/env python3
"""Run the remaining CPU-safe Phase 1 jobs and refresh gated artifacts.

This queue fills the missing Tier C Delta-SCF pair using matched atom counts,
regenerates the Phase 1 synthesis and digest, then launches the Tier C LDOS
queue so emission contrast is accepted only when real LDOS artifacts exist.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
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
SEED = 20260604
QUEUE_PATH = Path(__file__).resolve().parent / "08_single_node_bigdft_queue.py"
sys.path.insert(0, str(REPO / "src"))
SPEC = importlib.util.spec_from_file_location("single_node_queue", QUEUE_PATH)
queue = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = queue
SPEC.loader.exec_module(queue)


@dataclass
class RemainingJob:
    name: str
    atoms: object
    hgrid: float
    rmult: tuple[float, float]
    max_seconds: int
    notes: str
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


def phase1_configs():
    """Build canonical clean/separated/recombined Phase 1 W(001) structures."""
    from electrodefect import build as build_mod
    from electrodefect import percolation as perc

    slab = build_mod.w_slab(nx_=3, ny=3, nz=4, vacuum=14.0, material="W")
    net = perc.dla_dendrite(nx_=3, ny=3, nz=4, a=build_mod.A_W, target_frac=0.10, seed=SEED)
    cfgs = build_mod.make_configs(slab, net, surface_band=7.0, material="W", a=build_mod.A_W)
    meta = {
        "builder": "electrodefect.build.make_configs",
        "network": "dla_dendrite",
        "target_frac": 0.10,
        "vacancy_indices": np.asarray(cfgs["vac_idx"], dtype=int).tolist(),
        "n_fp": int(len(cfgs["vac_idx"])),
        "seed": SEED,
        "atom_count_conserved": len(cfgs["separated"]) == len(cfgs["recombined"]) == len(cfgs["clean"]),
    }
    if not meta["atom_count_conserved"]:
        raise ValueError(f"canonical Phase 1 structures are not atom-count conserved: {meta}")
    return cfgs, meta


def phase1_pair():
    """Build atom-count-conserved separated/recombined W(001) EFP pair."""
    cfgs, meta = phase1_configs()
    return cfgs["separated"], cfgs["recombined"], meta


def write_bigdft_input(job: RemainingJob, job_dir: Path, extra_meta: dict | None = None) -> None:
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
            "itermax": 100,
            "nrepmax": 4,
            "gnrm_cv": 1.0e-4,
            "output_denspot": 22 if job.surface else 0,
        },
        "mix": {"tel": 0.72 / 27.211386245988, "occopt": 1},
    }
    (job_dir / "input.yaml").write_text(yaml.safe_dump(input_yaml, sort_keys=False))
    meta = {k: v for k, v in asdict(job).items() if k != "atoms"}
    if extra_meta:
        meta.update(extra_meta)
    meta["n_atoms"] = len(job.atoms)
    (job_dir / "job.json").write_text(json.dumps(meta, indent=2) + "\n")


def remaining_jobs():
    separated, recombined, meta = phase1_pair()
    common = {"extra_meta": {"phase1_pair": meta, "atom_count_matched": True}}
    return [
        (
            RemainingJob(
                "04_recombined_small_EFP_matched_nrep4",
                recombined,
                0.50,
                (4.5, 8.0),
                14400,
                "matched recombined small slab for Phase 1 E_FP",
            ),
            common["extra_meta"],
        ),
        (
            RemainingJob(
                "05_separated_small_EFP_matched_nrep4",
                separated,
                0.50,
                (4.5, 8.0),
                18000,
                "matched separated small slab for Phase 1 E_FP",
            ),
            common["extra_meta"],
        ),
    ]


def refresh_artifacts(run_root: Path, status: dict) -> None:
    heartbeat(run_root, "refresh synthesis start")
    synth = run_root / "synthesis"
    cmd = (
        "python scripts/10_sunday_synthesis.py "
        "--kpm-aggregate runs/kpm_aggregate_scope_upgrade_20260604_004115 "
        "--local-bigdft runs/single_bigdft_20260603_101509_local "
        "--extra-bigdft runs/bigdft_followup_20260603_171018_spark-5da5 "
        f"--extra-bigdft {run_root} "
        "--peer-bigdft-remote '' "
        f"--output {synth}"
    )
    rc = run(cmd, timeout=1200, log=run_root / "logs" / "refresh_synthesis.out").returncode
    status["refresh"]["synthesis"] = {"returncode": rc, "path": str(synth)}
    write_json(run_root / "status.json", status)
    if rc != 0:
        heartbeat(run_root, f"refresh synthesis failed rc={rc}")
        return

    tier_bc = run_root / "tier_bc"
    cmd = (
        "python scripts/03_bigdft_emission.py "
        f"--bigdft-observables {synth / 'bigdft_observables.csv'} "
        f"--output {tier_bc}"
    )
    rc = run(cmd, timeout=600, log=run_root / "logs" / "refresh_tier_bc.out").returncode
    status["refresh"]["tier_bc"] = {"returncode": rc, "path": str(tier_bc)}
    write_json(run_root / "status.json", status)
    if rc != 0:
        heartbeat(run_root, f"refresh tier_bc failed rc={rc}")
        return

    digest = run_root / "full_digest"
    cmd = (
        "python scripts/17_scope_upgrade_digest.py "
        f"--kpm-summary {synth / 'kpm_summary.csv'} "
        f"--bigdft-summary {synth / 'bigdft_observables.csv'} "
        f"--tier-bc-status {tier_bc / 'phase1_tier_bc_status.json'} "
        "--large-run runs/large_kpm_20260603_183240_local_large "
        "--large-run runs/large_kpm_20260603_183959_local_dend64 "
        "--kubo-run runs/scope_upgrade_imports/dgx-peer/kubo_mobility_20260603_183557_peer_kg "
        f"--output {digest}"
    )
    rc = run(cmd, timeout=600, log=run_root / "logs" / "refresh_digest.out").returncode
    status["refresh"]["digest"] = {"returncode": rc, "path": str(digest)}
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"refresh digest rc={rc}")


def launch_ldos_followup(run_root: Path, status: dict) -> None:
    synth_path = Path(status.get("refresh", {}).get("synthesis", {}).get("path", ""))
    bigdft_csv = synth_path / "bigdft_observables.csv"
    if not bigdft_csv.exists():
        status.setdefault("refresh", {})["ldos_queue"] = {
            "returncode": None,
            "reason": f"Missing BigDFT observables CSV: {bigdft_csv}",
        }
        write_json(run_root / "status.json", status)
        heartbeat(run_root, "skip LDOS follow-up: missing BigDFT observables CSV")
        return
    cmd = (
        f"{shlex.quote(sys.executable)} scripts/19_phase1_ldos_queue.py "
        f"--bigdft-observables {shlex.quote(str(bigdft_csv))}"
    )
    heartbeat(run_root, "launch LDOS follow-up start")
    rc = run(cmd, timeout=24000, log=run_root / "logs" / "refresh_ldos_queue.out").returncode
    status.setdefault("refresh", {})["ldos_queue"] = {"returncode": rc}
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"launch LDOS follow-up rc={rc}")


def main() -> int:
    label = os.environ.get("ELECTRODEFECT_PHASE1_LABEL", socket.gethostname())
    run_id = datetime.now().strftime(f"phase1_remaining_%Y%m%d_%H%M%S_{label}")
    run_root = RUNS / run_id
    log_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    activate = queue.bigdft_activation()
    jobs = remaining_jobs()
    status = {
        "state": "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "activate": activate,
        "jobs": [
            {"name": job.name, "notes": job.notes, "state": "pending", "elapsed_s": 0, "n_atoms": len(job.atoms)}
            for job, _ in jobs
        ],
        "refresh": {},
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "remaining Phase 1 queue start")

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

    refresh_artifacts(run_root, status)
    launch_ldos_followup(run_root, status)
    status["state"] = "completed"
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "remaining Phase 1 queue completed")
    print(f"remaining Phase 1 queue complete -> {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
