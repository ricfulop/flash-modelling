#!/usr/bin/env python3
"""Long GPU KPM sweep for Sunday compute pushes.

This script does not overwrite the Tier A adjudication outputs. It writes
repeat-resolved KPM DOS artifacts under runs/gpu_kpm_<timestamp>_<label>/.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from electrodefect import tier_a
from electrodefect import transport_kpm as tk


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "w_percolation.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def heartbeat(run_root: Path, message: str) -> None:
    with (run_root / "heartbeat.log").open("a") as fh:
        fh.write(f"{now()} {message}\n")


def case_disorder(cfg: dict, morphology: str) -> float:
    if morphology == "ordered":
        return float(cfg["tight_binding"]["disorder_W_ordered"])
    return float(cfg["tight_binding"]["disorder_W_random"])


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    if not tk._HAS_TORCH or not tk.torch.cuda.is_available():
        raise RuntimeError("CUDA torch device is required for this sweep")

    label = os.environ.get("ELECTRODEFECT_SWEEP_LABEL", socket.gethostname())
    run_id = datetime.now().strftime(f"gpu_kpm_%Y%m%d_%H%M%S_{label}")
    run_root = REPO_ROOT / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    moments = int(os.environ.get("ELECTRODEFECT_KPM_MOMENTS", "16384"))
    random_vectors = int(os.environ.get("ELECTRODEFECT_KPM_RANDOM_VECTORS", "96"))
    energies_count = int(os.environ.get("ELECTRODEFECT_KPM_ENERGIES", "6000"))
    repeats = int(os.environ.get("ELECTRODEFECT_KPM_REPEATS", "24"))
    base_seed = int(os.environ.get("ELECTRODEFECT_BASE_SEED", cfg["network"].get("seed", 0)))
    max_seconds = int(os.environ.get("ELECTRODEFECT_SWEEP_MAX_SECONDS", "21600"))

    status = {
        "state": "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "label": label,
        "moments": moments,
        "random_vectors": random_vectors,
        "energies": energies_count,
        "repeats_requested": repeats,
        "base_seed": base_seed,
        "cases": [],
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "sweep start")

    start_all = time.time()
    completed = 0
    try:
        for repeat in range(repeats):
            for morphology, frac in tier_a.iter_cases(cfg):
                if time.time() - start_all > max_seconds:
                    heartbeat(run_root, "max seconds reached")
                    raise TimeoutError("sweep time budget reached")

                case_dir = tier_a.case_dir(morphology, frac)
                if not (case_dir / "network.npz").exists():
                    net = tier_a.build_network(cfg, morphology, frac)
                    tier_a.save_network(
                        net,
                        case_dir,
                        {
                            "material": cfg["material"],
                            "morphology": morphology,
                            "target_frac": frac,
                            "seed": cfg["network"].get("seed", 0),
                            "supercell": cfg["network"]["supercell"],
                        },
                        spectral_t_max=int(os.environ.get("ELECTRODEFECT_DS_TMAX", "1000")),
                        spectral_walkers=int(os.environ.get("ELECTRODEFECT_DS_WALKERS", "1000")),
                    )
                net = tier_a.load_network(case_dir)
                seed = base_seed + repeat * 1000 + completed
                out_dir = run_root / f"repeat_{repeat:03d}" / f"{morphology}_{tier_a.frac_slug(frac)}"
                out_dir.mkdir(parents=True, exist_ok=True)

                heartbeat(run_root, f"start repeat={repeat} case={morphology} frac={frac}")
                case_start = time.time()
                H, _, _ = tk.build_tb(
                    net,
                    t_hop=float(cfg["tight_binding"]["t_hop"]),
                    disorder_W=case_disorder(cfg, morphology),
                    seed=seed,
                    device="cuda",
                )
                energies, dos = tk.kpm_dos(
                    H,
                    M=moments,
                    R=random_vectors,
                    n_energies=energies_count,
                    device="cuda",
                    seed=seed,
                )
                np.savez_compressed(out_dir / "kpm_dos.npz", energies=energies, dos=dos)
                elapsed = time.time() - case_start
                record = {
                    "repeat": repeat,
                    "morphology": morphology,
                    "target_frac": frac,
                    "seed": seed,
                    "elapsed_s": elapsed,
                    "output": str((out_dir / "kpm_dos.npz").relative_to(run_root)),
                }
                write_json(out_dir / "metadata.json", record)
                status["cases"].append(record)
                completed += 1
                status["completed_cases"] = completed
                status["updated_at"] = now()
                write_json(run_root / "status.json", status)
                heartbeat(run_root, f"finish repeat={repeat} case={morphology} frac={frac} elapsed={elapsed:.1f}s")
    except TimeoutError:
        status["state"] = "time_budget_reached"
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"failed {type(exc).__name__}: {exc}")
        raise
    else:
        status["state"] = "completed"

    status["elapsed_s"] = time.time() - start_all
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"sweep {status['state']} completed_cases={completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
