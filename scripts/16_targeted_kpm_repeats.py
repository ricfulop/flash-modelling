#!/usr/bin/env python3
"""Add targeted KPM repeats for groups with wide confidence bands."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from electrodefect import tier_a
from electrodefect import transport_kpm as tk


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "w_percolation.yaml"
RUNS = REPO_ROOT / "runs"
DEFAULT_SUMMARY = RUNS / "sunday_synthesis_20260603_171051" / "kpm_summary.csv"


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


def case_disorder(cfg: dict, morphology: str) -> float:
    if morphology == "ordered":
        return float(cfg["tight_binding"]["disorder_W_ordered"])
    return float(cfg["tight_binding"]["disorder_W_random"])


def select_targets(summary_path: Path, threshold: float) -> list[tuple[str, float, dict]]:
    summary = pd.read_csv(summary_path)
    targets = []
    for _, row in summary.iterrows():
        e0_mean = abs(float(row["dos_at_E0_mean"]))
        peak_mean = abs(float(row["peak_dos_mean"]))
        e0_ratio = float(row["dos_at_E0_ci95"]) / max(e0_mean, 1e-12)
        peak_ratio = float(row["peak_dos_ci95"]) / max(peak_mean, 1e-12)
        if max(e0_ratio, peak_ratio) >= threshold:
            targets.append(
                (
                    str(row["morphology"]),
                    float(row["target_frac"]),
                    {
                        "e0_ci_ratio": e0_ratio,
                        "peak_ci_ratio": peak_ratio,
                        "n_repeats_before": int(row["n_repeats"]),
                    },
                )
            )
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("ELECTRODEFECT_CI_RATIO_THRESHOLD", "0.30")))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("ELECTRODEFECT_TARGET_REPEATS", "12")))
    parser.add_argument("--moments", type=int, default=int(os.environ.get("ELECTRODEFECT_KPM_MOMENTS", "16384")))
    parser.add_argument("--random-vectors", type=int, default=int(os.environ.get("ELECTRODEFECT_KPM_RANDOM_VECTORS", "96")))
    parser.add_argument("--energies", type=int, default=int(os.environ.get("ELECTRODEFECT_KPM_ENERGIES", "6000")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ELECTRODEFECT_BASE_SEED", "30001")))
    parser.add_argument("--label", default=os.environ.get("ELECTRODEFECT_SWEEP_LABEL", socket.gethostname()))
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    if not tk._HAS_TORCH or not tk.torch.cuda.is_available():
        raise RuntimeError("CUDA torch device is required for targeted KPM repeats.")

    targets = select_targets(Path(args.summary), args.threshold)
    run_root = RUNS / datetime.now().strftime(f"targeted_kpm_%Y%m%d_%H%M%S_{args.label}")
    run_root.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "label": args.label,
        "summary": str(args.summary),
        "threshold": args.threshold,
        "targets": [
            {"morphology": m, "target_frac": f, **info}
            for m, f, info in targets
        ],
        "cases": [],
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"targeted KPM repeats start targets={len(targets)}")

    start_all = time.time()
    completed = 0
    for repeat in range(args.repeats):
        for morphology, frac, info in targets:
            seed = args.seed + repeat * 1000 + completed
            slug = f"repeat_{repeat:03d}/{morphology}_{frac:.2f}".replace(".", "p")
            out_dir = run_root / slug
            out_dir.mkdir(parents=True, exist_ok=True)
            heartbeat(run_root, f"start repeat={repeat} {morphology} frac={frac}")
            start = time.time()
            net = tier_a.load_network(tier_a.case_dir(morphology, frac))
            H, _, _ = tk.build_tb(
                net,
                t_hop=float(cfg["tight_binding"]["t_hop"]),
                disorder_W=case_disorder(cfg, morphology),
                seed=seed,
                device="cuda",
            )
            energies, dos = tk.kpm_dos(
                H,
                M=args.moments,
                R=args.random_vectors,
                n_energies=args.energies,
                device="cuda",
                seed=seed,
            )
            np.savez_compressed(out_dir / "kpm_dos.npz", energies=energies, dos=dos)
            elapsed = time.time() - start
            record = {
                "repeat": repeat,
                "morphology": morphology,
                "target_frac": frac,
                "seed": seed,
                "elapsed_s": elapsed,
                "selection": info,
                "output": str((out_dir / "kpm_dos.npz").relative_to(run_root)),
            }
            write_json(out_dir / "metadata.json", record)
            status["cases"].append(record)
            completed += 1
            status["completed_cases"] = completed
            status["updated_at"] = now()
            write_json(run_root / "status.json", status)
            heartbeat(run_root, f"finish repeat={repeat} {morphology} frac={frac} elapsed={elapsed:.1f}s")

    status["state"] = "completed"
    status["elapsed_s"] = time.time() - start_all
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"targeted KPM repeats completed cases={completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
