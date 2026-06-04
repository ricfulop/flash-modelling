#!/usr/bin/env python3
"""GPU Kubo-Greenwood and mobility-window diagnostics for Sunday.

This is an exploratory Tier D-lite extension. It uses the existing TB graph
contracts and writes new derived artifacts under runs/kubo_mobility_<timestamp>/.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from electrodefect import tier_a
from electrodefect import transport_kpm as tk


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "w_percolation.yaml"
RUNS = REPO_ROOT / "runs"


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


def mobility_windows(energies: np.ndarray, sigma: np.ndarray, threshold: float) -> list[dict]:
    finite = np.isfinite(sigma)
    if not np.any(finite):
        return []
    scale = float(np.nanmax(np.abs(sigma[finite])))
    if scale <= 0:
        return []
    mask = finite & ((sigma / scale) >= threshold)
    windows = []
    start = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        if start is not None and (not active or idx == len(mask) - 1):
            end = idx if active and idx == len(mask) - 1 else idx - 1
            windows.append(
                {
                    "energy_min": float(energies[start]),
                    "energy_max": float(energies[end]),
                    "width_eV_proxy": float(energies[end] - energies[start]),
                    "contains_E0": bool(energies[start] <= 0.0 <= energies[end]),
                }
            )
            start = None
    return windows


def plot_case(out_dir: Path, energies: np.ndarray, sigma: np.ndarray, windows: list[dict], title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    ax.plot(energies, sigma, color="#0072B2", linewidth=1.6)
    ax.axvline(0.0, color="0.25", linestyle="--", linewidth=0.8)
    for win in windows:
        ax.axvspan(win["energy_min"], win["energy_max"], color="#009E73", alpha=0.15)
    ax.set_xlabel("Energy (scaled tight-binding units)")
    ax.set_ylabel("Kubo-Greenwood sigma(E), arb.")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_dir / "sigma_energy.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "sigma_energy.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=os.environ.get("ELECTRODEFECT_SWEEP_LABEL", socket.gethostname()))
    parser.add_argument("--moments", type=int, default=int(os.environ.get("ELECTRODEFECT_KG_MOMENTS", "192")))
    parser.add_argument("--random-vectors", type=int, default=int(os.environ.get("ELECTRODEFECT_KG_RANDOM_VECTORS", "4")))
    parser.add_argument("--energies", type=int, default=int(os.environ.get("ELECTRODEFECT_KG_ENERGIES", "500")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ELECTRODEFECT_BASE_SEED", "20001")))
    parser.add_argument("--sigma-threshold", type=float, default=float(os.environ.get("ELECTRODEFECT_SIGMA_THRESHOLD", "0.05")))
    parser.add_argument("--case", action="append", default=[], help="Optional morphology:frac filter; may repeat.")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    if not tk._HAS_TORCH or not tk.torch.cuda.is_available():
        raise RuntimeError("CUDA torch device is required for Kubo-Greenwood diagnostics.")

    filters = set()
    for item in args.case:
        morphology, frac_s = item.split(":", 1)
        filters.add((morphology, float(frac_s)))

    run_root = RUNS / datetime.now().strftime(f"kubo_mobility_%Y%m%d_%H%M%S_{args.label}")
    run_root.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "label": args.label,
        "moments": args.moments,
        "random_vectors": args.random_vectors,
        "energies": args.energies,
        "sigma_threshold": args.sigma_threshold,
        "cases": [],
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "Kubo/mobility diagnostics start")

    rows = []
    start_all = time.time()
    for idx, (morphology, frac) in enumerate(tier_a.iter_cases(cfg)):
        if filters and (morphology, float(frac)) not in filters:
            continue
        slug = f"{morphology}_{frac:.2f}".replace(".", "p")
        out_dir = run_root / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        heartbeat(run_root, f"start {slug}")
        start = time.time()
        net = tier_a.load_network(tier_a.case_dir(morphology, frac))
        seed = args.seed + idx
        H, X, _ = tk.build_tb(
            net,
            t_hop=float(cfg["tight_binding"]["t_hop"]),
            disorder_W=case_disorder(cfg, morphology),
            seed=seed,
            device="cuda",
        )
        energies, sigma = tk.kubo_greenwood_dc(
            H,
            X,
            M=args.moments,
            R=args.random_vectors,
            n_energies=args.energies,
            device="cuda",
            seed=seed,
        )
        windows = mobility_windows(energies, sigma, args.sigma_threshold)
        zero_idx = int(np.nanargmin(np.abs(energies)))
        peak_idx = int(np.nanargmax(sigma))
        np.savez_compressed(out_dir / "kubo_sigma.npz", energies=energies, sigma=sigma)
        write_json(out_dir / "mobility_windows.json", {"windows": windows})
        plot_case(out_dir, energies, sigma, windows, f"{morphology} frac={frac:.2f}")
        elapsed = time.time() - start
        record = {
            "morphology": morphology,
            "target_frac": float(frac),
            "graph_nodes": int(net.graph.number_of_nodes()),
            "graph_edges": int(net.graph.number_of_edges()),
            "seed": seed,
            "elapsed_s": elapsed,
            "sigma_at_E0": float(sigma[zero_idx]),
            "sigma_peak": float(sigma[peak_idx]),
            "sigma_peak_energy": float(energies[peak_idx]),
            "n_mobility_windows": len(windows),
            "has_E0_window": bool(any(w["contains_E0"] for w in windows)),
            "output": str((out_dir / "kubo_sigma.npz").relative_to(run_root)),
        }
        write_json(out_dir / "metadata.json", record)
        rows.append(record)
        status["cases"].append(record)
        status["completed_cases"] = len(status["cases"])
        status["updated_at"] = now()
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"finish {slug} elapsed={elapsed:.1f}s")
        try:
            tk.torch.cuda.empty_cache()
        except Exception:
            pass

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.to_csv(run_root / "kubo_mobility_summary.csv", index=False)
    status["state"] = "completed"
    status["elapsed_s"] = time.time() - start_all
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "Kubo/mobility diagnostics completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
