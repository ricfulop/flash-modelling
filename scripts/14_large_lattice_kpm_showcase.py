#!/usr/bin/env python3
"""Large-lattice GPU KPM showcase for Sunday transport scaling.

Writes independent artifacts under runs/large_kpm_<timestamp>_<label>/ and
does not modify the frozen Tier A data products.
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

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from electrodefect import percolation
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


def parse_case(spec: str) -> tuple[str, float, tuple[int, int, int]]:
    """Parse morphology:frac:nx or morphology:frac:nx,ny,nz."""
    morphology, frac_s, shape_s = spec.split(":", 2)
    if "," in shape_s:
        shape = tuple(int(x) for x in shape_s.split(","))
        if len(shape) != 3:
            raise ValueError(f"shape must have 3 entries: {spec}")
    else:
        n = int(shape_s)
        shape = (n, n, n)
    return morphology, float(frac_s), shape


def build_network(morphology: str, frac: float, shape: tuple[int, int, int], a: float, seed: int):
    nx_, ny, nz = shape
    if morphology == "ordered":
        return percolation.ordered_superlattice(nx_=nx_, ny=ny, nz=nz, a=a, frac=frac, seed=seed)
    if morphology == "random":
        return percolation.random_percolation(nx_=nx_, ny=ny, nz=nz, a=a, frac=frac, seed=seed)
    if morphology in {"dendritic", "dendrite"}:
        return percolation.dla_dendrite(nx_=nx_, ny=ny, nz=nz, a=a, target_frac=frac, seed=seed)
    raise ValueError(f"unknown morphology: {morphology}")


def case_disorder(cfg: dict, morphology: str) -> float:
    if morphology == "ordered":
        return float(cfg["tight_binding"]["disorder_W_ordered"])
    return float(cfg["tight_binding"]["disorder_W_random"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case as morphology:frac:nx or morphology:frac:nx,ny,nz. May repeat.",
    )
    parser.add_argument("--label", default=os.environ.get("ELECTRODEFECT_SWEEP_LABEL", socket.gethostname()))
    parser.add_argument("--moments", type=int, default=int(os.environ.get("ELECTRODEFECT_LARGE_KPM_MOMENTS", "8192")))
    parser.add_argument("--random-vectors", type=int, default=int(os.environ.get("ELECTRODEFECT_LARGE_KPM_RANDOM_VECTORS", "48")))
    parser.add_argument("--energies", type=int, default=int(os.environ.get("ELECTRODEFECT_LARGE_KPM_ENERGIES", "4000")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ELECTRODEFECT_BASE_SEED", "10001")))
    parser.add_argument("--max-seconds", type=int, default=int(os.environ.get("ELECTRODEFECT_LARGE_KPM_MAX_SECONDS", "21600")))
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    if not tk._HAS_TORCH or not tk.torch.cuda.is_available():
        raise RuntimeError("CUDA torch device is required for large-lattice KPM.")

    case_specs = args.case or [
        "ordered:0.35:32",
        "random:0.35:32",
        "dendritic:0.35:32",
        "ordered:0.35:64",
        "random:0.35:64",
    ]
    cases = [parse_case(spec) for spec in case_specs]

    run_root = RUNS / datetime.now().strftime(f"large_kpm_%Y%m%d_%H%M%S_{args.label}")
    run_root.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "label": args.label,
        "moments": args.moments,
        "random_vectors": args.random_vectors,
        "energies": args.energies,
        "seed": args.seed,
        "cases": [],
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "large-lattice KPM showcase start")

    start_all = time.time()
    for idx, (morphology, frac, shape) in enumerate(cases):
        if time.time() - start_all > args.max_seconds:
            status["state"] = "time_budget_reached"
            heartbeat(run_root, "max seconds reached")
            break
        slug = f"{morphology}_{frac:.2f}_{shape[0]}x{shape[1]}x{shape[2]}".replace(".", "p")
        out_dir = run_root / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = args.seed + idx
        heartbeat(run_root, f"start {slug}")
        start = time.time()
        net = build_network(morphology, frac, shape, float(cfg["lattice_a"]), seed)
        build_elapsed = time.time() - start
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
        elapsed = time.time() - start
        zero_idx = int(np.nanargmin(np.abs(energies)))
        peak_idx = int(np.nanargmax(dos))
        np.savez_compressed(
            out_dir / "kpm_dos.npz",
            energies=energies,
            dos=dos,
            shape=np.asarray(shape, dtype=int),
        )
        record = {
            "morphology": morphology,
            "target_frac": frac,
            "shape": shape,
            "total_lattice_sites": int(len(net.coords)),
            "defect_graph_nodes": int(net.graph.number_of_nodes()),
            "defect_graph_edges": int(net.graph.number_of_edges()),
            "seed": seed,
            "build_elapsed_s": build_elapsed,
            "elapsed_s": elapsed,
            "dos_at_E0": float(dos[zero_idx]),
            "peak_energy": float(energies[peak_idx]),
            "peak_dos": float(dos[peak_idx]),
            "output": str((out_dir / "kpm_dos.npz").relative_to(run_root)),
        }
        write_json(out_dir / "metadata.json", record)
        status["cases"].append(record)
        status["completed_cases"] = len(status["cases"])
        status["updated_at"] = now()
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"finish {slug} elapsed={elapsed:.1f}s nodes={record['defect_graph_nodes']}")

    if status.get("state") == "running":
        status["state"] = "completed"
    status["elapsed_s"] = time.time() - start_all
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"large-lattice KPM showcase {status['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
