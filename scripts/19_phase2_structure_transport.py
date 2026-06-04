#!/usr/bin/env python3
"""Opt-in geometry-proxy KPM diagnostics on Phase 2 structures.

Use scripts/20_phase2_wannier_transport.py for the primary Wannier/downfolded
Hamiltonian path. This script is only a fallback plumbing smoke.
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
from electrodefect import transport_kpm as tk


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
CONFIG = REPO / "config" / "w_percolation.yaml"


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


def discover_structures(phase2_root: Path) -> list[Path]:
    patterns = [
        "mace_pilot/*/*/annealed_separated.xyz",
        "mace_pilot/*/*/relaxed_separated.xyz",
        "sed_pilot/*/*/sed_final.xyz",
        "neb_pilot/*/*/initial.xyz",
        "neb_pilot/*/*/final.xyz",
    ]
    paths: list[Path] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(phase2_root.glob(pattern)):
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def structure_tags(path: Path, phase2_root: Path) -> dict:
    rel = path.relative_to(phase2_root)
    parts = rel.parts
    tags = {"source_relative": str(rel), "stage": parts[0] if parts else None}
    if len(parts) >= 4:
        tags.update({"material": parts[1], "morphology": parts[2], "structure": path.stem})
    return tags


def safe_slug(path: Path, phase2_root: Path) -> str:
    rel = path.relative_to(phase2_root)
    return "_".join(rel.with_suffix("").parts).replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-root", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--structure", action="append", default=[], type=Path)
    parser.add_argument("--max-structures", type=int, default=0)
    parser.add_argument("--moments", type=int, default=512)
    parser.add_argument("--random-vectors", type=int, default=8)
    parser.add_argument("--energies", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cutoff-scale", type=float, default=1.35)
    parser.add_argument("--kubo", action="store_true")
    parser.add_argument("--allow-geometry-proxy", action="store_true")
    args = parser.parse_args()
    if not args.allow_geometry_proxy:
        raise SystemExit(
            "Refusing geometry-proxy transport without --allow-geometry-proxy. "
            "Use scripts/20_phase2_wannier_transport.py for Wannier/downfolded transport."
        )

    cfg = yaml.safe_load(CONFIG.read_text())
    phase2_root = args.phase2_root.resolve()
    if args.device == "cuda" and (not tk._HAS_TORCH or not tk.torch.cuda.is_available()):
        raise RuntimeError("CUDA torch device is required; pass --device scipy for a CPU graph smoke.")
    structures = [p.resolve() for p in args.structure] if args.structure else discover_structures(phase2_root)
    if args.max_structures:
        structures = structures[: int(args.max_structures)]
    run_root = args.run_root or RUNS / datetime.now().strftime(
        f"phase2_structure_transport_%Y%m%d_%H%M%S_{socket.gethostname()}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "running",
        "run_root": str(run_root),
        "phase2_root": str(phase2_root),
        "host": socket.gethostname(),
        "moments": args.moments,
        "random_vectors": args.random_vectors,
        "energies": args.energies,
        "device": args.device,
        "kubo": bool(args.kubo),
        "geometry_proxy_warning": "Edges come from relaxed structure distances; not a DFT/Wannier transport model.",
        "cases": [],
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "phase2 structure transport start")

    from ase.io import read

    start_all = time.time()
    try:
        for idx, path in enumerate(structures):
            tags = structure_tags(path, phase2_root)
            slug = safe_slug(path, phase2_root)
            out_dir = run_root / slug
            out_dir.mkdir(parents=True, exist_ok=True)
            heartbeat(run_root, f"start {tags['source_relative']}")
            start = time.time()
            atoms = read(path)
            material = tags.get("material") or cfg.get("material", "W")
            lattice_a = float(cfg.get("lattice_a", 3.165))
            if material == "Mo":
                lattice_a = 3.147
            net = tk.network_from_atoms(atoms, lattice_a=lattice_a, cutoff_scale=args.cutoff_scale)
            H, X, _ = tk.build_tb(
                net,
                t_hop=float(cfg["tight_binding"]["t_hop"]),
                disorder_W=0.0,
                seed=args.seed + idx,
                device=args.device,
            )
            if args.device == "scipy":
                record = {
                    **tags,
                    "n_atoms": len(atoms),
                    "graph_nodes": int(net.graph.number_of_nodes()),
                    "graph_edges": int(net.graph.number_of_edges()),
                    "elapsed_s": time.time() - start,
                    "graph_only": True,
                    "output": str((out_dir / "metadata.json").relative_to(run_root)),
                }
                write_json(out_dir / "metadata.json", record)
                status["cases"].append(record)
                status["completed_cases"] = len(status["cases"])
                status["updated_at"] = now()
                write_json(run_root / "status.json", status)
                heartbeat(run_root, f"finish graph-only {tags['source_relative']} elapsed={record['elapsed_s']:.1f}s")
                continue
            energies, dos = tk.kpm_dos(
                H,
                M=args.moments,
                R=args.random_vectors,
                n_energies=args.energies,
                device=args.device,
                seed=args.seed + idx,
            )
            arrays = {"energies": energies, "dos": dos}
            record = {
                **tags,
                "n_atoms": len(atoms),
                "graph_nodes": int(net.graph.number_of_nodes()),
                "graph_edges": int(net.graph.number_of_edges()),
                "elapsed_s": time.time() - start,
                "dos_at_E0": float(dos[int(np.nanargmin(np.abs(energies)))]),
                "output": str((out_dir / "structure_kpm.npz").relative_to(run_root)),
            }
            if args.kubo:
                kg_energies, sigma = tk.kubo_greenwood_dc(
                    H,
                    X,
                    M=max(16, min(args.moments, 256)),
                    R=max(1, min(args.random_vectors, 8)),
                    n_energies=args.energies,
                    device=args.device,
                    seed=args.seed + idx,
                )
                arrays.update({"kubo_energies": kg_energies, "sigma": sigma})
                zero_idx = int(np.nanargmin(np.abs(kg_energies)))
                record["sigma_at_E0"] = float(sigma[zero_idx])
            np.savez_compressed(out_dir / "structure_kpm.npz", **arrays)
            write_json(out_dir / "metadata.json", record)
            status["cases"].append(record)
            status["completed_cases"] = len(status["cases"])
            status["updated_at"] = now()
            write_json(run_root / "status.json", status)
            heartbeat(run_root, f"finish {tags['source_relative']} elapsed={record['elapsed_s']:.1f}s")
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"failed {status['error']}")
        raise

    status["state"] = "completed"
    status["elapsed_s"] = time.time() - start_all
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "phase2 structure transport completed")
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
