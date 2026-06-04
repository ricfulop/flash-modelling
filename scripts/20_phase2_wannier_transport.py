#!/usr/bin/env python3
"""Wannier/downfolded-Hamiltonian KPM diagnostics for Phase 2 transport."""
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from electrodefect import dft_bigdft
from electrodefect import transport_kpm as tk


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"


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


def discover_artifacts(root: Path) -> list[Path]:
    """Find Wannier90 or BigDFT matrix-export Hamiltonian artifacts."""
    patterns = [
        "**/*_hr.dat",
        "**/*hr.dat",
        "**/*Hamiltonian*.mtx",
        "**/*hamiltonian*.mtx",
        "**/*ham*.mtx",
    ]
    artifacts: list[Path] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path not in seen:
                artifacts.append(path)
                seen.add(path)
    return artifacts


def classify_artifact(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith("_hr.dat") or lower.endswith("hr.dat"):
        return "wannier90_hr"
    if lower.endswith(".mtx"):
        return "matrix_market_hamiltonian"
    raise ValueError(f"unsupported downfolded Hamiltonian artifact: {path}")


def load_hamiltonian(path: Path, matrix_unit: str):
    kind = classify_artifact(path)
    if kind == "wannier90_hr":
        H = tk.load_wannier90_hr(path)
    elif kind == "matrix_market_hamiltonian":
        from scipy.io import mmread

        H = mmread(path).tocoo()
    else:
        raise ValueError(f"unsupported artifact kind: {kind}")
    if matrix_unit == "Ha":
        H = H * dft_bigdft.EV_PER_HA
    return kind, H


def safe_slug(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    return "_".join(rel.with_suffix("").parts).replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=Path, default=[])
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--label", default=os.environ.get("ELECTRODEFECT_WANNIER_LABEL", socket.gethostname()))
    parser.add_argument("--max-artifacts", type=int, default=0)
    parser.add_argument("--moments", type=int, default=1024)
    parser.add_argument("--random-vectors", type=int, default=16)
    parser.add_argument("--energies", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--device", default="cuda", choices=["cuda", "scipy"])
    parser.add_argument("--matrix-unit", default="eV", choices=["eV", "Ha"])
    args = parser.parse_args()

    roots = [root.resolve() for root in args.source_root]
    artifacts = [path.resolve() for path in args.artifact]
    for root in roots:
        artifacts.extend(discover_artifacts(root))
    # Deduplicate while preserving order.
    artifacts = list(dict.fromkeys(artifacts))
    if args.max_artifacts:
        artifacts = artifacts[: int(args.max_artifacts)]
    if not artifacts:
        raise SystemExit(
            "No Wannier/downfolded Hamiltonian artifacts found. Expected *_hr.dat "
            "or MatrixMarket Hamiltonian .mtx files from BigDFT/Wannier export."
        )
    if args.device == "cuda" and (not tk._HAS_TORCH or not tk.torch.cuda.is_available()):
        raise RuntimeError("CUDA torch device is required for KPM; use --device scipy for scalar-only smoke.")

    run_root = args.run_root or RUNS / datetime.now().strftime(
        f"phase2_wannier_transport_%Y%m%d_%H%M%S_{args.label}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "label": args.label,
        "source_roots": [str(root) for root in roots],
        "device": args.device,
        "moments": args.moments,
        "random_vectors": args.random_vectors,
        "energies": args.energies,
        "matrix_unit": args.matrix_unit,
        "geometry_proxy": False,
        "cases": [],
        "updated_at": now(),
    }
    write_json(run_root / "status.json", status)
    heartbeat(run_root, f"wannier transport start n_artifacts={len(artifacts)}")

    start_all = time.time()
    try:
        for idx, artifact in enumerate(artifacts):
            root = next((r for r in roots if artifact.is_relative_to(r)), artifact.parent)
            slug = safe_slug(artifact, root)
            out_dir = run_root / slug
            out_dir.mkdir(parents=True, exist_ok=True)
            heartbeat(run_root, f"start {artifact}")
            start = time.time()
            kind, H = load_hamiltonian(artifact, args.matrix_unit)
            scalars = tk.downfolded_hamiltonian_scalars(H)
            record = {
                "source": str(artifact),
                "source_kind": kind,
                "artifact_relative": str(artifact.relative_to(root)) if artifact.is_relative_to(root) else artifact.name,
                "elapsed_s": None,
                "scalars": scalars,
                "geometry_proxy": False,
            }
            if args.device == "cuda":
                H_torch = tk.matrix_to_torch_sparse(H, device=args.device)
                energies, dos = tk.kpm_dos(
                    H_torch,
                    M=args.moments,
                    R=args.random_vectors,
                    n_energies=args.energies,
                    device=args.device,
                    seed=args.seed + idx,
                )
                np.savez_compressed(out_dir / "wannier_kpm.npz", energies=energies, dos=dos)
                record["output"] = str((out_dir / "wannier_kpm.npz").relative_to(run_root))
                record["dos_at_E0"] = float(dos[int(np.nanargmin(np.abs(energies)))])
            else:
                record["scalar_only"] = True
            record["elapsed_s"] = time.time() - start
            write_json(out_dir / "metadata.json", record)
            status["cases"].append(record)
            status["completed_cases"] = len(status["cases"])
            status["updated_at"] = now()
            write_json(run_root / "status.json", status)
            heartbeat(run_root, f"finish {artifact} elapsed={record['elapsed_s']:.1f}s")
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["updated_at"] = now()
        write_json(run_root / "status.json", status)
        heartbeat(run_root, f"failed {status['error']}")
        raise

    status["state"] = "completed"
    status["elapsed_s"] = time.time() - start_all
    status["updated_at"] = now()
    write_json(run_root / "status.json", status)
    heartbeat(run_root, "wannier transport completed")
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
