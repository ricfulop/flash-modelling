#!/usr/bin/env python3
"""Phase 2 MLIP staging launcher.

This script is deliberately gate-based. It runs tiny smoke/pilot jobs first and
only writes promotion manifests when the production calculator is present and
pilot sanity checks pass.
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
from electrodefect import build, mlip_al, percolation


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
CONFIG = REPO / "config" / "w_percolation.yaml"
SEED = 20260603


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_root(prefix: str = "phase2_mlip") -> Path:
    label = os.environ.get("ELECTRODEFECT_PHASE2_LABEL", socket.gethostname())
    return RUNS / datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_{label}")


def load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def ensure_run_root(path: str | None) -> Path:
    root = Path(path) if path else default_run_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def heartbeat(run_root: Path, message: str) -> None:
    with (run_root / "heartbeat.log").open("a") as fh:
        fh.write(f"{now()} {message}\n")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def update_status(run_root: Path, key: str, payload: dict) -> None:
    status_path = run_root / "status.json"
    status = read_json(status_path, {"state": "running", "run_root": str(run_root), "steps": {}})
    status["updated_at"] = now()
    status["host"] = socket.gethostname()
    status["steps"][key] = payload
    if all(step.get("state") in {"completed", "blocked"} for step in status["steps"].values()):
        status["state"] = "completed"
    mlip_al.write_json(status_path, status)


def build_network(morphology: str, frac: float, shape: tuple[int, int, int], a: float, seed: int):
    nx_, ny, nz = shape
    if morphology == "ordered":
        return percolation.ordered_superlattice(nx_=nx_, ny=ny, nz=nz, a=a, frac=frac, seed=seed)
    if morphology == "random":
        return percolation.random_percolation(nx_=nx_, ny=ny, nz=nz, a=a, frac=frac, seed=seed)
    if morphology in {"dendritic", "dendrite"}:
        return percolation.dla_dendrite(nx_=nx_, ny=ny, nz=nz, a=a, target_frac=frac, seed=seed)
    raise ValueError(f"unknown morphology: {morphology}")


def run_env_smoke(args) -> Path:
    run_root = ensure_run_root(args.run_root)
    heartbeat(run_root, "env-smoke start")
    payload = mlip_al.environment_smoke(device=args.device)
    payload["state"] = "completed"
    payload["production_mace_ready"] = bool(payload.get("mace_available") and payload.get("torch", {}).get("cuda_available"))
    mlip_al.write_json(run_root / "env_smoke.json", payload)
    update_status(run_root, "env_smoke", payload)
    heartbeat(run_root, "env-smoke completed")
    print(run_root)
    return run_root


def run_seed_configs(args) -> Path:
    from ase.io import write

    cfg = load_cfg()
    run_root = ensure_run_root(args.run_root)
    heartbeat(run_root, "seed-configs start")
    material = args.material
    params = build.material_params(material)
    frac = float(args.frac)
    shape = tuple(args.shape)
    morphologies = args.morphology or ["ordered", "random", "dendritic"]
    seed_root = run_root / "seed_configs" / material
    records = []

    for idx, morphology in enumerate(morphologies):
        net = build_network(morphology, frac, shape, params["a"], SEED + idx)
        slab = build.w_slab(nx_=shape[0], ny=shape[1], nz=shape[2], a=params["a"],
                            vacuum=float(args.vacuum), material=material)
        cfgs = build.make_configs(slab, net, surface_band=float(args.surface_band),
                                  material=material, a=params["a"])
        out_dir = seed_root / morphology
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "network.npz",
            coords=net.coords,
            mask=net.defect_mask,
            a=np.asarray(net.a),
            nx_shape=np.asarray(net.nx_shape, dtype=int),
        )
        for name in ("clean", "separated", "recombined"):
            write(out_dir / f"{name}.xyz", cfgs[name], format="extxyz")
        record = {
            "material": material,
            "morphology": morphology,
            "target_frac": frac,
            "shape": shape,
            "seed": SEED + idx,
            "defect_fraction_realized": net.frac,
            "defect_nodes": int(net.graph.number_of_nodes()),
            "defect_edges": int(net.graph.number_of_edges()),
            "vacancy_sites": int(len(cfgs["vac_idx"])),
            "configs": {
                name: mlip_al.geometry_summary(cfgs[name], material=material)
                for name in ("clean", "separated", "recombined")
            },
            "config_dir": out_dir,
        }
        mlip_al.write_json(out_dir / "seed_summary.json", record)
        records.append(record)

    manifest = {
        "state": "completed",
        "config": str(CONFIG),
        "material": material,
        "cfg_material_default": cfg.get("material"),
        "records": records,
        "exploratory": True,
    }
    mlip_al.write_json(run_root / "seed_manifest.json", manifest)
    update_status(run_root, "seed_configs", manifest)
    heartbeat(run_root, "seed-configs completed")
    print(run_root)
    return run_root


def run_mace_pilot(args) -> Path:
    from ase.io import read, write

    run_root = ensure_run_root(args.run_root)
    heartbeat(run_root, "mace-pilot start")
    seed_manifest = read_json(run_root / "seed_manifest.json")
    if not seed_manifest:
        raise SystemExit(f"missing seed manifest under {run_root}; run seed-configs first")
    calc, calc_meta = mlip_al.select_calculator(
        model_path=args.model,
        device=args.device,
        material=args.material,
        allow_toy=args.allow_toy,
    )
    records = []
    for seed_record in seed_manifest["records"]:
        morphology = seed_record["morphology"]
        cfg_dir = Path(seed_record["config_dir"])
        out_dir = run_root / "mace_pilot" / args.material / morphology
        out_dir.mkdir(parents=True, exist_ok=True)
        case = {
            "material": args.material,
            "morphology": morphology,
            "calculator": calc_meta,
            "observables": {},
            "relaxed": {},
            "anneal": {},
        }
        for name in ("clean", "separated", "recombined"):
            atoms = read(cfg_dir / f"{name}.xyz")
            obs = mlip_al.calculator_observables(atoms, calc)
            case["observables"][name] = obs
            relaxed = build.relax(
                atoms,
                calc,
                fmax=float(args.fmax),
                steps=int(args.relax_steps),
                lattice_a=build.material_params(args.material)["a"],
            )
            write(out_dir / f"relaxed_{name}.xyz", relaxed, format="extxyz")
            case["relaxed"][name] = {
                "geometry": mlip_al.geometry_summary(relaxed, material=args.material),
                "observables": mlip_al.calculator_observables(relaxed, calc),
                "file": out_dir / f"relaxed_{name}.xyz",
            }
        sep = read(out_dir / "relaxed_separated.xyz")
        md = mlip_al.short_langevin_sample(
            sep,
            calc,
            T_K=float(args.temperature_K),
            steps=int(args.md_steps),
            sample_interval=int(args.sample_interval),
            seed=SEED,
        )
        np.savez_compressed(
            out_dir / "anneal_sample.npz",
            positions=md["positions"],
            velocities=md["velocities"],
            masses=md["masses"],
            dt_fs=np.asarray(md["dt_fs"]),
        )
        write(out_dir / "annealed_separated.xyz", md["atoms"], format="extxyz")
        case["anneal"] = {
            "rows": md["rows"],
            "trajectory_file": out_dir / "anneal_sample.npz",
            "final_file": out_dir / "annealed_separated.xyz",
            "finite": bool(all(row.get("finite") for row in md["rows"])),
        }
        finite = [
            obs.get("finite", False)
            for obs in case["observables"].values()
        ] + [
            rec["observables"].get("finite", False)
            for rec in case["relaxed"].values()
        ] + [case["anneal"]["finite"]]
        case["gate"] = "pilot_passed" if all(finite) else "blocked"
        mlip_al.write_json(out_dir / "mace_pilot_summary.json", case)
        records.append(case)

    payload = {
        "state": "completed" if all(r["gate"] == "pilot_passed" for r in records) else "blocked",
        "calculator": calc_meta,
        "records": records,
        "promotion_ready": bool(calc_meta["production_ready"] and all(r["gate"] == "pilot_passed" for r in records)),
        "exploratory": not bool(calc_meta["production_ready"]),
    }
    mlip_al.write_json(run_root / "mace_pilot_summary.json", payload)
    update_status(run_root, "mace_pilot", payload)
    heartbeat(run_root, f"mace-pilot {payload['state']}")
    print(run_root)
    return run_root


def run_label_queue(args) -> Path:
    run_root = ensure_run_root(args.run_root)
    heartbeat(run_root, "label-queue start")
    pilot = read_json(run_root / "mace_pilot_summary.json")
    if not pilot:
        raise SystemExit(f"missing pilot summary under {run_root}; run mace-pilot first")
    items = []
    for record in pilot.get("records", []):
        if record.get("gate") != "pilot_passed":
            continue
        morphology = record["morphology"]
        for key in ("relaxed", "anneal"):
            if key == "relaxed":
                source = record[key]["separated"]["file"]
                label_kind = "relaxed_separated"
            else:
                source = record[key]["final_file"]
                label_kind = "annealed_separated"
            items.append(
                {
                    "material": record["material"],
                    "morphology": morphology,
                    "label_kind": label_kind,
                    "source_atoms": source,
                    "bigdft_state": "queued",
                    "production_note": "CPU BigDFT label only; GPU BigDFT remains rejected for production.",
                }
            )
    manifest = {
        "state": "completed",
        "items": items[: int(args.max_labels)],
        "n_items": min(len(items), int(args.max_labels)),
        "n_available": len(items),
        "execute_bigdft": False,
    }
    mlip_al.write_json(run_root / "label_queue_manifest.json", manifest)
    update_status(run_root, "label_queue", manifest)
    heartbeat(run_root, "label-queue completed")
    print(run_root)
    return run_root


def run_promote(args) -> Path:
    run_root = ensure_run_root(args.run_root)
    heartbeat(run_root, "promote start")
    pilot = read_json(run_root / "mace_pilot_summary.json", {})
    labels = read_json(run_root / "label_queue_manifest.json", {})
    promotion_ready = bool(pilot.get("promotion_ready"))
    commands = []
    if promotion_ready:
        commands = [
            "source env/activate_dgx.sh",
            f"python3 scripts/01_mlip_active_learning.py mace-pilot --run-root {run_root} --material {args.material} --relax-steps {args.production_relax_steps} --md-steps {args.production_md_steps}",
            f"python3 scripts/02_md_phonons_neb.py sed-pilot --run-root {run_root} --material {args.material} --md-steps {args.production_md_steps}",
        ]
    payload = {
        "state": "completed" if promotion_ready else "blocked",
        "promotion_ready": promotion_ready,
        "reason": None if promotion_ready else "MACE production calculator or pilot finite-check gate is not ready.",
        "label_queue_items": labels.get("n_items", 0),
        "production_commands": commands,
        "long_compute_launched": False,
    }
    mlip_al.write_json(run_root / "production_manifest.json", payload)
    update_status(run_root, "promotion", payload)
    heartbeat(run_root, f"promote {payload['state']}")
    print(run_root)
    return run_root


def run_all_pilot(args) -> int:
    run_root = ensure_run_root(args.run_root)
    args.run_root = str(run_root)
    run_env_smoke(args)
    run_seed_configs(args)
    run_mace_pilot(args)
    run_label_queue(args)
    run_promote(args)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", default=None)
    p.add_argument("--material", default="W", choices=["W", "Mo"])
    p.add_argument("--device", default=os.environ.get("ELECTRODEFECT_DEVICE", "cuda"))
    p.add_argument("--model", default="mace-mp-0")
    p.add_argument("--allow-toy", action="store_true", help="Use Lennard-Jones fallback for smoke plumbing only.")
    p.add_argument("--shape", nargs=3, type=int, default=[3, 3, 4])
    p.add_argument("--frac", type=float, default=0.25)
    p.add_argument("--morphology", action="append", default=None)
    p.add_argument("--vacuum", type=float, default=8.0)
    p.add_argument("--surface-band", type=float, default=5.0)
    p.add_argument("--relax-steps", type=int, default=5)
    p.add_argument("--fmax", type=float, default=0.20)
    p.add_argument("--temperature-K", type=float, default=1000.0)
    p.add_argument("--md-steps", type=int, default=20)
    p.add_argument("--sample-interval", type=int, default=5)
    p.add_argument("--max-labels", type=int, default=6)
    p.add_argument("--production-relax-steps", type=int, default=400)
    p.add_argument("--production-md-steps", type=int, default=2000)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env-smoke")
    sub.add_parser("seed-configs")
    sub.add_parser("mace-pilot")
    sub.add_parser("label-queue")
    sub.add_parser("promote")
    sub.add_parser("all-pilot")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.cmd == "env-smoke":
        run_env_smoke(args)
    elif args.cmd == "seed-configs":
        run_seed_configs(args)
    elif args.cmd == "mace-pilot":
        run_mace_pilot(args)
    elif args.cmd == "label-queue":
        run_label_queue(args)
    elif args.cmd == "promote":
        run_promote(args)
    elif args.cmd == "all-pilot":
        return run_all_pilot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
