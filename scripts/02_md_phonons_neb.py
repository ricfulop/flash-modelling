#!/usr/bin/env python3
"""Phase 2 MD/phonon/NEB staging launcher."""
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
from electrodefect import build, mlip_al, percolation


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
SEED = 20260603


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_root(prefix: str = "phase2_md") -> Path:
    label = os.environ.get("ELECTRODEFECT_PHASE2_LABEL", socket.gethostname())
    return RUNS / datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_{label}")


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
    status_path = run_root / "md_status.json"
    status = read_json(status_path, {"state": "running", "run_root": str(run_root), "steps": {}})
    status["updated_at"] = now()
    status["host"] = socket.gethostname()
    status["steps"][key] = payload
    if all(step.get("state") in {"completed", "blocked"} for step in status["steps"].values()):
        status["state"] = "completed"
    mlip_al.write_json(status_path, status)


def material_payload(run_root: Path, summary_name: str, material: str, payload: dict) -> dict:
    """Merge per-material summaries without losing W when Mo runs in the same root."""
    summary_path = run_root / summary_name
    summary = read_json(summary_path, {"materials": {}})
    summary.setdefault("materials", {})[material] = payload
    states = [entry.get("state") for entry in summary["materials"].values()]
    summary["state"] = "completed" if states and all(state == "completed" for state in states) else "blocked"
    summary["promotion_ready"] = bool(
        summary["materials"]
        and all(entry.get("promotion_ready") for entry in summary["materials"].values())
    )
    summary["exploratory"] = bool(any(entry.get("exploratory") for entry in summary["materials"].values()))
    summary["latest_material"] = material
    return summary


def seed_atoms(material: str, morphology: str = "ordered"):
    """Create a tiny separated pilot cell if no MLIP seed artifacts exist yet."""
    params = build.material_params(material)
    net = percolation.ordered_superlattice(nx_=3, ny=3, nz=4, a=params["a"], frac=0.25, seed=SEED)
    if morphology == "random":
        net = percolation.random_percolation(nx_=3, ny=3, nz=4, a=params["a"], frac=0.25, seed=SEED)
    elif morphology in {"dendritic", "dendrite"}:
        net = percolation.dla_dendrite(nx_=3, ny=3, nz=4, a=params["a"], target_frac=0.25, seed=SEED)
    slab = build.w_slab(nx_=3, ny=3, nz=4, a=params["a"], vacuum=8.0, material=material)
    return build.make_configs(slab, net, surface_band=5.0, material=material, a=params["a"])["separated"]


def load_pilot_atoms(run_root: Path, material: str, morphology: str):
    from ase.io import read
    candidates = [
        run_root / "mace_pilot" / material / morphology / "annealed_separated.xyz",
        run_root / "mace_pilot" / material / morphology / "relaxed_separated.xyz",
        run_root / "seed_configs" / material / morphology / "separated.xyz",
    ]
    for path in candidates:
        if path.exists():
            return read(path), path
    return seed_atoms(material, morphology), None


def make_neb_endpoint(atoms, material: str, displacement: float):
    """Move a mobile near-surface atom toward a lateral neighbour/vacancy-like site."""
    params = build.material_params(material)
    initial = mlip_al.apply_bottom_constraint(atoms, lattice_a=params["a"])
    final = initial.copy()
    fixed = mlip_al.fixed_atom_mask(initial)
    positions = final.get_positions()
    mobile_candidates = np.flatnonzero(~fixed)
    if len(mobile_candidates) == 0:
        raise ValueError("NEB endpoint generation has no mobile atoms")
    z = positions[:, 2]
    top_mobile = mobile_candidates[np.argmax(z[mobile_candidates])]
    direction = np.array([1.0, 1.0, 0.0], dtype=float)
    direction /= np.linalg.norm(direction)
    positions[top_mobile] = positions[top_mobile] + float(displacement) * direction
    final.set_positions(positions)
    final.set_constraint(initial.constraints)
    return initial, final, int(top_mobile)


def run_neb_pilot(args) -> Path:
    from ase.io import write

    run_root = ensure_run_root(args.run_root)
    heartbeat(run_root, "neb-pilot start")
    calc, calc_meta = mlip_al.select_calculator(
        model_path=args.model,
        device=args.device,
        material=args.material,
        allow_toy=args.allow_toy,
    )
    records = []
    params = build.material_params(args.material)
    for morphology in args.morphology:
        atoms, source = load_pilot_atoms(run_root, args.material, morphology)
        relax_summary = None
        if int(args.pre_relax_steps) > 0:
            atoms, relax_summary = mlip_al.relax_pilot_structure(
                atoms,
                calc,
                lattice_a=params["a"],
                fmax=float(args.pre_relax_fmax),
                steps=int(args.pre_relax_steps),
            )
        initial, final, mobile = make_neb_endpoint(atoms, args.material, float(args.displacement))
        out_dir = run_root / "neb_pilot" / args.material / morphology
        out_dir.mkdir(parents=True, exist_ok=True)
        write(out_dir / "initial.xyz", initial, format="extxyz")
        write(out_dir / "final.xyz", final, format="extxyz")
        try:
            e_initial = mlip_al.calculator_observables(initial, calc)["energy_eV"]
            e_final = mlip_al.calculator_observables(final, calc)["energy_eV"]
            barrier, energies = mlip_al.neb_hop_barrier(
                initial,
                final,
                calc,
                n_images=int(args.n_images),
                fmax=float(args.fmax),
                steps=int(args.neb_steps),
            )
            finite = bool(np.isfinite(barrier) and np.isfinite(energies).all())
            endpoint_delta = float(e_final - e_initial)
            record = {
                "state": "completed" if finite else "blocked",
                "material": args.material,
                "morphology": morphology,
                "source_atoms": source,
                "mobile_atom_index": mobile,
                "displacement_A": args.displacement,
                "endpoint_delta_eV": endpoint_delta,
                "endpoint_energies_eV": [e_initial, e_final],
                "barrier_eV": barrier,
                "image_energies_eV": energies,
                "pre_relax": relax_summary,
                "finite": finite,
                "calculator": calc_meta,
            }
        except Exception as exc:
            record = {
                "state": "blocked",
                "material": args.material,
                "morphology": morphology,
                "source_atoms": source,
                "error": f"{type(exc).__name__}: {exc}",
                "calculator": calc_meta,
            }
        mlip_al.write_json(out_dir / "neb_pilot_summary.json", record)
        records.append(record)
    material_summary = {
        "state": "completed" if all(r["state"] == "completed" for r in records) else "blocked",
        "material": args.material,
        "records": records,
        "promotion_ready": bool(calc_meta["production_ready"] and all(r["state"] == "completed" for r in records)),
        "exploratory": not bool(calc_meta["production_ready"]),
    }
    payload = material_payload(run_root, "neb_pilot_summary.json", args.material, material_summary)
    mlip_al.write_json(run_root / "neb_pilot" / args.material / "neb_pilot_summary.json", material_summary)
    mlip_al.write_json(run_root / "neb_pilot_summary.json", payload)
    update_status(run_root, "neb_pilot", payload)
    heartbeat(run_root, f"neb-pilot {material_summary['state']}")
    print(run_root)
    return run_root


def sed_summary(sed: dict) -> list[dict]:
    rows = []
    for k, (omega, spec) in sed.items():
        peak = int(np.nanargmax(spec))
        rows.append(
            {
                "k": k,
                "peak_omega_rad_s": float(omega[peak]),
                "peak_power": float(spec[peak]),
                "finite": bool(np.isfinite(omega).all() and np.isfinite(spec).all()),
            }
        )
    return rows


def run_sed_pilot(args) -> Path:
    from ase.io import write

    run_root = ensure_run_root(args.run_root)
    heartbeat(run_root, "sed-pilot start")
    calc, calc_meta = mlip_al.select_calculator(
        model_path=args.model,
        device=args.device,
        material=args.material,
        allow_toy=args.allow_toy,
    )
    records = []
    params = build.material_params(args.material)
    kpoints = [
        np.array([0.05, 0.0, 0.0]),
        np.array([0.0, 0.05, 0.0]),
        np.array([0.0, 0.0, 0.05]),
    ]
    for morphology in args.morphology:
        atoms, source = load_pilot_atoms(run_root, args.material, morphology)
        relax_summary = None
        if int(args.pre_relax_steps) > 0:
            atoms, relax_summary = mlip_al.relax_pilot_structure(
                atoms,
                calc,
                lattice_a=params["a"],
                fmax=float(args.pre_relax_fmax),
                steps=int(args.pre_relax_steps),
            )
        out_dir = run_root / "sed_pilot" / args.material / morphology
        out_dir.mkdir(parents=True, exist_ok=True)
        md = mlip_al.short_langevin_sample(
            atoms,
            calc,
            T_K=float(args.temperature_K),
            steps=int(args.md_steps),
            sample_interval=int(args.sample_interval),
            dt_fs=float(args.dt_fs),
            friction=float(args.friction),
            seed=SEED,
            lattice_a=params["a"],
        )
        sed = mlip_al.spectral_energy_density(
            md["velocities"],
            md["atoms"].get_positions(),
            kpoints,
            md["dt_fs"],
            md["masses"],
        )
        rows = sed_summary(sed)
        np.savez_compressed(
            out_dir / "sed_pilot_arrays.npz",
            positions=md["positions"],
            velocities=md["velocities"],
            masses=md["masses"],
            dt_fs=np.asarray(md["dt_fs"]),
        )
        write(out_dir / "sed_final.xyz", md["atoms"], format="extxyz")
        max_disp = max((row.get("max_displacement_A", 0.0) for row in md["rows"]), default=0.0)
        max_force = max((row.get("max_force_eV_A", 0.0) for row in md["rows"]), default=0.0)
        peak_power = max((row.get("peak_power", 0.0) for row in rows), default=0.0)
        finite = bool(
            all(r["finite"] for r in rows)
            and all(r.get("finite") for r in md["rows"])
            and max_disp > 1.0e-5
            and max_disp <= float(args.max_displacement_A)
            and max_force <= float(args.max_force_eV_A)
            and peak_power > 0.0
        )
        record = {
            "state": "completed" if finite else "blocked",
            "material": args.material,
            "morphology": morphology,
            "source_atoms": source,
            "calculator": calc_meta,
            "pre_relax": relax_summary,
            "md_rows": md["rows"],
            "sed_rows": rows,
            "max_displacement_A": max_disp,
            "max_force_eV_A": max_force,
            "max_sed_peak_power": peak_power,
            "stability_gates": {
                "max_displacement_A": float(args.max_displacement_A),
                "max_force_eV_A": float(args.max_force_eV_A),
            },
            "finite": finite,
        }
        mlip_al.write_json(out_dir / "sed_pilot_summary.json", record)
        records.append(record)
    material_summary = {
        "state": "completed" if all(r["state"] == "completed" for r in records) else "blocked",
        "material": args.material,
        "records": records,
        "promotion_ready": bool(calc_meta["production_ready"] and all(r["state"] == "completed" for r in records)),
        "exploratory": not bool(calc_meta["production_ready"]),
    }
    payload = material_payload(run_root, "sed_pilot_summary.json", args.material, material_summary)
    mlip_al.write_json(run_root / "sed_pilot" / args.material / "sed_pilot_summary.json", material_summary)
    mlip_al.write_json(run_root / "sed_pilot_summary.json", payload)
    update_status(run_root, "sed_pilot", payload)
    heartbeat(run_root, f"sed-pilot {material_summary['state']}")
    print(run_root)
    return run_root


def run_promote(args) -> Path:
    run_root = ensure_run_root(args.run_root)
    heartbeat(run_root, "md promote start")
    neb = read_json(run_root / "neb_pilot_summary.json", {})
    sed = read_json(run_root / "sed_pilot_summary.json", {})
    ready = bool(neb.get("promotion_ready") and sed.get("promotion_ready"))
    commands = []
    if ready:
        commands = [
            "source env/activate_dgx.sh",
            f"python3 scripts/02_md_phonons_neb.py neb-pilot --run-root {run_root} --material {args.material} --neb-steps {args.production_neb_steps}",
            f"python3 scripts/02_md_phonons_neb.py sed-pilot --run-root {run_root} --material {args.material} --md-steps {args.production_md_steps}",
        ]
    payload = {
        "state": "completed" if ready else "blocked",
        "promotion_ready": ready,
        "reason": None if ready else "NEB and/or SED pilot did not pass with a production MACE calculator.",
        "production_commands": commands,
        "long_compute_launched": False,
    }
    mlip_al.write_json(run_root / "md_production_manifest.json", payload)
    update_status(run_root, "promotion", payload)
    heartbeat(run_root, f"md promote {payload['state']}")
    print(run_root)
    return run_root


def run_all_pilot(args) -> int:
    run_root = ensure_run_root(args.run_root)
    args.run_root = str(run_root)
    run_neb_pilot(args)
    run_sed_pilot(args)
    run_promote(args)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", default=None)
    p.add_argument("--material", default="W", choices=["W", "Mo"])
    p.add_argument("--device", default=os.environ.get("ELECTRODEFECT_DEVICE", "cuda"))
    p.add_argument("--model", default="mace-mp-0")
    p.add_argument("--allow-toy", action="store_true", help="Use Lennard-Jones fallback for smoke plumbing only.")
    p.add_argument("--morphology", action="append", default=None)
    p.add_argument("--temperature-K", type=float, default=1000.0)
    p.add_argument("--md-steps", type=int, default=20)
    p.add_argument("--sample-interval", type=int, default=5)
    p.add_argument("--dt-fs", type=float, default=0.25)
    p.add_argument("--friction", type=float, default=0.05)
    p.add_argument("--n-images", type=int, default=3)
    p.add_argument("--neb-steps", type=int, default=5)
    p.add_argument("--fmax", type=float, default=1.0)
    p.add_argument("--displacement", type=float, default=0.8)
    p.add_argument("--pre-relax-steps", type=int, default=50)
    p.add_argument("--pre-relax-fmax", type=float, default=1.0)
    p.add_argument("--max-displacement-A", type=float, default=10.0)
    p.add_argument("--max-force-eV-A", type=float, default=2000.0)
    p.add_argument("--production-md-steps", type=int, default=2000)
    p.add_argument("--production-neb-steps", type=int, default=100)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("neb-pilot")
    sub.add_parser("sed-pilot")
    sub.add_parser("promote")
    sub.add_parser("all-pilot")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.morphology is None:
        args.morphology = ["ordered", "random", "dendritic"]
    if args.cmd == "neb-pilot":
        run_neb_pilot(args)
    elif args.cmd == "sed-pilot":
        run_sed_pilot(args)
    elif args.cmd == "promote":
        run_promote(args)
    elif args.cmd == "all-pilot":
        return run_all_pilot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
