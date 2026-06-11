#!/usr/bin/env python3
"""Repair/relax the frozen W/Mo Frenkel-pair candidate pairs.

This is the first confirmatory execution step after the timestamp freeze in:
docs/science-superpowers/preregistrations/2026-06-07-matterchat-first-electrodefect-freeze-note.md

The script reads the fixed CIFs from the geometry-QC run, relaxes separated and
recombined structures with a selected MLIP, and classifies each material /
morphology pair for DFT promotion. MLIP output is screening only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import FIRE

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from electrodefect import build, chgnet_field, mlip_al  # noqa: E402

SEED = 20260607
DEFAULT_INPUT_ROOT = REPO / "runs" / "matterchat_geometry_qc_20260607_022909" / "matterchat_inputs"
MORPHOLOGIES = ("ordered", "random", "dendritic")
MATERIALS = ("W", "Mo")


@dataclass
class GeometryCheck:
    n_atoms: int
    finite_positions: bool
    min_distance_a: float | None
    min_pair: list[int] | None
    z_min_a: float | None
    z_max_a: float | None
    atom_count_ok: bool
    no_vacuum_collapse: bool
    pass_geometry: bool
    notes: list[str]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n")
    tmp.replace(path)


def min_distance(atoms) -> tuple[float | None, list[int] | None]:
    if len(atoms) < 2:
        return None, None
    try:
        distances = atoms.get_all_distances(mic=bool(any(atoms.pbc)))
    except Exception:
        distances = atoms.get_all_distances(mic=False)
    distances[distances <= 1.0e-12] = np.inf
    idx = np.unravel_index(np.argmin(distances), distances.shape)
    value = float(distances[idx])
    if not math.isfinite(value):
        return None, None
    return value, [int(idx[0]), int(idx[1])]


def geometry_check(atoms, reference_atoms, min_distance_a: float, allow_dumbbell: bool) -> GeometryCheck:
    pos = np.asarray(atoms.get_positions(), dtype=float)
    dmin, pair = min_distance(atoms)
    finite = bool(np.isfinite(pos).all())
    notes: list[str] = []
    atom_count_ok = len(atoms) == len(reference_atoms)
    if not atom_count_ok:
        notes.append(f"atom count changed from {len(reference_atoms)} to {len(atoms)}")
    z_min = float(pos[:, 2].min()) if len(pos) else None
    z_max = float(pos[:, 2].max()) if len(pos) else None
    ref_pos = np.asarray(reference_atoms.get_positions(), dtype=float)
    ref_z_min = float(ref_pos[:, 2].min()) if len(ref_pos) else 0.0
    ref_z_max = float(ref_pos[:, 2].max()) if len(ref_pos) else 0.0
    no_vacuum_collapse = z_min is not None and z_max is not None and z_min >= ref_z_min - 2.0 and z_max <= ref_z_max + 2.0
    if not no_vacuum_collapse:
        notes.append("z extent moved more than 2 A outside starting slab range")
    distance_ok = dmin is not None and (dmin >= min_distance_a or allow_dumbbell)
    if dmin is None:
        notes.append("minimum distance unavailable")
    elif dmin < min_distance_a and allow_dumbbell:
        notes.append(f"minimum distance {dmin:.3f} A below threshold but allowed as dumbbell-screening case")
    elif dmin < min_distance_a:
        notes.append(f"minimum distance {dmin:.3f} A below threshold {min_distance_a:.3f} A")
    pass_geometry = bool(finite and atom_count_ok and no_vacuum_collapse and distance_ok)
    return GeometryCheck(
        n_atoms=len(atoms),
        finite_positions=finite,
        min_distance_a=round(dmin, 6) if dmin is not None else None,
        min_pair=pair,
        z_min_a=round(z_min, 6) if z_min is not None else None,
        z_max_a=round(z_max, 6) if z_max is not None else None,
        atom_count_ok=atom_count_ok,
        no_vacuum_collapse=bool(no_vacuum_collapse),
        pass_geometry=pass_geometry,
        notes=notes,
    )


def apply_bottom_constraint(atoms, material: str):
    constrained = atoms.copy()
    params = build.material_params(material)
    mask = build.bottom_contact_mask(constrained, lattice_a=params["a"], max_fixed_fraction=0.35)
    if mask.any():
        constrained.set_constraint(FixAtoms(mask=mask))
    return constrained


def make_calculator(kind: str, material: str, device: str, allow_toy: bool):
    if kind == "chgnet":
        return chgnet_field.make_chgnet_calculator(device=device), {
            "kind": "chgnet",
            "device": device,
            "production_ready": True,
        }
    if kind == "mace":
        calc, meta = mlip_al.select_calculator(
            model_path="mace-mp-0",
            device=device,
            material=material,
            allow_toy=allow_toy,
        )
        return calc, meta
    if kind == "toy":
        calc, meta = mlip_al.select_calculator(
            model_path="mace-mp-0",
            device="cpu",
            material=material,
            allow_toy=True,
        )
        meta["forced_toy"] = True
        return calc, meta
    raise ValueError(f"unknown calculator kind: {kind}")


def relax_atoms(atoms, calc, fmax: float, steps: int, maxstep: float, logfile: Path):
    relaxed = atoms.copy()
    relaxed.calc = calc
    opt = FIRE(relaxed, logfile=str(logfile), maxstep=maxstep)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    energy = float(relaxed.get_potential_energy())
    forces = np.asarray(relaxed.get_forces(), dtype=float)
    fmag = np.linalg.norm(forces, axis=1) if len(forces) else np.array([])
    return relaxed, {
        "converged": converged,
        "energy_eV": energy,
        "max_force_eV_A": float(fmag.max()) if len(fmag) else None,
        "mean_force_eV_A": float(fmag.mean()) if len(fmag) else None,
        "finite": bool(np.isfinite(energy) and np.isfinite(forces).all()),
    }


def run_pair(args, run_root: Path, material: str, morphology: str) -> dict:
    pair_dir = run_root / "pairs" / material / morphology
    pair_dir.mkdir(parents=True, exist_ok=True)
    calc, calc_meta = make_calculator(args.calculator, material, args.device, args.allow_toy)
    record = {
        "material": material,
        "morphology": morphology,
        "state": "running",
        "calculator": calc_meta,
        "inputs": {},
        "structures": {},
        "started_at": now(),
    }
    write_json(pair_dir / "pair_status.json", record)
    source_paths = {
        "separated": args.input_root / f"{material}_{morphology}_separated.cif",
        "recombined": args.input_root / f"{material}_{morphology}_recombined.cif",
    }
    for config_name, source in source_paths.items():
        record["inputs"][config_name] = str(source.relative_to(REPO) if source.is_relative_to(REPO) else source)
        atoms0 = read(source)
        atoms = apply_bottom_constraint(atoms0, material=material)
        initial_check = geometry_check(
            atoms,
            atoms0,
            min_distance_a=args.min_distance,
            allow_dumbbell=(config_name == "separated" and args.allow_dumbbell),
        )
        try:
            relaxed, obs = relax_atoms(
                atoms,
                calc,
                fmax=args.fmax,
                steps=args.steps,
                maxstep=args.maxstep,
                logfile=pair_dir / f"{config_name}_fire.log",
            )
            out_xyz = pair_dir / f"relaxed_{config_name}.xyz"
            out_cif = pair_dir / f"relaxed_{config_name}.cif"
            write(out_xyz, relaxed, format="extxyz")
            write(out_cif, relaxed, format="cif")
            final_check = geometry_check(
                relaxed,
                atoms0,
                min_distance_a=args.min_distance,
                allow_dumbbell=(config_name == "separated" and args.allow_dumbbell),
            )
            state = "completed" if obs["finite"] and obs["converged"] and final_check.pass_geometry else "blocked"
            record["structures"][config_name] = {
                "state": state,
                "initial_geometry": asdict(initial_check),
                "final_geometry": asdict(final_check),
                "observables": obs,
                "relaxed_xyz": str(out_xyz.relative_to(REPO)),
                "relaxed_cif": str(out_cif.relative_to(REPO)),
            }
        except Exception as exc:
            record["structures"][config_name] = {
                "state": "blocked",
                "initial_geometry": asdict(initial_check),
                "error": f"{type(exc).__name__}: {exc}",
            }
    states = [entry["state"] for entry in record["structures"].values()]
    record["state"] = "promote_to_dft" if states and all(state == "completed" for state in states) else "blocked_geometry"
    record["finished_at"] = now()
    write_json(pair_dir / "pair_status.json", record)
    return record


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--materials", nargs="+", default=list(MATERIALS), choices=list(MATERIALS))
    p.add_argument("--morphologies", nargs="+", default=list(MORPHOLOGIES), choices=list(MORPHOLOGIES))
    p.add_argument("--calculator", choices=["chgnet", "mace", "toy"], default="chgnet")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fmax", type=float, default=0.08)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--maxstep", type=float, default=0.03)
    p.add_argument("--min-distance", type=float, default=1.8)
    p.add_argument("--allow-dumbbell", action="store_true", help="Allow separated dumbbell-like close contacts below min distance.")
    p.add_argument("--allow-toy", action="store_true", help="Permit toy fallback for plumbing only.")
    p.add_argument("--max-pairs", type=int, default=0)
    return p


def main() -> int:
    args = parser().parse_args()
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    np.random.seed(SEED)
    run_root = args.run_root or REPO / "runs" / datetime.now().strftime("matterchat_geometry_repair_%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "state": "running",
        "run_root": str(run_root),
        "host": socket.gethostname(),
        "started_at": now(),
        "seed": SEED,
        "freeze_note": "docs/science-superpowers/preregistrations/2026-06-07-matterchat-first-electrodefect-freeze-note.md",
        "input_root": str(args.input_root),
        "parameters": {
            "materials": args.materials,
            "morphologies": args.morphologies,
            "calculator": args.calculator,
            "device": args.device,
            "fmax": args.fmax,
            "steps": args.steps,
            "maxstep": args.maxstep,
            "min_distance": args.min_distance,
            "allow_dumbbell": args.allow_dumbbell,
        },
        "pairs": [],
    }
    write_json(run_root / "repair_manifest.json", manifest)
    count = 0
    for material in args.materials:
        for morphology in args.morphologies:
            if args.max_pairs and count >= args.max_pairs:
                break
            pair = run_pair(args, run_root, material, morphology)
            manifest["pairs"].append(pair)
            manifest["updated_at"] = now()
            write_json(run_root / "repair_manifest.json", manifest)
            count += 1
        if args.max_pairs and count >= args.max_pairs:
            break
    states = [pair["state"] for pair in manifest["pairs"]]
    manifest["state"] = "completed" if states and all(state == "promote_to_dft" for state in states) else "completed_with_blocks"
    manifest["finished_at"] = now()
    manifest["promotion_ready_pairs"] = [
        {"material": pair["material"], "morphology": pair["morphology"]}
        for pair in manifest["pairs"]
        if pair["state"] == "promote_to_dft"
    ]
    write_json(run_root / "repair_manifest.json", manifest)
    print(run_root)
    print(json.dumps({"state": manifest["state"], "promotion_ready_pairs": manifest["promotion_ready_pairs"]}, indent=2))
    return 0 if manifest["promotion_ready_pairs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
