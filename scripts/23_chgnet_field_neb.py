#!/usr/bin/env python3
"""Preregistered CHGNet field-wrapper NEB runner for near-surface W."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from electrodefect import build, mlip_al, percolation  # noqa: E402
from electrodefect.chgnet_field import (  # noqa: E402
    ElectricFieldCalculator,
    ZeroCalculator,
    make_chgnet_calculator,
    mobile_atom_charges,
    normalized_vector,
)


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
SEED = 20260603

PRIMARY = {
    "material": "W",
    "morphology": "ordered",
    "shape": [3, 3, 4],
    "frac": 0.25,
    "vacuum_A": 8.0,
    "surface_band_A": 5.0,
    "displacement_A": 0.45,
    "endpoint_direction": [1.0, 1.0, 0.0],
    "field_strength_V_A": 0.2,
    "mobile_charge_e": 1.0,
    "n_intermediate_images": 5,
    "fmax_eV_A": 0.05,
    "neb_steps": 60,
    "min_pair_distance_A": 1.5,
    "max_adjacent_mobile_displacement_A": 1.0,
    "positive_threshold_eV": -0.05,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_root() -> Path:
    label = os.environ.get("ELECTRODEFECT_CHGNET_LABEL", socket.gethostname())
    return RUNS / datetime.now().strftime(f"chgnet_field_neb_%Y%m%d_%H%M%S_{label}")


def ensure_run_root(path: str | None) -> Path:
    root = Path(path) if path else default_run_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def heartbeat(run_root: Path, message: str) -> None:
    with (run_root / "heartbeat.log").open("a") as fh:
        fh.write(f"{now()} {message}\n")


def make_seed_atoms():
    material = PRIMARY["material"]
    params = build.material_params(material)
    nx_, ny, nz = PRIMARY["shape"]
    net = percolation.ordered_superlattice(
        nx_=nx_,
        ny=ny,
        nz=nz,
        a=params["a"],
        frac=PRIMARY["frac"],
        seed=SEED,
    )
    slab = build.w_slab(
        nx_=nx_,
        ny=ny,
        nz=nz,
        a=params["a"],
        vacuum=PRIMARY["vacuum_A"],
        material=material,
    )
    cfgs = build.make_configs(
        slab,
        net,
        surface_band=PRIMARY["surface_band_A"],
        material=material,
        a=params["a"],
    )
    return cfgs["separated"], net


def build_endpoints(run_root: Path) -> dict:
    from ase.io import write

    heartbeat(run_root, "build-endpoints start")
    atoms, net = make_seed_atoms()
    params = build.material_params(PRIMARY["material"])
    initial = mlip_al.apply_bottom_constraint(atoms, lattice_a=params["a"])
    final = initial.copy()
    fixed = mlip_al.fixed_atom_mask(initial)
    positions = final.get_positions()
    mobile_candidates = np.flatnonzero(~fixed)
    if len(mobile_candidates) == 0:
        raise ValueError("endpoint generation has no mobile atoms")
    z = positions[:, 2]
    mobile_atom_index = int(mobile_candidates[np.argmax(z[mobile_candidates])])
    direction = normalized_vector(PRIMARY["endpoint_direction"])
    displacement = float(PRIMARY["displacement_A"]) * direction
    positions[mobile_atom_index] += displacement
    final.set_positions(positions)
    final.set_constraint(initial.constraints)

    endpoint_dir = run_root / "endpoints"
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    write(endpoint_dir / "initial.xyz", initial, format="extxyz")
    write(endpoint_dir / "final.xyz", final, format="extxyz")

    charges = mobile_atom_charges(
        len(initial),
        mobile_atom_index,
        charge_e=PRIMARY["mobile_charge_e"],
    )
    manifest = {
        "state": "completed",
        "created_at": now(),
        "primary": PRIMARY,
        "seed": SEED,
        "defect_fraction_realized": float(net.frac),
        "atom_count": len(initial),
        "fixed_atom_count": int(fixed.sum()),
        "mobile_atom_index": mobile_atom_index,
        "mobile_atom_symbol": initial[mobile_atom_index].symbol,
        "mobile_initial_position_A": initial.get_positions()[mobile_atom_index],
        "mobile_final_position_A": final.get_positions()[mobile_atom_index],
        "displacement_vector_A": displacement,
        "field_direction": direction,
        "charge_nonzero_indices": np.flatnonzero(charges),
        "charge_nonzero_values_e": charges[np.flatnonzero(charges)],
        "initial_path": endpoint_dir / "initial.xyz",
        "final_path": endpoint_dir / "final.xyz",
    }
    manifest["endpoint_qc"] = validate_endpoints(initial, final, mobile_atom_index, fixed)
    write_json(run_root / "endpoint_manifest.json", manifest)
    heartbeat(run_root, "build-endpoints completed")
    return manifest


def validate_endpoints(initial, final, mobile_atom_index: int, fixed_mask: np.ndarray) -> dict:
    displacement = np.linalg.norm(
        final.get_positions()[mobile_atom_index] - initial.get_positions()[mobile_atom_index]
    )
    fixed_unchanged = True
    if fixed_mask.any():
        fixed_unchanged = bool(
            np.allclose(initial.get_positions()[fixed_mask], final.get_positions()[fixed_mask])
        )
    qc = {
        "finite_initial": bool(np.isfinite(initial.get_positions()).all()),
        "finite_final": bool(np.isfinite(final.get_positions()).all()),
        "same_atom_count": len(initial) == len(final),
        "same_cell": bool(np.allclose(initial.cell.array, final.cell.array)),
        "mobile_displacement_A": float(displacement),
        "mobile_displacement_ok": bool(abs(displacement - PRIMARY["displacement_A"]) <= 1e-6),
        "min_pair_initial_A": mlip_al.min_pair_distance(initial),
        "min_pair_final_A": mlip_al.min_pair_distance(final),
        "fixed_atoms_unchanged": fixed_unchanged,
    }
    qc["passes"] = bool(
        qc["finite_initial"]
        and qc["finite_final"]
        and qc["same_atom_count"]
        and qc["same_cell"]
        and qc["mobile_displacement_ok"]
        and qc["min_pair_initial_A"] > PRIMARY["min_pair_distance_A"]
        and qc["min_pair_final_A"] > PRIMARY["min_pair_distance_A"]
        and qc["fixed_atoms_unchanged"]
    )
    return qc


def run_setup_smoke(run_root: Path, device: str) -> dict:
    import importlib.metadata as md
    import torch

    heartbeat(run_root, "setup-smoke start")
    calc = make_chgnet_calculator(device=device)
    payload = {
        "state": "completed",
        "created_at": now(),
        "purpose": "setup_smoke_only",
        "device_requested": device,
        "packages": {pkg: md.version(pkg) for pkg in ["chgnet", "ase", "pymatgen", "numpy", "torch"]},
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()),
        "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "calculator": type(calc).__name__,
    }
    write_json(run_root / "setup_smoke.json", payload)
    heartbeat(run_root, "setup-smoke completed")
    return payload


def validate_wrapper(run_root: Path) -> dict:
    from ase import Atoms

    heartbeat(run_root, "wrapper-validation start")
    atoms = Atoms("WW", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.calc = ElectricFieldCalculator(
        ZeroCalculator(),
        charges=np.array([1.0, -1.0]),
        field_strength=0.2,
        direction=(1.0, 0.0, 0.0),
    )
    forces = atoms.get_forces()
    expected = np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])

    atoms_zero = Atoms("WW", positions=[[0, 0, 0], [1, 0.5, 0]])
    atoms_zero.calc = ElectricFieldCalculator(
        ZeroCalculator(),
        charges=np.zeros(2),
        field_strength=0.2,
        direction=(1.0, 0.0, 0.0),
    )
    zero_energy = float(atoms_zero.get_potential_energy())
    zero_forces = atoms_zero.get_forces()

    payload = {
        "state": "completed",
        "force_sign_shape_ok": bool(forces.shape == (2, 3) and np.allclose(forces, expected)),
        "zero_charge_energy_eV": zero_energy,
        "zero_charge_forces": zero_forces,
        "zero_charge_identity_ok": bool(zero_energy == 0.0 and np.allclose(zero_forces, 0.0)),
    }
    payload["passes"] = bool(payload["force_sign_shape_ok"] and payload["zero_charge_identity_ok"])
    write_json(run_root / "validation_summary.json", payload)
    heartbeat(run_root, f"wrapper-validation {'passed' if payload['passes'] else 'failed'}")
    return payload


def load_endpoints(run_root: Path):
    from ase.io import read

    manifest_path = run_root / "endpoint_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing endpoint manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    initial = read(manifest["initial_path"])
    final = read(manifest["final_path"])
    return initial, final, manifest


def make_images(initial, final):
    images = [initial.copy()]
    images.extend(initial.copy() for _ in range(PRIMARY["n_intermediate_images"]))
    images.append(final.copy())
    return images


def run_neb(images, calc, out_dir: Path):
    from ase.io import write
    from ase.mep import NEB
    from ase.optimize import FIRE

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        neb = NEB(images, allow_shared_calculator=True)
    except TypeError:
        neb = NEB(images)
    neb.interpolate()
    for image in images:
        image.calc = calc
    opt = FIRE(neb, logfile=str(out_dir / "optimizer.log"))
    opt.run(fmax=PRIMARY["fmax_eV_A"], steps=PRIMARY["neb_steps"])
    energies = [float(image.get_potential_energy()) for image in images]
    barrier = float(max(energies) - energies[0])
    for idx, image in enumerate(images):
        write(out_dir / f"image_{idx:02d}.xyz", image, format="extxyz")
    return {
        "state": "completed",
        "barrier_eV": barrier,
        "image_energies_eV": energies,
        "finite": bool(np.isfinite(barrier) and np.isfinite(energies).all()),
        "image_paths": [out_dir / f"image_{idx:02d}.xyz" for idx in range(len(images))],
    }


def image_qc(images, mobile_atom_index: int) -> dict:
    min_pairs = [mlip_al.min_pair_distance(image) for image in images]
    finite_positions = [bool(np.isfinite(image.get_positions()).all()) for image in images]
    mobile_positions = np.array([image.get_positions()[mobile_atom_index] for image in images])
    adjacent_mobile = np.linalg.norm(np.diff(mobile_positions, axis=0), axis=1)
    atom_counts = [len(image) for image in images]
    cells_match = [bool(np.allclose(images[0].cell.array, image.cell.array)) for image in images]
    qc = {
        "finite_positions_all": bool(all(finite_positions)),
        "min_pair_distances_A": min_pairs,
        "min_pair_pass": bool(min(min_pairs) > PRIMARY["min_pair_distance_A"]),
        "adjacent_mobile_displacements_A": adjacent_mobile,
        "adjacent_mobile_pass": bool(
            len(adjacent_mobile) == 0
            or np.nanmax(adjacent_mobile) < PRIMARY["max_adjacent_mobile_displacement_A"]
        ),
        "same_atom_count_all": bool(len(set(atom_counts)) == 1),
        "same_cell_all": bool(all(cells_match)),
    }
    qc["passes"] = bool(
        qc["finite_positions_all"]
        and qc["min_pair_pass"]
        and qc["adjacent_mobile_pass"]
        and qc["same_atom_count_all"]
        and qc["same_cell_all"]
    )
    return qc


def run_primary_nebs(run_root: Path, device: str) -> dict:
    heartbeat(run_root, "run-neb start")
    initial, final, manifest = load_endpoints(run_root)
    mobile = int(manifest["mobile_atom_index"])

    no_field_images = make_images(initial, final)
    no_field = run_neb(
        no_field_images,
        make_chgnet_calculator(device=device),
        run_root / "no_field",
    )
    no_field["image_qc"] = image_qc(no_field_images, mobile)
    write_json(run_root / "no_field" / "neb_summary.json", no_field)

    charges = mobile_atom_charges(len(initial), mobile, PRIMARY["mobile_charge_e"])
    field_calc = ElectricFieldCalculator(
        make_chgnet_calculator(device=device),
        charges=charges,
        field_strength=PRIMARY["field_strength_V_A"],
        direction=manifest["field_direction"],
    )
    field_images = make_images(initial, final)
    field = run_neb(field_images, field_calc, run_root / "field_primary")
    field["image_qc"] = image_qc(field_images, mobile)
    field["field_spec"] = field_calc.field_spec(len(initial)).__dict__
    write_json(run_root / "field_primary" / "neb_summary.json", field)

    comparison = summarize_comparison(run_root)
    heartbeat(run_root, f"run-neb {comparison['classification']}")
    return comparison


def summarize_comparison(run_root: Path) -> dict:
    no_field = read_json(run_root / "no_field" / "neb_summary.json")
    field = read_json(run_root / "field_primary" / "neb_summary.json")
    endpoint_manifest = read_json(run_root / "endpoint_manifest.json")
    barrier_delta = float(field["barrier_eV"] - no_field["barrier_eV"])
    qc_passes = bool(
        endpoint_manifest["endpoint_qc"]["passes"]
        and no_field["finite"]
        and field["finite"]
        and no_field["image_qc"]["passes"]
        and field["image_qc"]["passes"]
    )
    blocking_reasons = []
    if not endpoint_manifest["endpoint_qc"]["passes"]:
        blocking_reasons.append("endpoint_qc_failed")
    if not no_field["finite"]:
        blocking_reasons.append("no_field_non_finite")
    if not field["finite"]:
        blocking_reasons.append("field_primary_non_finite")
    if not no_field["image_qc"]["passes"]:
        blocking_reasons.append("no_field_image_qc_failed")
    if not field["image_qc"]["passes"]:
        blocking_reasons.append("field_primary_image_qc_failed")
    if not qc_passes:
        classification = "blocked"
    elif barrier_delta <= PRIMARY["positive_threshold_eV"]:
        classification = "positive_exploratory"
    else:
        classification = "negative"
    comparison = {
        "state": "completed",
        "classification": classification,
        "barrier_no_field_eV": no_field["barrier_eV"],
        "barrier_field_eV": field["barrier_eV"],
        "barrier_delta_eV": barrier_delta,
        "positive_threshold_eV": PRIMARY["positive_threshold_eV"],
        "qc_passes": qc_passes,
        "blocking_reasons": blocking_reasons,
        "primary": PRIMARY,
        "no_field_summary": run_root / "no_field" / "neb_summary.json",
        "field_summary": run_root / "field_primary" / "neb_summary.json",
    }
    write_json(run_root / "comparison_summary.json", comparison)
    write_promotion_recommendation(run_root, comparison, no_field, field)
    return comparison


def write_promotion_recommendation(run_root: Path, comparison: dict, no_field: dict, field: dict) -> None:
    if comparison["classification"] == "positive_exploratory":
        energies = np.asarray(field["image_energies_eV"], dtype=float)
        saddle_idx = int(np.nanargmax(energies))
        recommendation = (
            "# BigDFT promotion recommendation\n\n"
            "Classification: `positive_exploratory`.\n\n"
            "Promote the field-primary initial, saddle-like, and final images for BigDFT validation.\n\n"
            f"- Initial: `{field['image_paths'][0]}`\n"
            f"- Saddle-like image: `{field['image_paths'][saddle_idx]}`\n"
            f"- Final: `{field['image_paths'][-1]}`\n"
            f"- Barrier delta: `{comparison['barrier_delta_eV']:.6f} eV`\n"
        )
    else:
        recommendation = (
            "# BigDFT promotion recommendation\n\n"
            f"Classification: `{comparison['classification']}`.\n\n"
            "Do not promote images to BigDFT from this primary screen.\n\n"
            f"- No-field barrier: `{comparison['barrier_no_field_eV']:.6f} eV`\n"
            f"- Field barrier: `{comparison['barrier_field_eV']:.6f} eV`\n"
            f"- Barrier delta: `{comparison['barrier_delta_eV']:.6f} eV`\n"
            f"- QC passes: `{comparison['qc_passes']}`\n"
            f"- Blocking reasons: `{', '.join(comparison['blocking_reasons']) or 'none'}`\n"
        )
    (run_root / "promotion_recommendation.md").write_text(recommendation)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["setup-smoke", "validate", "build-endpoints", "run-neb", "summarize", "all"])
    p.add_argument("--run-root", default=None)
    p.add_argument("--device", default="cuda")
    return p


def main() -> int:
    args = parser().parse_args()
    run_root = ensure_run_root(args.run_root)
    if args.cmd == "setup-smoke":
        run_setup_smoke(run_root, args.device)
    elif args.cmd == "validate":
        validate_wrapper(run_root)
    elif args.cmd == "build-endpoints":
        build_endpoints(run_root)
    elif args.cmd == "run-neb":
        run_primary_nebs(run_root, args.device)
    elif args.cmd == "summarize":
        summarize_comparison(run_root)
    elif args.cmd == "all":
        run_setup_smoke(run_root, args.device)
        validation = validate_wrapper(run_root)
        if not validation["passes"]:
            raise SystemExit("wrapper validation failed")
        manifest = build_endpoints(run_root)
        if not manifest["endpoint_qc"]["passes"]:
            raise SystemExit("endpoint validation failed")
        run_primary_nebs(run_root, args.device)
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
