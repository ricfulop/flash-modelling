#!/usr/bin/env python3
"""Generate small W/Mo Frenkel-pair candidates and gate them for MatterChat.

This script is intentionally geometry-only. It writes MatterChat input CIFs
only for structures that pass a hard minimum-distance gate, while retaining
blocked structures separately for debugging.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.io import write

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from electrodefect import build, percolation  # noqa: E402


@dataclass
class QcRow:
    material: str
    morphology: str
    config: str
    n_atoms: int
    n_vacancy_sites: int
    min_distance_a: float | None
    min_pair: list[int] | None
    pass_min_distance: bool
    warn_close_contact: bool
    matterchat_ready: bool
    dft_relaxation_required: bool
    cif_path: str | None
    note: str


def min_distance(atoms) -> tuple[float | None, list[int] | None]:
    """Return the closest atom-atom distance and pair indices."""
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


def build_network(material: str, morphology: str, frac: float, size: tuple[int, int, int], seed: int):
    params = build.material_params(material)
    kwargs = {
        "nx_": size[0],
        "ny": size[1],
        "nz": size[2],
        "a": params["a"],
        "seed": seed,
    }
    if morphology == "ordered":
        return percolation.ordered_superlattice(frac=frac, **kwargs)
    if morphology == "random":
        return percolation.random_percolation(frac=frac, **kwargs)
    if morphology in {"dendrite", "dendritic"}:
        return percolation.dla_dendrite(target_frac=frac, **kwargs)
    raise ValueError(f"unsupported morphology: {morphology}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--materials", nargs="+", default=["W", "Mo"], choices=["W", "Mo"])
    p.add_argument("--morphologies", nargs="+", default=["ordered", "random", "dendritic"])
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--network-size", type=int, nargs=3, default=[4, 4, 6])
    p.add_argument("--slab-size", type=int, nargs=3, default=[4, 4, 6])
    p.add_argument("--vacuum", type=float, default=10.0)
    p.add_argument("--surface-band", type=float, default=6.0)
    p.add_argument(
        "--min-distance",
        type=float,
        default=1.5,
        help="Hard block threshold in Angstrom; structures below this do not enter MatterChat inputs.",
    )
    p.add_argument(
        "--warn-distance",
        type=float,
        default=1.8,
        help="Warning threshold in Angstrom; passing structures below this require relaxation before DFT.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=None)
    return p


def main() -> int:
    args = parser().parse_args()
    run_id = datetime.now().strftime("matterchat_geometry_qc_%Y%m%d_%H%M%S")
    out_dir = args.output_dir or REPO / "runs" / run_id
    input_dir = out_dir / "matterchat_inputs"
    fail_dir = out_dir / "failed_geometry"
    input_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    rows: list[QcRow] = []
    ready_paths: list[str] = []
    for material in args.materials:
        params = build.material_params(material)
        slab = build.w_slab(
            nx_=args.slab_size[0],
            ny=args.slab_size[1],
            nz=args.slab_size[2],
            a=params["a"],
            vacuum=args.vacuum,
            material=material,
        )
        for morphology in args.morphologies:
            net = build_network(material, morphology, args.fraction, tuple(args.network_size), args.seed)
            cfgs = build.make_configs(
                slab,
                net,
                surface_band=args.surface_band,
                material=material,
                a=params["a"],
            )
            n_vac = int(len(cfgs["vac_idx"]))
            for config_name in ("clean", "separated", "recombined"):
                atoms = cfgs[config_name]
                dmin, pair = min_distance(atoms)
                passes = dmin is not None and dmin >= args.min_distance
                warns = passes and dmin < args.warn_distance
                target_dir = input_dir if passes else fail_dir
                cif_name = f"{material}_{morphology}_{config_name}.cif"
                cif_path = target_dir / cif_name
                write(cif_path, atoms, format="cif")
                rel_path = str(cif_path.relative_to(REPO))
                if passes:
                    ready_paths.append(rel_path)
                rows.append(
                    QcRow(
                        material=material,
                        morphology=morphology,
                        config=config_name,
                        n_atoms=len(atoms),
                        n_vacancy_sites=n_vac,
                        min_distance_a=round(dmin, 6) if dmin is not None else None,
                        min_pair=pair,
                        pass_min_distance=bool(passes),
                        warn_close_contact=bool(warns),
                        matterchat_ready=bool(passes),
                        dft_relaxation_required=bool(warns),
                        cif_path=rel_path,
                        note=(
                            "allowed for MatterChat; relax or repair before DFT"
                            if warns
                            else "allowed for MatterChat qualitative prompt"
                            if passes
                            else "blocked from MatterChat/DFT until repaired or relaxed"
                        ),
                    )
                )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "geometry-only MatterChat input gate; no physics claims",
        "parameters": {
            "materials": args.materials,
            "morphologies": args.morphologies,
            "fraction": args.fraction,
            "network_size": args.network_size,
            "slab_size": args.slab_size,
            "vacuum_a": args.vacuum,
            "surface_band_a": args.surface_band,
            "min_distance_a": args.min_distance,
            "warn_distance_a": args.warn_distance,
            "seed": args.seed,
        },
        "n_total": len(rows),
        "n_matterchat_ready": len(ready_paths),
        "matterchat_ready_cifs": ready_paths,
        "rows": [asdict(row) for row in rows],
    }
    (out_dir / "geometry_qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        "# MatterChat Geometry QC\n\n"
        "This run generated small W/Mo Frenkel-pair candidate structures and applied "
        "a minimum-distance gate before MatterChat prompting. Passing CIFs are in "
        "`matterchat_inputs/`; blocked CIFs are retained in `failed_geometry/` for "
        "debugging only and should not be used for MatterChat or DFT claims.\n\n"
        f"- Total structures: {len(rows)}\n"
        f"- MatterChat-ready structures: {len(ready_paths)}\n"
        f"- Hard minimum-distance threshold: {args.min_distance} A\n"
        f"- Close-contact warning threshold: {args.warn_distance} A\n"
    )
    print(out_dir)
    print(json.dumps({"n_total": len(rows), "n_matterchat_ready": len(ready_paths)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
