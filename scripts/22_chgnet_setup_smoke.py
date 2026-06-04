#!/usr/bin/env python3
"""Setup-only smoke test for the CHGNet + ASE environment.

This verifies imports, CUDA visibility, pymatgen-to-ASE conversion, and the
installed CHGNet ASE calculator API. It intentionally does not run MD or NEB.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
from pathlib import Path

import torch
from chgnet.model import CHGNetCalculator
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor


def build_bcc_structure(element: str, lattice_a: float) -> Structure:
    return Structure(
        Lattice.cubic(float(lattice_a)),
        [element, element],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


def package_versions() -> dict[str, str]:
    packages = ["chgnet", "ase", "pymatgen", "numpy", "torch"]
    return {package: md.version(package) for package in packages}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--element", default="W", choices=["W", "Mo"])
    parser.add_argument("--lattice-a", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    lattice_a = args.lattice_a
    if lattice_a is None:
        lattice_a = {"W": 3.1652, "Mo": 3.147}[args.element]

    structure = build_bcc_structure(args.element, lattice_a)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = CHGNetCalculator()

    # Setup smoke only: force a calculator call but do not report the energy as a result.
    atoms.get_potential_energy()

    payload = {
        "state": "completed",
        "purpose": "setup_smoke_only",
        "element": args.element,
        "lattice_a_A": lattice_a,
        "atom_count": len(atoms),
        "packages": package_versions(),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()),
        "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "chgnet_calculator": "CHGNetCalculator",
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
