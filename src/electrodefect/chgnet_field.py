"""CHGNet calculator helpers for exploratory fixed-charge field NEB runs.

The electric-field wrapper is a qualitative perturbation only. It does not make
CHGNet a polarizable or field-aware potential, and it must not infer charges
from CHGNet magnetic moments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


def make_chgnet_calculator(device: str = "cuda"):
    """Construct the installed CHGNet ASE calculator."""
    from chgnet.model import CHGNetCalculator

    return CHGNetCalculator(use_device=device)


def normalized_vector(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"expected a 3-vector, got shape {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("direction must be a finite non-zero vector")
    return arr / norm


def validate_explicit_charges(charges, n_atoms: int) -> np.ndarray:
    """Validate explicit charges and reject CHGNet magmom-derived shortcuts."""
    if isinstance(charges, str):
        if charges.lower() == "magmom":
            raise ValueError("CHGNet magmom values must not be used as electric charges")
        raise ValueError("charges must be an explicit numeric array, not a string label")
    arr = np.asarray(charges, dtype=float)
    if arr.shape != (int(n_atoms),):
        raise ValueError(f"expected {n_atoms} charges, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("charges must be finite")
    return arr


def mobile_atom_charges(n_atoms: int, mobile_atom_index: int, charge_e: float = 1.0) -> np.ndarray:
    charges = np.zeros(int(n_atoms), dtype=float)
    idx = int(mobile_atom_index)
    if idx < 0 or idx >= int(n_atoms):
        raise IndexError(f"mobile atom index {idx} is outside [0, {n_atoms})")
    charges[idx] = float(charge_e)
    return charges


@dataclass(frozen=True)
class FieldSpec:
    field_strength: float
    direction: tuple[float, float, float]
    nonzero_charge_count: int
    charge_sum_e: float


class ElectricFieldCalculator(Calculator):
    """Wrap an ASE calculator and add a fixed-charge linear field term."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, charges, field_strength=0.2, direction=(1.0, 0.0, 0.0)):
        super().__init__()
        self.base_calc = base_calc
        self._charges_input = charges
        self.field_strength = float(field_strength)
        self.direction = normalized_vector(direction)
        if not np.isfinite(self.field_strength):
            raise ValueError("field_strength must be finite")
        if self.field_strength < 0:
            raise ValueError("field_strength must be non-negative; reverse direction instead")

    @property
    def efield(self) -> np.ndarray:
        return self.field_strength * self.direction

    def field_spec(self, n_atoms: int) -> FieldSpec:
        charges = validate_explicit_charges(self._charges_input, n_atoms)
        return FieldSpec(
            field_strength=self.field_strength,
            direction=tuple(float(x) for x in self.direction),
            nonzero_charge_count=int(np.count_nonzero(charges)),
            charge_sum_e=float(charges.sum()),
        )

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("atoms must be provided")

        charges = validate_explicit_charges(self._charges_input, len(atoms))
        self.base_calc.calculate(atoms, properties, system_changes)
        results = dict(self.base_calc.results)

        positions = np.asarray(atoms.get_positions(), dtype=float)
        base_forces = np.asarray(results["forces"], dtype=float)
        field_forces = charges[:, None] * self.efield[None, :]
        field_energy = -float(np.sum(charges * (positions @ self.efield)))

        results["forces"] = base_forces + field_forces
        results["energy"] = float(results["energy"]) + field_energy
        self.results = results


class ZeroCalculator(Calculator):
    """Deterministic toy calculator for field-wrapper validation."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("atoms must be provided")
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3), dtype=float),
        }
