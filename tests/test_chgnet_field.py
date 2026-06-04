import os
import sys

import numpy as np
import pytest
from ase import Atoms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from electrodefect.chgnet_field import (  # noqa: E402
    ElectricFieldCalculator,
    ZeroCalculator,
    mobile_atom_charges,
    validate_explicit_charges,
)


def test_electric_field_forces_follow_qe_sign_and_shape():
    atoms = Atoms("WW", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.calc = ElectricFieldCalculator(
        ZeroCalculator(),
        charges=np.array([1.0, -1.0]),
        field_strength=0.2,
        direction=(1.0, 0.0, 0.0),
    )
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    assert np.allclose(forces, [[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])


def test_zero_charges_reproduce_zero_base_calculator():
    atoms = Atoms("WW", positions=[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
    atoms.calc = ElectricFieldCalculator(
        ZeroCalculator(),
        charges=np.zeros(2),
        field_strength=0.2,
        direction=(1.0, 0.0, 0.0),
    )
    assert atoms.get_potential_energy() == 0.0
    assert np.allclose(atoms.get_forces(), np.zeros((2, 3)))


def test_magmom_charge_shortcut_is_rejected():
    with pytest.raises(ValueError, match="magmom"):
        validate_explicit_charges("magmom", n_atoms=2)


def test_mobile_atom_charge_assignment_is_explicit():
    charges = mobile_atom_charges(n_atoms=4, mobile_atom_index=2, charge_e=1.0)
    assert np.allclose(charges, [0.0, 0.0, 1.0, 0.0])
