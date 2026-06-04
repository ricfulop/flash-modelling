import os
import sys
import importlib.util

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from electrodefect import build, mlip_al, percolation, transport_kpm


def test_w_slab_centering_works_with_current_ase():
    slab = build.w_slab(nx_=2, ny=2, nz=2, vacuum=4.0)
    assert len(slab) == 8
    assert np.isfinite(slab.get_positions()).all()


def test_mo_seed_configs_preserve_material_symbol():
    params = build.material_params("Mo")
    net = percolation.ordered_superlattice(nx_=2, ny=2, nz=2, a=params["a"], frac=0.25)
    slab = build.w_slab(nx_=2, ny=2, nz=2, a=params["a"], material="Mo", vacuum=4.0)
    cfgs = build.make_configs(slab, net, material="Mo", a=params["a"])
    assert set(cfgs["separated"].get_chemical_symbols()) == {"Mo"}
    assert len(cfgs["separated"]) == len(cfgs["clean"]) == len(cfgs["recombined"])


def test_bottom_contact_mask_leaves_mobile_atoms_on_tiny_slab():
    slab = build.w_slab(nx_=3, ny=3, nz=4, vacuum=4.0)
    mask = build.bottom_contact_mask(slab, lattice_a=build.A_W)
    assert 0 < mask.sum() < len(slab)
    assert len(slab) - mask.sum() >= 1


def test_spectral_energy_density_accepts_complex_projected_signal():
    velocities = np.ones((4, 2, 3))
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]])
    masses = np.array([1.0, 2.0])
    sed = mlip_al.spectral_energy_density(
        velocities,
        positions,
        [np.array([0.1, 0.2, 0.3])],
        dt_fs=1.0,
        masses=masses,
    )
    omega, spec = sed[(0.1, 0.2, 0.3)]
    assert np.isfinite(omega).all()
    assert np.isfinite(spec).all()
    assert spec.shape[0] == omega.shape[0]


def test_short_langevin_sample_has_mobile_degrees_of_freedom():
    from ase.build import bulk
    from ase.calculators.emt import EMT

    atoms = bulk("Cu", cubic=True) * (2, 2, 2)
    sample = mlip_al.short_langevin_sample(
        atoms,
        EMT(),
        T_K=300,
        steps=4,
        sample_interval=2,
        lattice_a=3.6,
    )
    assert len(sample["rows"]) == 2
    assert max(row["rms_velocity_A_fs"] for row in sample["rows"]) > 0.0
    assert sample["velocities"].shape[0] == 2


def test_network_from_atoms_produces_transport_graph_contract():
    from ase.build import bulk

    atoms = bulk("Cu", cubic=True) * (2, 2, 2)
    net = transport_kpm.network_from_atoms(atoms, lattice_a=3.6)
    assert len(net.coords) == len(atoms)
    assert net.defect_mask.all()
    assert net.graph.number_of_nodes() == len(atoms)
    assert net.graph.number_of_edges() > 0


def test_wannier90_hr_loader_extracts_home_cell_hamiltonian(tmp_path):
    hr = tmp_path / "toy_hr.dat"
    hr.write_text(
        "\n".join(
            [
                "toy wannier file",
                "2",
                "1",
                "1",
                "0 0 0 1 1 0.100000 0.000000",
                "0 0 0 2 1 -0.050000 0.000000",
                "0 0 0 1 2 -0.050000 0.000000",
                "0 0 0 2 2 0.200000 0.000000",
                "",
            ]
        )
    )
    H = transport_kpm.load_wannier90_hr(hr)
    scalars = transport_kpm.downfolded_hamiltonian_scalars(H)
    assert H.shape == (2, 2)
    assert np.allclose(H.toarray(), [[0.1, -0.05], [-0.05, 0.2]])
    assert scalars["n_orbitals"] == 2
    assert scalars["max_hop_eV"] == 0.05
    assert scalars["bandwidth_eV"] > 0.0


def test_phase2_bigdft_labels_request_matrix_export_by_default(tmp_path):
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "18_phase2_bigdft_labels.py")
    spec = importlib.util.spec_from_file_location("phase2_labels", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    job = module.LabelJob(
        name="toy",
        material="W",
        morphology="ordered",
        label_kind="relaxed",
        source_atoms=str(tmp_path / "atoms.xyz"),
        hgrid=0.55,
        rmult=(4.0, 7.0),
        max_seconds=10,
    )
    from ase import Atoms
    from ase.io import write

    write(job.source_atoms, Atoms("W", positions=[[0.0, 0.0, 0.0]], cell=[8, 8, 8], pbc=[False, False, False]))
    psp = tmp_path / "psppar.W.yaml"
    psp.write_text("# toy psp\n")
    module.find_psp = lambda material: psp
    module.write_bigdft_input(job, tmp_path / "job")
    text = (tmp_path / "job" / "input.yaml").read_text()
    assert "import: linear" in text
    assert "lin_general:" in text
    assert "output_mat: 1" in text


def test_toy_calculator_is_explicitly_not_production_ready():
    calc, meta = mlip_al.select_calculator(material="W", allow_toy=True)
    assert calc is not None
    assert meta["kind"] == "toy_lennard_jones"
    assert meta["production_ready"] is False
