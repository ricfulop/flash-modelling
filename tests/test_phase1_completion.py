import json
import os
import sys
import importlib.util

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from electrodefect import dft_bigdft, phase1


def load_ldos_queue_module():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "19_phase1_ldos_queue.py")
    spec = importlib.util.spec_from_file_location("phase1_ldos_queue", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_remaining_queue_module():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "18_phase1_remaining_queue.py")
    spec = importlib.util.spec_from_file_location("phase1_remaining_queue", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase1_selects_nrep4_and_computes_phi_delta():
    rows = [
        {"job": "01_clean_w001_small_phi", "energy_Ha": "-1", "phi_proxy_eV": "3.0", "infocode": "1", "warning_count": "5", "has_nan": "False"},
        {"job": "01_clean_w001_small_phi_nrep4", "energy_Ha": "-1", "phi_proxy_eV": "3.1", "infocode": "1", "warning_count": "5", "has_nan": "False"},
        {"job": "02_defect_w001_small_phi_nrep4", "energy_Ha": "-2", "phi_proxy_eV": "2.3", "infocode": "1", "warning_count": "5", "has_nan": "False"},
    ]
    best = phase1.select_best_rows(rows)
    assert best["clean_phi"]["job"].endswith("nrep4")
    wf = phase1.work_function_summary(best)
    assert wf["status"] == "provisional"
    assert np.isclose(wf["deltas"]["defect_phi_minus_clean_eV"]["delta_eV"], -0.8)


def test_efp_requires_paired_delta_scf_jobs():
    best = phase1.select_best_rows(
        [{"job": "04_separated_small_EFP_nrep4", "energy_Ha": "-10", "infocode": "0", "warning_count": "0", "has_nan": "False"}]
    )
    summary = phase1.efp_summary(best)
    assert summary["status"] == "missing"
    assert summary["E_FP_eV"] is None


def test_emission_summary_uses_ldos_contract(tmp_path):
    z = np.linspace(0, 30, 50)
    energies = np.array([-1.0, 6.0])

    def save(path, leak):
        rho = np.ones((2, len(z)))
        rho[1, z > 18.0] *= leak
        np.savez(path, energies=energies, density_z=rho, z_grid=z, z_surface=18.0, E_F=0.0)

    clean = tmp_path / "clean.npz"
    recombined = tmp_path / "recombined.npz"
    save(clean, 0.01)
    save(recombined, 10.0)
    out = phase1.emission_summary(clean, recombined, phi_eff=4.5, Te_eV=0.72)
    assert out["status"] == "accepted"
    assert out["contrast"]["ratio"] > 5


def test_work_function_from_archived_planar_average(tmp_path):
    run = tmp_path / "job"
    rundata = run / "rundata"
    rundata.mkdir(parents=True)
    (run / "log.yaml").write_text("Fermi Energy : -0.1\nEnergy (Hartree) : -1.0\nBigDFT infocode : 0\n")
    z = np.linspace(0, 10, 10)
    v = np.linspace(-0.2, 0.0, 10)
    np.savetxt(rundata / "local_potential_avg_z.dat", np.column_stack([np.arange(10), z, v]))
    out = dft_bigdft.work_function(run)
    assert np.isclose(out["phi_eV"], 0.1 * dft_bigdft.EV_PER_HA)


def test_cube_density_z_normalizes():
    origin = np.array([0.0, 0.0, 0.0])
    axes = np.array([[2, 1.0, 0.0, 0.0], [2, 0.0, 1.0, 0.0], [4, 0.0, 0.0, 0.5]])
    values = np.ones((2, 2, 4))
    z, density_z = dft_bigdft.cube_density_z(origin, axes, values)
    assert len(z) == 4
    assert np.isclose(np.trapezoid(density_z, z), 1.0)


def test_ldos_queue_reduces_orbital_cubes_to_stage03_contract(tmp_path):
    ldos_queue = load_ldos_queue_module()
    run = tmp_path / "job"
    run.mkdir()
    (run / "posinp.xyz").write_text("1 angstroem\nsurface 4 4 8\nW 0 0 1.5\n")
    (run / "log.yaml").write_text("Fermi Energy : 0.0\nEnergy (Hartree) : -1.0\nBigDFT infocode : 0\n")
    cube = run / "orbital_001.cube"
    cube.write_text(
        "\n".join(
            [
                "cube",
                "orbital",
                "1 0.0 0.0 0.0",
                "2 1.0 0.0 0.0",
                "2 0.0 1.0 0.0",
                "4 0.0 0.0 0.5",
                "74 0.0 0.0 0.0 1.5",
                "1 1 1 1 1 1 1 1",
                "1 1 1 1 1 1 1 1",
            ]
        )
        + "\n"
    )
    out = tmp_path / "clean_ldos.npz"
    status = ldos_queue.write_ldos_npz(run, [(cube, 6.0)], out)
    data = np.load(out)
    assert status["status"] == "accepted"
    assert data["energies"].tolist() == [6.0]
    assert data["density_z"].shape == (1, 4)
    assert np.isclose(np.trapezoid(data["density_z"][0], data["z_grid"]), 1.0)
    assert (run / "cubes" / "state_energies.json").exists()


def test_ldos_queue_rejects_missing_actual_eigenvalue_window(tmp_path):
    ldos_queue = load_ldos_queue_module()
    run = tmp_path / "job"
    run.mkdir()
    (run / "log.yaml").write_text(
        "\n".join(
            [
                "Fermi Energy : 0.0",
                "#Eigenvalues and New Occupation Numbers",
                "Orbitals: [",
                "{e:  1.000000000000E-01, f:  0.0000},  # 00001",
                "{e:  1.500000000000E-01, f:  0.0000}] # 00002",
            ]
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="below emission threshold"):
        ldos_queue.selected_emission_bands(run, phi_eff=6.0, min_above=1, max_bands=4)


def test_phase1_efp_pair_uses_atom_conserved_canonical_configs():
    remaining = load_remaining_queue_module()
    separated, recombined, meta = remaining.phase1_pair()
    assert meta["builder"] == "electrodefect.build.make_configs"
    assert meta["atom_count_conserved"] is True
    assert len(separated) == len(recombined)
    assert meta["n_fp"] > 0


def test_ldos_jobs_use_true_clean_and_defect_source_geometries():
    ldos_queue = load_ldos_queue_module()
    jobs = ldos_queue.ldos_jobs(nvirt=4, nplot=4, Te_eV=0.72)
    clean_job, clean_meta = jobs[0]
    source_job, source_meta = jobs[1]
    assert clean_meta["geometry_role"] == "clean"
    assert source_meta["geometry_role"] == "separated_frenkel_pair"
    assert len(clean_job.atoms) == len(source_job.atoms)
    assert not np.allclose(clean_job.atoms.get_positions(), source_job.atoms.get_positions())


def test_ldos_bigdft_input_places_wavefunction_output_keys_at_top_level(tmp_path):
    ldos_queue = load_ldos_queue_module()
    jobs = ldos_queue.ldos_jobs(nvirt=4, nplot=4, Te_eV=0.72)
    job, meta = jobs[0]
    psp = tmp_path / "psppar.W.yaml"
    psp.write_text("# toy psp\n")
    ldos_queue.queue.find_w_psp = lambda: psp
    ldos_queue.write_bigdft_input(job, tmp_path / "job", meta)
    payload = yaml.safe_load((tmp_path / "job" / "input.yaml").read_text())
    assert "output_wf" not in payload["dft"]
    assert payload["dft"]["nvirt"] == 4
    assert payload["dft"]["nplot"] == 4
    assert payload["output"]["orbitals"] == "ETSF"
    assert payload["output"]["outputpsiid"] == "orbitals"
