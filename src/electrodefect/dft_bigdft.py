"""
dft_bigdft.py — BigDFT driver (Stage 4). SHELL-OUT to the separate bigdft env.

Architectural constraint: BigDFT lives in its own micromamba env
(.local-bigdft/envs/bigdft, activated by use_bigdft.sh) and CANNOT share a Python
process with torch/MACE. So this module:
  1. writes ASE Atoms -> BigDFT input (posinp + input.yaml) in a run dir
  2. shells out: `bash use_bigdft.sh && bigdft -n <run>`  (subprocess, separate env)
  3. parses the BigDFT logfile back into python on return

Why BigDFT for this problem (vs a plane-wave code):
  - SURFACE boundary conditions (periodic in x,y; free in z) give a true vacuum and a
    physical work function with no spurious slab images — exactly what the emission
    test needs for phi_eff and near-surface states.
  - LINEAR-SCALING (O(N)) support-function mode reaches ~1000-atom defective cells and
    its density-kernel truncation *natively measures* the density-matrix decay length xi.
  - Finite electronic temperature (Fermi-Dirac smearing) implements the DRIVEN Te proxy.

KEY API POINTS TO VERIFY against the installed PyBigDFT (flagged ===VERIFY===):
  - input dict keys for surface BC, linear scaling, electronic temperature
  - Logfile attribute names for energies / eigenvalues
  - how to dump KS wavefunctions / LDOS on a real-space grid (bigdft-tool / *.cube)
This skeleton encodes the INTENT and the data contract; let Cursor fill API specifics
from `python -c "import BigDFT; help(BigDFT.Calculators)"` on the Spark.
"""
from __future__ import annotations
import json
import os, subprocess
from pathlib import Path
import numpy as np
import re

EV_PER_HA = 27.211386245988
BIGDFT_ACTIVATE = os.environ.get(
    "USE_BIGDFT",
    str(Path(__file__).resolve().parents[3] / "use_bigdft.sh"),
)


# --------------------------------------------------------------------------- #
# Input generation
# --------------------------------------------------------------------------- #
def write_inputs(atoms, run_dir, Te_eV=0.72, linear_scaling=True,
                 hgrids=0.35, surface=True, charge=0):
    """
    Write BigDFT posinp + input.yaml for a surface-BC, finite-Te run.

    Te_eV : electronic temperature (Fermi-Dirac smearing) — the DRIVEN-state proxy.
            0.72 eV = 8358 K (your Mo Saha-Boltzmann electron temperature).
    linear_scaling : O(N) support-function mode for big defective cells.
    """
    os.makedirs(run_dir, exist_ok=True)
    # --- posinp ---
    from ase.io import write
    # VERIFIED-PARTIAL: xyz is accepted by the installed BigDFT smoke path.
    # Surface geocode/cell encoding still needs a slab-specific posinp check.
    write(os.path.join(run_dir, "posinp.xyz"), atoms)

    # --- input.yaml ---
    # VERIFIED: finite electronic temperature is mix.tel in Hartree; occopt
    # selects the smearing convention. output_denspot=22 writes density and
    # potential cubes for work-function post-processing.
    inp = {
        "dft": {
            "hgrids": hgrids,
            "ixc": "PBE",
            "qcharge": charge,
            "nspin": 1,
            "elecfield": [0.0, 0.0, 0.0],
            "output_denspot": 22 if surface else 0,
        },
        "mix": {
            "tel": Te_eV / EV_PER_HA,
            "occopt": 1,
        },
    }
    if linear_scaling:
        inp["import"] = "linear"
        inp["lin_general"] = {"output_mat": 1}  # formatted sparse H/S matrices
    with open(os.path.join(run_dir, "input.yaml"), "w") as f:
        import yaml
        yaml.safe_dump(inp, f)
    return run_dir


def run(run_dir, nproc=40, name="run"):
    """
    Shell out into the bigdft env and run. nproc spans both nodes' Grace cores via MPI.
    """
    cmd = (f"source {BIGDFT_ACTIVATE} && cd {run_dir} && "
           f"mpirun -n {nproc} bigdft -n {name} > bigdft.out 2>&1")
    # VERIFIED: `bigdft -n <name>` reads named inputs and writes log-<name>.yaml.
    # Multi-node hostfile policy remains cluster-specific.
    subprocess.run(["bash", "-lc", cmd], check=True)
    return os.path.join(run_dir, "bigdft.out")


# --------------------------------------------------------------------------- #
# Logfile parsing  (data contract for the rest of the pipeline)
# --------------------------------------------------------------------------- #
def parse_total_energy(run_dir):
    """Return total energy (eV). Verified against local smoke log."""
    from BigDFT import Logfiles
    log = Logfiles.Logfile(os.path.join(run_dir, "log-run.yaml"))
    return float(log.energy) * EV_PER_HA


def parse_eigenvalues(run_dir):
    """KS eigenvalues (eV) and Fermi level. Verified against local smoke log."""
    from BigDFT import Logfiles
    log = Logfiles.Logfile(os.path.join(run_dir, "log-run.yaml"))
    blocks = [np.asarray(block, dtype=float).ravel() for block in log.evals]
    evals = np.concatenate(blocks) * EV_PER_HA if blocks else np.array([])
    e_fermi = float(getattr(log, "fermi_level", 0.0)) * EV_PER_HA
    return evals, e_fermi


def _read_log_text(run_dir):
    """Read the BigDFT log text from common archived run layouts."""
    run_dir = Path(run_dir)
    for name in ("log-run.yaml", "log.yaml"):
        path = run_dir / name
        if path.exists():
            return path.read_text(errors="replace")
    return ""


def _regex_last(pattern: str, text: str):
    hits = re.findall(pattern, text, flags=re.IGNORECASE)
    if not hits:
        return None
    value = hits[-1]
    if isinstance(value, tuple):
        value = value[-1]
    try:
        return float(str(value).strip())
    except Exception:
        return None


def parse_log_scalars(run_dir):
    """Parse energy, Fermi level, charge, and infocode from archived BigDFT logs."""
    text = _read_log_text(run_dir)
    return {
        "energy_Ha": _regex_last(r"Energy \(Hartree\)\s*:\s*([+\-0-9.EeNaInf]+)", text),
        "fermi_Ha": _regex_last(r"Fermi Energy\s*:\s*([+\-0-9.Ee]+)", text),
        "charge_e": _regex_last(r"Total electronic charge\s*:\s*([+\-0-9.EeNaInf]+)", text),
        "infocode": _regex_last(r"BigDFT infocode\s*:\s*([+\-0-9]+)", text),
    }


def parse_planar_average(path):
    """Parse BigDFT `*_avg_z.dat` style planar-average files."""
    path = Path(path)
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"expected at least three columns in {path}")
    return arr[:, 1].astype(float), arr[:, 2].astype(float)


def vacuum_level_from_planar_average(z, potential_Ha, tail_fraction=0.10):
    """Estimate V_vac from a planar averaged potential.

    The max is intentionally conservative for asymmetric one-sided slabs: it
    avoids averaging metallic and vacuum tails together when only +z is vacuum.
    """
    z = np.asarray(z, dtype=float)
    v = np.asarray(potential_Ha, dtype=float)
    if len(z) != len(v) or len(v) < 5:
        raise ValueError("z and potential arrays must have matching length >= 5")
    n_tail = max(5, int(tail_fraction * len(v)))
    return {
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "vacuum_left_Ha": float(np.mean(v[:n_tail])),
        "vacuum_right_Ha": float(np.mean(v[-n_tail:])),
        "vacuum_max_Ha": float(np.max(v)),
        "vacuum_tail_mean_Ha": float(np.mean([np.mean(v[:n_tail]), np.mean(v[-n_tail:])])),
    }


def export_ks_for_emission(run_dir, states="above_fermi", z_axis=2):
    """
    Export KS wavefunctions / LDOS on a real-space z-grid for emission.emittable_flux().

    Returns dict: {energies, density_z (M,Z), z_grid, z_surface, E_F}.

    Implementation path (===VERIFY===): use bigdft-tool to dump selected KS orbitals to
    .cube, then integrate |psi|^2 over (x,y) per z-plane here. For O(N) runs, dump the
    LDOS in energy windows instead of individual orbitals.
    """
    if z_axis != 2:
        raise NotImplementedError("Only z-axis LDOS reduction is implemented.")
    run_dir = Path(run_dir)
    cube_dir = run_dir / "cubes"
    energy_map = cube_dir / "state_energies.json"
    if not cube_dir.exists() or not energy_map.exists():
        raise FileNotFoundError(
            "Missing exported KS/LDOS cubes. Expected `cubes/*.cube` plus "
            "`cubes/state_energies.json` mapping cube filenames to energies in eV."
        )
    energies_by_file = json.loads(energy_map.read_text())
    z_grid = None
    densities = []
    energies = []
    for cube_name, energy in sorted(energies_by_file.items(), key=lambda item: float(item[1])):
        cube_path = cube_dir / cube_name
        origin, axes, values = read_cube(cube_path)
        z, density_z = cube_density_z(origin, axes, values, square_values=True)
        if z_grid is None:
            z_grid = z
        elif len(z_grid) != len(z) or not np.allclose(z_grid, z):
            raise ValueError(f"inconsistent z grid in {cube_path}")
        energies.append(float(energy))
        densities.append(density_z)
    scalars = parse_log_scalars(run_dir)
    e_fermi = float(scalars["fermi_Ha"]) * EV_PER_HA if scalars["fermi_Ha"] is not None else 0.0
    z_surface = infer_surface_z(run_dir)
    return {
        "energies": np.asarray(energies, dtype=float),
        "density_z": np.asarray(densities, dtype=float),
        "z_grid": np.asarray(z_grid, dtype=float),
        "z_surface": float(z_surface),
        "E_F": float(e_fermi),
    }


def infer_surface_z(run_dir):
    """Infer the top atomic z coordinate from `posinp.xyz`."""
    path = Path(run_dir) / "posinp.xyz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; pass z_surface explicitly downstream")
    coords = []
    for line in path.read_text().splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                coords.append(float(parts[3]))
            except ValueError:
                pass
    if not coords:
        raise ValueError(f"no atomic z coordinates parsed from {path}")
    return max(coords)


def read_cube(path):
    """Read a Gaussian cube file into `(origin, axes, values)`.

    Axes are a `(3, 4)` array with columns `(n, dx, dy, dz)`.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    if len(lines) < 6:
        raise ValueError(f"cube file too short: {path}")
    nat_line = lines[2].split()
    natoms = abs(int(nat_line[0]))
    origin = np.asarray([float(x) for x in nat_line[1:4]], dtype=float)
    axes = []
    counts = []
    for idx in range(3, 6):
        parts = lines[idx].split()
        n = int(parts[0])
        counts.append(abs(n))
        axes.append([abs(n), float(parts[1]), float(parts[2]), float(parts[3])])
    data_start = 6 + natoms
    values = np.fromstring(" ".join(lines[data_start:]), sep=" ", dtype=float)
    expected = counts[0] * counts[1] * counts[2]
    if values.size != expected:
        raise ValueError(f"cube data size mismatch in {path}: got {values.size}, expected {expected}")
    return origin, np.asarray(axes, dtype=float), values.reshape(tuple(counts))


def cube_density_z(origin, axes, values, square_values=True):
    """Integrate cube values over x/y to a normalized density along z."""
    values = np.asarray(values, dtype=float)
    density = values ** 2 if square_values else values
    nx, ny, nz = values.shape
    z_axis = np.asarray(axes[2, 1:4], dtype=float)
    dz = float(np.linalg.norm(z_axis))
    z0 = float(origin[2])
    z = z0 + np.arange(nz) * dz
    # For orthogonal BigDFT cubes, summing over x/y and normalizing over z is
    # sufficient for the emission contrast; absolute xy area cancels in ratios.
    density_z = density.sum(axis=(0, 1))
    norm = np.trapezoid(density_z, z)
    if norm > 0:
        density_z = density_z / norm
    return z, density_z


# --------------------------------------------------------------------------- #
# High-level physics helpers
# --------------------------------------------------------------------------- #
def E_FP_delta_scf(sep_dir, recomb_dir):
    """Frenkel-pair recombination enthalpy E_FP = E(separated) - E(recombined), eV."""
    return parse_total_energy(sep_dir) - parse_total_energy(recomb_dir)


def work_function(slab_dir, z_axis=2):
    """
    phi_eff = V_vacuum - E_F. Needs the planar-averaged local potential in vacuum.
    Returns a conservative proxy from `rundata/local_potential_avg_z.dat`.
    """
    if z_axis != 2:
        raise NotImplementedError("Only z-axis slab work-function parsing is implemented.")
    slab_dir = Path(slab_dir)
    pot_path = slab_dir / "rundata" / "local_potential_avg_z.dat"
    if not pot_path.exists():
        raise FileNotFoundError(f"missing planar potential: {pot_path}")
    scalars = parse_log_scalars(slab_dir)
    if scalars["fermi_Ha"] is None:
        # Fall back to PyBigDFT parser for unarchived `log-run.yaml` layouts.
        _, fermi_eV = parse_eigenvalues(str(slab_dir))
        fermi_Ha = fermi_eV / EV_PER_HA
    else:
        fermi_Ha = float(scalars["fermi_Ha"])
    z, potential = parse_planar_average(pot_path)
    vac = vacuum_level_from_planar_average(z, potential)
    phi_eV = (vac["vacuum_max_Ha"] - fermi_Ha) * EV_PER_HA
    return {
        **scalars,
        **vac,
        "fermi_Ha": fermi_Ha,
        "phi_eV": float(phi_eV),
        "potential_file": str(pot_path),
        "note": "Conservative work-function proxy; review convergence and vacuum plateau before final reporting.",
    }


def export_support_hamiltonian(run_dir):
    """
    Export the O(N) support-function H and S matrices (sparse) for transport_kpm.build_tb
    as an alternative to a TB fit. ===VERIFY=== lin_general output_mat file format
    (NTPoly / CheSS matrices). This is the 'trust the DFT Hamiltonian directly' fork.
    """
    raise NotImplementedError("Parse output_mat (NTPoly/CheSS) -> scipy sparse H,S.")


if __name__ == "__main__":
    print(__doc__)
    print("Skeleton. Verify ===VERIFY=== points against installed PyBigDFT, then wire to scripts/03.")
