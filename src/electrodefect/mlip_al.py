"""
mlip_al.py — MACE active-learning loop + MD/phonon/NEB (Stages 2-3).

MACE-MP-0 has never seen 25 mol% defective W with Frenkel cores at Te~8000 K — it is
far out of distribution. So we DO NOT trust it cold. Active-learning loop:

  1. start from MACE-MP-0 (foundation) or a small fine-tune
  2. run MD on the defect/FP configs; flag high-uncertainty frames
     (uncertainty = force std across a committee of N models)
  3. label flagged frames with BigDFT (dft_bigdft.run -> energies + forces)
  4. retrain / fine-tune the committee on the growing labeled set
  5. repeat until held-out force RMSE on FP/defect configs is acceptable
  wandb tracks uncertainty, RMSE, and dataset growth.

Then use the converged potential for:
  - morphology relaxation / annealing of the percolating network (build.relax)
  - phonon lifetimes via spectral energy density (SED) -> settles superhighway vs
    coherent-slow (narrow SED peaks + LOW group velocity = coherent-slow, NOT ballistic)
  - e-ph coupling beta at the damping ridge q*/q_D ~ 0.73 -> k_soft = 1 - 0.533*beta
  - NEB migration barriers E_hop for the escape budget (emission.escape_budget)

Run on the torch/MACE env, GPU. mace-torch==0.3.16, ase==3.28.
KEY API POINTS (===VERIFY===): MACECalculator construction, fine-tune invocation.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Calculators / committee
# --------------------------------------------------------------------------- #
def get_calculator(model_path="mace-mp-0", device="cuda", default_dtype="float32"):
    """Single MACE calculator. ===VERIFY=== mace-torch 0.3.16 API."""
    from mace.calculators import MACECalculator, mace_mp
    if model_path == "mace-mp-0":
        return mace_mp(model="medium", device=device, default_dtype=default_dtype)
    return MACECalculator(model_paths=model_path, device=device, default_dtype=default_dtype)


def write_json(path, payload):
    """Write a JSON artifact with numpy values converted to builtins."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_builtin(payload), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def to_builtin(value):
    """Convert numpy, pathlib, and tuple values for JSON artifacts."""
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def module_available(name: str) -> bool:
    """Return whether a top-level module can be imported without importing it now."""
    import importlib.util
    return importlib.util.find_spec(name) is not None


def environment_smoke(device="cuda") -> dict:
    """Collect Phase 2 environment readiness without mixing BigDFT into torch work."""
    bigdft_activate = Path(os.environ.get(
        "USE_BIGDFT",
        Path(__file__).resolve().parents[3] / "use_bigdft.sh",
    ))
    info = {
        "ase_available": module_available("ase"),
        "mace_available": module_available("mace"),
        "bigdft_python_available_in_current_env": module_available("BigDFT"),
        "bigdft_shell": {
            "activate": bigdft_activate,
            "activate_exists": bigdft_activate.exists(),
        },
        "nvidia_smi": None,
        "torch": {"available": module_available("torch")},
    }
    if module_available("torch"):
        try:
            import torch
            info["torch"].update(
                {
                    "version": torch.__version__,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "device_count": int(torch.cuda.device_count()),
                    "devices": [
                        torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())
                    ]
                    if torch.cuda.is_available()
                    else [],
                    "requested_device": device,
                }
            )
        except Exception as exc:
            info["torch"]["error"] = f"{type(exc).__name__}: {exc}"
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
        )
        info["nvidia_smi"] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip().splitlines(),
            "stderr": proc.stderr.strip().splitlines(),
        }
    if bigdft_activate.exists():
        try:
            proc = subprocess.run(
                [
                    "bash",
                    "-lc",
                    f"source {str(bigdft_activate)!r} >/tmp/phase2_bigdft_activate.out && "
                    "python -c 'from BigDFT import Logfiles; print(\"BigDFT shell ok\")'",
                ],
                text=True,
                capture_output=True,
                timeout=30,
            )
            info["bigdft_shell"].update(
                {
                    "returncode": proc.returncode,
                    "stdout": proc.stdout.strip().splitlines(),
                    "stderr": proc.stderr.strip().splitlines(),
                    "ready": proc.returncode == 0,
                }
            )
        except Exception as exc:
            info["bigdft_shell"].update(
                {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
            )
    return info


def toy_calculator(material="W"):
    """Fallback calculator for pipeline smoke only; never use for physics claims."""
    from ase.calculators.lj import LennardJones
    # Parameters are deliberately generic: this only tests plumbing and stability.
    sigma = {"W": 2.74, "Mo": 2.72}.get(str(material), 2.7)
    return LennardJones(epsilon=0.20, sigma=sigma, rc=3.0 * sigma)


def select_calculator(model_path="mace-mp-0", device="cuda", material="W",
                      allow_toy=False):
    """
    Select a production MACE calculator when available, else an explicit toy fallback.

    Returns (calculator, metadata). Toy mode is only for smoke/pilot plumbing.
    """
    if module_available("mace"):
        try:
            return get_calculator(model_path=model_path, device=device), {
                "kind": "mace",
                "model_path": model_path,
                "device": device,
                "production_ready": True,
            }
        except Exception as exc:
            if not allow_toy:
                raise
            fallback_error = f"{type(exc).__name__}: {exc}"
        else:
            fallback_error = None
    else:
        fallback_error = "mace is not importable"
    if not allow_toy:
        raise RuntimeError("mace is not importable; install mace-torch before production Phase 2 runs")
    return toy_calculator(material=material), {
        "kind": "toy_lennard_jones",
        "model_path": None,
        "device": "cpu",
        "production_ready": False,
        "fallback_error": fallback_error,
        "warning": "Toy fallback exercises pipeline plumbing only; not a physical W/Mo potential.",
    }


def min_pair_distance(atoms) -> float:
    """Minimum interatomic distance in Angstrom, ignoring self-distances."""
    pos = atoms.get_positions()
    if len(pos) < 2:
        return float("nan")
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    dist[dist == 0.0] = np.inf
    return float(np.min(dist))


def geometry_summary(atoms, material="W") -> dict:
    """Small geometry sanity summary for pilot artifacts."""
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    fixed = fixed_atom_mask(atoms)
    return {
        "material": material,
        "n_atoms": len(atoms),
        "n_fixed": int(fixed.sum()),
        "n_mobile": int(len(atoms) - fixed.sum()),
        "symbols": sorted(set(atoms.get_chemical_symbols())),
        "cell": cell,
        "position_min": pos.min(axis=0) if len(pos) else np.full(3, np.nan),
        "position_max": pos.max(axis=0) if len(pos) else np.full(3, np.nan),
        "min_pair_distance_A": min_pair_distance(atoms),
        "finite_positions": bool(np.isfinite(pos).all()),
    }


def calculator_observables(atoms, calc) -> dict:
    """Evaluate finite energy/force observables for one ASE Atoms object."""
    a = atoms.copy()
    a.calc = calc
    energy = float(a.get_potential_energy())
    forces = np.asarray(a.get_forces(), dtype=float)
    fmag = np.linalg.norm(forces, axis=1) if len(forces) else np.array([])
    return {
        "energy_eV": energy,
        "max_force_eV_A": float(fmag.max()) if len(fmag) else float("nan"),
        "mean_force_eV_A": float(fmag.mean()) if len(fmag) else float("nan"),
        "finite": bool(np.isfinite(energy) and np.isfinite(forces).all()),
    }


def fixed_atom_mask(atoms) -> np.ndarray:
    """Best-effort mask for atoms fixed by ASE FixAtoms constraints."""
    mask = np.zeros(len(atoms), dtype=bool)
    for constraint in getattr(atoms, "constraints", []) or []:
        if hasattr(constraint, "index"):
            idx = np.asarray(constraint.index, dtype=int)
            mask[idx] = True
        elif hasattr(constraint, "get_indices"):
            idx = np.asarray(constraint.get_indices(), dtype=int)
            mask[idx] = True
    return mask


def apply_bottom_constraint(atoms, lattice_a, max_fixed_fraction=0.35):
    """Apply a bottom contact constraint while guaranteeing at least one mobile atom."""
    from ase.constraints import FixAtoms
    from . import build

    a = atoms.copy()
    mask = build.bottom_contact_mask(a, lattice_a=lattice_a, max_fixed_fraction=max_fixed_fraction)
    if mask.any():
        a.set_constraint(FixAtoms(mask=mask))
    else:
        a.set_constraint()
    return a


def relax_pilot_structure(
    atoms,
    calc,
    lattice_a=3.165,
    fmax=1.0,
    steps=50,
    maxstep=0.03,
    max_force_growth=5.0,
):
    """Shortly relax a pilot structure before MD/NEB so smoke runs do not explode."""
    from ase.optimize import FIRE

    start = apply_bottom_constraint(atoms, lattice_a=lattice_a)
    start.calc = calc
    before = calculator_observables(start, calc)
    a = start.copy()
    a.calc = calc
    FIRE(a, logfile=None, maxstep=float(maxstep)).run(fmax=float(fmax), steps=int(steps))
    after = calculator_observables(a, calc)
    force_limit = max(float(fmax), float(max_force_growth) * before["max_force_eV_A"])
    accepted = bool(after["finite"] and after["max_force_eV_A"] <= force_limit)
    chosen = a if accepted else start
    return chosen, {
        "before": before,
        "after": after,
        "accepted": accepted,
        "force_limit_eV_A": force_limit,
        "steps": int(steps),
        "fmax": float(fmax),
        "maxstep_A": float(maxstep),
    }


def short_langevin_sample(atoms, calc, T_K=1000.0, steps=20, sample_interval=5,
                          dt_fs=1.0, friction=0.02, seed=20260603,
                          lattice_a=3.165, fix_bottom=True):
    """Run a tiny Langevin trajectory and return positions, velocities, and summary rows."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    np.random.seed(int(seed))
    a = apply_bottom_constraint(atoms, lattice_a=lattice_a) if fix_bottom else atoms.copy()
    a.calc = calc
    if len(a) - int(fixed_atom_mask(a).sum()) <= 0:
        raise ValueError("MD setup has no mobile atoms after constraints")
    start_positions = a.get_positions().copy()
    MaxwellBoltzmannDistribution(a, temperature_K=T_K)
    Stationary(a)
    ZeroRotation(a)
    dyn = Langevin(a, dt_fs * units.fs, temperature_K=T_K, friction=friction)
    positions, velocities, rows = [], [], []
    for step in range(0, int(steps), int(sample_interval)):
        dyn.run(int(sample_interval))
        positions.append(a.get_positions().copy())
        velocities.append(a.get_velocities().copy())
        obs = calculator_observables(a, calc)
        obs["step"] = step + int(sample_interval)
        obs["rms_velocity_A_fs"] = float(np.sqrt(np.mean(np.asarray(a.get_velocities()) ** 2)))
        obs["max_displacement_A"] = float(np.max(np.linalg.norm(a.get_positions() - start_positions, axis=1)))
        rows.append(obs)
    return {
        "atoms": a,
        "positions": np.asarray(positions),
        "velocities": np.asarray(velocities),
        "rows": rows,
        "masses": a.get_masses(),
        "dt_fs": dt_fs * sample_interval,
    }


def committee(model_paths, device="cuda"):
    """List of calculators for uncertainty estimation."""
    from mace.calculators import MACECalculator
    return [MACECalculator(model_paths=[p], device=device, default_dtype="float32")
            for p in model_paths]


def force_uncertainty(atoms, calcs):
    """Std of per-atom force magnitude across the committee (eV/Ang). High => label it."""
    F = []
    for c in calcs:
        a = atoms.copy(); a.calc = c
        F.append(a.get_forces())
    F = np.array(F)                      # (n_models, n_atoms, 3)
    fmag = np.linalg.norm(F, axis=2)     # (n_models, n_atoms)
    return float(fmag.std(axis=0).max())


# --------------------------------------------------------------------------- #
# Active-learning loop
# --------------------------------------------------------------------------- #
def active_learning_loop(seed_configs, calcs, label_fn, retrain_fn,
                         n_rounds=6, md_steps=2000, T_L=1000.0,
                         unc_threshold=0.15, wandb_run=None):
    """
    seed_configs : list[Atoms] starting defect/FP cells
    calcs        : committee of MACE calculators
    label_fn     : f(Atoms)-> (energy, forces) via BigDFT (dft_bigdft)
    retrain_fn   : f(dataset)-> new model_paths
    unc_threshold: force-std (eV/Ang) above which a frame is labeled
    """
    from ase.md.langevin import Langevin
    from ase import units
    dataset = []
    for rnd in range(n_rounds):
        flagged = []
        for atoms in seed_configs:
            a = atoms.copy(); a.calc = calcs[0]
            dyn = Langevin(a, 1.0 * units.fs, temperature_K=T_L, friction=0.02)
            for _ in range(md_steps // 20):
                dyn.run(20)
                u = force_uncertainty(a, calcs)
                if u > unc_threshold:
                    flagged.append(a.copy())
        # label flagged frames with DFT
        for fr in flagged:
            e, f = label_fn(fr)
            fr.info["energy"] = e; fr.arrays["forces"] = f
            dataset.append(fr)
        model_paths = retrain_fn(dataset)
        calcs = committee(model_paths)
        if wandb_run:
            wandb_run.log(dict(round=rnd, n_flagged=len(flagged),
                               dataset=len(dataset)))
        print(f"[AL round {rnd}] flagged={len(flagged)} dataset={len(dataset)}")
    return calcs, dataset


# --------------------------------------------------------------------------- #
# Phonon SED (superhighway vs coherent-slow) and e-ph coupling
# --------------------------------------------------------------------------- #
def spectral_energy_density(traj_velocities, positions, kpoints, dt_fs,
                            masses):
    """
    SED(k, w) from MD velocities. Peak WIDTH -> phonon lifetime tau(k); peak POSITION
    along k -> dispersion (group velocity = d omega/dk).

    Interpretation for the manuscript:
      narrow peaks (long tau) + LOW group velocity  => coherent-slow (defect miniband)
      narrow peaks + HIGH group velocity            => the (unlikely) ballistic case
    The bad-metal premise needs the former; report whichever the data show.
    """
    v = np.asarray(traj_velocities)              # (n_frames, n_atoms, 3)
    pos = np.asarray(positions)
    nfr = v.shape[0]
    sed = {}
    for k in kpoints:
        phase = np.exp(1j * (pos @ np.asarray(k)))         # (n_atoms,)
        qk = np.einsum('fad,a,a->fd', v, np.sqrt(masses), phase)  # collective coord
        spec_full = np.abs(np.fft.fft(qk, axis=0)) ** 2
        w_full = np.fft.fftfreq(nfr, d=dt_fs * 1e-15) * 2 * np.pi
        keep = w_full >= 0.0
        spec = spec_full[keep]
        w = w_full[keep]
        sed[tuple(k)] = (w, spec.sum(axis=1))
    return sed


def eph_coupling_beta(band_shift_per_mode, mode_q, q_D):
    """
    Extract the phonon-softening coupling beta near the damping ridge q*/q_D ~ 0.73 from
    frozen-phonon electronic band shifts, then check k_soft = 1 - 0.533*beta against your
    measured Vc. Returns beta and the self-consistency residual.
    """
    ridge = np.argmin(np.abs(np.asarray(mode_q) / q_D - 0.73))
    beta = float(np.asarray(band_shift_per_mode)[ridge])
    k_soft = 1 - 0.533 * beta
    return dict(beta=beta, k_soft=k_soft, ridge_mode=int(ridge))


def neb_hop_barrier(initial, final, calc, n_images=7, fmax=0.05, steps=100):
    """NEB migration barrier E_hop (eV) between adjacent defect sites (for escape budget)."""
    from ase.mep import NEB
    from ase.optimize import FIRE
    images = [initial.copy()] + [initial.copy() for _ in range(n_images - 2)] + [final.copy()]
    try:
        neb = NEB(images, allow_shared_calculator=True)
    except TypeError:
        neb = NEB(images)
    neb.interpolate()
    for im in images:
        im.calc = calc
    FIRE(neb, logfile='-').run(fmax=fmax, steps=steps)
    energies = [im.get_potential_energy() for im in images]
    return float(max(energies) - energies[0]), energies


if __name__ == "__main__":
    print(__doc__)
    print("Skeleton. Verify ===VERIFY=== MACE API points, then wire to scripts/01-02.")
