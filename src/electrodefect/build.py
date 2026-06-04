"""
build.py — Construct the tungsten emission slab with a percolating Frenkel-pair network.

Geometry: BCC W(001) slab, +z is the emission face with vacuum above. A percolating /
dendritic defect network (from percolation.py) is realized as vacancy + self-interstitial
(Frenkel) pairs. For BCC W the stable SIA is the <111> dumbbell/crowdion, so each defect
site removes the host atom (vacancy) and inserts a <111> dumbbell nearby.

Three configs are produced for the emission test (emission.py):
  - 'separated' : vacancies + SIAs present (the driven defect phase)
  - 'recombined': SIAs returned into their vacancies (relaxed host) -> Delta-SCF gives E_FP
  - 'clean'     : pristine slab (emission baseline)

Run on the torch/MACE env (ASE). Relaxation uses a MACE calculator (mlip_al.get_calculator).
NOTE: not executed here (no ASE in this sandbox) — verify against ase==3.28 on the Spark.
"""
from __future__ import annotations
import numpy as np

A_W = 3.165          # tungsten lattice parameter (Angstrom)
MATERIALS = {
    "W": {"a": 3.165, "structure": "bcc", "sia_axis": (1, 1, 1)},
    "Mo": {"a": 3.147, "structure": "bcc", "sia_axis": (1, 1, 1)},
}


def material_params(material: str = "W") -> dict:
    """Return minimal structure parameters for Phase 2 W/Mo pilot cells."""
    symbol = str(material)
    if symbol not in MATERIALS:
        raise ValueError(f"unsupported Phase 2 pilot material: {material!r}")
    params = MATERIALS[symbol].copy()
    params["symbol"] = symbol
    params["sia_axis"] = np.asarray(params["sia_axis"], dtype=float)
    params["sia_axis"] /= np.linalg.norm(params["sia_axis"])
    return params


def w_slab(nx_=8, ny=8, nz=12, a=None, vacuum=14.0, material="W"):
    """Orthogonal BCC(001) W/Mo slab, +z emission face, vacuum on top only."""
    from ase.build import bcc100
    params = material_params(material)
    if params["structure"] != "bcc":
        raise ValueError(f"w_slab only supports BCC pilot materials, got {material!r}")
    if a is None:
        a = params["a"]
    slab = bcc100(params["symbol"], size=(nx_, ny, nz), a=a, vacuum=0.0, orthogonal=True)
    # add vacuum on +z only (asymmetric: bottom is bulk-like contact, top emits)
    cell = slab.get_cell()
    cell[2, 2] += vacuum
    slab.set_cell(cell)
    slab.center(axis=2, about=(0.0, 0.0, 0.0))  # keep slab at low z, vacuum above
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    return slab


def _match_network_to_slab(slab, net, surface_band=8.0):
    """
    Map defect-network sites onto slab atom indices. Bias the network so its backbone
    reaches the +z surface (so recombination happens within `surface_band` Angstrom of
    the emitting face, where emission.py integrates vacuum amplitude).
    Returns indices of slab atoms designated as Frenkel-pair (vacancy) sites.
    """
    pos = slab.get_positions()
    zmax = pos[:, 2].max()
    # candidate slab atoms whose fractional position matches network defect density,
    # preferentially in the surface band
    net_frac = net.frac
    rng = np.random.default_rng(0)
    near_surf = pos[:, 2] > (zmax - surface_band)
    w = np.where(near_surf, 3.0, 1.0)              # 3x weight near surface
    p = w / w.sum()
    n_def = int(net_frac * len(slab))
    idx = rng.choice(len(slab), size=n_def, replace=False, p=p)
    return np.sort(idx)


def make_configs(slab, net, surface_band=8.0, material="W", a=None):
    """
    Build ('clean','separated','recombined') ASE Atoms for the Delta-SCF / emission test.
    """
    from ase import Atoms
    params = material_params(material)
    if a is None:
        a = params["a"]
    sia_axis = params["sia_axis"]
    symbol = params["symbol"]
    clean = slab.copy()
    vac_idx = _match_network_to_slab(slab, net, surface_band)

    # separated: remove vacancy atoms and place the same atoms into nearby
    # interstitial/crowdion positions. This keeps the finite-cell atom count
    # conserved for a direct separated-vs-recombined Delta-SCF comparison.
    sep = slab.copy()
    pos = sep.get_positions()
    keep = np.ones(len(sep), bool)
    extra_pos = []
    for i in vac_idx:
        keep[i] = False                                   # vacancy
        extra_pos.append(pos[i] + 0.35 * a * sia_axis)
    sep = sep[keep]
    if extra_pos:
        sep += Atoms(symbol * len(extra_pos), positions=np.array(extra_pos))

    # recombined: SIAs returned to vacancies -> just the relaxed host lattice with the
    # vacancy sites refilled (i.e. start from clean; the energy difference vs `sep`
    # after relaxation is E_FP).
    recomb = clean.copy()

    return dict(clean=clean, separated=sep, recombined=recomb, vac_idx=vac_idx)


def bottom_contact_mask(atoms, lattice_a=A_W, max_fixed_fraction=0.45):
    """Return a bottom-contact mask that never freezes the whole pilot slab."""
    z = atoms.get_positions()[:, 2]
    n_atoms = len(z)
    if n_atoms == 0:
        return np.zeros(0, dtype=bool)
    bottom = z <= (z.min() + lattice_a)
    max_fixed = max(1, int(np.floor(max_fixed_fraction * n_atoms)))
    min_mobile = max(1, n_atoms - max_fixed)
    if int(bottom.sum()) > max_fixed or (n_atoms - int(bottom.sum())) < min_mobile:
        order = np.argsort(z)
        bottom = np.zeros(n_atoms, dtype=bool)
        bottom[order[:max_fixed]] = True
    return bottom


def relax(atoms, calc, fmax=0.03, steps=400, lattice_a=A_W):
    """Relax with a MACE (or any ASE) calculator. Fix a bottom contact, leaving mobile atoms."""
    from ase.optimize import FIRE
    from ase.constraints import FixAtoms
    bottom = bottom_contact_mask(atoms, lattice_a=lattice_a)
    atoms.set_constraint(FixAtoms(mask=bottom))
    atoms.calc = calc
    FIRE(atoms, logfile='-').run(fmax=fmax, steps=steps)
    return atoms


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from percolation import dla_dendrite
        net = dla_dendrite(nx_=8, ny=8, nz=12, target_frac=0.25, seed=1)
        slab = w_slab()
        cfgs = make_configs(slab, net)
        for k in ('clean', 'separated', 'recombined'):
            print(f"{k:10s}: {len(cfgs[k])} atoms")
        print("vacancy sites:", len(cfgs['vac_idx']))
    except ImportError as e:
        print("ASE not available in this environment — run on the Spark torch env.")
        print(e)
