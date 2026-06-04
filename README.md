# flash-electrodefect-sim

Simulate hot-electron emission from Frenkel-pair recombination in a percolating defect phase
of driven tungsten, on two DGX Sparks. Companion to the *electrodefect effect* manuscript.

**Start with `CURSOR_SPEC.md`** — it is the full spec (physics, hardware split, data
contracts, ===VERIFY=== API list, acceptance criteria, manuscript edits). This README is the
quick operational guide.

## What's here

| Module | Status | What it does |
|---|---|---|
| `percolation.py` | tested | DLA-dendrite / random FP network; D_f, spectral dim d_s, spanning |
| `mechanism_table.py` | tested | σ(T) & I-V mechanism elimination + selection criteria + verdict |
| `regime_decision.py` | tested | PRE-REGISTERED blind regime rule (miniband/Anderson/fracton) + W/Mo check |
| `emission.py` | tested | vacuum-LDOS above φ_eff at driven Te; clean-vs-recombined contrast; escape budget |
| `transport_kpm.py` | written | torch KPM DOS / Kubo-Greenwood σ(E) / IPR / ⟨r⟩ / ξ (run on GPU) |
| `build.py` | written | W(001) slab + vacuum; vacancy+⟨111⟩ SIA; clean/separated/recombined configs |
| `dft_bigdft.py` | skeleton | shell-out BigDFT driver (surface BC, finite-Te, vacuum-LDOS export) |
| `mlip_al.py` | skeleton | MACE committee active-learning; SED phonon lifetimes; e-ph β; NEB E_hop |

"tested" = executed and verified in authoring. "written" = complete but validate on first run
(no ASE/torch in the authoring sandbox). "skeleton" = intent + data contract; fill the
===VERIFY=== API points against the installed PyBigDFT / mace-torch.

## Environments (two, by design)

- **Torch/MACE stack** (Stages 0,1,2,4,5,6): `source env/activate_dgx.sh`. Has mace-torch,
  torch+cu130, ase, pymatgen, numpy/scipy/networkx, lammps ML-IAP, wandb.
- **BigDFT** (Stage 3): its own micromamba env, `source use_bigdft.sh`. `dft_bigdft.py`
  shells out to it — never imported alongside torch.

## Run order

```bash
source env/activate_dgx.sh
python scripts/00_build_percolation.py      # geometry + d_s classification  (CPU, instant)
python scripts/01_mlip_active_learning.py   # MACE AL: label high-σ frames w/ BigDFT (GPU+CPU)
python scripts/02_md_phonons_neb.py         # morphology, SED lifetimes, e-ph β, E_hop (GPU)
python scripts/03_bigdft_emission.py        # GO/NO-GO: Δ-SCF E_FP + vacuum-LDOS at Te (CPU shell-out)
python scripts/04_transport.py              # KPM σ(E,Te), IPR, ⟨r⟩, ξ: ordered vs disordered (GPU)
python scripts/05_mechanism_table.py        # dominant transport mechanism + verdict (CPU)
```

All parameters live in `config/w_percolation.yaml` (defect fraction, Te=0.72 eV, slab size,
t_hop, KPM moments, MPI ranks, etc.). Edit there, not in code.

## The central result (Stage 3, acceptance A3)

PASS if `emission.emission_contrast(clean, recombined) > 5` at Te = 0.72 eV, with
Δ-SCF E_FP ∈ [8,13] eV for W and escape budget E_escape > 0. That is the static proof that
**recombination, not temperature**, fills vacuum-coupled states above the surface barrier.

## Honest scope

Static Δ-SCF energetics + elevated-Te vacuum-LDOS **demonstrate the channel**; they are a
proxy for the driven non-equilibrium state, not that state itself. Real-time emission
dynamics (RT-TDDFT/Ehrenfest) and full non-equilibrium transport (NEGF/Floquet-Keldysh) are
out of scope on this hardware. Say so in Methods.
