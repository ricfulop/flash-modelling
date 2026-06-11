# Deviation note: DFT label engine pivot (BigDFT → QE GPU primary)

**Date:** 2026-06-10
**Applies to:** `2026-06-07-matterchat-first-electrodefect-modelling.md` (frozen preregistration), Task 3 (DFT recombination energetics).

## What changed

The frozen plan specified BigDFT CPU as the conservative primary label path with QE 7.5 GPU as cross-check. Eight BigDFT queue attempts (serial, parallel, energy-only, coarse-grid, local and dgx-peer) all timed out with zero total-energy scalars; the failures are recorded in `runs/autonomous_electrodefect_20260607_041730/autonomous_manifest.json` under `timed_out_queues`. The primary label engine is now **QE 7.5 GPU** (`scripts/27_phase2_qe_gpu_labels.py`, validated NVHPC CUDA build), on the identical frozen label queue (`geometry_repair_convergence_gate/label_queue_manifest.json`, 4 matched W/Mo random separated/recombined jobs).

## What did not change

- The four label structures, their geometry gate, and the pairing rule.
- The recombination-energy definition `E_FP = E_sep − E_rec` on matched converged labels only.
- Escape-budget assumptions (`phi_eff = 4.55 eV`, frozen `n_hop`, `E_hop`, loss scenarios).
- Emission contrast gate (`> 5`), transport gate (DFT-derived Hamiltonians required), TDDFT exclusion.
- No outcome (energy, escape budget, contrast) had been observed from any engine when this pivot was decided; the pivot is engine-availability-driven, not result-driven.

## Status implication

Because the engine differs from the frozen registration, the first QE-labelled results are reported as **engine-deviation confirmatory** (same hypotheses, same gates, documented engine swap) rather than silently confirmatory. A single pre-specified SCF retry (mixing_beta 0.15, electron_maxstep 200) and one cutoff-escalation check (ecutwfc 60 / ecutrho 480 / kgrid 3 3 1) are fixed now, before any QE outcome is seen.

## New exploratory additions (not confirmatory)

- KPM defect-fraction sweep (0.25–0.70, n≥3) for onset/Ioffe-Regel mapping per `docs/science-superpowers/plans/2026-06-10-onset-ir-emission-modelling-loop.md`.
- Ordered-0.55 DOS bimodality anomaly investigation (no data dropped).
