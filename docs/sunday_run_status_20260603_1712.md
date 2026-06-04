# Sunday Run Status - 2026-06-03 17:12 EDT

## Current package

- Latest synthesis package: `runs/sunday_synthesis_20260603_171051/`
- Core report: `runs/sunday_synthesis_20260603_171051/SUNDAY_SYNTHESIS.md`
- Manifest: `runs/sunday_synthesis_20260603_171051/manifest.json`
- Figures:
  - `figures/kpm_dos_E0_by_morphology.png`
  - `figures/tier_d_lite_localized_fraction.png`
  - `figures/bigdft_phi_proxy.png`

## Completed Sunday-critical lanes

- KPM aggregation: complete for 9 morphology/fraction groups with 48 repeats per group from local+peer sweep outputs.
- Tier A mechanism/regime package: complete from frozen `regime_decision` and `mechanism_table` outputs.
- BigDFT extraction: complete for currently accessible local CPU BigDFT outputs.
- Tier D-lite proxy: complete as exploratory IPR/mobility-window and Anderson-Holstein proxy outputs.

## Active compute

- Local BigDFT convergence follow-up is running:
  - `runs/bigdft_followup_20260603_171018_spark-5da5/`
  - queued jobs: clean, defect, stepped W(001) small slabs with `nrepmax=4`
  - goal: improve convergence for the provisional work-function comparison
- Watchdog is running:
  - log: `runs/sunday_watchdog.log`
  - behavior: checks active follow-ups every 15 minutes and rebuilds synthesis when a follow-up completes

## Known blockers

- Peer host alias resolution is currently unavailable from this shell (`dgx-peer` and `spark-0808` do not resolve), so the current BigDFT synthesis is local-only.
- The synthesis script is peer-ready and will retry peer import when the watchdog runs with peer import enabled.
- Existing slab BigDFT outputs have usable potential files but nonzero BigDFT infocode/warnings, so work-function numbers remain provisional until follow-up outputs are parsed.

## Deliberately not running

- CUDA BigDFT production jobs: CUDA-linked builds produced NaN/abort validation behavior and are not scientifically safe.
- Two-node MPI BigDFT: not needed for Sunday throughput and was the primary blocker.
- Full Wannier90, EPW/DFPT, RT-TDDFT/NAMD, production Anderson-Holstein, and kMC: Phase 2 scope, not Sunday scope.
