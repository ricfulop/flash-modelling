# Pre-registration: Phase 1 remaining-run completion

**Frozen at commit:** not applicable; `flash-modelling` is not currently inside a git repository.  
**Frozen timestamp:** 2026-06-04T01:39Z.  
**Question doc:** `/home/ricfulop/Desktop/Cursor/flash-modelling/CURSOR_SPEC.md`  
**Current status artifact:** `/home/ricfulop/Desktop/Cursor/flash-modelling/runs/phase1_full_digest_current/claim_boundaries.csv`

## Scope

This registration covers the remaining Phase 1 execution needed after the existing Tier A/D-lite and provisional Tier B package:

- Tier B convergence-qualified work-function differences for clean, defect, and stepped W(001) slabs.
- Tier C paired Delta-SCF `E_FP(W)` from separated and recombined small slabs.
- Tier C emission GO/NO-GO only if BigDFT LDOS/KS cube artifacts are actually produced and pass the `emission.py` contrast gate.
- Final synthesis of all Phase 1 tiers into a single manifest and claim-boundary table.

It does not cover Phase 2 MLIP morphology selection, Wannier/DFPT, RT-TDDFT/NAMD, production Anderson-Holstein dynamics, or kMC.

## Hypotheses / gates

- H0: Remaining Phase 1 cannot produce accepted Tier B/C gate artifacts with the current CPU BigDFT path and available run infrastructure.
- H1: Remaining Phase 1 produces accepted or explicitly bounded artifacts for every Tier A/B/C/D deliverable, with any nonaccepted component labeled provisional, missing, or blocked.

## Primary execution

1. Use CPU BigDFT only for production DFT. CUDA BigDFT is excluded unless the existing validation matrix is passed first.
2. Run only missing/remaining jobs required by the current gate table:
   - recombined small EFP job matching the existing separated small EFP job;
   - optional LDOS/export probe jobs only if the required cube/energy-map contract can be produced without changing the scientific gates;
   - convergence follow-up only for slab roles still needed to evaluate `phase1.py`.
3. Parse all available BigDFT rows with `scripts/10_sunday_synthesis.py`.
4. Generate Tier B/C status with `scripts/03_bigdft_emission.py`.
5. Generate the final digest with `scripts/17_scope_upgrade_digest.py`.

## Decision rules

- Tier A is accepted if the existing 9 morphology/fraction frozen `regime_decision` and `mechanism_summary` artifacts are present and reproduced by the synthesis inputs.
- Tier B is accepted only if the selected clean/defect/stepped slab rows have finite energy/charge/phi, `infocode=0`, no warnings, and no NaN. Otherwise Tier B remains provisional with numeric deltas labeled as such.
- Tier C `E_FP` is accepted only if paired separated and recombined Delta-SCF totals are finite and both jobs pass the same accepted quality gate. Otherwise it is missing/provisional/failed.
- Tier C emission PASS is accepted only if clean and recombined LDOS artifacts exist, satisfy the `emission.py` data contract, and `emission_contrast(...).ratio > 5`. A bounded escape budget without LDOS is not an emission PASS.
- Tier D-lite remains exploratory unless replaced by a true Wannier/DFPT/Anderson-Holstein calculation, which is out of scope here.

## Stopping and failure handling

- Do not retune thresholds, seeds, slab role matching, or selection rules based on favorable output.
- If a BigDFT job fails or produces non-finite output, preserve logs and classify the component as failed/blocked unless the root cause is a clear code/input bug that can be fixed without changing the registered scientific gate.
- If hostname/peer access fails, continue on the local CPU path and record the peer blocker.
- The run stops only when the final digest has been regenerated from the latest completed/failed artifacts and every Phase 1 tier has an accepted/provisional/missing/blocked status.

## Exploratory outputs

Kubo/mobility-window diagnostics, Tier D-lite activation proxies, configured-range escape budgets, and nonaccepted BigDFT deltas are exploratory or bounded support only.
