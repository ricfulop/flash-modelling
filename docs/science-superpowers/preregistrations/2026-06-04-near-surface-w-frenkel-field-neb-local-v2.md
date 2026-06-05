# Pre-registration: Near-surface W local Frenkel-pair field NEB v2

**Frozen at commit:** not applicable; `flash-modelling` has a dirty working tree and many pre-existing uncommitted files.
**Frozen before v2 outcome inspection:** yes; no no-field or field-wrapper CHGNet NEB barriers for the local-v2 endpoint have been computed before this registration.
**Question doc:** `docs/science-superpowers/questions/2026-06-04-near-surface-w-frenkel-field-neb.md`
**v1 blocked-run diagnostic:** `runs/chgnet_field_neb_20260604_1119/neb_continuity_diagnostic.md`
**Analysis plan:** `docs/science-superpowers/plans/2026-06-04-near-surface-w-frenkel-field-neb-local-v2.md`

## Hypotheses

- H0: The explicit-field CHGNet wrapper does not produce a practical qualitative barrier reduction for the fixed local-v2 near-surface W pathway, or the local-v2 run is blocked by non-finite outputs, failed geometry QC, or atom-role exchange/path instability.
- H1: The explicit-field CHGNet wrapper produces a practical qualitative barrier reduction for the fixed local-v2 near-surface W pathway, with finite outputs and geometry QC sufficient to promote selected images to BigDFT validation.

## Primary analysis

- Model/test: paired NEB comparison on identical local-v2 endpoints.
- Outcome: `barrier_delta_eV = barrier_field_eV - barrier_no_field_eV`.
- Profile: `local_v2`.
- No-field control: `CHGNetCalculator` only.
- Field-primary run: `CHGNetCalculator` wrapped by an ASE calculator adding fixed-charge `F = qE` and corresponding linear field energy.
- Material: `W`.
- Morphology: `ordered`.
- Cell shape: `(3, 3, 4)`.
- Defect fraction target: `0.25`; realized fraction in the frozen local-v2 endpoint is `0.3333333333333333`, as recorded in `runs/chgnet_field_neb_local_v2_20260604_1219/endpoint_manifest.json` before v2 NEB barrier execution.
- Endpoint construction: generated near-surface pilot `separated` configuration; selected mobile atom is the highest-`z` atom not fixed by the bottom-contact constraint; final endpoint displaces this atom by `0.20 A` along normalized `[1, 1, 0]`.
- Field direction: normalized endpoint displacement direction.
- Field strength: `0.2 V/A`.
- Charge assignment: selected mobile atom charge `+1.0 e`; every other atom `0.0 e`.
- NEB images: `7` intermediate images plus endpoints.
- Optimizer: ASE `FIRE`.
- Convergence and stopping: `fmax=0.05 eV/A`, maximum `80` optimizer steps.

## Prediction

- Direction: `barrier_delta_eV < 0`.
- Practical threshold: `barrier_delta_eV <= -0.02 eV`.
- Rationale: the local-v2 endpoint displacement is smaller than v1, so the maximum direct linear field-energy scale is smaller. A `0.02 eV` threshold is used as a practical screen while remaining above meV-scale numerical noise.

## Decision rule

- Confirm H1 as a **positive exploratory screen** if:
  - no-field and field-primary NEBs both complete,
  - both barriers are finite,
  - endpoint and image geometry QC checks pass,
  - continuity diagnostic does not identify atom-role exchange or endpoint path instability,
  - `barrier_delta_eV <= -0.02 eV`.
- Disconfirm / null if:
  - both NEBs complete,
  - QC and continuity diagnostics pass,
  - `barrier_delta_eV > -0.02 eV`.
- Blocked if:
  - either NEB fails,
  - any required barrier or image energy is non-finite,
  - endpoint or image geometry QC fails,
  - continuity diagnostic identifies atom-role exchange or endpoint path instability.

## Geometry QC

Endpoint QC:

- finite positions,
- same atom count and cell between endpoints,
- selected mobile atom displacement is `0.20 +/- 1e-6 A`,
- endpoint minimum pair distance is greater than `1.5 A`,
- bottom-contact fixed atoms are unchanged.

Image QC:

- finite positions for every image,
- minimum pair distance greater than `1.5 A` for every image,
- maximum adjacent selected-mobile displacement less than `0.5 A`,
- same atom count and cell across all images,
- selected mobile atom remains nearest the intended local linear target in the post-run continuity diagnostic.

## Sample size and stopping

- N is fixed at one local-v2 pathway and two primary paired NEB runs: no-field and field-primary.
- No optional stopping.
- Failed, blocked, or negative local-v2 runs will not be replaced by a different endpoint to relabel the primary test as positive.

## Multiplicity

- One confirmatory exploratory comparison: field-primary versus no-field on local-v2.
- V1 remains a blocked exploratory predecessor and is not combined with v2.
- Any additional endpoint displacement, field strength, charge magnitude, image count, or path constraint is exploratory unless separately preregistered.

## Secondary and exploratory

- Continuity diagnostics are required for QC and do not change the primary barrier calculation.
- Alternative atom-exchange pathway definitions may be designed after this run, but they are not part of local-v2.

## Planned deviations handling

Any deviation from the endpoint, field, charge, NEB, QC, or decision rules above will be documented in the report and renders the affected result exploratory only.
