# Pre-registration: Near-surface W Frenkel-pair field NEB

**Frozen at commit:** not applicable; `flash-modelling` is not currently inside a git repository.
**Frozen before outcome inspection:** yes; no no-field or field-wrapper CHGNet NEB barriers for this fixed endpoint have been computed before this registration.
**Question doc:** `docs/science-superpowers/questions/2026-06-04-near-surface-w-frenkel-field-neb.md`
**Prior-work note:** `docs/science-superpowers/prior-work/2026-06-04-near-surface-w-frenkel-field-neb.md`
**Analysis plan:** `docs/science-superpowers/plans/2026-06-04-near-surface-w-frenkel-field-neb.md`

## Hypotheses

- H0: The explicit-field CHGNet wrapper does not produce a practical qualitative barrier reduction for the fixed near-surface W pathway, or the field run is blocked by non-finite outputs or failed geometry QC.
- H1: The explicit-field CHGNet wrapper produces a practical qualitative barrier reduction for the fixed near-surface W pathway, with finite outputs and geometry QC sufficient to promote selected images to BigDFT validation.

## Primary analysis

- Model/test: paired NEB comparison on identical endpoints.
- Outcome: `barrier_delta_eV = barrier_field_eV - barrier_no_field_eV`.
- No-field control: `CHGNetCalculator` only.
- Field-primary run: `CHGNetCalculator` wrapped by an ASE calculator adding fixed-charge `F = qE` and corresponding linear field energy.
- Material: `W`.
- Morphology: `ordered`.
- Cell shape: `(3, 3, 4)`.
- Defect fraction target: `0.25`.
- Realized defect fraction in the frozen tiny ordered endpoint: `0.3333333333333333`, as recorded in `runs/chgnet_field_neb_20260604_1119/endpoint_manifest.json` before NEB barrier execution.
- Endpoint construction: generated near-surface pilot `separated` configuration; selected mobile atom is the highest-`z` atom not fixed by the bottom-contact constraint; final endpoint displaces this atom by `0.45 A` along normalized `[1, 1, 0]`.
- Field direction: normalized endpoint displacement direction.
- Field strength: `0.2 V/A`.
- Charge assignment: selected mobile atom charge `+1.0 e`; every other atom `0.0 e`.
- NEB images: `5` intermediate images plus endpoints.
- Optimizer: ASE `FIRE`.
- Convergence and stopping: `fmax=0.05 eV/A`, maximum `60` optimizer steps.

## Prediction

- Direction: `barrier_delta_eV < 0`.
- Practical threshold: `barrier_delta_eV <= -0.05 eV`.
- Rationale: changes smaller than `0.05 eV` are not promotion-worthy for this first screen because W SIA barriers can be meV-scale and CHGNet transition-state error may exceed such changes.

## Decision rule

- Confirm H1 as a **positive exploratory screen** if:
  - no-field and field-primary NEBs both complete,
  - both barriers are finite,
  - all endpoint and image geometry QC checks pass,
  - `barrier_delta_eV <= -0.05 eV`.
- Disconfirm / null if:
  - both NEBs complete and QC passes but `barrier_delta_eV > -0.05 eV`.
- Blocked if:
  - either NEB fails,
  - any required barrier or image energy is non-finite,
  - endpoint or image geometry QC fails.

## Geometry QC

Endpoint QC:

- finite positions,
- same atom count and cell between endpoints,
- selected mobile atom displacement is `0.45 +/- 1e-6 A`,
- endpoint minimum pair distance is greater than `1.5 A`,
- bottom-contact fixed atoms are unchanged.

Image QC:

- finite positions for every image,
- minimum pair distance greater than `1.5 A` for every image,
- maximum adjacent-image displacement of the selected mobile atom less than `1.0 A`,
- same atom count and cell across all images.

## Sample size and stopping

- N is fixed at one primary pathway and two primary paired NEB runs: no-field and field-primary.
- No optional stopping.
- Failed or negative primary runs will not be replaced by a different endpoint to relabel the primary test as positive.

## Multiplicity

- One confirmatory exploratory comparison: field-primary versus no-field.
- Any additional field strengths, charge magnitudes, directions, morphologies, materials, or endpoint definitions are exploratory and reported separately.

## Secondary and exploratory

- Zero-charge field wrapper is validation only and should reproduce no-field forces/energies.
- Alternative charge values, including `+0.5 e` or `-1.0 e` on the mobile atom, may be run only after the primary comparison and must be labeled exploratory.
- Surface-normal field direction may be run only after the primary comparison and must be labeled exploratory.

## Planned deviations handling

Any deviation from the endpoint, field, charge, NEB, QC, or decision rules above will be documented in the report and renders the affected result exploratory only.
