# Near-Surface W Frenkel-Pair Field NEB Analysis Plan

> **For agentic workers:** REQUIRED SUB-SKILL: pre-register this plan with science-superpowers:preregistering-analysis BEFORE execution. Then use science-superpowers:subagent-driven-analysis (recommended) or science-superpowers:executing-analysis to run it step-by-step. Steps use checkbox (`- [ ]`) syntax for tracking.

**Question:** For a near-surface tungsten Frenkel-pair hop or recombination pathway, does an explicit-field ASE calculator wrapper applied to CHGNet qualitatively lower the NEB migration barrier relative to the no-field CHGNet control, and is any observed trend strong enough to justify BigDFT validation?

**Design:** Paired exploratory computational screen; the same fixed initial/final pathway is evaluated with a no-field CHGNet calculator and a fixed-charge CHGNet electric-field wrapper.

**Data:** One generated near-surface W pilot pathway from the existing `flash-modelling` structure builders; unit of analysis is one fixed NEB pathway and field/charge assignment.

**Primary analysis:** `barrier_delta_eV = barrier_field_eV - barrier_no_field_eV` for the fixed endpoint, same image count, same optimizer, and same convergence settings.

**Decision rule:** The screen is positive only if both NEBs complete with finite barriers, image geometries pass QC, and `barrier_delta_eV <= -0.05 eV`. Otherwise the result is negative or blocked. Any positive result is exploratory and only promotes images to BigDFT validation; it is not a manuscript claim.

---

## Prior-work basis

This plan uses `docs/science-superpowers/prior-work/2026-06-04-near-surface-w-frenkel-field-neb.md`.

Key design implications:

- Self-interstitial motion in W can be meV-scale, while vacancy migration is around `1.6-1.8 eV`; endpoint geometry can dominate barrier interpretation.
- CHGNet barriers are exploratory because universal MLIPs can soften high-energy transition states.
- A fixed-charge `F = qE` wrapper is not a physical electric-field model for metallic W; it is only a qualitative perturbation.
- No CHGNet magnetic moments are used as charges.

## Fixed primary configuration

- Material: `W`
- Structure source: existing Phase-2 seed builder equivalent to `scripts/02_md_phonons_neb.py`
- Morphology: `ordered`
- Defect fraction target: `0.25`
- Realized defect fraction in the frozen tiny ordered endpoint: `0.3333333333333333`
- Cell shape: `(3, 3, 4)`
- Surface/vacuum model: existing W slab builder with `vacuum=8.0 A`
- Endpoint construction: load or generate the pilot `separated` configuration, optionally pre-relax it, then move the selected top mobile atom by `0.45 A` along normalized `[1, 1, 0]`
- Mobile atom selection: highest-`z` atom not fixed by the bottom-contact constraint
- Field direction: normalized mobile-atom displacement direction from initial to final
- Primary field strength: `0.2 V/A`
- Primary charge model: mobile atom has charge `+1.0 e`; all other atoms have charge `0.0 e`
- Calculator: `chgnet.model.CHGNetCalculator`
- NEB images: `5` intermediate images, plus initial and final
- NEB optimizer: ASE `FIRE`
- NEB convergence: `fmax=0.05 eV/A`, max `60` optimizer steps

## Artifacts and data flow

### Task 1: Setup and code validation

**Artifacts:**
- Reads: `env/activate_flash_chgnet.sh`
- Reads: `scripts/22_chgnet_setup_smoke.py`
- Creates: `src/electrodefect/chgnet_field.py`
- Creates: `scripts/23_chgnet_field_neb.py`
- Writes: `runs/chgnet_field_neb_<timestamp>/setup_smoke.json`
- Writes: `runs/chgnet_field_neb_<timestamp>/validation_summary.json`

- [ ] **Step 1: Validate the CHGNet environment**

Run:

```bash
source env/activate_flash_chgnet.sh
python scripts/22_chgnet_setup_smoke.py --output runs/<run_root>/setup_smoke.json
```

Expected: JSON with `state: completed`, `torch_cuda_available: true`, and `chgnet_calculator: CHGNetCalculator`.

- [ ] **Step 2: Validate the electric-field wrapper on a toy calculator**

Use a deterministic two-atom toy calculator with zero base forces. For charges `[+1, -1]`, `field_strength=0.2`, and direction `[1, 0, 0]`, expected field forces are `[[0.2, 0, 0], [-0.2, 0, 0]]`.

- [ ] **Step 3: Validate zero-charge identity**

With all charges zero, wrapper energy and forces must match the base calculator to numerical tolerance. This is required before any real NEB output is inspected.

### Task 2: Endpoint freeze

**Artifacts:**
- Creates: `runs/<run_root>/endpoints/initial.xyz`
- Creates: `runs/<run_root>/endpoints/final.xyz`
- Creates: `runs/<run_root>/endpoint_manifest.json`

- [ ] **Step 1: Generate endpoints**

Use the fixed primary configuration above. Write `initial.xyz`, `final.xyz`, and an `endpoint_manifest.json` containing material, morphology, target defect fraction, realized defect fraction, mobile atom index, displacement vector, charge vector summary, image count, and field settings.

- [ ] **Step 2: Validate endpoint geometry**

Expected validation checks:

- finite positions
- same atom count and cell between initial/final
- mobile atom displacement norm within `0.45 +/- 1e-6 A`
- minimum pair distance in each endpoint greater than `1.5 A`
- fixed bottom-contact atoms unchanged

### Task 3: Primary NEB comparison

**Artifacts:**
- Reads: frozen endpoint files
- Writes: `runs/<run_root>/no_field/neb_summary.json`
- Writes: `runs/<run_root>/field_primary/neb_summary.json`
- Writes: `runs/<run_root>/comparison_summary.json`

- [ ] **Step 1: Run no-field control**

Use `CHGNetCalculator` with no field wrapper. Run NEB with the fixed image count, optimizer, `fmax`, and step limit.

- [ ] **Step 2: Run field-primary NEB**

Use the same endpoints and NEB settings, but wrap the base CHGNet calculator with the fixed-charge electric-field calculator.

- [ ] **Step 3: Compare barriers**

Compute:

```text
barrier_delta_eV = barrier_field_eV - barrier_no_field_eV
```

Do not change the decision threshold after computing this value.

### Task 4: Quality control and promotion gate

**Artifacts:**
- Writes: `runs/<run_root>/promotion_recommendation.md`

- [ ] **Step 1: Apply finite-output checks**

Both NEBs must have finite endpoint energies, finite image energies, and finite barriers.

- [ ] **Step 2: Apply geometry checks**

For every saved image, check:

- finite positions
- minimum pair distance greater than `1.5 A`
- maximum adjacent-image mobile-atom displacement less than `1.0 A`
- no atom count or cell mismatch

- [ ] **Step 3: Apply decision rule**

Classify:

- `positive_exploratory`: finite runs, geometry QC pass, and `barrier_delta_eV <= -0.05 eV`
- `negative`: finite runs but `barrier_delta_eV > -0.05 eV`
- `blocked`: non-finite outputs, failed geometry QC, or execution failure

- [ ] **Step 4: BigDFT promotion recommendation**

Recommend BigDFT validation only for `positive_exploratory` and include the selected initial, highest-energy image, and final image paths.

## Scope exclusions

- No manuscript claim is made from CHGNet field-wrapper barriers.
- No CHGNet `magmom` values are used as charges.
- No endpoint, field strength, charge assignment, image count, or decision threshold is changed after the preregistration.
- Any additional charge assignments or field strengths are exploratory and must be reported separately from the primary comparison.
