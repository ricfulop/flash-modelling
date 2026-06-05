# Near-Surface W Frenkel-Pair Local Field NEB v2 Analysis Plan

> **For agentic workers:** REQUIRED SUB-SKILL: pre-register this plan with science-superpowers:preregistering-analysis BEFORE execution. Then use science-superpowers:subagent-driven-analysis (recommended) or science-superpowers:executing-analysis to run it step-by-step. Steps use checkbox (`- [ ]`) syntax for tracking.

**Question:** For a smaller local near-surface tungsten displacement pathway that avoids the v1 atom-exchange instability, does the explicit-field ASE wrapper applied to CHGNet qualitatively lower the NEB barrier relative to the no-field CHGNet control?

**Design:** Paired exploratory computational screen on a revised, unused endpoint; the same local initial/final pathway is evaluated with no-field CHGNet and a fixed-charge CHGNet electric-field wrapper.

**Data:** One generated near-surface W local-displacement pathway from the existing `flash-modelling` structure builders; unit of analysis is one fixed NEB pathway and field/charge assignment.

**Primary analysis:** `barrier_delta_eV = barrier_field_eV - barrier_no_field_eV` for the fixed local endpoint, same image count, same optimizer, and same convergence settings.

**Decision rule:** The v2 screen is positive only if both NEBs complete with finite barriers, endpoint and image geometries pass QC, and `barrier_delta_eV <= -0.02 eV`. Otherwise the result is negative or blocked. Any positive result is exploratory and only promotes images to BigDFT validation.

---

## Why v2 exists

The v1 screen in `runs/chgnet_field_neb_20260604_1119` is blocked. Its diagnostic report, `neb_continuity_diagnostic.md`, classified the failure as `endpoint_path_instability_or_atom_role_exchange`: the selected mobile atom was not consistently nearest the intended linear path after NEB relaxation.

This v2 plan does not loosen v1 QC. It creates a fresh, smaller local endpoint before observing any v2 NEB barrier.

## Fixed v2 primary configuration

- Profile: `local_v2`
- Material: `W`
- Structure source: same generated Phase-2 seed-builder path as v1
- Morphology: `ordered`
- Defect fraction target: `0.25`
- Cell shape: `(3, 3, 4)`
- Surface/vacuum model: existing W slab builder with `vacuum=8.0 A`
- Endpoint construction: generated near-surface pilot `separated` configuration; selected mobile atom is the highest-`z` atom not fixed by the bottom-contact constraint
- Endpoint displacement: `0.20 A` along normalized `[1, 1, 0]`
- Field direction: normalized endpoint displacement direction
- Primary field strength: `0.2 V/A`
- Primary charge model: selected mobile atom has charge `+1.0 e`; all other atoms have charge `0.0 e`
- Calculator: `chgnet.model.CHGNetCalculator`
- NEB images: `7` intermediate images, plus initial and final
- NEB optimizer: ASE `FIRE`
- NEB convergence: `fmax=0.05 eV/A`, max `80` optimizer steps
- Minimum pair-distance QC threshold: `1.5 A`
- Adjacent selected-mobile displacement QC threshold: `< 0.5 A`

## Artifacts and data flow

### Task 1: Setup and validation

**Artifacts:**
- Reads: `env/activate_flash_chgnet.sh`
- Reads: `scripts/23_chgnet_field_neb.py`
- Writes: `runs/chgnet_field_neb_local_v2_<timestamp>/setup_smoke.json`
- Writes: `runs/chgnet_field_neb_local_v2_<timestamp>/validation_summary.json`

- [ ] **Step 1: Validate CHGNet setup**

Run `setup-smoke` with profile `local_v2`. Expected: CUDA visible and `CHGNetCalculator` available.

- [ ] **Step 2: Validate the field wrapper**

Run `validate`. Expected: sign/shape and zero-charge identity tests pass.

### Task 2: Endpoint freeze

**Artifacts:**
- Writes: `runs/<run_root>/endpoints/initial.xyz`
- Writes: `runs/<run_root>/endpoints/final.xyz`
- Writes: `runs/<run_root>/endpoint_manifest.json`

- [ ] **Step 1: Generate local-v2 endpoints**

Write initial/final structures and a manifest that records the realized defect fraction, mobile atom index, displacement vector, charge summary, field settings, image count, and profile name.

- [ ] **Step 2: Validate endpoint geometry before NEB**

Endpoint QC must pass:

- finite positions
- same atom count and cell
- selected mobile atom displacement is `0.20 +/- 1e-6 A`
- minimum pair distance greater than `1.5 A`
- bottom-contact fixed atoms unchanged

### Task 3: Primary v2 NEB comparison

**Artifacts:**
- Writes: `runs/<run_root>/no_field/neb_summary.json`
- Writes: `runs/<run_root>/field_primary/neb_summary.json`
- Writes: `runs/<run_root>/comparison_summary.json`

- [ ] **Step 1: Run no-field control**

Use `CHGNetCalculator` only.

- [ ] **Step 2: Run field-primary**

Use the same endpoints and settings, wrapping the CHGNet calculator with the explicit fixed-charge field calculator.

- [ ] **Step 3: Compare barriers**

Compute `barrier_delta_eV = barrier_field_eV - barrier_no_field_eV`.

### Task 4: QC and promotion gate

**Artifacts:**
- Writes: `runs/<run_root>/promotion_recommendation.md`
- Writes: `runs/<run_root>/neb_continuity_diagnostic.json`
- Writes: `runs/<run_root>/neb_continuity_diagnostic.md`

- [ ] **Step 1: Apply finite-output checks**

Both runs must have finite endpoint energies, image energies, and barriers.

- [ ] **Step 2: Apply geometry/image checks**

For every saved image, check:

- finite positions
- minimum pair distance greater than `1.5 A`
- maximum adjacent selected-mobile displacement less than `0.5 A`
- selected mobile atom remains nearest the intended local linear target in the post-run diagnostic
- no atom-count or cell mismatch

- [ ] **Step 3: Apply decision rule**

Classify:

- `positive_exploratory`: finite runs, QC pass, continuity diagnostic does not detect atom-role exchange, and `barrier_delta_eV <= -0.02 eV`
- `negative`: finite runs and QC pass but `barrier_delta_eV > -0.02 eV`
- `blocked`: non-finite outputs, failed geometry QC, execution failure, or continuity diagnostic indicating atom-role exchange/path instability

- [ ] **Step 4: BigDFT promotion recommendation**

Recommend BigDFT validation only for `positive_exploratory` and include the selected initial, highest-energy image, and final image paths.

## Scope exclusions

- V2 does not rescue or relabel the blocked v1 run.
- No manuscript claim is made from CHGNet field-wrapper barriers.
- No CHGNet `magmom` values are used as charges.
- Any alternative endpoint, displacement, charge magnitude, field strength, or image count is exploratory unless separately preregistered.
