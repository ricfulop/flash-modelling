# Pre-registration: MatterChat-first electrodefect modelling

**Frozen at:** 2026-06-07T04:15:43-04:00 before confirmatory execution
**Question doc:** `docs/science-superpowers/questions/2026-06-07-matterchat-first-electrodefect-modelling.md`
**Analysis plan:** `docs/science-superpowers/plans/2026-06-07-matterchat-first-electrodefect-modelling.md`
**Prior-work note:** `docs/science-superpowers/prior-work/2026-06-07-matterchat-first-electrodefect-modelling.md`

## Status before freeze

The following setup artifacts already exist and are treated as setup or exploratory inputs, not confirmatory outcomes:

- Geometry-QC run: `runs/matterchat_geometry_qc_20260607_022909/geometry_qc_summary.json`
- ASE gallery: `runs/matterchat_geometry_qc_20260607_022909/ase_gallery/index.html`
- Pilot MatterChat response: `runs/matterchat_geometry_qc_20260607_022909/matterchat_responses/W_dendritic_separated_response.json`
- Pilot MatterChat response index: `runs/matterchat_geometry_qc_20260607_022909/matterchat_response_index.json`
- MatterChat plan-review synthesis: `runs/matterchat_geometry_qc_20260607_022909/matterchat_plan_review/review_synthesis.md`

The pilot MatterChat response suggested three qualitative motifs: vacancy-interstitial proximity, under-coordinated surface atoms, and crowdion-like alignments. A later MatterChat plan-review pass was mostly non-admissible: overlong prompts failed, shortened prompts returned empty responses, and native structure prompts returned generic or incoherent output. The motifs are therefore advisory annotations only; they are not evidence for electrodefect emission and do not define the confirmatory candidate set.

## Hypotheses

- H0: The pre-specified W/Mo Frenkel-pair modelling variants will not pass the full physics evidence chain. At least one required link will fail, remain proxy-only, or be blocked by geometry, DFT convergence, insufficient escape budget, weak emission proxy, or lack of DFT-derived transport support.
- H1 (directional): At least one geometry-repaired W or Mo Frenkel-pair modelling variant from the pre-specified material/morphology grid will pass the full evidence chain: stable geometry, converged DFT recombination energetics, positive escape budget, static emission-proxy contrast above the project gate, and DFT-derived transport/localization diagnostics consistent with the electrodefect mechanism. MatterChat feedback may annotate why a candidate is interesting only if it passes the response-quality gate.

## Primary analysis (exact)

- Model/test: staged deterministic gate, not a statistical model.
- Unit of analysis: one material/morphology/configuration family with matched separated and recombined structures.
- Candidate materials: `W`, `Mo`.
- Candidate morphologies: `ordered`, `random`, `dendritic`.
- Advisory MatterChat motif annotations from the pilot:
  - vacancy-interstitial proximity;
  - under-coordinated surface atoms;
  - crowdion-like alignments.
- MatterChat response-quality gate:
  - response state must be `completed`;
  - output must be non-empty;
  - output must be structure- or material-specific;
  - output must contain no obvious material hallucination;
  - output must map to a geometry, MLIP, DFT, transport, or TDDFT check.
- If MatterChat output fails this gate, it is logged as non-admissible and cannot alter confirmatory candidate choice, thresholds, or decision rules.
- Required structure family: matched `separated` and `recombined` structures generated from the same material, morphology, slab size, boundary setup, and defect builder.
- Confirmatory pipeline:
  1. MLIP relaxation/repair as screening only.
  2. Geometry validation after repair.
  3. BigDFT CPU labels as the conservative DFT path.
  4. QE GPU labels as cross-check only for geometry-clean cases.
  5. Recombination energy from matched converged labels.
  6. Escape-budget calculation under fixed assumptions.
  7. Static emission proxy using existing emission code and converged artifacts.
  8. Transport/localization using DFT-derived Hamiltonians; geometry-proxy KPM remains exploratory.

## Variables and operationalizations

- Geometry pass: finite positions and cell; intended atom count preserved; no atom collapse into vacuum; post-repair minimum pair distance `>= 1.8 A`, unless explicitly documented as a stable dumbbell motif.
- DFT convergence: finite energies, matched settings within a separated/recombined pair, no NaN values, and no unresolved warnings that prevent interpretation.
- Recombination energy: `E_FP = E_separated - E_recombined` for matched converged DFT labels.
- Escape budget: `E_escape = E_FP - n_hop * E_hop - phi_eff`.
- Primary `phi_eff`: repo default `4.55 eV` for W unless replaced by a converged work-function label before outcome inspection for that pair. For Mo, record the chosen value before calculating `E_escape`.
- Primary loss scenario: `n_hop = 1`, `E_hop = 1.0 eV`.
- Sensitivity loss scenarios: `(n_hop, E_hop) = (0, 0.0 eV)`, `(2, 1.0 eV)`, `(3, 1.0 eV)`.
- Static emission proxy: existing project gate `emission_contrast(clean, recombined) > 5`, using only converged DFT artifacts.
- Transport/localization: DOS, Kubo conductivity, IPR, level-spacing ratio, and localization length using DFT-derived Wannier/downfolded or equivalent Hamiltonians.

## Inclusion/exclusion criteria

- Include only structures derived from `scripts/29_matterchat_geometry_qc.py` or a documented repair run that links back to those CIFs.
- Include only matched separated/recombined pairs for DFT recombination-energy claims.
- Exclude any DFT label with non-convergence, NaN, mismatched settings, missing required artifacts, or unresolved warnings.
- Exclude geometry-proxy KPM from confirmatory transport claims.
- Exclude MatterChat text from all quantitative evidence gates.
- Exclude non-admissible MatterChat outputs from candidate selection and plan revision.
- Exclude any post-hoc motif, threshold, loss scenario, or morphology added after confirmatory execution begins from confirmatory claims.

## Prediction

- Direction: the strongest candidate, if any, is expected to be a separated near-surface Frenkel-pair structure from the pre-specified W/Mo grid that remains geometrically stable after relaxation and has `E_escape > 0` under the primary loss scenario.
- Expected magnitude: no defensible prior quantitative effect size is available for MatterChat usefulness. The physics magnitude gate is categorical: `E_escape > 0` and `emission_contrast > 5`.

## Decision rule

- Confirm H1 if at least one matched material/morphology pair satisfies all of the following:
  - geometry repair/validation passes;
  - matched DFT labels converge with interpretable warnings;
  - `E_FP` is finite and positive;
  - primary-scenario `E_escape > 0`;
  - static emission contrast is `> 5`;
  - DFT-derived transport/localization diagnostics support localization/Ioffe-Regel-compatible behavior under frozen thresholds.
- Disconfirm H1 if all geometry-valid matched pairs either have non-positive primary-scenario `E_escape`, emission contrast `<= 5`, or DFT-derived transport diagnostics that contradict localization/Ioffe-Regel-compatible behavior.
- Classify as blocked, not negative, if no matched pair reaches the DFT/emission/transport gates because of geometry failure, non-convergence, missing Hamiltonians, or missing artifacts.
- Classify as partial support if DFT energetics and escape budget pass but emission proxy or DFT-derived transport remains missing, warning-limited, or proxy-only.

## Sample size and stopping

- Fixed candidate set for the first confirmatory pass: `W` and `Mo` across `ordered`, `random`, and `dendritic` morphologies, with matched `separated` and `recombined` structures derived from `runs/matterchat_geometry_qc_20260607_022909`.
- No optional stopping by success. If one pair passes early, still evaluate the remaining pre-specified candidate set or explicitly mark unevaluated pairs as not run.
- No candidate expansion after seeing DFT, emission, or transport outcomes. New candidates require a new preregistration.

## Multiplicity

- Number of confirmatory family-level tests: one evidence-chain test per material/morphology family, maximum six in the first pass.
- Primary conclusion is family-level: at least one full-chain pass supports the existence of a plausible modelling route, not universality across all morphologies.
- No p-value correction is applicable because the primary decision is a deterministic gate. The multiplicity risk is handled by reporting every family, including failures and blocked cases, rather than selecting only the best result.

## Secondary and exploratory analyses

These may be run and reported separately, but cannot support the primary confirmatory claim:

- Additional MatterChat prompts beyond the fixed motif set.
- Any MatterChat response that fails the response-quality gate.
- Geometry-proxy KPM without a DFT-derived Hamiltonian.
- Alternative loss scenarios beyond the primary scenario.
- SALMON-TDDFT or Octopus prototypes unless separately preregistered.
- Alternative MLIP calculators, field wrappers, charge assignments, or endpoint construction variants.
- Any newly discovered motif not in the fixed three-motif list.

## Planned deviations handling

Any deviation from this preregistration will be documented in the final report. The affected analysis becomes exploratory unless a fresh preregistration is created before unused outcomes are inspected.

## Verification checklist

- [x] Hypotheses written, with directional H1.
- [x] Primary analysis fully specified as a staged deterministic gate.
- [x] A disconfirming or blocked result is stated explicitly.
- [x] Decision rule is exact and result-independent.
- [x] Candidate set and stopping rule are fixed.
- [x] Multiplicity is handled by full reporting across all six material/morphology families.
- [x] Exploratory analyses are listed and labeled.
- [ ] Pre-registration committed to git before confirmatory DFT/emission/transport outcomes are observed.
