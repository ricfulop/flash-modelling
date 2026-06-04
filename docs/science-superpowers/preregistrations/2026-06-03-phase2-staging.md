# Pre-registration: Phase 2 staging and promotion gates

**Frozen at commit:** not applicable; `flash-modelling` is not currently inside a git repository.
**Question doc:** `/home/ricfulop/Desktop/Cursor/flash-modelling/CURSOR_SPEC.md`
**Analysis plan:** `/home/ricfulop/.cursor/plans/phase_2_staging_eda54fe5.plan.md`

## Scope

This registration covers Phase 2 smoke tests and pilot promotion gates only. Smoke-test
outputs are exploratory engineering artifacts. They do not support a confirmatory
scientific claim about which defect morphology forms.

## Hypotheses

- H0: The current Phase 2 environment and API wiring are not sufficient to run MLIP-gated
  morphology, NEB, SED, and material cross-check pilots without intervention.
- H1: The current environment and API wiring are sufficient for at least pilot-scale
  execution of those tracks with finite energies, forces, trajectories, and structured
  output artifacts.

## Primary analysis

- Run environment/API smoke checks for ASE, torch/CUDA, MACE, and the BigDFT shell boundary.
- Run toy-scale W pilots for morphology relaxation/anneal, NEB, and SED.
- Promote a track only if its pilot writes complete status artifacts and all numeric
  sanity checks are finite.

## Decision rule

- A track is **promotion-ready** if the intended production calculator is available,
  all pilot outputs are finite, and no geometry sanity guard fails.
- A track is **pilot-only** if it passes with a toy fallback calculator but lacks the
  intended production calculator.
- A track is **blocked** if imports, geometry construction, MD, NEB, or output validation
  fail.

## Sample size and stopping

- Pilot sample is fixed before outcome inspection: W ordered/random/dendritic seed
  morphologies, one tiny NEB path, and one short SED trajectory.
- No optional extension of the pilot set will be used to relabel a blocked track as
  promotion-ready.

## Multiplicity

Four tracks are checked independently: MLIP morphology, NEB barriers, phonon SED/e-ph,
and Mo/Pt cross-check staging. Each track receives its own gate status; no combined
success claim is made.

## Exploratory outputs

Any calculator fallback, toy-force-field result, or small-cell pilot result is exploratory.
Production claims require a later registration using MACE or DFT labels on cells not used
to tune this staging workflow.
