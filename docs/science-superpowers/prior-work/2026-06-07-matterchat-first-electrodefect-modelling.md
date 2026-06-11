# Prior-work note: MatterChat-first electrodefect modelling

**Question doc:** `docs/science-superpowers/questions/2026-06-07-matterchat-first-electrodefect-modelling.md`

## What needs grounding

1. Whether MatterChat can be used in the workflow without becoming evidence.
2. Which local modelling tools are appropriate for each link in the electrodefect chain.
3. Which artifacts are already known to threaten W/Mo Frenkel-pair modelling.
4. Which gates are needed before any result can support the manuscript mechanism.

## Method constraints to carry forward

1. **MatterChat is a structure-to-text assistant, not a physics solver.** The local MatterChat installation accepts a prompt plus either a validation sample or a CIF visible inside the CUDA container. It cannot model applied current, electric field response, electron emission rates, hot-electron energy distributions, work functions, or transport. Use it only to suggest local motifs worth testing, and only when the response is non-empty, structure-specific, and non-hallucinated.

2. **Geometry QC must precede both MatterChat and DFT.** Prior QE label attempts on W/Mo defect slabs encountered very short contacts around `1.25 A`, causing SCF instability. The first gate must therefore be minimum-distance and visual/gallery inspection, not larger DFT throughput.

3. **MLIPs are screening tools for these defect densities.** MACE and CHGNet are available for relaxation, MD, phonon, and NEB prototypes, but dense Frenkel-pair networks, surfaces, and high-energy transition states are out-of-distribution enough that MLIP results cannot be final energetics.

4. **DFT validation must separate convergence from physics.** BigDFT CPU is the conservative path for finite-temperature labels and density/potential artifacts. QE 7.5 GPU is available for plane-wave SCF and Wannier bridge attempts once structures are repaired. GPU BigDFT was not validated for claims on this CUDA/GB10 stack.

5. **Transport claims need real Hamiltonians.** Geometry-proxy KPM is useful for plumbing and exploratory morphology triage. Confirmatory localization or Ioffe-Regel claims require Wannier/downfolded or equivalent DFT-derived Hamiltonians.

6. **Static emission proxies are not real-time emission dynamics.** The existing `emission.py` path tests separated/recombined energetics, escape budget, and vacuum-accessible LDOS contrast. SALMON-TDDFT GPU or Octopus CPU/MPI are follow-up tools only if static energetics and transport survive QC.

## Adopted method

- Use MatterChat only after structures pass minimum-distance QC and only with short prompts that request qualitative motifs. The corrective plan-review pass showed that broad plan-review prompts failed or returned low-quality output.
- Convert an admissible MatterChat suggestion into an explicit modelling annotation or check before running confirmatory tools. If MatterChat output is empty, generic, or hallucinated, fall back to the pre-specified W/Mo material/morphology grid.
- Keep the physics evidence chain on the existing `flash-modelling` stack: builders -> geometry QC -> MLIP screening -> BigDFT/QE labels -> emission proxy -> transport diagnostics -> optional TDDFT cross-check.
- Preserve negative and blocked outcomes. A failed geometry, non-converged DFT label, insufficient escape budget, or proxy-only transport result must stop or narrow the claim.

## Prior effect size or smallest effect of interest

There is no defensible quantitative prior effect size for "MatterChat usefulness" in electrodefect modelling. MatterChat success is therefore procedural: it must generate at least one auditable, structure-specific motif suggestion that can be translated into a pre-specified modelling check without changing the physics gates. If it does not, the physics workflow remains valid without MatterChat input.

For the physics chain, use existing project gates rather than a new effect size:

- Minimum-distance gate: no structure with unphysical close contacts proceeds to MatterChat or DFT.
- Emission-proxy gate: use the existing static contrast criterion from the repo README, `emission_contrast(clean, recombined) > 5`, only after DFT artifacts are converged and warning-clean enough to interpret.
- Escape-budget gate: `E_escape` must remain positive under documented work-function and loss assumptions.
- Transport gate: confirmatory claims require DFT-derived Hamiltonians; proxy KPM can only motivate follow-up.

## Sources and in-house precedent

- `main.pdf` and `supplementary.pdf`: electrodefect mechanism, W/Mo/Cu evidence chain, onset/escape equations, and static escape-budget framing.
- `LATTICE_MODELING_TOOLS.md`: installed local tools, environment caveats, MatterChat worker workflow, DFT/MLIP/TDDFT availability, and prior W/Mo short-contact warnings.
- `README.md`: current staged electrodefect simulation contract and central static emission-proxy gate.
- `docs/science-superpowers/questions/2026-06-04-near-surface-w-frenkel-field-neb.md`: precedent for treating CHGNet/field wrappers as exploratory only.
- `docs/science-superpowers/prior-work/2026-06-04-near-surface-w-frenkel-field-neb.md`: precedent for using smallest-effect and promotion gates when no defensible prior effect size exists.
- `runs/matterchat_geometry_qc_20260607_022909/matterchat_plan_review/review_synthesis.md`: corrective MatterChat review showing prompt-length failures, empty responses, and generic/hallucinated output; motivates the response-quality gate.
