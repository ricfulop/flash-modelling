# MatterChat Plan-Review Prompts

These prompts ask MatterChat to critique the modelling plan. Responses are advisory and must be checked against the project code, tool constraints, and preregistration before any plan change is treated as valid.

## Shared context

```text
You are reviewing an electrodefect modelling plan as a materials-structure assistant. Treat your response as advisory brainstorming, not proof. Do not claim electron emission is proven. Do not invent numerical emission rates, work functions, transport coefficients, or activation barriers. If you are unsure, say what should be checked with DFT, MLIP, transport, or TDDFT.

Current plan summary:
- Generate W/Mo BCC slab structures with clean, separated Frenkel-pair, and recombined configurations.
- Use geometry QC before MatterChat, MLIP, or DFT.
- Ask MatterChat for qualitative motifs only.
- Use MACE/CHGNet only for repair/screening.
- Use BigDFT CPU and QE GPU for matched separated-vs-recombined DFT labels.
- Compute Frenkel-pair recombination energy, escape budget, and static LDOS/work-function emission proxy.
- Use KPM/Wannier/downfolded Hamiltonians for transport/localization; geometry-proxy transport is exploratory only.
- Use SALMON/Octopus TDDFT only as a later separately preregistered cross-check.

MatterChat pilot motifs so far:
- vacancy-interstitial proximity;
- under-coordinated surface atoms;
- crowdion-like alignments.
```

## Prompt A: Gaps and Failure Modes

```text
Using the shared context and this W dendritic separated Frenkel-pair slab structure, critique the plan. What are the most important missing checks, controls, or failure modes before treating the modelling chain as credible? Return a concise prioritized list. Keep every suggestion testable with MLIP, DFT, transport, or TDDFT.
```

## Prompt B: First Calculations

```text
Using the shared context and this W dendritic separated Frenkel-pair slab structure, what should the first three modelling calculations be and why? Prefer calculations that de-risk geometry, recombination energetics, surface/vacuum coupling, or localization. Do not propose quantitative conclusions; propose checks.
```

## Prompt C: Controls

```text
Using the shared context and this W dendritic separated Frenkel-pair slab structure, what control structures or negative controls should be paired with this candidate so that a later DFT/transport result is interpretable? Return controls and what each would rule out.
```
