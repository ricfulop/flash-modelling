# MatterChat-First Electrodefect Modelling

**Research question:** Can geometry-vetted W/Mo Frenkel-pair defect structures support a defect-recombination route to hot carriers, with MatterChat used only when its qualitative feedback is non-empty, structure-specific, and testable?

**Background / motivation:** The electrodefect manuscript argues that a driven, defect-rich metal state can generate nonthermal carriers through Frenkel-pair recombination, with Mo providing a same-material bridge between transport anomaly and nonthermal optical signatures. The local modelling stack now includes W/Mo BCC slab builders, defect-network generators, MLIP screening, BigDFT/QE validation paths, transport/KPM tools, SALMON/Octopus TDDFT options, and a locally installed MatterChat CUDA worker. MatterChat is useful for structure-to-text brainstorming, but it has no native emission, current-drive, hot-electron, or transport physics and therefore cannot prove the mechanism.

**Hypotheses:**
- H0 (null): The pre-specified W/Mo Frenkel-pair modelling variants do not pass the full physics evidence chain; any apparent support is blocked by geometry failure, non-converged DFT, insufficient recombination/escape energy, weak emission proxy, or proxy-only transport.
- H1 (alternative): At least one geometry-repaired W or Mo Frenkel-pair variant from the fixed material/morphology grid has DFT energetics, work-function/LDOS proxies, and transport/localization diagnostics that coherently support a defect-recombination route to eV-scale hot carriers. MatterChat may annotate candidate motifs only if its response passes the quality gate.

**Population & unit of analysis:** The population is W and Mo BCC slab configurations generated from the existing `flash-modelling` defect builders, including clean, separated Frenkel-pair, and recombined structures across ordered, random, and dendritic defect morphologies. The unit of analysis is one material/morphology/configuration tuple that has passed minimum-distance QC and, for confirmatory physics claims, has associated converged DFT and transport artifacts.

**Key variables (operationalized):**
- MatterChat input: a geometry-QC-passing CIF or structure file plus a bounded natural-language prompt asking for qualitative local motifs, coordination changes, surface effects, or localization-relevant features.
- MatterChat output: qualitative motif suggestions only, recorded as prompts and responses; never used as a measured outcome. Empty, generic, or hallucinated responses are non-admissible and cannot alter confirmatory choices.
- Geometry QC outcome: minimum pair distance, atom count, cell/PBC status, and whether the structure is allowed to proceed to MatterChat or DFT.
- DFT energetics outcome: separated-versus-recombined Frenkel-pair energy difference and any associated warnings/convergence status.
- Escape-budget outcome: whether `E_escape = E_FP - losses - phi_eff` remains compatible with eV-scale carriers under pre-specified loss/work-function assumptions.
- Emission-proxy outcome: vacuum-accessible LDOS/work-function contrast at elevated electronic temperature, using the existing static proxy path.
- Transport/localization outcome: KPM/Wannier/downfolded-Hamiltonian diagnostics for Ioffe-Regel/Anderson-like behavior; geometry-proxy transport is exploratory only.

**What counts as an answer:** MatterChat is useful only if it produces bounded, auditable, structure-specific qualitative suggestions that can be translated into pre-specified modelling checks without changing success criteria after results are seen. If MatterChat output fails that quality gate, the investigation proceeds with the physics-driven W/Mo material/morphology grid. The electrodefect modelling chain is supported only if geometry QC, DFT energetics, escape budget, emission proxy, and transport/localization diagnostics pass their gates. The chain is negative or blocked if any essential link fails, remains proxy-only, or cannot be converged.

**Scope & exclusions:** This investigation does not use MatterChat as evidence for electron emission, hot-electron energies, work functions, defect barriers, or transport. It does not treat unrepaired short-contact geometries as physical. It does not claim real-time emission dynamics unless SALMON/Octopus follow-up is separately planned and preregistered. It does not tune KPM, DFT, or emission thresholds after seeing outcomes.

**Open questions for prior-work survey:** Identify which parts of the local stack are standard enough for confirmatory claims, which are screening-only, what known artifacts threaten W/Mo Frenkel-pair modelling, and which modelling gates are needed before MatterChat-inspired motifs can be promoted.

## Self-review

- Vagueness scan: "useful MatterChat output" is operationalized as bounded, auditable, structure-specific qualitative suggestions; "support" is tied to explicit geometry, DFT, escape, emission-proxy, and transport gates.
- Falsifiability: H1 is disconfirmed or blocked by geometry failure, DFT non-convergence, insufficient escape budget, proxy-only transport, or incoherent diagnostics.
- Scope check: The investigation is one modelling workflow question, not a claim that MatterChat or any single calculation proves the whole manuscript.
- Unit clarity: The unit is one material/morphology/configuration tuple, with MatterChat responses treated as annotations rather than outcomes.
