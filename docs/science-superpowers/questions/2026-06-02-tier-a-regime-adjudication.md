# Tier A Regime Adjudication

**Research question:** For W Frenkel-pair defect networks at transient defect fractions 0.25, 0.30, and 0.35, do the ordered, random, and dendritic morphology hypotheses fall into distinct frozen transport-regime categories under the pre-registered `regime_decision` thresholds?

**Background / motivation:** The manuscript should not assume an Anderson regime before simulation. Tier A provides a defensible, DFT-independent floor result for the Sunday deadline: the regime verdict and mechanism-elimination table conditional on the fixed morphology hypotheses and tight-binding parameters.

**Hypotheses:**
- H0 (null): The Tier A diagnostics do not separate the morphology hypotheses into a coherent regime taxonomy; verdicts are inconclusive or insensitive to morphology/fraction.
- H1 (alternative): At least one morphology/fraction class receives a non-inconclusive frozen verdict, and the verdicts distinguish miniband-like, disorder-localized, or quantum-percolation/fracton mechanisms in a way that can guide the manuscript language.

**Population & unit of analysis:** The population is the hypothesis basis set of W defect-phase structures specified in `config/w_percolation.yaml`. The unit of analysis is one morphology/fraction case: `ordered`, `random`, or `dendritic` at defect fraction 0.25, 0.30, or 0.35.

**Key variables (operationalized):**
- Outcome: transport regime -> `regime_decision.adjudicate(...)` verdict using the frozen thresholds in `src/electrodefect/regime_decision.py`.
- Predictors / exposures: morphology hypothesis -> ordered lamellar/superlattice proxy, random percolation cluster, or DLA dendritic cluster; defect fraction -> 0.25, 0.30, or 0.35.
- Diagnostic variables: geometry -> spanning, `D_f`, `d_s`; electronic -> level-spacing ratio `<r>`, participation fraction near `E_F`, density-matrix decay length `xi/a`, `W/B`, and a tight-binding `kF_l` proxy pending DFT/Kubo calibration.
- Fixed inputs: W lattice parameter, `t_hop`, disorder strengths, KPM moments/random vectors, and all regime thresholds from config/source. These are not retuned after seeing outputs.

**What counts as an answer:** A case-level answer is the frozen verdict plus rationale emitted by `regime_decision`. A Tier A answer exists if all 9 cases produce geometry and transport artifacts, with any inconclusive verdicts reported rather than repaired by threshold changes.

**Scope & exclusions:** This Tier A question does not determine which morphology flash physically forms, does not launch production DFT, does not claim a DFT-calibrated absolute `kF_l`, and does not retune thresholds. BigDFT work in this pass is API verification only, preparing Tier B.

**Open questions for prior-work survey:** The main open methodological questions are how to calibrate the tight-binding `kF_l` proxy against DFT/Kubo outputs, how BigDFT exposes support-function matrices and density-kernel decay, and how to phrase a morphology-conditional Tier A result in the manuscript without overclaiming physical morphology selection.
