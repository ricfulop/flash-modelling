# CURSOR_SPEC.md — Flash Electrodefect Simulation (single source of truth)

The only spec. Read top to bottom. Companion to *"Electrodefect effect: electron emission from
defect recombination in driven localized metals"* (Fulop & Ferralis, 2026).

**DEADLINE: paper submits Sunday Jun 7 (Gordon Conference, following week).** We are now using the
available time to run the full feasible Sunday program, not merely the minimum submittable subset.
The operating rule is: launch every scientifically useful computation that can plausibly finish or
produce a defensible bounded artifact by Sunday, while keeping claim gates strict. Failed, timed-out,
proxy-only, or placeholder-derived artifacts must remain marked as such and must not enter accepted
claims. Resolve the `===VERIFY===` API points (§8) by introspection as needed, then execute the
expanded Sunday critical path (§0).

**Goal.** (1) Identify the electron-transport mechanism in a high-concentration defect phase of
driven W beyond the Ioffe-Regel limit; (2) enumerate and eliminate the candidate charge-transport
mechanisms (hopping, miniband, VRH, polaron, percolation, field-driven); (3) build the DFT
energetics that rank those mechanisms and pick the favored one. Plus the static emission/φ result.

**Stance.** The draft leans on an Anderson / Ioffe-Regel picture, but we do not know the regime —
that is why we simulate. **This pipeline adjudicates; it does not confirm the draft.** Decision
thresholds are frozen before results (`regime_decision.py`). Do not lower the bar to make a story
work: no hardcoded estimates, no synthetic LDOS/eigenvalue windows, no placeholder energetics, and
no post-hoc threshold tuning in accepted claims. Under the deadline, the submitted paper should make
the regime-agnostic claim ("beyond-IR localization") unless a sharper regime label is in hand and
clean.

---

## 0. Sunday critical path (priority order)

Run in this order, with independent queues in parallel when resources permit. Each tier is
independently submittable, but the plan is no longer to stop early. Continue through the full list
until Sunday or until a task is scientifically blocked. Claim strength is determined by gates, not
by optimism.

**TIER A — guaranteed, runs today, no DFT (tested modules).**
1. `00` build the W triplet (`ordered`/`random`/`dendritic`) at frac 0.25/0.30/0.35.
2. `04` diagnostics (kF·ℓ, IPR, ⟨r⟩, ξ, d_s) + `regime_decision` per structure.
3. `05` mechanism-elimination table per structure (Goal 2 + Goal 3 selection).
   → **Deliverable A:** one figure/table — "three candidate structures, regime each lands in,
   dominant mechanism, why." Conditional on TB parameters (absolute kF·ℓ pending Tier B).
   This is the floor: a real Goal-1a/2/3 result no matter what DFT does.

**TIER B — high value, needs BigDFT up fast (launch day 1, runs overnight).**
4. DFT-parameterize the TB: transfer integral `t`, bandwidth `B`, on-site disorder `W` from a
   ~200–300-atom defective-W cell → re-run Tier A so kF·ℓ and the regime verdict are
   DFT-grounded, not guessed. → upgrades Deliverable A to a quantitative result.
5. Work-function drop Δφ (three slab calcs, §4.3) → **Deliverable B**, the highest
   confidence-per-effort new number; feeds Eq. 6 and the step-bunching/punctate-emission story.
6. `E_FP(W)` single self-consistent Δ-SCF value → can replace the literature-E_v + DFT-E_SIA
   composite in Fig. 4A. → **Deliverable C.**

**TIER C — required push for a solid Phase 1, not a placeholder.**
7. vacuum-LDOS η_e estimate at Te=0.72 eV (§4.4) → first-principles electronic branching, addresses
   draft caveat 3.
8. Assemble the E_escape budget (Eq. 6) with the computed φ and η_e.

**TIER D-lite — run for Sunday as exploratory bridge, with explicit labels.**
9. Mobility-window diagnostic from the existing TB/KPM artifacts: energy-resolved IPR near and
   above E_F, a pre-declared IPR threshold, and a tentative mobility-edge window E_c. Output is a
   figure/table that says which energy windows are extended-like vs localized-like in the assumed
   defect network.
10. Simplified Anderson-Holstein parameter sketch on small subclusters only: use current TB site
    energies and t_ij proxies, plus literature/DFT-bounded polaron energy λ_M, to produce an
    activation-energy distribution P(E_a) and percolation-threshold E_a estimate. This is explicitly
    an exploratory bridge to the 33/44/59 meV Arrhenius comparison, not a Wannier/DFPT result.
11. Write the Phase-2 protocol box: Wannier90 W-d projection, site ε_i/t_ij, DFPT/EPW e-ph,
    Anderson-Holstein ED/TEBD/HEOM, RT-TDDFT/NAMD recombination source term, and kMC upscaling.

**Sunday extended scope — start every tractable Phase-2-staging track, but gate claims hard.**
12. MLIP/anneal staging for "which structure flash actually forms": prepare or run the W/Mo
    morphology matrix labels that are already wired, and use finished rows only as exploratory
    morphology-selection evidence.
13. Mo cross-check and fraction/annealed-cell sweeps where code and inputs are ready; these are
    falsifiability/staging artifacts unless complete and reviewed.
14. Cheap NEB E_hop search or smoke tests if the queue is available; do not use unreviewed barriers
    as final mechanism selection scalars.
15. Protocol-only or smoke-only if full production is not feasible by Sunday: Wannier90 W-d
    projection, EPW/DFPT electron-phonon coupling, production Anderson-Holstein ED/DMRG/HEOM,
    real-time TDDFT/NAMD, kMC upscaling, phonon SED lifetimes, and real-time emission.

---

## 1. Goals → modules (what computes what)

| Goal | Question | Module(s) | Phase |
|---|---|---|---|
| 2 | list of candidate transport mechanisms | `mechanism_table` (enumerate+eliminate) | 1 — done |
| 1a | regime *given* a structure (beyond IR) | `transport_kpm` + `regime_decision` | 1 — Sunday |
| 3 | DFT energetics ranking the mechanisms | `transport_kpm` scalars + `dft_bigdft` | 1 — Sunday |
| — | emission GO/NO-GO, φ-drop, E_FP | `emission` + `dft_bigdft` | 1 — Sunday (Tier B/C) |
| 1b | which structure flash actually forms | `mlip_al` anneal → re-diagnose | Sunday extended / Phase 2 staging |

Goal 3 mechanics: the favored mechanism falls out of comparing energy scales computed on the
fixed cell — kF·ℓ ≲ 1 kills band; E_p/2t > 1 picks small-polaron; W/B > 16.5 picks Anderson;
d_s ≈ 4/3 picks fracton; Arrhenius E_hop sets NN-hopping. `mechanism_table.selection_criteria`
+ `regime_decision` assemble this into a ranked verdict per structure.

---

## 2. Operating conditions and how they map to the model

| Physical condition | Value (manuscript) | How the sim represents it |
|---|---|---|
| Lattice temperature T_L | 1300–1721 K (W) | ion config from a thermostatted snapshot at T_L |
| Electronic temperature Te | 0.72 eV (8358 K), Saha | Fermi–Dirac smearing in BigDFT + occupation weight in `emission` |
| Current density J | 65–110 A/mm² | **NOT a DFT input** — enters as the defect concentration it produces (Assumption 1) |
| Drive "exceeds damping" | flash onset, ΔB ≤ 0 | the 25–35 mol% transient FP fraction is the proxy for "deep in flash" |
| Local field | a few V/m | negligible; **explicitly not applied** (the whole point vs. field emission) |

**Assumption 1 — drive enters as the defect population it creates, not as a current.** A static/
elevated-Te DFT model cannot ingest J. The 25–35 mol% transient (draft line 331) *is* the encoding
of "high J, in flash, drive > damping," so `experiment.frac_sweep` **is the drive axis**. The
J→fraction map is an inference; report kF·ℓ and the regime **as a function of FP fraction**, which
also fixes the draft's kF·ℓ at the true operating point (25–35%) rather than the 1.4% remnant.

**Assumption 2 — atmosphere is a witness, not modeled.** W/Mo vacuum, Pt air; OES lines are
residual-gas witnesses. The emission test computes escape into **vacuum**; no gas-phase plasma is
modeled. φ_eff is facet-dependent (4.36–4.95 eV across bcc {100}/{110}/{111}); default W≈4.55 eV,
report the escape budget across the facet range, not at one value.

---

## 3. The three regimes and the frozen decision rule

| Regime | Meaning | Decided by |
|---|---|---|
| `EXTENDED_MINIBAND` | coherent Bloch/miniband, not localized (slow ≠ localized) | ⟨r⟩→GOE, PR/N(E_F) large, ξ large |
| `ANDERSON_LOCALIZED` | disorder-driven exponential localization | ⟨r⟩→Poisson, PR/N small, ξ finite, **random** |
| `QUANTUM_PERCOLATION_FRACTON` | geometry-driven critical localization (Alexander-Orbach) | localized **+ d_s≈4/3 on a spanning backbone** |

Anderson-vs-fracton is **geometric** — the draft's single umbrella does not split it.
`regime_decision.adjudicate` uses pre-registered thresholds (R_GOE=0.50, R_POISSON=0.42,
PR_EXTENDED=0.10, PR_LOCALIZED=0.01, XI_EXTENDED=5, DS_FRACTON∈[1.0,1.9], WB_ANDERSON=16.5,
KFL_BAND_SAFE=2.0). **Do not retune.** Report what it returns, including INCONCLUSIVE.
("Fracton" = Alexander-Orbach 1982, not Pretko-Nandkishore topological order — do not conflate.)

---

## 4. Phase 1 — DFT-direct (Sunday)

### 4.1 Structures (hypothesis basis set)
Three W(001) cells, **same** 25–35 mol% vacancy+SIA Frenkel content, differing only in correlation
structure: `ordered` superlattice / `random` / `dendritic` DLA backbone. Built by hand
(`build.py` + `percolation.py`). These are hypotheses; **which one is physical is Goal 1b = Phase 2.**
State this conditionality plainly in the paper.

### 4.2 Transport diagnostics + selection scalars (Goals 1a, 3)
On each structure's Hamiltonian (DFT support-function H if Tier B is up, else DFT-parameterized TB):
- regime battery: kF·ℓ, energy-resolved IPR, ⟨r⟩ level statistics, density-matrix decay ξ, d_s.
- selection scalars: transfer integral t, bandwidth B, on-site disorder W, N(E_F), polaron binding
  E_p (add e⁻, relax, ΔE), a few NEB E_hop. → `regime_decision` + `mechanism_table` → ranked verdict.

### 4.3 Work-function drop Δφ (Deliverable B — do this even if other DFT slips)
φ = V_vac − E_F, reported as **differences** (systematic vacuum-level errors cancel):
1. bulk defect-phase: clean W(001) vs W(001) with high near-surface FP density.
2. facet selection: per-facet φ for the flash-selected low-index planes ({110} bcc); ties to the
   SEM faceting + literature facet φ already in the draft.
3. step/edge lowering: a vicinal/stepped slab (Smoluchowski) → the DFT backing for "grooves as
   preferential emitters."
Computing the φ *consequence* of observed faceting is static DFT and easy; deriving that flash
*causes* the bunching is surface kinetics = Phase 2. Take the SEM faceting as input.

### 4.4 Emission GO/NO-GO + η_e (Tier B/C)
Δ-SCF E_FP (8–13 eV W) → vacuum-LDOS above φ_eff at Te=0.72 eV → `emission.emission_contrast`
(PASS > 5) → η_e estimate → E_escape budget (Eq. 6). Static proxy, not real-time; say so. BigDFT
surface BC gives a true vacuum and physical φ — why it beats a plane-wave code here.

### 4.5 Mobility window + Anderson-Holstein bridge (Tier D-lite, Sunday)
This is a bridge analysis, not a full first-principles polaron calculation. Run it for Sunday, but
keep the figure and language explicitly exploratory unless Wannier/DFPT inputs actually exist.

1. From the existing TB/KPM/diagnostic artifacts, compute or tabulate IPR(E) around E_F and above
   E_F. Use a fixed threshold before plotting to mark localized-like states and estimate a tentative
   mobility edge/window E_c. Report threshold sensitivity as exploratory, not confirmatory.
2. Treat localized states as defect-network sites with proxy ε_i and t_ij from the current TB model.
   Use a bounded λ_M / Huang-Rhys proxy from literature or completed DFT energetics; do not claim
   site-resolved e-ph unless Wannier/DFPT has actually been run.
3. On small subclusters, compute an exploratory activation-energy distribution P(E_a) and the
   critical percolating-path E_a using the Ambegaokar-Halperin-Langer logic. Compare the resulting
   scale to measured Arrhenius barriers (Mo 33 meV, W 44 meV, Pt 59 meV) as a plausibility check.
4. If the scale is wrong by an order of magnitude, report this as evidence that the mechanism class
   or proxy parameterization is incomplete. Do not retune the proxy to match the barriers.

---

## 5. Sunday extended / Phase 2 staging

The prior "future papers only" work is now a Sunday extended queue where tractable. It can improve
the paper if complete, but it must not contaminate Phase 1 claims if partial.

- **Goal 1b — which structure flash forms:** MACE active-learning loop or current MLIP staging
  (force RMSE target about 0.1 eV/Å before strong claims), then anneal each imposed cell at T_L and
  see which is selected; check vs quench remnant (Mo Magnéli=ordered, W restored). Only reviewed,
  completed rows can say anything about the *physical* state.
- BigDFT matrix-label rows for W/Mo morphology staging: run while CPU is available; failed/time-out
  rows stay failed and do not get imputed.
- NEB E_hop search across available configs; fraction sweep on annealed cells where the pipeline is
  ready.
- Phonon SED lifetimes + e-ph β if runnable; otherwise preserve as a protocol box.
- Mo cross-check as different-regime falsifiability; Pt remains lower priority unless W/Mo are
  complete.
- Real-time emission dynamics (RT-TDDFT/NAMD) is a protocol/smoke-test target on this hardware, not
  a required production result for Sunday.

`mlip_al.py` and the AL stages are no longer ignored under the Sunday clock, but any output from
them is exploratory until the RMSE/validation gate is met.

---

## 6. Hardware split (GB10)

DFT (FP64) → CPU/OpenMP/MPI BigDFT. Transport linear algebra → GPU (torch KPM, FP32). **Two-env
constraint:** BigDFT in its own micromamba env (`use_bigdft.sh`), cannot share a process with torch;
`dft_bigdft.py` shells out, interchange is files. Keep CPU BigDFT and GPU transport/MLIP work
overlapped where possible. Low GPU utilization during BigDFT is expected unless a separate
GPU-capable KPM/MLIP task is running.

---

## 7. File map & data contracts

```
src/electrodefect/
  percolation.py     [tested]   ordered/random/dendritic; D_f, d_s, spanning
  transport_kpm.py   [written]  KPM DOS, Kubo σ(E), kF·l, IPR, <r>, ξ  (GPU; offline exact path tested)
  regime_decision.py [tested]   frozen blind regime rule + W/Mo material check
  mechanism_table.py [tested]   σ(T)/I-V elimination + selection criteria + verdict
  emission.py        [tested]   vacuum-LDOS above φ_eff at Te; contrast; escape budget
  build.py           [written]  W(001) slab+vacuum; vacancy+<111> SIA; clean/separated/recombined
  dft_bigdft.py      [skeleton] shell-out driver; ===VERIFY=== PyBigDFT API
  mlip_al.py         [skeleton] Sunday extended / Phase 2 staging
config/w_percolation.yaml   single parameter source (experiment: morphologies, frac_sweep, materials)
scripts/00..05 ; tests/test_pipeline.py
```
Contracts: network npz {coords,mask,a}; emission ← {energies,density_z,z_grid,z_surface,E_F};
`regime_decision.adjudicate(percolation.classify(net), {r_stat,pr_frac_EF,xi_over_a,kF_l,W_over_B})`;
σ(T) npz {T,sigma} → `mechanism_table.fit_temperature_laws`.

---

## 8. ===VERIFY=== (resolve day 1 by introspection on the box)

PyBigDFT: input.yaml keys for surface BC / linear-scaling / Fermi-Dirac Te; Logfile energy,
eigenvalues, Fermi level (+Ha→eV); bigdft-tool cube export for KS-LDOS and local potential
(φ=V_vac−E_F); support-function H,S export for the TB; 2-node mpirun. ASE 3.28: BigDFT posinp
writer; `ase.mep.NEB`. Resolve via `python -c "import BigDFT; help(BigDFT.Calculators)"` etc.
(MACE/`mlip_al` ===VERIFY=== is Sunday extended: resolve enough to run staging safely, but do not
promote MLIP outputs to accepted claims without validation.)

---

## 9. Acceptance / what is defensible to submit Sunday

- **A (floor):** Tier A figure produced — regime per structure + mechanism table. Even on
  DFT-parameterized TB this is a real Goal-1a/2/3 result.
- **B:** Δφ work-function drop reported as differences across bulk-defect / facet / step.
- **C:** E_FP(W) single value; emission contrast > 5; η_e estimate; E_escape budget.
- **D-lite:** IPR/mobility-window figure and exploratory Anderson-Holstein activation distribution,
  clearly labeled as proxy-based and not a Wannier/DFPT calculation.
- **E / Phase-2 staging:** MLIP/anneal/matrix-label, NEB, Mo, phonon/e-ph, or real-time-emission
  outputs may be included only as explicitly exploratory or protocol material unless the relevant
  validation gates are met before Sunday.
- **Integrity:** every structure through `regime_decision` with frozen thresholds; report the
  verdict even if it contradicts the draft; no post-hoc tuning. If W is not cleanly localized,
  the submitted text says "beyond-IR" not "Anderson."
- **No placeholders:** accepted claims require computed artifacts. Configured constants, literature
  bounds, and proxy model parameters are allowed only when labeled as inputs/bounds/proxies.

---

## 10. Manuscript edits (Sunday)

- Replace asserted "kFℓ ≲ 3–5" with the computed value at 25–35 mol% (Tier A/B).
- Add Δφ work-function drop (Tier B) → strengthens Eq. 6 and the punctate/step-bunching argument.
- Replace Fig. 4A composite E_FP with the self-consistent BigDFT E_FP(W) only if the recombined and
  separated Δ-SCF pair both complete cleanly; otherwise state the computed side that exists and keep
  the literature/proxy range labeled.
- Add one cautious Tier D-lite paragraph/box: localized energy windows from IPR and an
  exploratory activation-barrier distribution; explicitly reserve Wannier/DFPT Anderson-Holstein
  rates for future work.
- Add a short "extended Sunday staging" paragraph only for completed, reviewed MLIP/NEB/Mo/phonon
  artifacts; otherwise keep those as Methods/protocol future work.
- **Generalize "Anderson is central" → "beyond-Ioffe-Regel localization"** unless a clean regime
  verdict is in hand by Saturday. This makes the paper robust to the sim instead of betting on it.
- "Superhighway" edit already done — the current draft adopts suppressed-relaxation framing.
- Frame the three-structure result as conditional unless the Sunday extended morphology-selection
  queue completes and passes validation; frame novelty as the driven recombination-fed regime, not
  a new conduction mechanism (filament corpus is prior art for the mechanism menu).

---

## 11. Honest scope (Methods)

Static Δ-SCF + elevated-Te vacuum-LDOS demonstrate the channel but are a proxy for the driven
non-equilibrium state. The three structures are hypotheses; Sunday extended runs may begin to test
which one flash produces, but the favored-mechanism ranking remains conditional on the assumed
structure until MLIP/anneal validation is complete. Real-time emission and full non-equilibrium
transport (NEGF/Floquet) remain protocol-level unless a runnable, reviewed calculation lands before
Sunday.
