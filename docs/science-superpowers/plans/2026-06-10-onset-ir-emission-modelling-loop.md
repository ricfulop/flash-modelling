# Onset → Ioffe-Regel → Hot-Electron Emission Modelling Loop Plan

> **For agentic workers:** REQUIRED SUB-SKILL: this plan extends the frozen preregistration `docs/science-superpowers/preregistrations/2026-06-07-matterchat-first-electrodefect-modelling.md`. The DFT-engine pivot is recorded in `docs/science-superpowers/preregistrations/2026-06-10-dft-label-pivot-deviation-note.md`. Confirmatory gates are unchanged; everything new added here is exploratory until separately preregistered. Steps use checkbox (`- [ ]`) syntax for tracking.

**Question:** Does increasing Frenkel-pair defect fraction in W/Mo drive the chain claimed by the voltivity manuscript — conduction onset, then an Ioffe-Regel (IR) crossover near defect fraction `c_V ≈ 0.32`, then defect-recombination hot electrons energetically able to escape — and does an ordered defect superstructure retain a conductive defect band where disordered morphologies localize?

**Design:** Computational comparison across morphology (`ordered`, `random`, `dendritic`) × defect-fraction grid (0.25–0.70), with exploratory geometry-proxy KPM transport on both DGX GPUs and confirmatory QE DFT recombination/escape labels replacing the failed BigDFT path.

**Data:** Frozen CIF inputs in `runs/matterchat_geometry_qc_20260607_022909/matterchat_inputs` (checksums in autonomous manifest); generated percolation lattices with recorded seeds; label queue `runs/autonomous_electrodefect_20260607_041730/geometry_repair_convergence_gate/label_queue_manifest.json`.

**Primary analysis:** (a) DOS(E≈0) and localization diagnostics vs `c_V` per morphology (KPM, exploratory); (b) matched separated/recombined QE total energies → `E_FP = E_sep − E_rec`; (c) escape budget `E_escape = E_FP − n_hop·E_hop − phi_eff` with frozen `phi_eff = 4.55 eV` and frozen loss scenarios; (d) emission-proxy contrast gate `> 5` (frozen, unchanged).

**Decision rule (unchanged from frozen prereg):** The chain is supported only if geometry, converged DFT energetics, positive escape budget, emission contrast, and transport gates all pass; confirmatory IR/localization claims require DFT-derived Hamiltonians, not geometry-proxy KPM. Manuscript-anchored exploratory expectations: `E_FP` in 3–12 eV; disordered morphologies show DOS/transport suppression deepening through `c_V ≈ 0.25–0.35`; ordered superstructure retains a defect band (already seen as ~15–300× DOS(E=0) excess, exploratory). A converged `E_FP < phi_eff` for all pairs, or no morphology-dependent IR-region separation in confirmatory transport, disconfirms the modelled chain.

---

## Manuscript anchors (Ordered Defect Condensation / voltivity)

- IR crossover estimate `c_V ≈ 1/π ≈ 0.32` (range 0.25–0.35) — our fraction grid must resolve 0.25→0.45 finely.
- Frenkel-pair recombination releases 3–12 eV; hot electrons recaptured or escaping over a work function near 4.55 eV (W).
- OES signature: bi-Maxwellian with supra-thermal excitation energies to 23.1 eV (the experimental anchor; modelling supports, never proves, this number).
- Ordered Ridge-Templated Superstructure: ordered defect placement should produce a conductive defect band (anti-localization), unlike random/dendritic.

## Loop architecture

One agent-driven loop (dynamic schedule via `AGENT_LOOP_WAKE_MODELLING_LOOP` sentinel) that on every wake:

1. **Harvest:** read `status.json` of every active queue (local QE, local/peer KPM); record completions in the autonomous manifest; never reinterpret thresholds.
2. **Aggregate:** refresh KPM aggregate (`scripts/07_aggregate_kpm_sweeps.py`) when new KPM cases completed; recompute `E_FP`/escape table when new converged QE pairs exist.
3. **Gate-check:** apply frozen gates (geometry / DFT / escape / emission / transport). Advance a stage only when its gate passes.
4. **Refill:** relaunch the highest-value idle-resource job from the priority table below, targeting ≥80% utilization on both DGX systems.
5. **Anomaly discipline:** any surprising value goes through investigating-anomalous-results phases; no data dropped, no thresholds moved.
6. **Re-arm:** schedule the next wake sized to the shortest expected job completion (45–180 min).

### Resource priority table

| Resource | Primary | Fallback |
|---|---|---|
| Local GPU (spark-5da5) | QE GPU label queue (confirmatory-critical) | KPM fraction sweep batches |
| Local CPU | QE host-side work, aggregation, manifests | MLIP screening repeats |
| Peer GPU (spark-0808) | KPM IR-crossing fraction sweep | KPM robustness repeats |
| Peer CPU | KPM network builds (dendritic builds are CPU-heavy) | rsync/harvest |

## Task 1: QE DFT label pivot (replaces BigDFT primary path)

**Artifacts:**
- Reads: `geometry_repair_convergence_gate/label_queue_manifest.json` (4 jobs: W/Mo random separated/recombined)
- Runs: `scripts/27_phase2_qe_gpu_labels.py` with `pw.x` at `.local-qe/src/q-e-qe-7.5/bin/pw.x`, pseudos in `data/qe_pseudos/`
- Writes: `runs/phase2_qe_gpu_labels_<timestamp>_<label>/status.json`

- [ ] **Step 1:** Launch the 4-job QE queue: `--root <convergence_gate> --max-seconds 7200` with default smoke-grid setting (ecutwfc 40, kgrid 2 2 1, mv smearing). Energy-only; no matrix export.
- [ ] **Step 2:** Each loop wake, parse `converged`, `total_energy_ry`, error markers. A job is usable only if `convergence has been achieved` and `JOB DONE.` are both present.
- [ ] **Step 3:** When both members of a material pair converge with identical settings, compute `E_FP = E_sep − E_rec` (Ry→eV, ×13.6057), record per pair.
- [ ] **Step 4:** Apply frozen DFT gate (`dft_supported` / `dft_partial` / `blocked_dft`). If a job fails SCF, retry once with `mixing_beta 0.15` and `electron_maxstep 200` (single pre-specified retry; further changes require a new deviation note).
- [ ] **Step 5:** If converged, escalate cutoffs (`ecutwfc 60, ecutrho 480, kgrid 3 3 1`) as a convergence check on one W pair before trusting `E_FP` beyond ±0.5 eV.

## Task 2: Onset → IR-crossing map (exploratory KPM, both GPUs)

**Artifacts:**
- Runs: `scripts/14_large_lattice_kpm_showcase.py` (64³ lattices, moments 65536, R 192)
- Writes: `runs/large_kpm_<timestamp>_<label>/status.json`; aggregates via `scripts/07_aggregate_kpm_sweeps.py`

- [ ] **Step 1:** Fill the fraction grid to n≥3 per morphology at `c_V ∈ {0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}`. Launch order (peer GPU first, local GPU when QE idle): 0.30 trio → 0.45 trio → 0.50 trio → 0.60 trio → 0.65 trio → repeats to n=3 ascending by current n.
- [ ] **Step 2:** After each aggregate refresh, plot DOS(E=0) vs `c_V` per morphology with CI bands; identify where random/dendritic curves flatten to the localized floor vs where ordered retains its defect band. Label all of this `exploratory_proxy`.
- [ ] **Step 3:** Ordered-0.55 bimodality (5 low ~1.4–1.8e−5 vs 2 high ~2.7e−4) continues under anomaly investigation: next probes vary disorder seed at fixed network seed (when the case is scheduled), and 0.50/0.60 ordered points bracket the fraction. No data dropped.

## Task 3: Escape budget and emission proxy (gated on Task 1)

- [ ] **Step 1:** With converged `E_FP`, compute `E_escape = E_FP − n_hop·E_hop − phi_eff` for the frozen loss-scenario set; compare against the manuscript 3–12 eV release window.
- [ ] **Step 2:** Run the static emission proxy (`src/electrodefect/emission.py` path) only on converged QE artifacts; gate stays `contrast > 5`.
- [ ] **Step 3:** Report whether escaped-electron energy headroom is consistent with supra-thermal tails (exploratory comparison to 23.1 eV OES anchor; no fitting to the anchor).

## Task 4: Confirmatory transport bridge (gated; not run until Task 1 passes)

- [ ] **Step 1:** From converged QE outputs, attempt Wannier/downfolded Hamiltonian per `scripts/20_phase2_wannier_transport.py`.
- [ ] **Step 2:** Localization diagnostics (DOS, Kubo, IPR, level-spacing) with frozen thresholds; only these can support confirmatory IR claims.

## Task 5: TDDFT
Unchanged: requires a separate preregistration; not run under this plan.

## Reporting cadence

Each wake appends to the autonomous manifest; daily (or on user return) a synthesis note `runs/<...>/onset_ir_emission_synthesis.md` summarizes: fraction-sweep curves, `E_FP` table, escape budgets, gate states, anomalies, and what remains blocked.

## Self-review

- Coverage: onset mapping (Task 2), IR crossing (Tasks 2+4), hot-electron emission (Tasks 1+3) — the full manuscript chain has a task.
- Degrees of freedom: QE retry policy, fraction grid, seeds, and gates are fixed above before results are seen; anything else is a recorded deviation.
- Confounds: geometry-proxy vs real-Hamiltonian transport kept separate; finite-size and seed effects handled by n≥3 repeats; QE convergence checked by cutoff escalation.
- Placeholders: none; every step names its script, inputs, and thresholds.
