"""
regime_decision.py — PRE-REGISTERED blind regime adjudication.

Purpose: decide which transport regime the driven defect state is in, WITHOUT
assuming the answer the manuscript currently carries. The thresholds below are fixed
*before* looking at any run output. Do not tune them to match a desired verdict — that
is the whole point. If the data put a material in a different regime than the draft
asserts, the draft is what changes.

Three mutually exclusive regimes (+ inconclusive):
  EXTENDED_MINIBAND          coherent Bloch/miniband states; not localized
  ANDERSON_LOCALIZED         disorder-driven exponential localization
  QUANTUM_PERCOLATION_FRACTON geometry-driven critical localization on a fractal backbone
  INCONCLUSIVE               diagnostics disagree -> report honestly, gather more

The discriminator between ANDERSON and QUANTUM_PERCOLATION is geometric: Anderson needs
spatial randomness; fracton/quantum-percolation needs a correlated percolating backbone
(d_s ~ 4/3). Both are "localized but not Anderson-vs-Anderson" — distinguishing them is
exactly what the manuscript's single "Anderson-Ioffe-Regel" umbrella does NOT do.

Inputs are the regime-agnostic diagnostics:
  geometry : output of percolation.classify(net)   -> spanning, D_f, d_s
  elec     : dict with
      r_stat      : <r> level-spacing ratio (GOE~0.531 extended, Poisson~0.386 localized)
      pr_frac_EF  : participation-ratio fraction PR/N for states at E_F (extended ->O(1), localized ->~0)
      xi_over_a   : density-matrix decay length / lattice spacing (large -> extended)
      kF_l        : Ioffe-Regel product (>~2 band may survive; <~1 band invalid)
      W_over_B    : disorder strength / bandwidth (3D Anderson threshold ~16.5)

Pure numpy. Tested standalone.
"""
from __future__ import annotations
from dataclasses import dataclass

# ---- PRE-REGISTERED THRESHOLDS (do not change after seeing results) ---------
R_GOE, R_POISSON = 0.50, 0.42          # <r> extended above / localized below
PR_EXTENDED, PR_LOCALIZED = 0.10, 0.01  # PR/N at E_F
XI_EXTENDED = 5.0                       # xi/a above this => effectively extended
DS_FRACTON_LO, DS_FRACTON_HI = 1.0, 1.9 # spectral dim window for fracton/percolation
DS_DENSE = 2.6                          # d_s above this => dense extended network
KFL_BAND_SAFE = 2.0                     # kF*l above this => band transport plausible
WB_ANDERSON = 16.5                      # 3D simple-cubic Anderson threshold


@dataclass
class Verdict:
    regime: str
    confidence: str          # "strong" | "weak" | "split"
    rationale: list
    flags: list


def adjudicate(geometry: dict, elec: dict) -> Verdict:
    r = elec.get("r_stat")
    pr = elec.get("pr_frac_EF")
    xi = elec.get("xi_over_a")
    kfl = elec.get("kF_l")
    wb = elec.get("W_over_B")
    d_s = geometry.get("d_s")
    spanning = geometry.get("spanning")

    rationale, flags, votes = [], [], {"extended": 0, "localized": 0}

    # --- Ioffe-Regel premise gate -------------------------------------------
    if kfl is not None:
        if kfl > KFL_BAND_SAFE:
            rationale.append(f"kF*l={kfl:.2f} > {KFL_BAND_SAFE}: band transport plausible "
                             "(the near-IR premise itself is in question)")
            flags.append("premise: not clearly at Ioffe-Regel limit")
        else:
            rationale.append(f"kF*l={kfl:.2f} <= {KFL_BAND_SAFE}: at/below Ioffe-Regel; "
                             "band transport not a safe zeroth-order picture")

    # --- electronic localization votes --------------------------------------
    if r is not None:
        if r >= R_GOE: votes["extended"] += 1; rationale.append(f"<r>={r:.3f} >= {R_GOE} (GOE/extended)")
        elif r <= R_POISSON: votes["localized"] += 1; rationale.append(f"<r>={r:.3f} <= {R_POISSON} (Poisson/localized)")
        else: rationale.append(f"<r>={r:.3f} intermediate (no vote)")
    if pr is not None:
        if pr >= PR_EXTENDED: votes["extended"] += 1; rationale.append(f"PR/N(E_F)={pr:.3f} >= {PR_EXTENDED} (extended at E_F)")
        elif pr <= PR_LOCALIZED: votes["localized"] += 1; rationale.append(f"PR/N(E_F)={pr:.3f} <= {PR_LOCALIZED} (localized at E_F)")
        else: rationale.append(f"PR/N(E_F)={pr:.3f} near mobility edge (no vote)")
    if xi is not None:
        if xi >= XI_EXTENDED: votes["extended"] += 1; rationale.append(f"xi/a={xi:.1f} >= {XI_EXTENDED} (delocalized)")
        else: votes["localized"] += 1; rationale.append(f"xi/a={xi:.1f} < {XI_EXTENDED} (finite localization length)")

    extended = votes["extended"] > votes["localized"]
    localized = votes["localized"] > votes["extended"]
    split = votes["extended"] == votes["localized"]

    # --- assign regime -------------------------------------------------------
    if split or (votes["extended"] + votes["localized"]) == 0:
        return Verdict("INCONCLUSIVE", "split", rationale,
                       flags + [f"localization votes tied: {votes}"])

    if extended:
        regime = "EXTENDED_MINIBAND"
        if d_s is not None and d_s >= DS_DENSE:
            rationale.append(f"d_s={d_s:.2f} >= {DS_DENSE}: dense extended network, consistent")
        conf = "strong" if votes["extended"] >= 2 else "weak"
        return Verdict(regime, conf, rationale, flags)

    # localized: Anderson vs quantum-percolation/fracton, decided GEOMETRICALLY
    if d_s is not None and DS_FRACTON_LO <= d_s <= DS_FRACTON_HI and spanning:
        regime = "QUANTUM_PERCOLATION_FRACTON"
        rationale.append(f"d_s={d_s:.2f} in [{DS_FRACTON_LO},{DS_FRACTON_HI}] on a spanning "
                         "backbone: geometry-driven (fracton), NOT disorder-driven Anderson")
        flags.append("manuscript 'Anderson' label would be imprecise here -> use beyond-IR/percolation taxonomy")
    elif wb is not None and wb >= WB_ANDERSON:
        regime = "ANDERSON_LOCALIZED"
        rationale.append(f"W/B={wb:.1f} >= {WB_ANDERSON} and no fractal backbone: disorder-driven Anderson")
    else:
        regime = "ANDERSON_LOCALIZED"
        rationale.append("localized, structure not fractal-percolating -> disorder-driven (Anderson) by default")
        if wb is not None:
            rationale.append(f"(note W/B={wb:.1f} below the {WB_ANDERSON} 3D threshold; localization may be marginal)")
    conf = "strong" if votes["localized"] >= 2 else "weak"
    return Verdict(regime, conf, rationale, flags)


def material_implication(material: str, regime: str) -> str:
    """
    The draft's own V-SIA migration asymmetry (Table S4) predicts the materials may
    NOT share one regime. This flags consistency / tension with that prediction.
    """
    expect = {
        "W":  ("ANDERSON_LOCALIZED",
               "W has extreme SIA/V migration asymmetry (~43) and restores its lattice post-flash "
               "(no ordered child phase) -> disordered FP transient is the natural expectation."),
        "Mo": ("EXTENDED_MINIBAND",
               "Mo freezes vacancies into ordered Magneli/crystallographic-shear phases post-flash "
               "-> an ordered defect lattice (miniband) is plausible, at least in the remnant."),
        "Cu": ("INCONCLUSIVE",
               "Cu recombines near-completely (reversible) -> little persistent defect phase."),
    }
    exp_regime, why = expect.get(material, (None, "no prior expectation encoded"))
    if exp_regime is None:
        return why
    if regime == exp_regime:
        return f"CONSISTENT with draft's migration-asymmetry expectation. {why}"
    return (f"TENSION with draft expectation ({exp_regime}). {why} "
            f"If the sim holds, the manuscript's single 'Anderson-Ioffe-Regel' umbrella "
            f"should be split per material.")


if __name__ == "__main__":
    # three synthetic cases spanning the regimes
    cases = {
        "W disordered (Anderson-like)": (
            dict(d_s=2.1, spanning=True),
            dict(r_stat=0.39, pr_frac_EF=0.005, xi_over_a=2.0, kF_l=0.9, W_over_B=18.0)),
        "Mo ordered (miniband-like)": (
            dict(d_s=2.8, spanning=True),
            dict(r_stat=0.55, pr_frac_EF=0.30, xi_over_a=20.0, kF_l=1.4, W_over_B=1.0)),
        "dendritic (fracton/percolation)": (
            dict(d_s=1.45, spanning=True),
            dict(r_stat=0.40, pr_frac_EF=0.02, xi_over_a=3.0, kF_l=1.0, W_over_B=6.0)),
    }
    for name, (geo, elec) in cases.items():
        v = adjudicate(geo, elec)
        print(f"\n=== {name} ===")
        print(f"  REGIME: {v.regime}  ({v.confidence})")
        for r in v.rationale: print(f"    - {r}")
        for f in v.flags: print(f"    ! {f}")
        mat = "W" if name.startswith("W") else ("Mo" if name.startswith("Mo") else "?")
        if mat in ("W", "Mo"):
            print(f"    material check: {material_implication(mat, v.regime)}")
