#!/usr/bin/env python
"""Stage 05: consolidate Tier A mechanism-elimination criteria."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from electrodefect import mechanism_table as mt
from electrodefect import tier_a


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "w_percolation.yaml"


def _polaron_proxy(cfg):
    lo, hi = cfg["emission"]["E_FP_expected"]
    return 0.5 * (float(lo) + float(hi))


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    rows = []
    details = {}
    for morphology, frac in tier_a.iter_cases(cfg):
        out_dir = tier_a.case_dir(morphology, frac)
        transport_path = out_dir / "transport.json"
        if not transport_path.exists():
            raise FileNotFoundError(
                f"missing {transport_path}; run scripts/04_transport.py first"
            )
        transport = tier_a.read_json(transport_path)
        elec = transport["electronic"]
        disorder_W = (
            cfg["tight_binding"]["disorder_W_ordered"]
            if morphology == "ordered"
            else cfg["tight_binding"]["disorder_W_random"]
        )
        crit = mt.selection_criteria(
            kF_l=float(elec["kF_l"]),
            W_disorder=float(disorder_W),
            B_bandwidth=float(elec["bandwidth_eV"]),
            E_polaron=_polaron_proxy(cfg),
            t_hop=float(cfg["tight_binding"]["t_hop"]),
            xi=float(elec["xi_over_a"]) * float(cfg["lattice_a"]),
        )
        crit_path = out_dir / "mechanism_criteria.csv"
        crit.to_csv(crit_path, index=False)
        regime = transport["regime_decision"]["regime"]
        rows.append(
            {
                "morphology": morphology,
                "target_frac": frac,
                "regime": regime,
                "confidence": transport["regime_decision"]["confidence"],
                "bandwidth_eV": elec["bandwidth_eV"],
                "kF_l_proxy": elec["kF_l"],
                "W_over_B": elec["W_over_B"],
                "polaron_proxy_eV": _polaron_proxy(cfg),
                "dominant_transport": _dominant_mechanism(regime, crit),
                "case_dir": str(out_dir.relative_to(REPO_ROOT)),
            }
        )
        details[f"{morphology}_{tier_a.frac_slug(frac)}"] = {
            "criteria": crit.to_dict(orient="records"),
            "note": elec["diagnostic_note"],
        }
        print(
            f"{morphology:10s} frac={frac:.2f} regime={regime:28s} "
            f"dominant={rows[-1]['dominant_transport']}"
        )

    summary = pd.DataFrame(rows)
    summary_path = tier_a.TIER_A_DIR / "mechanism_summary.csv"
    summary.to_csv(summary_path, index=False)
    tier_a.write_json(
        tier_a.TIER_A_DIR / "mechanism_summary.json",
        {"cases": rows, "details": details},
    )
    print(f"\nsaved mechanism summary -> {summary_path.relative_to(REPO_ROOT)}")


def _dominant_mechanism(regime: str, crit: pd.DataFrame) -> str:
    if regime == "EXTENDED_MINIBAND":
        return "coherent/miniband candidate"
    if regime == "QUANTUM_PERCOLATION_FRACTON":
        return "geometry-driven quantum percolation/fracton"
    polaron = crit[crit["criterion"].str.contains("Holstein")]["verdict"].iloc[0]
    if "small-polaron" in polaron:
        return "small-polaron/activated hopping candidate"
    return "disorder-localized hopping candidate"


if __name__ == "__main__":
    main()
