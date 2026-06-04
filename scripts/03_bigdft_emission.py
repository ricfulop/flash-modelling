#!/usr/bin/env python3
"""Stage 03: Tier B/C BigDFT post-processing and emission gates.

This script does not launch DFT. It consumes completed BigDFT observables and
optional vacuum-LDOS artifacts, then writes explicit Phase 1 status outputs:

- Tier B work-function differences, with convergence/infocode guardrails.
- Tier C Delta-SCF E_FP if paired separated/recombined jobs exist.
- Tier C emission GO/NO-GO if clean and recombined LDOS artifacts exist.
- Escape-budget artifact using Delta-SCF E_FP, or the configured expected range.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from electrodefect import phase1


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
CONFIG = REPO / "config" / "w_percolation.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def flatten_summary(status: dict) -> pd.DataFrame:
    wf = status["tier_b_work_function"]
    efp = status["tier_c_E_FP"]
    em = status["tier_c_emission"]
    budget = status["tier_c_escape_budget"]
    rows = []
    rows.append(
        {
            "tier": "B",
            "component": "work_function",
            "status": wf["status"],
            "primary_value": None,
            "claim_boundary": wf["claim_boundary"],
        }
    )
    for key, payload in wf["deltas"].items():
        rows.append(
            {
                "tier": "B",
                "component": key,
                "status": payload["status"],
                "primary_value": payload["delta_eV"],
                "claim_boundary": wf["claim_boundary"],
            }
        )
    rows.append(
        {
            "tier": "C",
            "component": "E_FP",
            "status": efp["status"],
            "primary_value": efp.get("E_FP_eV"),
            "claim_boundary": efp["claim_boundary"],
        }
    )
    rows.append(
        {
            "tier": "C",
            "component": "emission_contrast",
            "status": em["status"],
            "primary_value": None if em.get("contrast") is None else em["contrast"]["ratio"],
            "claim_boundary": em["claim_boundary"],
        }
    )
    rows.append(
        {
            "tier": "C",
            "component": "escape_budget",
            "status": budget["status"],
            "primary_value": None if budget.get("budget") is None else budget["budget"]["E_escape"],
            "claim_boundary": "Delta-SCF budget if available; configured range is a bounded proxy.",
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bigdft-observables",
        required=True,
        help="CSV or JSON from scripts/10_sunday_synthesis.py.",
    )
    parser.add_argument("--clean-ldos", default=None, help="Optional clean LDOS npz matching emission.py contract.")
    parser.add_argument("--recombined-ldos", default=None, help="Optional recombined LDOS npz matching emission.py contract.")
    parser.add_argument("--n-hop", type=int, default=2)
    parser.add_argument("--E-hop", type=float, default=0.4, help="Per-hop energy loss in eV for Eq. 6 budget.")
    parser.add_argument("--output", default=str(RUNS / datetime.now().strftime("phase1_tier_bc_%Y%m%d_%H%M%S")))
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    rows = phase1.read_bigdft_rows(args.bigdft_observables)
    expected = tuple(float(x) for x in cfg["emission"]["E_FP_expected"])
    status = phase1.phase1_status(
        rows,
        phi_eff_default=float(cfg["emission"]["phi_eff"]),
        Te_eV=float(cfg["drive_proxy"]["Te_eV"]),
        n_hop=args.n_hop,
        E_hop=args.E_hop,
        expected_E_FP=expected,
        clean_ldos=args.clean_ldos,
        recombined_ldos=args.recombined_ldos,
    )
    status["created_at"] = now()
    status["inputs"] = {
        "bigdft_observables": args.bigdft_observables,
        "clean_ldos": args.clean_ldos,
        "recombined_ldos": args.recombined_ldos,
        "config": str(CONFIG),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "phase1_tier_bc_status.json", status)
    flatten_summary(status).to_csv(output / "phase1_tier_bc_summary.csv", index=False)
    phase1.write_phase1_report(output / "PHASE1_TIER_BC_REPORT.md", status)
    write_json(
        output / "manifest.json",
        {
            "created_at": status["created_at"],
            "inputs": status["inputs"],
            "outputs": [str(p.relative_to(output)) for p in sorted(output.rglob("*")) if p.is_file()],
        },
    )
    print(f"phase1 Tier B/C status complete -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
