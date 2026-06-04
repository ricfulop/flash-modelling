#!/usr/bin/env python3
"""Create manuscript-facing digest for the upgraded Sunday scope.

This script consumes already-derived artifacts and writes a compact digest. It
does not alter the underlying KPM, BigDFT, or synthesis runs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def read_status(run: Path) -> dict:
    return json.loads((run / "status.json").read_text())


def collect_large_runs(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for run in paths:
        status = read_status(run)
        for row in status.get("cases", []):
            rows.append(
                {
                    "run": run.name,
                    "state": status.get("state"),
                    "morphology": row.get("morphology"),
                    "target_frac": row.get("target_frac"),
                    "shape": "x".join(str(x) for x in row.get("shape", [])),
                    "linear_size": int(row.get("shape", [0])[0]) if row.get("shape") else None,
                    "total_lattice_sites": row.get("total_lattice_sites"),
                    "defect_graph_nodes": row.get("defect_graph_nodes"),
                    "defect_graph_edges": row.get("defect_graph_edges"),
                    "dos_at_E0": row.get("dos_at_E0"),
                    "peak_energy": row.get("peak_energy"),
                    "peak_dos": row.get("peak_dos"),
                    "elapsed_s": row.get("elapsed_s"),
                    "output": str(run / row.get("output", "")),
                }
            )
    return pd.DataFrame(rows)


def collect_kubo_runs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for run in paths:
        summary = run / "kubo_mobility_summary.csv"
        if summary.exists():
            df = pd.read_csv(summary)
            df["run"] = run.name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_large_lattice(df: pd.DataFrame, fig_dir: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    for morphology, sub in df.sort_values(["morphology", "linear_size"]).groupby("morphology"):
        ax.plot(sub["linear_size"], sub["dos_at_E0"], marker="o", linewidth=1.8, label=morphology)
    ax.set_xlabel("BCC supercell linear size")
    ax.set_ylabel("DOS at E=0")
    ax.set_title("Large-lattice KPM finite-size showcase")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / "large_lattice_dos_E0_by_size.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "large_lattice_dos_E0_by_size.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    for morphology, sub in df.sort_values(["morphology", "linear_size"]).groupby("morphology"):
        ax.plot(sub["defect_graph_nodes"], sub["elapsed_s"], marker="o", linewidth=1.8, label=morphology)
    ax.set_xlabel("Defect graph nodes")
    ax.set_ylabel("Elapsed seconds")
    ax.set_title("GPU KPM scaling observed in upgraded run")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / "large_lattice_runtime_scaling.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "large_lattice_runtime_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_kubo(df: pd.DataFrame, fig_dir: Path) -> None:
    if df.empty:
        return
    order = df.sort_values(["target_frac", "morphology"]).reset_index(drop=True)
    labels = [f"{m}\n{f:.2f}" for m, f in zip(order["morphology"], order["target_frac"])]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.bar(x, order["sigma_at_E0"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("sigma(E=0), arb.")
    ax.set_title("Kubo-Greenwood conductivity proxy at E=0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / "kubo_sigma_E0_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "kubo_sigma_E0_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.bar(x, order["n_mobility_windows"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mobility-window count")
    ax.set_title("Tier D-lite+ mobility-window proxy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / "kubo_mobility_window_count.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "kubo_mobility_window_count.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize_bigdft(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"available": False}
    df = pd.read_csv(path)
    if df.empty:
        return {"available": False}
    warning_col = df["warning_count"].fillna(0) if "warning_count" in df else pd.Series([], dtype=float)
    infocode_col = df["infocode"].fillna(0) if "infocode" in df else pd.Series([], dtype=float)
    has_nan_col = df["has_nan"].fillna(False) if "has_nan" in df else pd.Series([], dtype=bool)
    caution = int(((warning_col > 0) | (infocode_col != 0) | has_nan_col.astype(bool)).sum())
    return {
        "available": True,
        "jobs_parsed": int(len(df)),
        "jobs_with_caution": caution,
        "finite_phi_proxy_jobs": int(df["phi_proxy_eV"].notna().sum()) if "phi_proxy_eV" in df else 0,
    }


def summarize_tier_bc(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"available": False}
    payload = json.loads(path.read_text())
    return {
        "available": True,
        "work_function_status": payload.get("tier_b_work_function", {}).get("status"),
        "E_FP_status": payload.get("tier_c_E_FP", {}).get("status"),
        "emission_status": payload.get("tier_c_emission", {}).get("status"),
        "escape_budget_status": payload.get("tier_c_escape_budget", {}).get("status"),
    }


def claim_table(kpm: pd.DataFrame, large: pd.DataFrame, kubo: pd.DataFrame, bigdft_summary: dict, tier_bc: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_area": "Frozen regime/mechanism adjudication",
                "status": "confirmatory-within-registered-proxy",
                "evidence": f"{len(kpm)} KPM morphology/fraction groups" if not kpm.empty else "missing KPM summary",
                "boundary": "Conditional on hypothesized defect morphology and TB parameterization.",
            },
            {
                "claim_area": "Large-lattice transport scaling",
                "status": "supporting-robustness",
                "evidence": f"{len(large)} large-lattice KPM cases up to {int(large['total_lattice_sites'].max()) if not large.empty else 0} lattice sites",
                "boundary": "KPM/TB showcase, not ab initio DFT at full lattice size.",
            },
            {
                "claim_area": "Kubo/mobility-window diagnostics",
                "status": "exploratory-tier-d-lite-plus",
                "evidence": f"{len(kubo)} Kubo-Greenwood cases" if not kubo.empty else "missing Kubo summary",
                "boundary": "Conductivity is in arbitrary units; mobility windows are proxy thresholds.",
            },
            {
                "claim_area": "BigDFT work-function / energetics anchor",
                "status": "provisional-until-convergence-reviewed",
                "evidence": f"{bigdft_summary.get('jobs_parsed', 0)} parsed jobs; {bigdft_summary.get('jobs_with_caution', 0)} caution jobs",
                "boundary": "Report differences only after convergence and vacuum-level extraction are reviewed.",
            },
            {
                "claim_area": "Emission GO/NO-GO and escape budget",
                "status": (
                    f"E_FP={tier_bc.get('E_FP_status')}; emission={tier_bc.get('emission_status')}; "
                    f"budget={tier_bc.get('escape_budget_status')}"
                    if tier_bc.get("available")
                    else "missing-tier-bc-status"
                ),
                "evidence": "Stage 03 Tier B/C status artifact" if tier_bc.get("available") else "run scripts/03_bigdft_emission.py",
                "boundary": "Emission PASS requires BigDFT LDOS contrast; budget-only or configured-range outputs are not emission proof.",
            },
        ]
    )


def write_report(output: Path, kpm: pd.DataFrame, large: pd.DataFrame, kubo: pd.DataFrame, claims: pd.DataFrame, bigdft_summary: dict, tier_bc: dict) -> None:
    lines = [
        "# Sunday Scope Upgrade Digest",
        "",
        f"Created: {now()}",
        "",
        "## What Changed",
        "- The Sunday scope is upgraded from the minimum Tier A/B package to A+/D-lite+.",
        "- Large-lattice GPU KPM and Kubo-Greenwood mobility diagnostics are manuscript-facing robustness/exploratory support.",
        "- BigDFT remains the CPU-backed DFT anchor; GPU BigDFT is excluded from production claims.",
        "",
        "## Evidence Inventory",
        f"- KPM summary groups: {len(kpm)}.",
        f"- Large-lattice KPM cases: {len(large)}.",
        f"- Largest lattice sites: {int(large['total_lattice_sites'].max()) if not large.empty else 0}.",
        f"- Kubo/mobility cases: {len(kubo)}.",
        f"- BigDFT parsed jobs: {bigdft_summary.get('jobs_parsed', 0)}.",
        f"- Tier B/C status artifact: {'yes' if tier_bc.get('available') else 'no'}.",
        "",
        "## Claim Boundaries",
    ]
    for _, row in claims.iterrows():
        lines.append(f"- **{row['claim_area']}**: {row['status']}. Evidence: {row['evidence']}. Boundary: {row['boundary']}")
    lines.extend(
        [
            "",
            "## Manuscript Language Guardrail",
            "Use \"beyond-Ioffe-Regel localization in hypothesized high-FP defect structures\" unless the frozen regime verdict and DFT anchoring justify a narrower mechanism label.",
            "",
            "## Outputs",
            "- `claim_boundaries.csv`",
            "- `large_lattice_summary.csv`",
            "- `kubo_mobility_summary.csv`",
            "- `figures/large_lattice_dos_E0_by_size.png`",
            "- `figures/large_lattice_runtime_scaling.png`",
            "- `figures/kubo_sigma_E0_summary.png`",
            "- `figures/kubo_mobility_window_count.png`",
        ]
    )
    (output / "SUNDAY_SCOPE_UPGRADE_DIGEST.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kpm-summary", required=True)
    parser.add_argument("--bigdft-summary", default=None)
    parser.add_argument("--tier-bc-status", default=None, help="Optional phase1_tier_bc_status.json from scripts/03.")
    parser.add_argument("--large-run", action="append", default=[])
    parser.add_argument("--kubo-run", action="append", default=[])
    parser.add_argument("--output", default=str(RUNS / datetime.now().strftime("sunday_scope_upgrade_%Y%m%d_%H%M%S")))
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fig_dir = output / "figures"
    fig_dir.mkdir(exist_ok=True)

    kpm = pd.read_csv(args.kpm_summary)
    large = collect_large_runs([Path(p) for p in args.large_run])
    kubo = collect_kubo_runs([Path(p) for p in args.kubo_run])
    bigdft_summary = summarize_bigdft(Path(args.bigdft_summary) if args.bigdft_summary else None)
    tier_bc = summarize_tier_bc(Path(args.tier_bc_status) if args.tier_bc_status else None)
    claims = claim_table(kpm, large, kubo, bigdft_summary, tier_bc)

    kpm.to_csv(output / "kpm_summary.csv", index=False)
    large.to_csv(output / "large_lattice_summary.csv", index=False)
    kubo.to_csv(output / "kubo_mobility_summary.csv", index=False)
    claims.to_csv(output / "claim_boundaries.csv", index=False)
    write_json(output / "manifest.json", {
        "created_at": now(),
        "kpm_summary": args.kpm_summary,
        "bigdft_summary": args.bigdft_summary,
        "tier_bc_status": args.tier_bc_status,
        "large_runs": args.large_run,
        "kubo_runs": args.kubo_run,
        "outputs": [],
    })
    plot_large_lattice(large, fig_dir)
    plot_kubo(kubo, fig_dir)
    write_report(output, kpm, large, kubo, claims, bigdft_summary, tier_bc)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["outputs"] = [str(p.relative_to(output)) for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "manifest.json", manifest)
    print(f"scope-upgrade digest complete -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
