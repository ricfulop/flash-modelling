#!/usr/bin/env python3
"""Aggregate repeat-resolved GPU KPM sweep outputs.

Reads one or more `runs/gpu_kpm_*` directories, combines all completed
`kpm_dos.npz` artifacts by morphology and defect fraction, and writes a
Sunday-ready summary with confidence bands and quick figures.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = REPO_ROOT / "runs"
PALETTE = {
    "ordered": "#0072B2",
    "random": "#D55E00",
    "dendritic": "#009E73",
    "dendrite": "#009E73",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def latest_local_gpu_runs(limit: int = 2) -> list[Path]:
    if not DEFAULT_RUNS.exists():
        return []
    runs = sorted(DEFAULT_RUNS.glob("gpu_kpm_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[:limit]


def import_remote_run(remote: str, import_root: Path) -> Path:
    """Copy a remote run directory locally and return the imported path."""
    if ":" not in remote:
        raise ValueError(f"remote run must look like host:/path, got {remote!r}")
    host, remote_path = remote.split(":", 1)
    dest = import_root / host / Path(remote_path).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        cmd = ["rsync", "-a", "--delete", f"{host}:{remote_path.rstrip('/')}/", str(dest) + "/"]
    else:
        if dest.exists():
            shutil.rmtree(dest)
        cmd = ["scp", "-r", f"{host}:{remote_path}", str(dest.parent)]
    subprocess.run(cmd, check=True)
    return dest


def load_status(run_dir: Path) -> dict:
    path = run_dir / "status.json"
    if not path.exists():
        return {"state": "unknown", "cases": []}
    return json.loads(path.read_text())


def iter_case_records(run_dir: Path):
    status = load_status(run_dir)
    for record in status.get("cases", []):
        output = run_dir / record["output"]
        if not output.exists():
            continue
        yield {
            **record,
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "node_label": status.get("label", run_dir.name),
            "host": status.get("host", "unknown"),
            "npz": str(output),
        }


def load_dos(record: dict, reference_energy: np.ndarray | None):
    data = np.load(record["npz"])
    energies = np.asarray(data["energies"], dtype=float)
    dos = np.asarray(data["dos"], dtype=float)
    if reference_energy is None:
        return energies, dos
    if len(energies) != len(reference_energy) or not np.allclose(energies, reference_energy):
        dos = np.interp(reference_energy, energies, dos)
    return reference_energy, dos


def group_key(record: dict) -> tuple[str, float]:
    return str(record["morphology"]), float(record["target_frac"])


def summarize_group(records: list[dict], out_dir: Path) -> dict:
    reference_energy = None
    dos_arrays = []
    for record in records:
        reference_energy, dos = load_dos(record, reference_energy)
        dos_arrays.append(dos)

    stack = np.vstack(dos_arrays)
    mean = stack.mean(axis=0)
    var = stack.var(axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(mean)
    std = np.sqrt(var)
    sem = std / np.sqrt(max(len(stack), 1))
    ci95 = 1.96 * sem
    q025, q975 = np.quantile(stack, [0.025, 0.975], axis=0)

    morphology, frac = group_key(records[0])
    slug = f"{morphology}_{frac:.2f}".replace(".", "p")
    stats_path = out_dir / "dos_stats" / f"{slug}.npz"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stats_path,
        energies=reference_energy,
        mean=mean,
        variance=var,
        std=std,
        sem=sem,
        ci95=ci95,
        q025=q025,
        q975=q975,
        n=np.array(len(stack), dtype=int),
    )

    peak_idx = int(np.nanargmax(mean))
    zero_idx = int(np.nanargmin(np.abs(reference_energy)))
    auc = float(np.trapezoid(mean, reference_energy))
    return {
        "morphology": morphology,
        "target_frac": frac,
        "n_repeats": len(records),
        "nodes": ",".join(sorted({r["node_label"] for r in records})),
        "energy_min": float(reference_energy.min()),
        "energy_max": float(reference_energy.max()),
        "dos_auc": auc,
        "dos_at_E0_mean": float(mean[zero_idx]),
        "dos_at_E0_ci95": float(ci95[zero_idx]),
        "peak_energy": float(reference_energy[peak_idx]),
        "peak_dos_mean": float(mean[peak_idx]),
        "peak_dos_ci95": float(ci95[peak_idx]),
        "stats_file": str(stats_path.relative_to(out_dir)),
    }


def plot_group_stats(summary: pd.DataFrame, aggregate_dir: Path) -> None:
    figures = aggregate_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    for frac, sub in summary.groupby("target_frac"):
        fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
        for _, row in sub.sort_values("morphology").iterrows():
            data = np.load(aggregate_dir / row["stats_file"])
            energies = data["energies"]
            mean = data["mean"]
            ci95 = data["ci95"]
            color = PALETTE.get(row["morphology"], None)
            label = f"{row['morphology']} (n={int(row['n_repeats'])})"
            ax.plot(energies, mean, label=label, color=color, linewidth=1.8)
            ax.fill_between(energies, mean - ci95, mean + ci95, color=color, alpha=0.18)
        ax.axvline(0.0, color="0.25", linewidth=0.8, linestyle="--")
        ax.set_title(f"KPM DOS by morphology, defect fraction {frac:.2f}")
        ax.set_xlabel("Energy (scaled tight-binding units)")
        ax.set_ylabel("DOS (mean ± 95% CI)")
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(figures / f"dos_by_morphology_frac_{frac:.2f}.png", dpi=300, bbox_inches="tight")
        fig.savefig(figures / f"dos_by_morphology_frac_{frac:.2f}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    x_labels = []
    x = np.arange(len(summary))
    colors = [PALETTE.get(m, "#666666") for m in summary["morphology"]]
    ax.bar(x, summary["dos_at_E0_mean"], yerr=summary["dos_at_E0_ci95"], color=colors, capsize=3)
    for _, row in summary.iterrows():
        x_labels.append(f"{row['morphology']}\n{row['target_frac']:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("DOS at E=0 (mean ± 95% CI)")
    ax.set_title("Fermi-proxy DOS summary across KPM repeats")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(figures / "dos_at_E0_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "dos_at_E0_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, aggregate_dir: Path, run_dirs: list[Path]) -> None:
    lines = [
        "# KPM Sweep Aggregate",
        "",
        f"Updated: {now()}",
        "",
        "## Source Runs",
        *[f"- `{p}`" for p in run_dirs],
        "",
        "## Summary",
    ]
    for _, row in summary.sort_values(["target_frac", "morphology"]).iterrows():
        lines.append(
            f"- `{row['morphology']}` frac={row['target_frac']:.2f}: "
            f"n={int(row['n_repeats'])}, DOS(E=0)={row['dos_at_E0_mean']:.4g} ± "
            f"{row['dos_at_E0_ci95']:.2g}, peak at E={row['peak_energy']:.3g}"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "- `summary.csv`: table for manuscript triage.",
            "- `dos_stats/*.npz`: mean, variance, SEM, CI, and quantile bands.",
            "- `figures/*.png` and `figures/*.pdf`: quick Sunday-ready plots.",
        ]
    )
    (aggregate_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="Local gpu_kpm run directory")
    parser.add_argument("--remote", action="append", default=[], help="Remote run as host:/absolute/path")
    parser.add_argument("--output", default=None, help="Output aggregate directory")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    aggregate_dir = Path(args.output) if args.output else DEFAULT_RUNS / datetime.now().strftime("kpm_aggregate_%Y%m%d_%H%M%S")
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    import_root = aggregate_dir / "imports"
    run_dirs = [Path(p) for p in args.run]
    for remote in args.remote:
        run_dirs.append(import_remote_run(remote, import_root))
    if not run_dirs:
        run_dirs = latest_local_gpu_runs()

    records = []
    for run_dir in run_dirs:
        records.extend(iter_case_records(run_dir))
    if not records:
        raise SystemExit("No completed KPM DOS records found.")

    records_df = pd.DataFrame(records)
    records_df.to_csv(aggregate_dir / "records.csv", index=False)

    summaries = []
    grouped = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)
    for key in sorted(grouped):
        summaries.append(summarize_group(grouped[key], aggregate_dir))

    summary = pd.DataFrame(summaries).sort_values(["target_frac", "morphology"])
    summary.to_csv(aggregate_dir / "summary.csv", index=False)
    write_json(
        aggregate_dir / "summary.json",
        {
            "created_at": now(),
            "source_runs": [str(p) for p in run_dirs],
            "n_records": len(records),
            "groups": summary.to_dict(orient="records"),
        },
    )
    if not args.no_figures:
        plot_group_stats(summary, aggregate_dir)
    write_markdown(summary, aggregate_dir, run_dirs)
    print(f"aggregated {len(records)} KPM records -> {aggregate_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
