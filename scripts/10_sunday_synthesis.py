#!/usr/bin/env python3
"""Assemble Sunday-ready outputs from completed KPM and BigDFT runs.

This script is deliberately conservative: it extracts what is present, labels
proxy analyses as exploratory, and never promotes unvalidated CUDA BigDFT.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
DATA = REPO / "data" / "tier_a"
CONFIG = REPO / "config" / "w_percolation.yaml"
HA_TO_EV = 27.211386245988


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def import_peer_tree(remote: str, destination: Path) -> Path:
    """Copy peer run tree locally so downstream parsing is reproducible."""
    if ":" not in remote:
        raise ValueError(f"remote path must be host:/path, got {remote!r}")
    host, path = remote.split(":", 1)
    target = destination / host / Path(path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        cmd = ["rsync", "-a", "--delete", f"{host}:{path.rstrip('/')}/", str(target) + "/"]
    else:
        if target.exists():
            shutil.rmtree(target)
        cmd = ["scp", "-r", f"{host}:{path}", str(target.parent)]
    subprocess.run(cmd, check=True)
    return target


def read_status(path: Path) -> dict:
    return json.loads((path / "status.json").read_text())


def regex_last(pattern: str, text: str, cast=float):
    hits = re.findall(pattern, text, flags=re.IGNORECASE)
    if not hits:
        return None
    value = hits[-1]
    if isinstance(value, tuple):
        value = value[-1]
    try:
        return cast(str(value).strip())
    except Exception:
        return str(value).strip()


def count_warnings(text: str) -> int:
    return len(re.findall(r"#WARNING|WARNING:", text, flags=re.IGNORECASE))


def contains_numeric_nan(text: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])(?:NaN|nan)(?![A-Za-z])", text))


def parse_avg_potential(path: Path):
    if not path.exists():
        return None
    try:
        arr = np.loadtxt(path)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    z = arr[:, 1]
    v = arr[:, 2]
    n_tail = max(5, int(0.1 * len(v)))
    return {
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "vacuum_left_Ha": float(np.mean(v[:n_tail])),
        "vacuum_right_Ha": float(np.mean(v[-n_tail:])),
        "vacuum_max_Ha": float(np.max(v)),
        "vacuum_tail_mean_Ha": float(np.mean([np.mean(v[:n_tail]), np.mean(v[-n_tail:])])),
    }


def parse_bigdft_job(job_dir: Path, node_label: str) -> dict:
    log = job_dir / "log.yaml"
    text = log.read_text(errors="replace") if log.exists() else ""
    job_meta = {}
    if (job_dir / "job.json").exists():
        job_meta = json.loads((job_dir / "job.json").read_text())
    fermi = regex_last(r"Fermi Energy\s*:\s*([+\-0-9.Ee]+)", text)
    energy = regex_last(r"Energy \(Hartree\)\s*:\s*([+\-0-9.EeNaInf]+)", text)
    charge = regex_last(r"Total electronic charge\s*:\s*([+\-0-9.EeNaInf]+)", text)
    infocode = regex_last(r"BigDFT infocode\s*:\s*([+\-0-9]+)", text, int)
    elapsed = regex_last(r"Elapsed time \(s\)\s*:\s*([+\-0-9.Ee]+)", text)
    atoms = None
    posinp = job_dir / "posinp.xyz"
    if posinp.exists():
        try:
            atoms = int(posinp.read_text().splitlines()[0].split()[0])
        except Exception:
            atoms = None
    pot = parse_avg_potential(job_dir / "rundata" / "local_potential_avg_z.dat")
    phi_proxy_eV = None
    if pot is not None and fermi is not None:
        phi_proxy_eV = (pot["vacuum_max_Ha"] - fermi) * HA_TO_EV
    return {
        "node": node_label,
        "job": job_dir.name,
        "notes": job_meta.get("notes"),
        "atoms": atoms,
        "energy_Ha": energy,
        "fermi_Ha": fermi,
        "charge_e": charge,
        "infocode": infocode,
        "warning_count": count_warnings(text),
        "has_nan": contains_numeric_nan(text),
        "elapsed_log_s": elapsed,
        "local_potential_present": pot is not None,
        "vacuum_proxy_Ha": pot["vacuum_max_Ha"] if pot else None,
        "phi_proxy_eV": phi_proxy_eV,
        "path": str(job_dir),
    }


def collect_bigdft(local_runs: list[tuple[str, Path]], peer_run: Path | None, output: Path) -> pd.DataFrame:
    rows = []
    all_runs: list[tuple[str, Path | None]] = [*local_runs, ("peer", peer_run)]
    for label, run in all_runs:
        if run is None:
            continue
        status_path = run / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("invalidated_reason") or status.get("state") == "blocked":
                continue
        work = run / "work"
        for job_dir in sorted(work.glob("*")):
            if job_dir.is_dir():
                rows.append(parse_bigdft_job(job_dir, label))
    df = pd.DataFrame(rows)
    out = output / "bigdft_observables.csv"
    df.to_csv(out, index=False)
    write_json(output / "bigdft_observables.json", {"jobs": df.to_dict(orient="records")})
    return df


def load_kpm_summary(kpm_aggregate: Path, output: Path) -> pd.DataFrame:
    src = kpm_aggregate / "summary.csv"
    df = pd.read_csv(src)
    df.to_csv(output / "kpm_summary.csv", index=False)
    return df


def load_mechanism_summary(output: Path) -> pd.DataFrame:
    df = pd.read_csv(DATA / "mechanism_summary.csv")
    df.to_csv(output / "mechanism_summary.csv", index=False)
    return df


def deterministic_subgraph(net, max_sites: int):
    if net.graph.number_of_nodes() <= max_sites:
        return net.graph.copy(), np.array(list(net.graph.nodes()), dtype=int)
    chosen = [
        n for n, _ in sorted(net.graph.degree, key=lambda item: (-item[1], item[0]))[:max_sites]
    ]
    sub = net.graph.subgraph(chosen).copy()
    return nx.convert_node_labels_to_integers(sub), np.array(chosen, dtype=int)


def union_find_spanning_threshold(edges, z_by_node, z_low, z_high):
    parent = {}
    low = {}
    high = {}

    def find(x):
        parent.setdefault(x, x)
        low.setdefault(x, z_by_node[x] <= z_low)
        high.setdefault(x, z_by_node[x] >= z_high)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        parent[rb] = ra
        low[ra] = low.get(ra, False) or low.get(rb, False)
        high[ra] = high.get(ra, False) or high.get(rb, False)
        return ra

    for ea, u, v in sorted(edges):
        r = union(u, v)
        if low.get(r, False) and high.get(r, False):
            return float(ea)
    return None


def run_tier_d_lite(output: Path, max_sites: int = 900) -> pd.DataFrame:
    sys.path.insert(0, str(REPO / "src"))
    from electrodefect import tier_a, transport_kpm as tk

    cfg = yaml.safe_load(CONFIG.read_text())
    rows = []
    details = {}
    lambda_grid = [0.05, 0.10, 0.25, 0.50]
    ef = 0.0
    ipr_pr_localized_threshold = 0.01
    for morphology, frac in tier_a.iter_cases(cfg):
        net = tier_a.load_network(tier_a.case_dir(morphology, frac))
        sub, original_nodes = deterministic_subgraph(net, max_sites)
        coords_all = tk.defect_node_coords(net)
        coords = coords_all[original_nodes]
        # Rebuild a DefectNetwork-like object for exact TB on the subgraph.
        from electrodefect.percolation import DefectNetwork
        work_net = DefectNetwork(coords, np.ones(len(coords), dtype=bool), net.a, net.nx_shape, sub)
        disorder = 0.0 if morphology == "ordered" else float(cfg["tight_binding"]["disorder_W_random"])
        H_sp, _, _ = tk.build_tb(
            work_net,
            t_hop=float(cfg["tight_binding"]["t_hop"]),
            disorder_W=disorder,
            seed=int(cfg["network"].get("seed", 0)),
            device="scipy",
        )
        H = H_sp.toarray()
        evals, ipr = tk.ipr_spectrum(H)
        n = len(evals)
        ipr_threshold = 1.0 / max(ipr_pr_localized_threshold * n, 1.0)
        above = evals >= ef
        extended_above = above & (ipr <= ipr_threshold)
        localized_above = above & (ipr > ipr_threshold)
        e_extended_onset = float(np.min(evals[extended_above] - ef)) if np.any(extended_above) else None
        localized_fraction_above = float(np.mean(localized_above[above])) if np.any(above) else None

        # Site-energy and edge activation proxy.
        rng = np.random.default_rng(int(cfg["network"].get("seed", 0)))
        eps = rng.uniform(-disorder / 2, disorder / 2, n) if disorder else np.zeros(n)
        z = coords[:, 2]
        z_low, z_high = np.quantile(z, [0.1, 0.9])
        percolation = {}
        for lam in lambda_grid:
            edge_records = []
            for u, v in sub.edges():
                delta = eps[v] - eps[u]
                ea = ((lam + delta) ** 2) / max(4.0 * lam, 1e-12)
                edge_records.append((float(ea), int(u), int(v)))
            critical = union_find_spanning_threshold(edge_records, z, z_low, z_high)
            percolation[f"lambda_{lam:.2f}_eV"] = critical

        row = {
            "morphology": morphology,
            "target_frac": frac,
            "n_sites_subgraph": n,
            "ipr_threshold": ipr_threshold,
            "localized_fraction_above_EF": localized_fraction_above,
            "extended_onset_above_EF_eV": e_extended_onset,
            **percolation,
        }
        rows.append(row)
        details[f"{morphology}_{tier_a.frac_slug(frac)}"] = {
            "energies": evals.tolist(),
            "ipr": ipr.tolist(),
            "note": "Exploratory Tier D-lite proxy; not Wannier/DFPT.",
        }

    df = pd.DataFrame(rows)
    df.to_csv(output / "tier_d_lite_summary.csv", index=False)
    write_json(output / "tier_d_lite_ipr_details.json", details)
    return df


def make_figures(kpm: pd.DataFrame, mechanism: pd.DataFrame, tier_d: pd.DataFrame, bigdft: pd.DataFrame, output: Path) -> None:
    fig_dir = output / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for morphology, sub in kpm.groupby("morphology"):
        ax.errorbar(sub["target_frac"], sub["dos_at_E0_mean"], yerr=sub["dos_at_E0_ci95"], marker="o", label=morphology)
    ax.set_xlabel("Defect fraction")
    ax.set_ylabel("DOS at E=0 (mean +/- 95% CI)")
    ax.set_title("KPM Fermi-proxy DOS by morphology")
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "kpm_dos_E0_by_morphology.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    plot_df = tier_d.copy()
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["localized_fraction_above_EF"].fillna(0.0))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{f:.2f}" for m, f in zip(plot_df["morphology"], plot_df["target_frac"])], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Localized fraction above E_F")
    ax.set_title("Tier D-lite IPR mobility-window proxy")
    fig.savefig(fig_dir / "tier_d_lite_localized_fraction.png", dpi=300)
    plt.close(fig)

    if "phi_proxy_eV" in bigdft:
        sub = bigdft[bigdft["phi_proxy_eV"].notna()].copy()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
            ax.bar(np.arange(len(sub)), sub["phi_proxy_eV"])
            ax.set_xticks(np.arange(len(sub)))
            ax.set_xticklabels(sub["job"], rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("V_vac proxy - E_F (eV)")
            ax.set_title("BigDFT work-function proxy (check convergence before reporting)")
            fig.savefig(fig_dir / "bigdft_phi_proxy.png", dpi=300)
            plt.close(fig)


def write_report(output: Path, kpm: pd.DataFrame, mechanism: pd.DataFrame, tier_d: pd.DataFrame, bigdft: pd.DataFrame) -> None:
    nonconverged = bigdft[(bigdft["infocode"].fillna(0) != 0) | (bigdft["warning_count"].fillna(0) > 0)]
    lines = [
        "# Sunday Synthesis Package",
        "",
        f"Created: {now()}",
        "",
        "## Completed Data",
        f"- KPM aggregate groups: {len(kpm)} morphology/fraction groups.",
        f"- KPM repeats per group: {', '.join(str(int(x)) for x in sorted(kpm['n_repeats'].unique()))}.",
        f"- BigDFT jobs parsed: {len(bigdft)}.",
        f"- Tier D-lite proxy rows: {len(tier_d)}.",
        "",
        "## Sunday-Critical Outputs",
        "- `kpm_summary.csv`: mean/CI DOS by morphology and fraction.",
        "- `mechanism_summary.csv`: frozen Tier A regime/mechanism table.",
        "- `bigdft_observables.csv`: energies, Fermi levels, convergence warnings, and potential proxies.",
        "- `tier_d_lite_summary.csv`: exploratory IPR/mobility-window and Anderson-Holstein proxy.",
        "- `figures/`: quick manuscript triage figures.",
        "",
        "## BigDFT Caution",
        f"- Jobs with warnings or nonzero infocode: {len(nonconverged)} / {len(bigdft)}.",
        "- Treat work-function proxies as provisional until convergence and vacuum-level extraction are reviewed.",
        "",
        "## Tier D-lite Caution",
        "- This is not Wannier90/DFPT/EPW. It uses TB proxy site energies and hopping on subgraphs.",
        "- Activation-energy estimates are exploratory and must not be tuned to match measured Arrhenius barriers.",
    ]
    (output / "SUNDAY_SYNTHESIS.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kpm-aggregate", default=str(RUNS / "kpm_aggregate_status_20260603_1549"))
    parser.add_argument("--local-bigdft", default=str(RUNS / "single_bigdft_20260603_101509_local"))
    parser.add_argument(
        "--extra-bigdft",
        action="append",
        default=[],
        help="Additional local BigDFT run roots to include, e.g. convergence follow-up queues.",
    )
    parser.add_argument("--peer-bigdft-remote", default="dgx-peer:/home/nvidia/Cursor/flash-modelling/runs/single_bigdft_20260603_101549_peer")
    parser.add_argument("--output", default=str(RUNS / datetime.now().strftime("sunday_synthesis_%Y%m%d_%H%M%S")))
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    imports = output / "imports"
    peer = None
    peer_import_error = None
    if args.peer_bigdft_remote:
        try:
            peer = import_peer_tree(args.peer_bigdft_remote, imports)
        except Exception as exc:
            peer_import_error = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: peer BigDFT import failed; continuing local-only: {peer_import_error}")

    kpm = load_kpm_summary(Path(args.kpm_aggregate), output)
    mechanism = load_mechanism_summary(output)
    local_runs = [("local", Path(args.local_bigdft))]
    local_runs.extend((f"local_extra_{idx}", Path(path)) for idx, path in enumerate(args.extra_bigdft, start=1))
    bigdft = collect_bigdft(local_runs, peer, output)
    tier_d = run_tier_d_lite(output)
    make_figures(kpm, mechanism, tier_d, bigdft, output)
    write_report(output, kpm, mechanism, tier_d, bigdft)
    write_json(
        output / "manifest.json",
        {
            "created_at": now(),
            "kpm_aggregate": args.kpm_aggregate,
            "local_bigdft": args.local_bigdft,
            "extra_bigdft": args.extra_bigdft,
            "peer_bigdft_import": str(peer) if peer else None,
            "peer_import_error": peer_import_error,
            "outputs": [str(p.relative_to(output)) for p in sorted(output.rglob("*")) if p.is_file()],
        },
    )
    print(f"sunday synthesis complete -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
