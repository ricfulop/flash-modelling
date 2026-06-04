"""Phase 1 completion helpers and claim gates.

This module turns derived Tier A/BigDFT/emission artifacts into a bounded
Phase 1 status package. It deliberately separates implemented pipeline state
from manuscript-strength scientific claims.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from . import emission

HA_TO_EV = 27.211386245988


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def read_bigdft_rows(path: str | Path) -> list[dict]:
    """Read a BigDFT observables CSV or JSON file."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        return list(payload.get("jobs", payload if isinstance(payload, list) else []))
    import csv

    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def classify_job_name(name: str) -> str | None:
    """Map a BigDFT job name onto the Phase 1 role it can support."""
    lower = name.lower()
    if "clean" in lower and "phi" in lower:
        return "clean_phi"
    if "defect" in lower and "phi" in lower:
        return "defect_phi"
    if "stepped" in lower and "phi" in lower:
        return "stepped_phi"
    if "separated" in lower and "efp" in lower:
        return "separated_efp"
    if "recombined" in lower and "efp" in lower:
        return "recombined_efp"
    if "w_atom" in lower:
        return "w_atom_smoke"
    return None


def row_quality(row: dict) -> dict:
    """Return deterministic quality flags for a parsed BigDFT row."""
    infocode = row.get("infocode")
    warning_count = row.get("warning_count", 0)
    has_nan = str(row.get("has_nan", "")).lower() in {"true", "1", "yes"}
    try:
        warning_count = int(float(warning_count or 0))
    except Exception:
        warning_count = 0
    try:
        infocode_i = int(float(infocode)) if infocode not in (None, "") else None
    except Exception:
        infocode_i = None
    finite_energy = _finite(row.get("energy_Ha"))
    finite_fermi = _finite(row.get("fermi_Ha"))
    finite_phi = _finite(row.get("phi_proxy_eV"))
    caution = has_nan or (infocode_i not in (None, 0)) or warning_count > 0
    if has_nan or not finite_energy:
        status = "failed"
    elif caution:
        status = "provisional"
    else:
        status = "accepted"
    return {
        "status": status,
        "infocode": infocode_i,
        "warning_count": warning_count,
        "has_nan": has_nan,
        "finite_energy": finite_energy,
        "finite_fermi": finite_fermi,
        "finite_phi": finite_phi,
    }


def select_best_rows(rows: Iterable[dict]) -> dict[str, dict]:
    """Select the best available row for each Phase 1 BigDFT role."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        role = classify_job_name(str(row.get("job", "")))
        if role:
            grouped.setdefault(role, []).append(row)

    def key(row: dict):
        quality = row_quality(row)
        name = str(row.get("job", ""))
        return (
            quality["status"] == "failed",
            quality["status"] == "provisional",
            0 if "nrep4" in name.lower() else 1,
            0 if "small" in name.lower() else 1,
            quality["warning_count"],
            -float(row.get("elapsed_log_s") or 0.0),
            str(row.get("path", "")),
        )

    return {role: sorted(candidates, key=key)[0] for role, candidates in grouped.items()}


def work_function_summary(best: dict[str, dict]) -> dict:
    """Compute clean-referenced phi differences from selected BigDFT rows."""
    clean = best.get("clean_phi")
    entries = {}
    for role in ("clean_phi", "defect_phi", "stepped_phi"):
        row = best.get(role)
        entries[role] = {
            "available": row is not None,
            "job": row.get("job") if row else None,
            "path": row.get("path") if row else None,
            "phi_proxy_eV": float(row["phi_proxy_eV"]) if row and _finite(row.get("phi_proxy_eV")) else None,
            "quality": row_quality(row) if row else None,
        }

    deltas = {}
    clean_phi = entries["clean_phi"]["phi_proxy_eV"]
    for role in ("defect_phi", "stepped_phi"):
        phi = entries[role]["phi_proxy_eV"]
        if clean_phi is None or phi is None:
            status = "missing"
            delta = None
        else:
            delta = float(phi - clean_phi)
            q1 = entries["clean_phi"]["quality"]["status"]
            q2 = entries[role]["quality"]["status"]
            status = "accepted" if q1 == q2 == "accepted" else "provisional"
        deltas[f"{role}_minus_clean_eV"] = {"delta_eV": delta, "status": status}

    return {
        "status": "missing" if clean is None else ("accepted" if all(v["status"] == "accepted" for v in deltas.values()) else "provisional"),
        "selected_jobs": entries,
        "deltas": deltas,
        "claim_boundary": "Use as quantitative only when selected slab jobs are accepted and vacuum extraction is reviewed.",
    }


def efp_summary(best: dict[str, dict]) -> dict:
    """Compute Delta-SCF E_FP if separated and recombined jobs are present."""
    separated = best.get("separated_efp")
    recombined = best.get("recombined_efp")
    if not separated or not recombined:
        return {
            "status": "missing",
            "E_FP_eV": None,
            "selected_jobs": {
                "separated_efp": separated.get("job") if separated else None,
                "recombined_efp": recombined.get("job") if recombined else None,
            },
            "claim_boundary": "Requires paired separated and recombined Delta-SCF totals.",
        }
    if not (_finite(separated.get("energy_Ha")) and _finite(recombined.get("energy_Ha"))):
        return {"status": "failed", "E_FP_eV": None, "claim_boundary": "Non-finite total energy."}
    e_fp = (float(separated["energy_Ha"]) - float(recombined["energy_Ha"])) * HA_TO_EV
    q_sep = row_quality(separated)["status"]
    q_rec = row_quality(recombined)["status"]
    return {
        "status": "accepted" if q_sep == q_rec == "accepted" else "provisional",
        "E_FP_eV": float(e_fp),
        "selected_jobs": {
            "separated_efp": separated.get("job"),
            "recombined_efp": recombined.get("job"),
        },
        "quality": {"separated": row_quality(separated), "recombined": row_quality(recombined)},
        "claim_boundary": "Use quantitatively only if both Delta-SCF jobs are accepted and geometries are comparable.",
    }


def load_ldos_npz(path: str | Path) -> dict:
    """Load an emission LDOS artifact following the `emission.py` data contract."""
    data = np.load(path)
    required = ["energies", "density_z", "z_grid", "z_surface", "E_F"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"LDOS artifact {path} missing keys: {missing}")
    return {
        "energies": np.asarray(data["energies"], dtype=float),
        "density_z": np.asarray(data["density_z"], dtype=float),
        "z_grid": np.asarray(data["z_grid"], dtype=float),
        "z_surface": float(np.asarray(data["z_surface"])),
        "E_F": float(np.asarray(data["E_F"])),
        "phi_eff": float(np.asarray(data["phi_eff"])) if "phi_eff" in data else None,
    }


def emission_summary(clean_ldos: str | Path | None, recombined_ldos: str | Path | None, *, phi_eff: float, Te_eV: float) -> dict:
    """Run the Tier C emission contrast if LDOS artifacts are available."""
    if not clean_ldos or not recombined_ldos:
        return {
            "status": "missing",
            "contrast": None,
            "claim_boundary": "Requires clean and recombined vacuum-LDOS artifacts from BigDFT.",
        }
    clean = load_ldos_npz(clean_ldos)
    recombined = load_ldos_npz(recombined_ldos)
    clean_flux = emission.emittable_flux(
        clean["energies"], clean["density_z"], clean["z_grid"], clean["z_surface"], clean["E_F"], phi_eff, Te_eV
    )
    recombined_flux = emission.emittable_flux(
        recombined["energies"],
        recombined["density_z"],
        recombined["z_grid"],
        recombined["z_surface"],
        recombined["E_F"],
        phi_eff,
        Te_eV,
    )
    contrast = emission.emission_contrast(clean_flux, recombined_flux)
    return {
        "status": "accepted" if contrast["ratio"] > 5 else "inconclusive",
        "contrast": contrast,
        "clean_ldos": str(clean_ldos),
        "recombined_ldos": str(recombined_ldos),
        "claim_boundary": "Static LDOS proxy, not real-time emission dynamics.",
    }


def escape_budget_summary(e_fp: dict, *, phi_eff: float, n_hop: int, E_hop: float, expected_range: tuple[float, float] | None = None) -> dict:
    """Compute final or bounded E_escape budgets."""
    if e_fp.get("E_FP_eV") is not None:
        budget = emission.escape_budget(float(e_fp["E_FP_eV"]), n_hop=n_hop, E_hop=E_hop, phi_eff=phi_eff)
        return {"status": e_fp.get("status", "provisional"), "budget": budget, "source": "Delta-SCF"}
    if expected_range is None:
        return {"status": "missing", "budget": None, "source": None}
    lo, hi = expected_range
    return {
        "status": "bounded-literature-proxy",
        "budget_range": [
            emission.escape_budget(float(lo), n_hop=n_hop, E_hop=E_hop, phi_eff=phi_eff),
            emission.escape_budget(float(hi), n_hop=n_hop, E_hop=E_hop, phi_eff=phi_eff),
        ],
        "source": "configured E_FP_expected range",
    }


def phase1_status(
    bigdft_rows: Iterable[dict],
    *,
    phi_eff_default: float,
    Te_eV: float,
    n_hop: int,
    E_hop: float,
    expected_E_FP: tuple[float, float] | None = None,
    clean_ldos: str | Path | None = None,
    recombined_ldos: str | Path | None = None,
) -> dict:
    """Assemble gated Tier B/C status from BigDFT and optional LDOS artifacts."""
    best = select_best_rows(bigdft_rows)
    wf = work_function_summary(best)
    e_fp = efp_summary(best)
    # Use defect slab phi if present; otherwise fall back to configured W default.
    defect_phi = wf["selected_jobs"].get("defect_phi", {}).get("phi_proxy_eV")
    phi_eff = float(defect_phi if defect_phi is not None else phi_eff_default)
    em = emission_summary(clean_ldos, recombined_ldos, phi_eff=phi_eff, Te_eV=Te_eV)
    budget = escape_budget_summary(e_fp, phi_eff=phi_eff, n_hop=n_hop, E_hop=E_hop, expected_range=expected_E_FP)
    return {
        "tier_b_work_function": wf,
        "tier_c_E_FP": e_fp,
        "tier_c_emission": em,
        "tier_c_escape_budget": budget,
        "parameters": {"phi_eff_used_eV": phi_eff, "Te_eV": Te_eV, "n_hop": n_hop, "E_hop_eV": E_hop},
    }


def write_phase1_report(path: str | Path, status: dict) -> None:
    """Write a compact markdown report for the Tier B/C status package."""
    path = Path(path)
    wf = status["tier_b_work_function"]
    efp = status["tier_c_E_FP"]
    em = status["tier_c_emission"]
    budget = status["tier_c_escape_budget"]
    lines = [
        "# Phase 1 Tier B/C Completion Status",
        "",
        "## Tier B Work Function",
        f"- Status: `{wf['status']}`.",
    ]
    for name, payload in wf["deltas"].items():
        lines.append(f"- `{name}`: `{payload['delta_eV']}` eV (`{payload['status']}`).")
    lines.extend(
        [
            "",
            "## Tier C E_FP",
            f"- Status: `{efp['status']}`.",
            f"- E_FP: `{efp.get('E_FP_eV')}` eV.",
            "",
            "## Tier C Emission Contrast",
            f"- Status: `{em['status']}`.",
            f"- Contrast: `{em.get('contrast')}`.",
            "",
            "## Escape Budget",
            f"- Status: `{budget['status']}`.",
            f"- Budget: `{budget.get('budget', budget.get('budget_range'))}`.",
            "",
            "## Claim Boundary",
            "- Accepted rows can support quantitative text after fresh rerun verification.",
            "- Provisional rows are implementation/artifact evidence only, not final DFT claims.",
            "- Missing LDOS means no emission PASS claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
