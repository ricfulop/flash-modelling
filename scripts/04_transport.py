#!/usr/bin/env python
"""Stage 04: Tier A KPM/transport diagnostics and frozen regime adjudication."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from electrodefect import percolation as perc
from electrodefect import regime_decision as rd
from electrodefect import tier_a
from electrodefect import transport_kpm as tk


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "w_percolation.yaml"


def _torch_device():
    if not tk._HAS_TORCH:
        return None
    if tk.torch.cuda.is_available():
        return "cuda"
    return None


def _case_disorder(cfg, morphology):
    if morphology == "ordered":
        return float(cfg["tight_binding"]["disorder_W_ordered"])
    return float(cfg["tight_binding"]["disorder_W_random"])


def _run_kpm_dos(net, cfg, out_dir, disorder_W, device):
    if device is None:
        return {"ran": False, "reason": "CUDA torch device not available"}
    try:
        kpm = cfg["kpm"]
        moments = int(os.environ.get("ELECTRODEFECT_KPM_MOMENTS", kpm["moments"]))
        random_vectors = int(os.environ.get("ELECTRODEFECT_KPM_RANDOM_VECTORS", kpm["random_vectors"]))
        energies_count = int(os.environ.get("ELECTRODEFECT_KPM_ENERGIES", kpm["energies"]))
        H, _, _ = tk.build_tb(
            net,
            t_hop=cfg["tight_binding"]["t_hop"],
            disorder_W=disorder_W,
            seed=cfg["network"].get("seed", 0),
            device=device,
        )
        energies, dos = tk.kpm_dos(
            H,
            M=moments,
            R=random_vectors,
            n_energies=energies_count,
            device=device,
            seed=cfg["network"].get("seed", 0),
        )
        np.savez(out_dir / "kpm_dos.npz", energies=energies, dos=dos)
        return {
            "ran": True,
            "device": device,
            "moments": moments,
            "random_vectors": random_vectors,
            "energies": energies_count,
            "dos_file": "kpm_dos.npz",
        }
    except Exception as exc:
        return {"ran": False, "device": device, "reason": f"{type(exc).__name__}: {exc}"}


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    device = _torch_device()
    max_exact = int(os.environ.get("ELECTRODEFECT_MAX_EXACT", "700"))
    spectral_t_max = int(os.environ.get("ELECTRODEFECT_DS_TMAX", "1000"))
    spectral_walkers = int(os.environ.get("ELECTRODEFECT_DS_WALKERS", "1000"))
    summaries = []

    for morphology, frac in tier_a.iter_cases(cfg):
        out_dir = tier_a.case_dir(morphology, frac)
        if not (out_dir / "network.npz").exists():
            net = tier_a.build_network(cfg, morphology, frac)
            tier_a.save_network(
                net,
                out_dir,
                {
                    "material": cfg["material"],
                    "morphology": morphology,
                    "target_frac": frac,
                    "seed": cfg["network"].get("seed", 0),
                    "supercell": cfg["network"]["supercell"],
                },
                spectral_t_max=spectral_t_max,
                spectral_walkers=spectral_walkers,
            )
        net = tier_a.load_network(out_dir)
        geometry = tier_a.read_json(out_dir / "geometry.json")
        disorder_W = _case_disorder(cfg, morphology)
        elec = tk.exact_diagnostics(
            net,
            t_hop=float(cfg["tight_binding"]["t_hop"]),
            disorder_W=disorder_W,
            E_F=0.0,
            Te_eV=float(cfg["drive_proxy"]["Te_eV"]),
            seed=cfg["network"].get("seed", 0),
            max_exact_sites=max_exact,
        )
        verdict = rd.adjudicate(geometry, elec)
        dos_info = _run_kpm_dos(net, cfg, out_dir, disorder_W, device)
        payload = {
            "material": cfg["material"],
            "morphology": morphology,
            "target_frac": frac,
            "geometry": geometry,
            "electronic": elec,
            "regime_decision": tier_a.to_builtin(verdict),
            "kpm_dos": dos_info,
        }
        tier_a.write_json(out_dir / "transport.json", payload)
        summaries.append(
            {
                "morphology": morphology,
                "target_frac": frac,
                "case_dir": str(out_dir.relative_to(REPO_ROOT)),
                "regime": verdict.regime,
                "confidence": verdict.confidence,
                "d_s": geometry.get("d_s"),
                "r_stat": elec["r_stat"],
                "pr_frac_EF": elec["pr_frac_EF"],
                "xi_over_a": elec["xi_over_a"],
                "kF_l": elec["kF_l"],
                "W_over_B": elec["W_over_B"],
                "bandwidth_eV": elec["bandwidth_eV"],
                "kpm_ran": dos_info["ran"],
            }
        )
        print(
            f"{morphology:10s} frac={frac:.2f} -> {verdict.regime:28s} "
            f"({verdict.confidence}); KPM={dos_info['ran']}"
        )

    tier_a.write_json(tier_a.TIER_A_DIR / "transport_summary.json", {"cases": summaries})
    print(f"\nsaved Tier A transport summary -> {(tier_a.TIER_A_DIR / 'transport_summary.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
