#!/usr/bin/env python
"""Stage 00: build the Tier A morphology/fraction network grid."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from electrodefect import tier_a


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "w_percolation.yaml"


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    spectral_t_max = int(os.environ.get("ELECTRODEFECT_DS_TMAX", "1000"))
    spectral_walkers = int(os.environ.get("ELECTRODEFECT_DS_WALKERS", "1000"))
    rows = []
    for morphology, frac in tier_a.iter_cases(cfg):
        out_dir = tier_a.case_dir(morphology, frac)
        net = tier_a.build_network(cfg, morphology, frac)
        metadata = {
            "material": cfg["material"],
            "morphology": morphology,
            "target_frac": frac,
            "seed": cfg["network"].get("seed", 0),
            "supercell": cfg["network"]["supercell"],
        }
        geometry = tier_a.save_network(
            net,
            out_dir,
            metadata,
            spectral_t_max=spectral_t_max,
            spectral_walkers=spectral_walkers,
        )
        rows.append((morphology, frac, geometry, out_dir))
        print(
            f"{morphology:10s} frac={frac:.2f} "
            f"n={geometry['n_defects']:4d} d_s={geometry['d_s']:.3f} "
            f"spanning={geometry['spanning']} -> {out_dir.relative_to(REPO_ROOT)}"
        )

    summary = [
        {
            "morphology": morphology,
            "target_frac": frac,
            "case_dir": str(out_dir.relative_to(REPO_ROOT)),
            **geometry,
        }
        for morphology, frac, geometry, out_dir in rows
    ]
    tier_a.write_json(tier_a.TIER_A_DIR / "geometry_summary.json", {"cases": summary})
    print(f"\nsaved Tier A geometry grid -> {tier_a.TIER_A_DIR.relative_to(REPO_ROOT)}")
    print("d_s ~ 4/3 => Alexander-Orbach/fracton; dense ordered networks test miniband behavior.")


if __name__ == "__main__":
    main()
