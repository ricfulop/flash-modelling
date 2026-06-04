import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from electrodefect import tier_a
from electrodefect import transport_kpm as tk


def _cfg():
    return {
        "material": "W",
        "lattice_a": 3.165,
        "network": {"supercell": [4, 4, 4], "seed": 1, "morphology": "dendrite", "target_frac": 0.25},
        "experiment": {
            "morphologies": ["ordered", "random", "dendritic"],
            "frac_sweep": [0.25, 0.30, 0.35],
        },
        "tight_binding": {"t_hop": 0.08, "disorder_W_ordered": 0.0, "disorder_W_random": 3.0},
        "drive_proxy": {"Te_eV": 0.72},
    }


def test_tier_a_enumerates_three_by_three_grid():
    cases = list(tier_a.iter_cases(_cfg()))
    assert len(cases) == 9
    assert cases[0] == ("ordered", 0.25)
    assert cases[-1] == ("dendritic", 0.35)


def test_case_dir_uses_stable_fraction_slug(tmp_path):
    path = tier_a.case_dir("random", 0.3, root=tmp_path)
    assert path == tmp_path / "random" / "frac_0p30"


def test_ordered_network_builds_connected_defect_graph():
    net = tier_a.build_network(_cfg(), "ordered", 0.25)
    assert net.frac > 0
    assert net.graph.number_of_nodes() == net.defect_mask.sum()
    assert net.graph.number_of_edges() > 0


def test_save_load_roundtrip_and_exact_diagnostics(tmp_path):
    cfg = _cfg()
    net = tier_a.build_network(cfg, "ordered", 0.25)
    out = tmp_path / "case"
    geom = tier_a.save_network(net, out, {"morphology": "ordered", "target_frac": 0.25})
    loaded = tier_a.load_network(out)
    assert loaded.defect_mask.sum() == net.defect_mask.sum()
    assert "d_s" in geom
    diag = tk.exact_diagnostics(loaded, max_exact_sites=40)
    assert diag["n_sites_exact"] <= 40
    assert diag["bandwidth_eV"] >= 0
    assert "r_stat" in diag
