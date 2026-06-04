"""Smoke tests for the self-contained (no torch/ASE/BigDFT) modules."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from electrodefect import percolation as perc, emission, mechanism_table as mt


def test_dendrite_is_fractal_and_spanning():
    net = perc.dla_dendrite(nx_=8, ny=8, nz=8, target_frac=0.25, seed=1)
    rep = perc.classify(net)
    assert rep["spanning"] is True
    assert 1.0 < rep["D_f"] < 3.0
    assert 0.8 < rep["d_s"] < 2.6        # fracton / quantum-percolation band


def test_emission_contrast_detects_recombination():
    z = np.linspace(0, 30, 200); zs, EF, phi, Te = 18.0, 0.0, 4.5, 0.72
    E = np.sort(np.linspace(-6, 14, 60))
    def state(c, leak):
        r = np.exp(-0.5*((z-c)/5)**2); r[z>zs]*=leak; return r/np.trapezoid(r, z)
    clean = emission.emittable_flux(E, np.array([state(9,0.02) for _ in E]), z, zs, EF, phi, Te)
    rec   = emission.emittable_flux(
        E, np.array([state(9 if (e-EF)<phi else 16, 0.02 if (e-EF)<phi else 0.6) for e in E]),
        z, zs, EF, phi, Te)
    assert emission.emission_contrast(clean, rec)["ratio"] > 5


def test_arrhenius_recovered():
    T = np.linspace(600, 1592, 25)
    sigma = 5e5*np.exp(-0.040/(mt.KB_EV*T))
    df = mt.fit_temperature_laws(T, sigma)
    # Arrhenius Ea recovered near 40 meV (best or within near-tie)
    arr = df[df.mechanism.str.contains("Arrhenius")].iloc[0]
    Ea = float(arr["param"].split("=")[1].split()[0])
    assert 35 < Ea < 45


def test_escape_budget_positive_for_W():
    b = emission.escape_budget(E_FP=11.0, n_hop=2, E_hop=0.4, phi_eff=4.5)
    assert b["escapes"] and b["E_escape"] > 0


if __name__ == "__main__":
    test_dendrite_is_fractal_and_spanning()
    test_emission_contrast_detects_recombination()
    test_arrhenius_recovered()
    test_escape_budget_positive_for_W()
    print("all smoke tests passed")


def test_regime_decision_separates_three_regimes():
    from electrodefect import regime_decision as rd
    anderson = rd.adjudicate(dict(d_s=2.1, spanning=True),
        dict(r_stat=0.39, pr_frac_EF=0.005, xi_over_a=2.0, kF_l=0.9, W_over_B=18.0))
    miniband = rd.adjudicate(dict(d_s=2.8, spanning=True),
        dict(r_stat=0.55, pr_frac_EF=0.30, xi_over_a=20.0, kF_l=1.4, W_over_B=1.0))
    fracton = rd.adjudicate(dict(d_s=1.45, spanning=True),
        dict(r_stat=0.40, pr_frac_EF=0.02, xi_over_a=3.0, kF_l=1.0, W_over_B=6.0))
    assert anderson.regime == "ANDERSON_LOCALIZED"
    assert miniband.regime == "EXTENDED_MINIBAND"
    assert fracton.regime == "QUANTUM_PERCOLATION_FRACTON"
