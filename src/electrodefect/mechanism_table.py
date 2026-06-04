"""
mechanism_table.py — Goal-2 / Goal-3 transport-mechanism elimination.

Ports the filament/breakdown community's "fit every candidate, keep what survives"
protocol (cf. Gismatulin et al., Sci. Rep. 2021) to a driven metal, and adds the
coherent-miniband and quantum-percolation options that an insulator analysis omits.

Two kinds of input:
  (1) sigma(T) at several electronic temperatures (from transport_kpm at various Te,
      or from measured transport-Arrhenius)  -> discriminates the *temperature law*
  (2) I-V data                               -> discriminates field-driven channels

Temperature-law candidates:
  - Arrhenius / nearest-neighbour hop:  sigma ~ exp(-Ea / kT)            [your W 44, Mo 33, Pt 59 meV band]
  - Mott 3D VRH:                        sigma ~ exp(-(T0/T)^(1/4))
  - Efros-Shklovskii VRH:               sigma ~ exp(-(T0/T)^(1/2))       [needs soft Coulomb gap]
  - metallic / miniband:                sigma ~ a - b*T  (or weak power), dsigma/dT < 0 weak

I-V candidates (per-electrode-area current density vs field E):
  - Poole-Frenkel:        ln(J/E) ~ + beta_PF * sqrt(E)/kT
  - Schottky:             ln(J)   ~ + beta_S  * sqrt(E)/kT   (beta_S = beta_PF/2)
  - Space-charge-limited: J ~ E^2 (Child) / higher slope on log-log
  - ohmic:                J ~ E

Energy-scale selection criteria (computed once per structure, from DFT + transport):
  - Ioffe-Regel:    kF*l <~ 1            -> Boltzmann/band invalid
  - Anderson (3D):  W_disorder / B >~ 16.5  -> exponentially localized
  - Holstein:       E_p / (2t)             -> small polaron if > 1
  - Mott vs NN:     compare optimal hop energy at T to xi^-1 site spacing

Output: a per-structure elimination table (pandas DataFrame) ranking the surviving
mechanism by fit quality (R^2 / AIC), plus a verdict.

Pure numpy/scipy/pandas. Tested standalone.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

KB_EV = 8.617333e-5  # eV/K


def _aic(y, yhat, k):
    n = len(y)
    rss = np.sum((y - yhat) ** 2)
    rss = max(rss, 1e-300)
    return n * np.log(rss / n) + 2 * k


def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Temperature-law fits.  Work in ln(sigma).
# --------------------------------------------------------------------------- #
def fit_temperature_laws(T: np.ndarray, sigma: np.ndarray) -> pd.DataFrame:
    """Fit Arrhenius / Mott / ES / metallic to sigma(T). Returns ranked table."""
    T = np.asarray(T, float)
    sigma = np.asarray(sigma, float)
    y = np.log(sigma)
    rows = []

    # Arrhenius: ln sigma = ln s0 - Ea/(kB T)
    def arr(T, ls0, Ea):
        return ls0 - Ea / (KB_EV * T)
    try:
        p, _ = curve_fit(arr, T, y, p0=[y.max(), 0.04], maxfev=20000)
        yhat = arr(T, *p)
        rows.append(dict(mechanism="Arrhenius / NN-hop", param=f"Ea={p[1]*1e3:.1f} meV",
                         R2=_r2(y, yhat), AIC=_aic(y, yhat, 2)))
    except Exception as e:
        rows.append(dict(mechanism="Arrhenius / NN-hop", param=f"fail:{e}", R2=np.nan, AIC=np.inf))

    # Mott 3D VRH: ln sigma = ln s0 - (T0/T)^(1/4)
    def mott(T, ls0, T0):
        return ls0 - (T0 / T) ** 0.25
    try:
        p, _ = curve_fit(mott, T, y, p0=[y.max(), 1e6], bounds=([-np.inf, 1.0], [np.inf, np.inf]), maxfev=20000)
        yhat = mott(T, *p)
        rows.append(dict(mechanism="Mott 3D VRH", param=f"T0={p[1]:.2e} K",
                         R2=_r2(y, yhat), AIC=_aic(y, yhat, 2)))
    except Exception as e:
        rows.append(dict(mechanism="Mott 3D VRH", param=f"fail:{e}", R2=np.nan, AIC=np.inf))

    # Efros-Shklovskii: ln sigma = ln s0 - (T0/T)^(1/2)
    def es(T, ls0, T0):
        return ls0 - (T0 / T) ** 0.5
    try:
        p, _ = curve_fit(es, T, y, p0=[y.max(), 1e4], bounds=([-np.inf, 1.0], [np.inf, np.inf]), maxfev=20000)
        yhat = es(T, *p)
        rows.append(dict(mechanism="Efros-Shklovskii VRH", param=f"T0={p[1]:.2e} K",
                         R2=_r2(y, yhat), AIC=_aic(y, yhat, 2)))
    except Exception as e:
        rows.append(dict(mechanism="Efros-Shklovskii VRH", param=f"fail:{e}", R2=np.nan, AIC=np.inf))

    # Metallic/miniband: sigma = a - b T  (fit in linear sigma, report on ln for AIC parity)
    def metal(T, a, b):
        return np.log(np.clip(a - b * T, 1e-12, None))
    try:
        p, _ = curve_fit(metal, T, y, p0=[sigma.max() * 1.1, np.ptp(sigma) / np.ptp(T)], maxfev=20000)
        yhat = metal(T, *p)
        rows.append(dict(mechanism="metallic / miniband", param=f"dsig/dT={-p[1]:.2e}",
                         R2=_r2(y, yhat), AIC=_aic(y, yhat, 2)))
    except Exception as e:
        rows.append(dict(mechanism="metallic / miniband", param=f"fail:{e}", R2=np.nan, AIC=np.inf))

    df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    df["dAIC_vs_best"] = df["AIC"] - df["AIC"].min()
    return df


# --------------------------------------------------------------------------- #
# Field-law fits (I-V).
# --------------------------------------------------------------------------- #
def fit_field_laws(E: np.ndarray, J: np.ndarray, T_K: float = 1533.0) -> pd.DataFrame:
    """Fit Poole-Frenkel / Schottky / SCLC / ohmic to J(E). Returns ranked table."""
    E = np.asarray(E, float)
    J = np.asarray(J, float)
    kT = KB_EV * T_K
    rows = []

    # Poole-Frenkel: ln(J/E) = c + beta_PF sqrt(E)/kT
    x = np.sqrt(E)
    yPF = np.log(J / E)
    bPF = np.polyfit(x, yPF, 1)
    rows.append(dict(mechanism="Poole-Frenkel", param=f"slope={bPF[0]:.3e}",
                     R2=_r2(yPF, np.polyval(bPF, x)), AIC=_aic(yPF, np.polyval(bPF, x), 2)))

    # Schottky: ln J = c + beta_S sqrt(E)/kT
    yS = np.log(J)
    bS = np.polyfit(x, yS, 1)
    rows.append(dict(mechanism="Schottky", param=f"slope={bS[0]:.3e}",
                     R2=_r2(yS, np.polyval(bS, x)), AIC=_aic(yS, np.polyval(bS, x), 2)))

    # SCLC / power law: log J = c + m log E   (m~1 ohmic, m~2 Child, m>2 trap-filled)
    lx, ly = np.log(E), np.log(J)
    bSC = np.polyfit(lx, ly, 1)
    rows.append(dict(mechanism=f"power-law (SCLC, m={bSC[0]:.2f})", param=f"m={bSC[0]:.2f}",
                     R2=_r2(ly, np.polyval(bSC, lx)), AIC=_aic(ly, np.polyval(bSC, lx), 2)))

    df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    df["dAIC_vs_best"] = df["AIC"] - df["AIC"].min()
    return df


# --------------------------------------------------------------------------- #
# Energy-scale selection criteria (computed from DFT + transport scalars).
# --------------------------------------------------------------------------- #
def selection_criteria(kF_l: float, W_disorder: float, B_bandwidth: float,
                       E_polaron: float, t_hop: float,
                       N_EF: float = None, xi: float = None) -> pd.DataFrame:
    """
    Apply the closed-form regime criteria. All energies in eV.
      kF_l        : kF * mean-free-path (Ioffe-Regel ~1)
      W_disorder  : on-site energy spread (disorder strength)
      B_bandwidth : defect-miniband full bandwidth
      E_polaron   : polaron binding energy E_p
      t_hop       : inter-site transfer integral t
      N_EF, xi    : optional, DOS at E_F (states/eV) and localization length (Angstrom)
    """
    rows = []
    rows.append(dict(criterion="Ioffe-Regel  (kF*l <~ 1 => band invalid)",
                     value=f"kF*l = {kF_l:.2f}",
                     verdict="band transport INVALID" if kF_l <= 1.5 else "band may survive"))
    WB = W_disorder / B_bandwidth if B_bandwidth else np.inf
    rows.append(dict(criterion="Anderson 3D  (W/B >~ 16.5 => localized)",
                     value=f"W/B = {WB:.1f}",
                     verdict="Anderson-localized" if WB >= 16.5 else "extended (not Anderson)"))
    ratio = E_polaron / (2 * t_hop) if t_hop else np.inf
    rows.append(dict(criterion="Holstein  (E_p/2t > 1 => small polaron)",
                     value=f"E_p/2t = {ratio:.2f}",
                     verdict="small-polaron hopping" if ratio > 1 else "large/itinerant carrier"))
    if N_EF is not None and xi is not None:
        # Mott T0 = 18 / (kB N(EF) xi^3); compare optimal hop energy at, say, 1533 K
        T0 = 18.0 / (N_EF * (xi * 1e-10) ** 3) / 11604.5  # crude, xi in m, -> K via kB
        rows.append(dict(criterion="Mott T0 (lower => VRH easier)",
                         value=f"T0 ~ {T0:.2e} K", verdict="info"))
    return pd.DataFrame(rows)


def verdict(temp_df: pd.DataFrame, crit_df: pd.DataFrame) -> str:
    """Plain-language synthesis."""
    best = temp_df.iloc[0]
    lines = [f"Dominant T-law: {best['mechanism']}  ({best['param']}, "
             f"R^2={best['R2']:.3f}, dAIC to next = {temp_df.iloc[1]['dAIC_vs_best']:.1f})"]
    for _, r in crit_df.iterrows():
        lines.append(f"  - {r['criterion']}: {r['verdict']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # synthetic self-test: generate Arrhenius-like sigma(T) with 40 meV barrier + noise
    rng = np.random.default_rng(0)
    T = np.linspace(600, 1592, 25)
    Ea_true = 0.040
    sigma = 5e5 * np.exp(-Ea_true / (KB_EV * T)) * np.exp(rng.normal(0, 0.03, T.size))
    tdf = fit_temperature_laws(T, sigma)
    print("Temperature-law elimination:")
    print(tdf.to_string(index=False))

    # synthetic Poole-Frenkel I-V
    E = np.linspace(1e6, 1e8, 30)
    J = 1e-3 * E * np.exp(2e-4 * np.sqrt(E) / (KB_EV * 1533))
    fdf = fit_field_laws(E, J)
    print("\nField-law elimination:")
    print(fdf.to_string(index=False))

    cdf = selection_criteria(kF_l=1.1, W_disorder=3.0, B_bandwidth=0.5,
                             E_polaron=0.30, t_hop=0.08, N_EF=2.0, xi=5.0)
    print("\nSelection criteria:")
    print(cdf.to_string(index=False))
    print("\nVERDICT:\n" + verdict(tdf, cdf))
