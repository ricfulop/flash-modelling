"""
emission.py — Static hot-electron-emission analysis from Frenkel-pair recombination.

This is the GO/NO-GO test. It does NOT simulate real-time emission dynamics
(that is RT-TDDFT/Ehrenfest, out of scope on Sparks and stated as such in Methods).
It demonstrates the channel statically, in three parts:

  (A) The enthalpy exists:
        E_FP = E(separated vacancy + SIA)  -  E(recombined)        [Delta-SCF totals]
        cross-check E_FP ~ E_v + E_SIA  (main-text Eq. 6); W expects 8-13 eV.

  (B) Accessible vacuum-coupled states exist at that energy:
        integrate the Kohn-Sham LDOS that has amplitude OUTSIDE the surface, above
        the barrier phi_eff, weighted by Fermi-Dirac occupation at the DRIVEN
        electronic temperature Te (~0.7 eV from Saha-Boltzmann). The claim is proven
        if this "emittable flux" is much larger in the recombined config than for the
        clean surface at the same Te.

  (C) The budget closes:
        E_escape = E_FP - n_hop * E_hop - phi_eff  > 0   (main-text Eq. 6)

Inputs (B) come from the BigDFT driver: KS eigenvalues, KS wavefunctions sampled on
a real-space z-grid, the surface z-location, E_F, and phi_eff (work function of the
*defective* surface). See dft_bigdft.export_ks_for_emission().

Pure numpy. Self-test uses synthetic KS states.
"""
from __future__ import annotations
import numpy as np

KB_EV = 8.617333e-5  # eV/K


def fermi_dirac(eps, E_F, Te_eV):
    """Occupation at electronic temperature Te (eV). Te ~ 0.72 eV = 8358 K (your Saha)."""
    x = np.clip((eps - E_F) / Te_eV, -500, 500)
    return 1.0 / (1.0 + np.exp(x))


def emittable_flux(ks_energies, ks_density_z, z_grid, z_surface,
                   E_F, phi_eff, Te_eV, energy_weight=True):
    """
    Energy-weighted population in vacuum-coupled states above the barrier.

    ks_energies   : (M,)   KS eigenvalues (eV)
    ks_density_z  : (M, Z) |psi_i(z)|^2 integrated over x,y for each state (1/Angstrom)
    z_grid        : (Z,)   z coordinates (Angstrom)
    z_surface     : float  z above which is vacuum
    E_F, phi_eff  : floats (eV) — Fermi level and effective surface barrier
    Te_eV         : float  driven electronic temperature (eV)

    Returns dict with total emittable flux and its energy spectrum.
    """
    ks_energies = np.asarray(ks_energies, float)
    ks_density_z = np.asarray(ks_density_z, float)
    occ = fermi_dirac(ks_energies, E_F, Te_eV)
    vac = z_grid > z_surface
    vac_weight = np.trapezoid(ks_density_z[:, vac], z_grid[vac], axis=1)  # amplitude in vacuum
    above = (ks_energies - E_F) >= phi_eff
    w = (ks_energies - E_F) if energy_weight else np.ones_like(ks_energies)
    contrib = occ * vac_weight * np.where(above, w, 0.0)
    return dict(
        total=float(contrib.sum()),
        spectrum=contrib,
        energies=ks_energies,
        vac_weight=vac_weight,
        occ=occ,
        n_above_barrier=int(above.sum()),
    )


def emission_contrast(clean, recombined):
    """
    Ratio of emittable flux (recombined / clean). The proof: >> 1 means recombination,
    not temperature, is filling vacuum-coupled states above phi_eff.
    """
    r = recombined["total"] / max(clean["total"], 1e-300)
    return dict(ratio=float(r),
                clean=clean["total"],
                recombined=recombined["total"],
                verdict=("EMISSION CHANNEL CONFIRMED" if r > 5
                         else "weak / inconclusive — increase defect density or Te"))


def escape_budget(E_FP, n_hop, E_hop, phi_eff):
    """E_escape = E_FP - n_hop*E_hop - phi_eff  (main-text Eq. 6). All eV."""
    Eesc = E_FP - n_hop * E_hop - phi_eff
    return dict(E_FP=E_FP, loss=n_hop * E_hop, phi_eff=phi_eff,
                E_escape=Eesc, escapes=(Eesc > 0))


def E_FP_from_totals(E_separated, E_recombined):
    """Frenkel-pair recombination enthalpy from Delta-SCF total energies (eV)."""
    return float(E_separated - E_recombined)


if __name__ == "__main__":
    # ---- synthetic self-test -------------------------------------------------
    # Build two fake "surfaces": a clean one (states bound, little vacuum amplitude
    # above E_F+phi) and a recombined one (a high-lying state with vacuum amplitude).
    rng = np.random.default_rng(0)
    Z = 200
    z = np.linspace(0, 30, Z)        # Angstrom; surface at z=18, vacuum above
    z_surf, E_F, phi = 18.0, 0.0, 4.5
    Te = 0.72                        # eV, your Saha-Boltzmann electronic temperature

    def gaussian_state(center, width, vac_amp):
        rho = np.exp(-0.5 * ((z - center) / width) ** 2)
        rho[z > z_surf] *= vac_amp   # how much leaks into vacuum
        return rho / np.trapezoid(rho, z)

    M = 60
    energies = np.sort(rng.uniform(-6, 14, M))
    # clean: all states peaked inside the metal, negligible vacuum leakage
    clean_rho = np.array([gaussian_state(9, 5, 0.02) for _ in energies])
    # recombined: high-energy states (eps-E_F > phi) get real vacuum amplitude
    rec_rho = []
    for e in energies:
        leak = 0.6 if (e - E_F) >= phi else 0.02
        rec_rho.append(gaussian_state(9 if (e - E_F) < phi else 16, 5, leak))
    rec_rho = np.array(rec_rho)

    clean = emittable_flux(energies, clean_rho, z, z_surf, E_F, phi, Te)
    rec = emittable_flux(energies, rec_rho, z, z_surf, E_F, phi, Te)
    print("clean emittable     :", f"{clean['total']:.4e}",
          f"({clean['n_above_barrier']} states above barrier)")
    print("recombined emittable:", f"{rec['total']:.4e}",
          f"({rec['n_above_barrier']} states above barrier)")
    print("contrast            :", emission_contrast(clean, rec))

    print("\nescape budget (W example, E_FP=11 eV, 2 hops x 0.4 eV, phi=4.5):")
    print("  ", escape_budget(E_FP=11.0, n_hop=2, E_hop=0.4, phi_eff=4.5))
