# Prior-work note: near-surface W Frenkel-pair field NEB

**Question doc:** `docs/science-superpowers/questions/2026-06-04-near-surface-w-frenkel-field-neb.md`

## Method constraints to carry forward

1. **W point-defect barriers span very different scales.** Prior DFT and interatomic-potential work reports monovacancy migration in bcc W near `1.6-1.8 eV`, while `<111>` self-interstitial migration can be meV-scale and rotation/reorientation barriers can be much higher. A near-surface Frenkel-pair recombination path can therefore be nearly barrierless if it is SIA-dominated, so endpoint construction and geometry sanity checks are as important as the scalar barrier.[^ahlgren2010][^dft2016][^defect2016]

2. **NEB is the standard pathway method, but small image counts are exploratory.** Published DFT migration studies commonly use NEB with fixed initial/final configurations and several intermediate images. For the first screen, five intermediate images is acceptable as a pilot, but any manuscript-grade barrier should be rerun with a DFT method and tighter convergence.[^renium2016]

3. **Universal MLIPs can soften transition states.** CHGNet, MACE, and related universal machine-learned interatomic potentials are trained largely on near-equilibrium relaxation data. Recent benchmarking reports systematic underprediction of high-energy transition-state, surface, defect, and migration-barrier energetics. CHGNet NEB barriers are therefore screening outputs, not final mechanism-selection scalars.[^chgnet2023][^softening2024]

4. **CHGNet magnetic moments are not electric charges.** CHGNet is charge-informed through magnetic-moment regularization, but the authors explicitly note ambiguity in using local magnetic moments as charges, especially for nonmagnetic ions and charge partitioning. The field-wrapper model must use explicit documented charges and must not infer charge from `magmom`.[^chgnet2023]

5. **Real electric-field response requires a polarizable or long-range model.** Fixed-charge forces `F = qE` are at best qualitative perturbations. Field-aware MLIP work uses polarization, Born effective charges, long-range electrostatics, or latent Ewald summation to infer electrical response and run external-field MD. This investigation can only decide whether a qualitative field-perturbed path is worth DFT validation.[^les2025]

## Adopted method

- Use CHGNet/ASE only for an exploratory no-field versus field-wrapper NEB screen on one fixed near-surface W pathway.
- Treat the field wrapper as a sensitivity perturbation, not a physical electric-field model.
- Require finite barriers, identical endpoints, stable images, and a practical negative `barrier_delta_eV` before promoting images to BigDFT.
- Label all CHGNet field-wrapper barriers exploratory unless BigDFT validation is completed under a separate gate.

## Prior effect size for design

There is no defensible prior effect size for the fixed-charge field wrapper in metallic W. The practical threshold will be a smallest-effect-of-interest rule, not a literature-powered effect size. Because SIA barriers can be meV-scale and universal MLIP errors can be much larger than meV, a field-associated change smaller than `0.05 eV` should not be treated as promotion-worthy in this first screen.

[^ahlgren2010]: Ahlgren et al. (2010). "Bond-order potential for point and extended defect simulations in tungsten." *Journal of Applied Physics*. https://www.mv.helsinki.fi/home/tahlgren/Ahlgren_BOP_W_JAP_107_2010_033516.pdf

[^dft2016]: "First-principles study on mono-vacancy self diffusion and recovery in tungsten crystal." *Fusion Engineering and Design*. https://www.sciencedirect.com/science/article/abs/pii/S0920379616301260

[^defect2016]: "First-principles study of vacancy, interstitial, noble gas atom interstitial and vacancy clusters in bcc-W." *Computational Materials Science*. https://www.sciencedirect.com/science/article/abs/pii/S0927025616303019

[^renium2016]: Xu et al. (2016). "Suppression of radiation-induced point defects by rhenium and osmium interstitials in tungsten." *Scientific Reports*. https://www.nature.com/articles/srep36738

[^chgnet2023]: Deng et al. (2023). "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling." *Nature Machine Intelligence*. https://www.nature.com/articles/s42256-023-00716-3

[^softening2024]: Deng et al. (2024). "Systematic softening in universal machine learning interatomic potentials." *npj Computational Materials*. https://www.nature.com/articles/s41524-024-01500-6

[^les2025]: Zhong et al. (2025). "Machine learning interatomic potential can infer electrical response." *npj Computational Materials*. https://www.nature.com/articles/s41524-025-01911-z
