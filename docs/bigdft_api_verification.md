# BigDFT API Verification

Date: 2026-06-02

Scope: introspection only. No production DFT jobs were launched.

## Environment

- Activation script: `/home/ricfulop/Desktop/Cursor/use_bigdft.sh`
- Python: `3.11.15` inside the BigDFT environment
- Executables found: `bigdft`, `bigdft-tool`, `mpirun`
- PyBigDFT modules import: `BigDFT`, `BigDFT.Calculators`, `BigDFT.Inputfiles`, `BigDFT.Logfiles`, `BigDFT.IO`

## Verified

- Launch CLI: `bigdft -n <name>` is supported and writes `log-<name>.yaml` when logfile output is enabled.
- `BigDFT.Logfiles.Logfile(path)` works on the local N2 smoke log.
- Smoke-log attributes exist:
  - `log.energy` in Hartree
  - `log.evals` as band arrays in Hartree
  - `log.fermi_level` in Hartree
- Finite electronic temperature is an input variable under `mix.tel` in Hartree.
- Smearing method is controlled by `mix.occopt`.
- Density/potential export is controlled by `dft.output_denspot`; value `22` writes density plus local/external/Hartree potentials in cube format.
- Linear-scaling matrix export is controlled by `lin_general.output_mat`; value `1` writes formatted sparse matrices.
- `bigdft-tool -a export-wf FILE --i-band ...` is available for wavefunction cube export.
- `bigdft-tool -a convert-field FROM TO` is available for field conversion.

## Still To Verify Before Tier B Production

- Surface/slab boundary encoding for W(001): local examples show BigDFT/PSolver uses geocode `S` for surface boundary conditions, but the current ASE `xyz` write path does not prove that the slab cell/geocode is encoded correctly.
- Exact output filenames and formats for `lin_general.output_mat=1` on the target linear-scaling W cells.
- Whether `bigdft-tool -a export-wf` or an LDOS/window workflow is better for near-vacuum state density on large O(N) runs.

## Driver Changes Made

- `src/electrodefect/dft_bigdft.py` now uses `mix.tel = Te_eV / Ha` and `mix.occopt = 1`.
- `dft.output_denspot=22` is used for surface runs so potential cubes are requested.
- `lin_general.output_mat=1` is used for sparse support-function matrix export.
- Logfile parsing now flattens `log.evals` blocks and converts all energies from Hartree to eV.
