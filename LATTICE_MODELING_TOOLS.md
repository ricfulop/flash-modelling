# Lattice Modeling Tools and Resources Handoff

Generated: 2026-06-05
Last updated: 2026-06-06 after QE GPU, MatterChat CUDA/container, ASE gallery, Atomsk, pseudopotential, Octopus, workflow/QC, CP2K, and Libint2 setup.

Purpose: give Claude a practical inventory of the local software, environments, scripts, and resources available for planning lattice/defect modeling work in this workspace. This is an environment and capability inventory; treat prior runs as setup/diagnostic evidence unless a later analysis explicitly validates them for scientific claims.

## Quick Planning Summary

Use the local tooling in layers:

1. Build and manipulate W/Mo BCC slabs and Frenkel-pair structures with ASE, pymatgen, and the project builders.
2. Use CHGNet or MACE for fast screening, relaxation, MD, and NEB prototypes, with strict caveats for out-of-distribution defect physics and field approximations.
3. Use Quantum ESPRESSO as the primary DFT route for W/Mo plane-wave SCF labels: CPU 7.4.1 for smoke tests and input validation, GPU 7.5 + NVHPC for production slab labels via `scripts/27_phase2_qe_gpu_labels.py`.
4. Use ABINIT 10.0.3 (`lattice-tools` conda env) as the fallback/cross-check DFT code when a second implementation is needed; PseudoDojo NC `.psp8` files under `.local-pseudopotentials` are the matching pseudopotential set.
5. Use BigDFT CPU only when finite-temperature/surface electronic-structure labels or density/potential cube artifacts specifically require the BigDFT workflow.
6. Use LAMMPS, phonopy, Atomsk, and OVITO when those tools are better suited than the existing project scripts.
7. Use the staged SSSP and PseudoDojo pseudopotential libraries for QE, ABINIT, SALMON, Octopus, and related validation workflows.
8. Use SALMON-TDDFT GPU for true real-time driven TDDFT prototypes: laser pulses, response dynamics, and Maxwell-TDDFT style workflows.
9. Use Octopus CPU/MPI as a real-space TDDFT cross-check when SALMON needs independent validation.
10. Use CP2K CPU/MPI for FIST and Quickstep CPU/MPI prototyping; the local build now includes CP2K-compatible Libint2 Fortran support for Gaussian integrals and hybrid/HFX workflows.
11. Use MatterChat from the NVIDIA PyTorch CUDA container for GPU inference; the host `.venv` is useful for import checks but remains CPU-only.
12. Use the ASE structure gallery for quick 3D inspection and close-contact triage before sending defect geometries to QE.
13. Use KPM/Wannier/downfolded-Hamiltonian scripts for transport diagnostics after a Hamiltonian artifact exists.

## Environment Status

| Area | Status | Practical note |
|---|---|---|
| System compilers | Ready | `gcc`, `gfortran`, `cmake`, `make` are on `PATH`. |
| MPI | Ready | `mpirun` and `mpiexec` are on `PATH` under `/usr/bin`. |
| CUDA toolkit | Installed | `nvcc` reports CUDA 13.0.88. |
| NVIDIA runtime | Ready outside restricted sandbox | With unrestricted device access, `nvidia-smi` sees `NVIDIA GB10`, driver `580.159.03`, CUDA `13.0`; repo PyTorch and CHGNet PyTorch both see one CUDA device. Restricted sandbox shell commands may still fail to access `/dev/nvidia*`. |
| Conda/Mamba | Installed locally | Miniforge is installed at `/home/ricfulop/Desktop/Cursor/.local-miniforge3`; `conda 26.3.2`, `mamba 2.5.0`. |
| QE 7.5 GPU | Built and script-integrated | NVHPC 25.11 / CUDA 13.0 build tree is under `/home/ricfulop/Desktop/Cursor/.local-qe/src/q-e-qe-7.5`; `scripts/27_phase2_qe_gpu_labels.py` sets the required runtime environment and writes GPU QE SCF jobs. |
| SALMON-TDDFT GPU | Installed locally | Built from SALMON-TDDFT/SALMON2 v2.2.2 with NVHPC OpenACC/CUDA runtime for `NVIDIA GB10`; source `/home/ricfulop/Desktop/Cursor/.local-salmon/src/SALMON2`, binary `/home/ricfulop/Desktop/Cursor/.local-salmon/install/nvhpc-openacc/bin/salmon`. |
| Atomsk | Installed locally | Built from source at `/home/ricfulop/Desktop/Cursor/.local-atomsk`; activate with `source /home/ricfulop/Desktop/Cursor/use_atomsk.sh`. |
| Pseudopotential libraries | Staged locally | SSSP 1.3.0 PBE efficiency/precision and PseudoDojo NC PBE scalar-relativistic libraries staged under `/home/ricfulop/Desktop/Cursor/.local-pseudopotentials`. |
| Octopus | Installed locally | CPU/MPI source build at `/home/ricfulop/Desktop/Cursor/.local-octopus/install`; activate with `source /home/ricfulop/Desktop/Cursor/use_octopus_cpu.sh`. |
| CP2K | Installed locally with Libint2 | CPU/MPI CMake build at `/home/ricfulop/Desktop/Cursor/.local-cp2k/install`; activate with `source /home/ricfulop/Desktop/Cursor/use_cp2k_cpu.sh`. `cp2k --version` reports `libint`, and a small H2/PBE0 Quickstep/HFX smoke test converged. |
| MatterChat CUDA | Installed/tested in NVIDIA PyTorch container | Zenodo code, dataset, checkpoint, and Mistral base model are staged under `/home/ricfulop/Desktop/Cursor/matterchat`; MatterChat imports and checkpoint loading were tested in `nvcr.io/nvidia/pytorch:25.11-py3` with CUDA on `NVIDIA GB10`. |
| Julia/R | Not found | `julia` and `R` are not on `PATH`. |

CUDA sandbox note: CUDA is working on the host and in unrestricted agent commands, but default sandboxed agent commands cannot access the NVIDIA device nodes. This modelling workstation disables Cursor sandboxing for future sessions via the user-level config `/home/ricfulop/.cursor/sandbox.json` and matching workspace configs such as `/home/ricfulop/Desktop/Cursor/.cursor/sandbox.json`:

```json
{
  "type": "insecure_none"
}
```

Already-running agent shells may still use their original sandbox until a new agent/backend session starts. Fresh Cursor agent sessions should start unsandboxed and see CUDA directly.

Practical CUDA options in Cursor:

- For agent-run GPU commands, run the command outside the sandbox / with all permissions. In this session, that is what made `nvidia-smi` and PyTorch CUDA probes work.
- For interactive work, use a normal IDE terminal, where `/dev/nvidia*` is visible.
- If you intentionally want to disable sandboxing for a workspace, Cursor supports sandbox configuration such as `.cursor/sandbox.json` with `{"type": "insecure_none"}`. Do not use that for untrusted commands; it is not GPU passthrough, it is disabling the sandbox.

## Python and MLIP Environments

### General Lattice Tools Environment

Activation:

```bash
source /home/ricfulop/Desktop/Cursor/.local-miniforge3/etc/profile.d/conda.sh
conda activate /home/ricfulop/Desktop/Cursor/.local-miniforge3/envs/lattice-tools
```

Reproducibility export:

```text
env/lattice_tools_environment.yml
```

Verified tools/packages:

| Tool/package | Version/status |
|---|---|
| `python` | 3.12 |
| `ase` | 3.28.0 |
| `pymatgen` | 2026.5.4 |
| `spglib` | 2.7.0 |
| `seekpath` | 2.2.1 |
| `phonopy` | 4.1.0 |
| `abinit` | 10.0.3 |
| `lmp` / LAMMPS | 11 Feb 2026, local Ubuntu ARM64 package extraction wrapped into this env |
| `mpi4py` | 4.1.2 |
| `numpy` | 2.4.6 |
| `scipy` | 1.17.1 |
| `matplotlib` | 3.10.9 |
| `pandas` | 3.0.3 |
| `networkx` | 3.6.1 |
| `h5py` | 3.16.0 |
| `signac` | 2.3.0 |
| `signac-flow` | 0.29.0 |
| `dvc` | 3.67.1 |

LAMMPS note: conda-forge does not currently provide `lammps` for `linux-aarch64`. LAMMPS was installed by downloading and locally extracting Ubuntu ARM64 packages into:

```text
/home/ricfulop/Desktop/Cursor/.local-lammps-deb/root
```

The wrapper at:

```text
/home/ricfulop/Desktop/Cursor/.local-miniforge3/envs/lattice-tools/bin/lmp
```

sets the required local `LD_LIBRARY_PATH` and then executes the extracted LAMMPS binary. It passed `lmp -help` and a tiny `run 0` smoke test outside the restricted sandbox.

### OVITO Environment

Activation:

```bash
source /home/ricfulop/Desktop/Cursor/.local-miniforge3/etc/profile.d/conda.sh
conda activate /home/ricfulop/Desktop/Cursor/.local-miniforge3/envs/ovito-tools
```

Reproducibility export:

```text
env/ovito_tools_environment.yml
```

Verified tool:

| Tool/package | Version/status |
|---|---|
| `ovito` CLI | 3.15.4 |

OVITO note: on `linux-aarch64`, `ovito` and `abinit` require incompatible `libnetcdf` builds in one conda environment. OVITO is therefore isolated in `ovito-tools`. The CLI is installed and `ovito --version` / `ovito --nogui --help` pass. The conda package does not expose an importable `ovito` Python module in this environment.

### ASE Structure Gallery

For quick visual inspection of generated structures, use the local ASE-to-HTML gallery:

```bash
cd /home/ricfulop/Desktop/Cursor/flash-modelling
/home/ricfulop/Desktop/Cursor/.local-bigdft/envs/bigdft/bin/python scripts/28_ase_structure_gallery.py --max-files 160
xdg-open runs/ase_structure_gallery_latest/index.html
```

The generated page is:

```text
runs/ase_structure_gallery_latest/index.html
```

It recursively scans `runs/` for ASE-readable structures (`.xyz`, `.extxyz`, `.cif`, `.traj`), including BigDFT `posinp.xyz` / `forces_posinp.xyz` files. The viewer lists newest structures first, renders them with 3Dmol.js, shows formula, atom count, periodic boundary flags, cell lengths, and minimum interatomic distance. Minimum-distance chips below about 1.8 Å are highlighted, which is useful for catching the short-contact geometries that destabilize direct QE SCF runs.

### MatterChat Environment and Weights

Local paths:

```text
/home/ricfulop/Desktop/Cursor/matterchat
/home/ricfulop/Desktop/Cursor/matterchat/src/MatterChat_code
/home/ricfulop/Desktop/Cursor/matterchat/.venv
/home/ricfulop/Desktop/Cursor/matterchat/submit_matterchat_query.py
/home/ricfulop/Desktop/Cursor/matterchat/src/MatterChat_code/matterchat_worker.py
```

Current status:

| Item | Status |
|---|---|
| Code archive | Downloaded: `/home/ricfulop/Desktop/Cursor/matterchat/MatterChat_code.zip` |
| Source tree | Extracted: `/home/ricfulop/Desktop/Cursor/matterchat/src/MatterChat_code/` |
| Host Python env | Installed enough for CPU/import checks at `/home/ricfulop/Desktop/Cursor/matterchat/.venv`, but its PyTorch is CPU-only |
| Dataset/checkpoint archive | Downloaded, checksum-verified, and extracted from Zenodo record `18735961` |
| Model checkpoint | Installed at `src/MatterChat_code/model_weight/model_weights.ckpt` |
| Mistral base model | Installed at `src/MatterChat_code/model_weight/Mistral-7B-Instruct-v0.3/` |
| Released configs | Patched to use `ckpt_path: ./model_weight/model_weights.ckpt` instead of the originally referenced `.pkl` filename |
| CUDA runtime | Use the NVIDIA PyTorch container, not the host `.venv`, for practical MatterChat inference on this GB10 workstation |

Installed runtime artifacts:

```text
/home/ricfulop/Desktop/Cursor/matterchat/Dataset_MatterChat.zip
/home/ricfulop/Desktop/Cursor/matterchat/Dataset_MatterChat/
/home/ricfulop/Desktop/Cursor/matterchat/src/MatterChat_code/model_weight/model_weights.ckpt
/home/ricfulop/Desktop/Cursor/matterchat/src/MatterChat_code/model_weight/Mistral-7B-Instruct-v0.3/
/home/ricfulop/Desktop/Cursor/matterchat/src/MatterChat_code/data_sample/Material_data_postprocess1_out_correct_train.pkl
/home/ricfulop/Desktop/Cursor/matterchat/src/MatterChat_code/data_sample/Material_data_postprocess1_out_correct_val.pkl
```

Verified importable packages in the MatterChat host `.venv`:

| Package | Version/status |
|---|---|
| `torch` | 2.3.1, currently CPU-only in this `.venv` |
| `transformers` | 4.42.4 |
| `datasets` | 2.20.0 |
| `pymatgen` | 2024.4.13 |
| `matminer` | 0.9.2 |
| `torch_geometric` | 2.5.3 |
| `faiss` / `faiss-cpu` | 1.9.0 |
| `mp-api` | 0.41.2 |
| `ase` | 3.28.0 |
| `spglib` | 2.4.0 |

MatterChat CUDA/container status:

| Check | Observed status |
|---|---|
| Container image | `nvcr.io/nvidia/pytorch:25.11-py3` |
| CUDA visibility | `torch.cuda.is_available() == True`, device `NVIDIA GB10` |
| Import smoke | `AutoConfig`, `LlamaTokenizer`, and `Blip2MistralInstruct` import/load checks pass inside the container |
| Checkpoint load | `MaterialLightningModule.load_from_checkpoint(...).to("cuda:0").eval()` succeeded |
| GPU memory | Moving the loaded MatterChat model to CUDA allocated about 14.9 GB |
| Compatibility fixes | `setuptools<81`, `transformers==4.46.3` in the container, and a vendored CHGNet `ExpCellFilter` import fallback for ASE 3.28 |

MatterChat interaction workflow:

```bash
# From a normal terminal, start a CUDA container with MatterChat mounted read/write.
# Add sudo if the user is not in the docker group.
docker run --rm -it --gpus all --name matterchat-cuda \
  -v /home/ricfulop/Desktop/Cursor/matterchat:/workspace/matterchat \
  -v /home/ricfulop/Desktop/Cursor/flash-modelling:/workspace/flash-modelling:ro \
  -w /workspace/matterchat/src/MatterChat_code \
  nvcr.io/nvidia/pytorch:25.11-py3 bash

# Inside the container, verify CUDA is visible.
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

# Inside the container, start the long-lived worker. This terminal stays occupied.
python matterchat_worker.py --config ./config/inference_MatterChat.yaml --device cuda:0
```

The worker loads the MatterChat checkpoint once, moves it to CUDA, writes `runtime/worker_status.json`, then watches a file queue. This is the preferred workflow for repeated prompts because each query avoids reloading the 7B model.

From the host, submit prompts through the mounted queue:

```bash
cd /home/ricfulop/Desktop/Cursor/matterchat

# Ask about one of the bundled MatterChat validation samples.
python submit_matterchat_query.py "Summarize this validation structure." --sample-index 0

# Use longer generation settings when needed.
python submit_matterchat_query.py \
  "Describe the likely structure-property relationships for this sample." \
  --sample-index 0 \
  --max-length 256 \
  --num-beams 3

# Ask about a specific CIF or structure file visible inside the container.
python submit_matterchat_query.py \
  "What is notable about this structure?" \
  --cif-path /workspace/flash-modelling/path/to/structure.cif \
  --max-length 256
```

The worker watches:

```text
/home/ricfulop/Desktop/Cursor/matterchat/src/MatterChat_code/runtime/requests
/home/ricfulop/Desktop/Cursor/matterchat/src/MatterChat_code/runtime/responses
/home/ricfulop/Desktop/Cursor/matterchat/src/MatterChat_code/runtime/processed
```

Request/response behavior:

- `submit_matterchat_query.py` writes a JSON request into `runtime/requests`, polls `runtime/responses`, prints the response JSON, and exits.
- Completed request files are moved into `runtime/processed` by the worker.
- Response JSON includes `state`, `prompt`, `output`, elapsed time, and CUDA memory when available.
- `--sample-index` selects from `Material_data_postprocess1_out_correct_val.pkl`.
- `--cif-path` must be a path visible from inside the container. With the container command above, use `/workspace/flash-modelling/...` for repo files and `/workspace/matterchat/...` for MatterChat files.
- The file-queue bridge lets the agent query a long-lived CUDA MatterChat process without direct Docker socket access.

Operational notes:

- If Docker socket access fails with permission denied, use `sudo docker ...` or add the user to the `docker` group and start a new group/session.
- If starting a fresh container image, verify the MatterChat dependency fixes from `SETUP_STATUS.md` are present before launching the worker, especially `setuptools<81`, `transformers==4.46.3`, and the ASE 3.28 `ExpCellFilter` fallback.
- For full inference, do not use the host `.venv`; it imports MatterChat dependencies but its PyTorch is CPU-only.

### Repo MACE/Torch Environment

Path:

```bash
/home/ricfulop/Desktop/Cursor/flash-modelling/.venv/bin/python
```

Activation helper:

```bash
cd /home/ricfulop/Desktop/Cursor/flash-modelling
source env/activate_dgx.sh
```

Installed/importable core packages:

| Package | Version/status |
|---|---|
| Python | 3.12 environment |
| `mace` / `mace-torch` | 0.3.16 |
| `torch` | 2.12.0+cu130 |
| `ase` | 3.28.0 |
| `numpy` | 2.4.6 |
| `scipy` | 1.17.1 |
| `networkx` | 3.6.1 |
| `pandas` | 3.0.3 |
| `PyYAML` | 6.0.3 |

Current CUDA status: outside the restricted sandbox, this environment reports `torch.cuda.is_available() == True`, one CUDA device, `NVIDIA GB10`.

Not installed in this environment: `chgnet`, `pymatgen`.

Use this environment for:

- MACE calculator and MACE-MP prototypes via `src/electrodefect/mlip_al.py`.
- ASE relaxation/NEB plumbing using MACE or the explicit toy fallback for pipeline smoke tests.
- KPM transport code paths that need torch; use unrestricted execution for GPU work, or `--device scipy` for scalar-only smoke paths.

### CHGNet Screening Environment

Activation helper:

```bash
cd /home/ricfulop/Desktop/Cursor/flash-modelling
source env/activate_flash_chgnet.sh
```

Environment path:

```bash
/home/ricfulop/Desktop/Cursor/.local-flash-chgnet/envs/flash_chgnet
```

Installed/importable core packages:

| Package | Version/status |
|---|---|
| Python | 3.10.20, per project setup note |
| `chgnet` | 0.4.2 |
| `ase` | 3.28.0 |
| `pymatgen` | 2025.10.7 |
| `torch` | 2.12.0 |
| `numpy` | 2.2.6 |
| `scipy` | 1.15.3 |
| `pandas` | 2.3.3 |
| `matplotlib` | 3.10.9 |
| `spglib` | 2.7.0 |
| `networkx` | 3.4.2 |

Use this environment for:

- Fast W/Mo structure relaxation and MD screening using CHGNet.
- ASE and pymatgen structure construction/conversion.
- The explicit fixed-charge field-wrapper NEB workflow.

Current CUDA status: outside the restricted sandbox, this environment reports `torch.cuda.is_available() == True`, one CUDA device, `NVIDIA GB10`.

Important scientific caveat: CHGNet is not a native electric-field or polarizable model. The project wrapper adds a fixed-charge linear field term and explicitly rejects deriving charges from CHGNet magnetic moments.

### BigDFT Environment

Activation helper:

```bash
source /home/ricfulop/Desktop/Cursor/use_bigdft.sh
```

Environment path:

```bash
/home/ricfulop/Desktop/Cursor/.local-bigdft/envs/bigdft
```

Installed/importable core packages and commands:

| Tool/package | Status |
|---|---|
| `bigdft` | Present in `.local-bigdft/envs/bigdft/bin` |
| `bigdft-tool` | Present in `.local-bigdft/envs/bigdft/bin` |
| `BigDFT` Python module | Importable |
| `futile` Python module | Importable |
| `ase` | 3.28.0 |
| `numpy` | 2.4.6 |
| `scipy` | 1.17.1 |
| `matplotlib` | 3.10.9 |
| `spglib` | 2.7.0 |
| `PyYAML` | 6.0.3 |

Use this environment for:

- BigDFT CPU validation labels.
- Finite electronic temperature inputs (`mix.tel`) and smearing controls (`mix.occopt`).
- Density/potential cube export with `dft.output_denspot=22`.
- Sparse matrix export with `lin_general.output_mat=1`.
- Wavefunction/field export utilities via `bigdft-tool`.

Known status:

- CPU BigDFT is the currently safer production path.
- GPU-linked BigDFT builds were compiled but failed validation on the current CUDA 13 / GB10 / ARM64 stack with NaNs or invalid controls. Do not use GPU BigDFT for scientific claims unless the smoke matrix is revalidated.

Reference doc:

```text
docs/bigdft_api_verification.md
docs/gpu_bigdft_fix_status_20260603_1745.md
```

## Quantum ESPRESSO

Local QE roots:

```text
/home/ricfulop/Desktop/Cursor/.local-qe
/home/ricfulop/Desktop/Cursor/.local-qe/build/qe-7.4.1-cpu/bin
/home/ricfulop/Desktop/Cursor/.local-qe/src/q-e-qe-7.5/bin/pw.x
/home/ricfulop/Desktop/Cursor/.local-qe/nvhpc/Linux_aarch64/25.11
```

CPU activation helper:

```bash
source /home/ricfulop/Desktop/Cursor/use_qe_cpu.sh
```

GPU note: `use_qe_cpu.sh` is only for the QE 7.4.1 CPU suite. The Phase-2 GPU label runner sets the NVHPC 25.11 / CUDA 13.0 runtime environment internally and looks for:

```text
/home/ricfulop/Desktop/Cursor/.local-qe/src/q-e-qe-7.5/bin/pw.x
```

Observed status:

| QE item | Status |
|---|---|
| QE 7.4.1 CPU suite | Built locally in `.local-qe/build/qe-7.4.1-cpu/bin`; `pw.x`, `ph.x`, `pp.x`, `dos.x`, `projwfc.x`, `bands.x`, `fs.x`, `matdyn.x`, `q2r.x`, `dynmat.x`, `epsilon.x`, `neb.x`, `cp.x`, `pw2wannier90.x`, `wannier90.x`, `postw90.x`, and `ld1.x` launch and report QE/Wannier versions. |
| QE 7.5 source/build tree | Present under `.local-qe/src/q-e-qe-7.5`. |
| QE 7.5 GPU `pw.x` | Present at `.local-qe/src/q-e-qe-7.5/bin/pw.x`; use through `scripts/27_phase2_qe_gpu_labels.py` or with the same NVHPC runtime variables. |
| QE GPU/NVHPC runtime tree | Present under `.local-qe/nvhpc`; project scripts know the expected NVHPC 25.11 CUDA 13.0 paths. |
| GPU validation | QE GPU plumbing was verified: runs reported GPU acceleration markers and `nvidia-smi` monitoring captured GPU activity. |
| Current scientific caveat | The attempted W/Mo defect slab label jobs did not produce trusted DFT labels because several input structures contain very short W-W or Mo-Mo contacts around 1.25 A, causing SCF instability/non-convergence even under more robust mixing and smearing settings. Repair/pre-relax geometries before scaling QE labels. |

Newly available QE tools:

```text
pw.x ph.x pp.x dos.x projwfc.x bands.x fs.x
matdyn.x q2r.x dynmat.x epsilon.x
neb.x cp.x ld1.x
pw2wannier90.x wannier90.x postw90.x
```

Project QE scripts:

```text
scripts/26_qe_smoke.py
scripts/27_phase2_qe_gpu_labels.py
```

Basic QE GPU label patterns:

```bash
cd /home/ricfulop/Desktop/Cursor/flash-modelling

# Prepare inputs without launching pw.x.
/home/ricfulop/Desktop/Cursor/.local-bigdft/envs/bigdft/bin/python scripts/27_phase2_qe_gpu_labels.py --prepare-only --max-jobs 1

# Run a short GPU smoke/validation job with CUDA launch lines in the QE output.
/home/ricfulop/Desktop/Cursor/.local-bigdft/envs/bigdft/bin/python scripts/27_phase2_qe_gpu_labels.py --max-jobs 1 --acc-notify --stop-on-failure

# More conservative SCF knobs that were tried for close-contact slabs.
/home/ricfulop/Desktop/Cursor/.local-bigdft/envs/bigdft/bin/python scripts/27_phase2_qe_gpu_labels.py \
  --max-jobs 1 \
  --kgrid 1 1 1 \
  --ecutwfc 30 \
  --ecutrho 240 \
  --diagonalization cg \
  --mixing-mode local-TF \
  --mixing-beta 0.05 \
  --mixing-ndim 12 \
  --smearing gaussian \
  --degauss 0.05 \
  --electron-maxstep 240 \
  --stop-on-failure
```

Each QE GPU run writes a timestamped directory under:

```text
runs/phase2_qe_gpu_labels_*/
```

Important output files include `status.json`, `heartbeat.log`, per-job `job.json`, generated QE input files, `qe_atoms.xyz`, `qe.out`, and `logs/*gpu*.log` monitor files.

Use QE for:

- Plane-wave SCF validation of W/Mo bulk/slab structures.
- Small W/Mo smoke tests before higher-volume label generation.
- Stress/force labels where the pseudopotentials and convergence settings are adequate.
- DFPT phonons and post-processing via `ph.x`, `q2r.x`, `matdyn.x`, `dynmat.x`, `dos.x`, `projwfc.x`, and `bands.x`.
- NEB pathways via `neb.x`.
- Wannier/downfolding bridge via `pw2wannier90.x`, `wannier90.x`, and `postw90.x`.

Local QE pseudopotentials staged in the repo:

```text
data/qe_pseudos/W.pbe-spn-kjpaw_psl.1.0.0.UPF
data/qe_pseudos/Mo.pbe-spn-kjpaw_psl.1.0.0.UPF
```

The QE smoke script defaults to this pseudopotential directory and supports W/Mo.

## SALMON-TDDFT GPU

Use this for true driven real-time TDDFT, including pulse-driven electron dynamics and SALMON Maxwell-TDDFT style workflows. This is the correct SALMON project:

```text
SALMON: Scalable Ab-initio Light-Matter simulator for Optics and Nanoscience
```

Do not install or use Ubuntu's `salmon` package for this purpose; that package is an unrelated RNA-seq transcript quantification tool.

Local paths:

```text
/home/ricfulop/Desktop/Cursor/.local-salmon/src/SALMON2
/home/ricfulop/Desktop/Cursor/.local-salmon/build/nvhpc-openacc
/home/ricfulop/Desktop/Cursor/.local-salmon/install/nvhpc-openacc/bin/salmon
```

Activation helper:

```bash
source /home/ricfulop/Desktop/Cursor/use_salmon_gpu.sh
```

Observed status:

| SALMON item | Status |
|---|---|
| Source version | SALMON-TDDFT/SALMON2 `v.2.2.2` |
| Compiler stack | NVHPC 25.11, CUDA 13.0, `NVIDIA GB10` |
| Build backend | `nvhpc-openacc` GPU backend, patched locally for current NVHPC flags and `cc121,cuda13.0` |
| CUDA linkage | `ldd salmon` resolves `libcublas.so.13`, `libcudart.so.13`, `libacchost.so`, `libaccdevice.so`, and related NVHPC CUDA/OpenACC libraries |
| Smoke test | Passed outside the sandbox using a tiny modified bundled C2H2 ground-state sample; `NVCOMPILER_ACC_NOTIFY=3` showed CUDA data transfers and kernel launches on `device=0` |

Basic run pattern:

```bash
source /home/ricfulop/Desktop/Cursor/use_salmon_gpu.sh
salmon < input.inp > stdout.log
```

For debug confirmation of GPU execution:

```bash
NVCOMPILER_ACC_NOTIFY=3 salmon < input.inp > stdout.log 2> stderr.log
```

Practical notes:

- SALMON requires a Fortran namelist input file and norm-conserving pseudopotentials (`.fhi`, `.vps`, `.psp8`, or NC `.upf` depending on the workflow).
- SALMON's own manual recommends CPU ground-state calculations and GPU TDDFT calculations for performance; validate that handoff per system before using results scientifically.
- The build was run with unrestricted device access. Fresh Cursor sessions should inherit the disabled sandbox config; if a shell cannot see `nvidia-smi`, run from a normal IDE terminal or an unrestricted agent command.

## Octopus CPU/MPI

Use this as an independent real-space TDDFT implementation for cross-checking SALMON or for finite-system/boundary-condition workflows where Octopus is the better fit.

Activation helper:

```bash
source /home/ricfulop/Desktop/Cursor/use_octopus_cpu.sh
```

Local paths:

```text
/home/ricfulop/Desktop/Cursor/.local-octopus/src/octopus
/home/ricfulop/Desktop/Cursor/.local-octopus/build/main
/home/ricfulop/Desktop/Cursor/.local-octopus/install/bin/octopus
/home/ricfulop/Desktop/Cursor/.local-miniforge3/envs/octopus-build
```

Observed status:

| Octopus item | Status |
|---|---|
| Source revision | `b74db58`, version string `octopus Chierchiae` |
| Build type | CPU/MPI source build with CMake/Ninja |
| Linked libraries | MPI, GSL, Libxc, FFTW, ADIOS2 |
| Smoke test | `octopus --version` passes after `source /home/ricfulop/Desktop/Cursor/use_octopus_cpu.sh` |
| Reproducibility export | `env/octopus_build_environment.yml` |

## CP2K CPU/MPI

Use this for CP2K/FIST, Quickstep CPU/MPI prototyping, and small Gaussian-integral/HFX checks. This is still a local CPU/MPI build rather than a fully optimized production CP2K stack, but Libint2 is enabled.

Activation helper:

```bash
source /home/ricfulop/Desktop/Cursor/use_cp2k_cpu.sh
```

Local paths:

```text
/home/ricfulop/Desktop/Cursor/.local-cp2k/src/cp2k
/home/ricfulop/Desktop/Cursor/.local-cp2k/build/cmake-libint2
/home/ricfulop/Desktop/Cursor/.local-cp2k/install/bin/cp2k.psmp
/home/ricfulop/Desktop/Cursor/.local-miniforge3/envs/cp2k-build
/home/ricfulop/Desktop/Cursor/.local-libint2-cp2k/install
```

Observed status:

| CP2K item | Status |
|---|---|
| Source revision | `dccd4e4`, version string `CP2K version 2026.1 (Development Version)` |
| Build type | CPU/MPI/OpenMP CMake build with CP2K-compatible Libint2 |
| Enabled flags | `omp`, `libint`, `fftw3`, `libxc`, `parallel`, `scalapack`, `spglib` |
| Libint2 | Built locally from CP2K's `libint-v2.13.1-cp2k-lmax-5` bundle with `LIBINT2_ENABLE_FORTRAN=ON` and `-fPIC`; installed at `/home/ricfulop/Desktop/Cursor/.local-libint2-cp2k/install` |
| Disabled/absent | LIBXS, LIBXSMM, GPU acceleration, HDF5, ELPA, PLUMED, SIRIUS, and other optional packages |
| Smoke tests | Tiny FIST two-Ar energy run passed; H2/PBE0 Quickstep/HFX run converged in 10 SCF steps with `ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -1.163870541557027` |
| Reproducibility export | `env/cp2k_build_environment.yml` |

Important CP2K caveat: this is now Libint2-enabled, but it is still a local CPU/MPI build without optional performance/features such as LIBXS, LIBXSMM, ELPA, HDF5, GPU acceleration, PLUMED, or SIRIUS. Use it for small Quickstep/HFX validation and prototyping; benchmark and harden the build before large production campaigns.

## Atomsk and Pseudopotential Libraries

Atomsk activation helper:

```bash
source /home/ricfulop/Desktop/Cursor/use_atomsk.sh
```

Atomsk status:

| Atomsk item | Status |
|---|---|
| Source | `/home/ricfulop/Desktop/Cursor/.local-atomsk/src/atomsk` |
| Binary | `/home/ricfulop/Desktop/Cursor/.local-atomsk/bin/atomsk` |
| Version | `master-2026-05-22 (Beta)` |
| Build/linkage | Source build with `/usr/bin/gfortran`, OpenMP, and system BLAS/LAPACK `.so.3` libraries |
| Smoke test | Created a tiny BCC W XSF file with `atomsk --create bcc 3.165 W` |

Pseudopotential library root:

```text
/home/ricfulop/Desktop/Cursor/.local-pseudopotentials
```

Staged resources:

| Library | Path/status |
|---|---|
| SSSP 1.3.0 PBE efficiency | `sssp/1.3.0/PBE/efficiency`, 103 UPF files plus JSON metadata |
| SSSP 1.3.0 PBE precision | `sssp/1.3.0/PBE/precision`, 103 UPF files plus JSON metadata |
| PseudoDojo NC PBE SR | `pseudo-dojo/ONCVPSP-PBE-SR`, 146 `.psp8` files; includes `Mo_std.psp8`, `Mo_str.psp8`, `W_std.psp8`, and `W_str.psp8` |
| Provenance/checksums | `provenance/README.md` and `provenance/CHECKSUMS.tsv` |

Notes:

- Use SSSP for QE-style plane-wave UPF workflows with the supplied cutoff metadata.
- Use PseudoDojo NC PBE scalar-relativistic `.psp8` files for ABINIT/SALMON-compatible norm-conserving workflows when applicable.
- Legacy ABINIT FHI archive URLs listed in SALMON documentation returned 404 during setup; prefer current PseudoDojo/SSSP resources unless a specific benchmark requires FHI format.

## Structure Builders and Defect Models

Project package:

```text
src/electrodefect/
```

Key modules:

| Module | Use |
|---|---|
| `src/electrodefect/build.py` | Builds orthogonal BCC W/Mo (001) slabs, vacuum geometry, clean/separated/recombined Frenkel-pair configurations, and bottom-contact constraints. |
| `src/electrodefect/percolation.py` | Builds ordered, random, and dendritic/DLA defect networks. |
| `src/electrodefect/chgnet_field.py` | CHGNet ASE calculator helpers and fixed-charge electric-field wrapper. |
| `src/electrodefect/mlip_al.py` | MACE/MLIP active-learning, calculator selection, geometry sanity checks, relaxation, MD/phonon/NEB helpers. |
| `src/electrodefect/dft_bigdft.py` | BigDFT driver and BigDFT log parsing. |
| `src/electrodefect/transport_kpm.py` | KPM DOS/Kubo/IPR/level-spacing/localization helpers and Wannier90 `*_hr.dat` loader. |

Useful scripts:

| Script | Use |
|---|---|
| `scripts/00_build_percolation.py` | Build Tier-A geometry/percolation summaries. |
| `scripts/02_md_phonons_neb.py` | Phase-2 MD, phonon, and NEB staging launcher. |
| `scripts/03_bigdft_emission.py` | BigDFT emission/DFT validation stage. |
| `scripts/04_transport.py` | KPM transport stage. |
| `scripts/18_phase2_bigdft_labels.py` | Phase-2 BigDFT labels. |
| `scripts/20_phase2_wannier_transport.py` | KPM diagnostics for Wannier90/downfolded Hamiltonian artifacts. |
| `scripts/22_bigdft_convergence_matrix.py` | BigDFT convergence matrix. |
| `scripts/23_chgnet_field_neb.py` | CHGNet fixed-charge field NEB runner. |
| `scripts/24_chgnet_neb_diagnostics.py` | Diagnostic tool for blocked NEB image continuity. |
| `scripts/25_phase2_w_bigdft_retry.py` | BigDFT retry workflow for W cases. |
| `scripts/26_qe_smoke.py` | Minimal QE W/Mo smoke runner. |
| `scripts/27_phase2_qe_gpu_labels.py` | QE GPU-label queue analogue for Phase 2. |
| `scripts/28_ase_structure_gallery.py` | Builds the interactive ASE/3Dmol.js HTML gallery for generated structures and close-contact triage. |

## Transport and Hamiltonian Resources

KPM transport code can operate on:

- Geometry-proxy graphs from ASE structures.
- Wannier90 `*_hr.dat` files.
- MatrixMarket Hamiltonian `.mtx` exports from BigDFT/downfolding workflows.

Relevant files:

```text
src/electrodefect/transport_kpm.py
scripts/20_phase2_wannier_transport.py
tests/fixtures/toy_hr.dat
```

Important caveat: geometry-proxy Hamiltonians are useful for plumbing and exploratory diagnostics, but are not substitutes for DFT/Wannier downfolded Hamiltonians when making electronic-transport claims.

## Existing Project Data and Results

Local results/resources already present:

```text
data/tier_a/
runs/sunday_synthesis_20260605_155229/SUNDAY_SYNTHESIS.md
runs/phase2_w_bigdft_retry_20260605_100808_retry_after_convergence_spark-5da5/
runs/phase2_qe_gpu_labels_20260605_133217_qe_gpu_robust_gamma_spark-5da5/
runs/ase_structure_gallery_latest/index.html
```

What they contain:

- Tier-A percolation, mechanism, and transport summaries for ordered/random/dendritic defect morphologies.
- BigDFT parsed observables and convergence warnings from prior runs.
- KPM summaries and exploratory Tier-D-lite proxy outputs.
- QE GPU label run status from setup/validation work, including GPU activity markers and non-convergence status for unrepaired short-contact geometries.
- The latest ASE structure gallery for quick visual inspection of recent `runs/` structures.

Do not treat prior exploratory outputs as confirmatory evidence without reviewing their preregistration, QC, and convergence warnings.

## General Crystal/Materials Resources

Local structure examples:

```text
/home/ricfulop/Desktop/Cursor/bigdft-suite/PyBigDFT/BigDFT/scripts/CIFs/W.cif
/home/ricfulop/Desktop/Cursor/bigdft-suite/PyBigDFT/BigDFT/scripts/CIFs/Mo.cif
/home/ricfulop/Desktop/Cursor/matterchat/src/MatterChat_code/data_sample/
```

Local QE/Wannier examples and pseudopotentials:

```text
/home/ricfulop/Desktop/Cursor/.local-qe/src/q-e-qe-7.4.1/external/wannier90/
/home/ricfulop/Desktop/Cursor/.local-qe/src/q-e-qe-7.4.1/external/wannier90/pseudo/
```

These are useful examples/resources, but not automatically validated for the current W/Mo flash-defect workflow.

## Tools Not Found or Not Ready

These were not available on `PATH` during this inventory:

- `vasp_std` / `vasp_gam`

Implication for Claude's plan:

- LAMMPS, phonopy, ABINIT, and OVITO are now available through the local Conda/Mamba environments above, not global system `PATH`.
- QE post-processing, phonon, NEB, CP, and Wannier90 tools are now available after `source /home/ricfulop/Desktop/Cursor/use_qe_cpu.sh`.
- QE 7.5 GPU `pw.x` is available through the NVHPC/CUDA runtime configured inside `scripts/27_phase2_qe_gpu_labels.py`; do not source `use_qe_cpu.sh` and assume that is the GPU environment.
- SALMON-TDDFT GPU is now available after `source /home/ricfulop/Desktop/Cursor/use_salmon_gpu.sh`; use it for true driven real-time TDDFT rather than QE linear-response TDDFT.
- Atomsk is now available after `source /home/ricfulop/Desktop/Cursor/use_atomsk.sh`.
- Octopus CPU/MPI is now available after `source /home/ricfulop/Desktop/Cursor/use_octopus_cpu.sh`.
- CP2K CPU/MPI is now available after `source /home/ricfulop/Desktop/Cursor/use_cp2k_cpu.sh`; this build includes CP2K-compatible Libint2 Fortran support and passed a small H2/PBE0 Quickstep/HFX smoke test.
- MatterChat's host `.venv` includes `matminer` and `mp-api` for CPU/import checks; full MatterChat inference should run inside the NVIDIA PyTorch CUDA container with the staged Zenodo weights.
- The ASE structure gallery at `runs/ase_structure_gallery_latest/index.html` is the quickest way to inspect generated geometries and find short-contact failures before launching more DFT.
- Do not assume VASP is available without an install/setup step and license.
- Prefer ASE/pymatgen for structure construction and conversion.
- Prefer MACE or CHGNet for local MLIP screening depending on whether the task needs MACE or pymatgen/CHGNet.
- Prefer QE 7.5 GPU for primary W/Mo slab SCF labels after geometry QC; use CPU QE 7.4.1 for smoke tests and input validation.
- Use ABINIT in `lattice-tools` for fallback/cross-check DFT on a subset of converged QE labels when a second code is needed.
- Reserve BigDFT CPU for finite-temperature labels or when BigDFT-specific density/potential artifacts are required.

## Recommended Prompt to Give Claude

```text
You are planning lattice/defect modeling in /home/ricfulop/Desktop/Cursor/flash-modelling.
Use LATTICE_MODELING_TOOLS.md as the environment inventory.

Please formulate a staged plan that:
1. Starts from the existing W/Mo BCC slab and Frenkel-pair builders.
2. Uses CHGNet or MACE only for screening/prototyping, with explicit out-of-distribution caveats.
3. Uses the ASE structure gallery and distance checks to reject or repair short-contact geometries before DFT.
4. Uses QE 7.5 GPU as the primary validation-label route and ABINIT as fallback/cross-check, with BigDFT CPU reserved for finite-temperature or BigDFT-specific artifacts.
5. Treats current unrepaired QE label outputs as non-converged setup/diagnostic runs, not trusted DFT labels.
6. Uses MatterChat only through the CUDA container/worker bridge for full inference; the host `.venv` is CPU-only.
7. Runs GPU-dependent steps outside restricted sandbox execution, because CUDA is available there but sandboxed commands may not see `/dev/nvidia*`.
8. Includes QC gates for geometry, convergence, pseudopotential consistency, and whether transport uses a geometry proxy or a true downfolded Hamiltonian.
9. Uses `lattice-tools` for ASE/pymatgen/LAMMPS/phonopy/ABINIT/signac/DVC and `ovito-tools` for OVITO.
10. Uses SALMON-TDDFT GPU for true driven TDDFT when laser/pulse dynamics are needed.
11. Uses Octopus CPU/MPI only when an independent real-space TDDFT check is needed.
12. Treats CP2K as available for CPU/MPI MM/FIST and small Quickstep/HFX prototyping with Libint2 enabled, while still noting the missing optional production/performance packages.
13. Clearly marks missing tools that would require installation, source builds, or licenses, especially VASP.
```
