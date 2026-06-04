# CUDA BigDFT Fix Research - 2026-06-03

## Local failure summary

The CUDA-linked BigDFT executable launches and gives correct results when BigDFT GPU/CUBLAS paths are disabled. It fails scientifically when `perf.blas: true` or `perf.accel: CUDAGPU` is enabled:

- N2 collapses to `Total electronic charge = 0.0` and ion-only/unphysical positive energy.
- W atom produces huge unphysical energy and NaN density spread.
- `OCLGPU` stops with `STOP FAKE init_acceleration_OCL`, meaning the current binary was not built with a real OpenCL backend.

Therefore the immediate issue is not activation or dynamic linking. It is BigDFT's GPU/CUBLAS/OpenCL backend correctness on this stack.

## Current hardware/software stack

- GPU: NVIDIA GB10 / DGX Spark, compute capability 12.1 (`sm_121`)
- Architecture: ARM64 / aarch64
- CUDA toolkit: 13.0
- Driver: 580.x
- OpenCL headers/runtime: not available locally (`libOpenCL.so`, `CL/cl.h` missing)

## What the web research says

1. BigDFT supports heterogeneous execution with CUDA/OpenCL, but GPU acceleration must be configured at build time and requested in the input/runtime setup. Official/source docs describe `Installer.py`/rcfile configuration and flags such as `--enable-cuda-gpu`, `--enable-opencl`, `--with-cuda-path`, and `--with-ocl-path`.

2. Historical BigDFT GPU acceleration was developed for CUDA/OpenCL kernels and CUBLAS offload, mainly wavelet/convolution and BLAS-heavy routines. This matches our failure site: the GPU/CUBLAS path corrupts the input-guess/subspace diagonalization before SCF is meaningful.

3. The documented CUDA BigDFT container path uses NVIDIA's old NGC image `nvcr.io/hpc/bigdft:cuda10-ubuntu1804-ompi4-mkl`. On GB10/Blackwell this is risky because Blackwell requires compatible PTX or a rebuild with modern architecture support; old cubin-only CUDA 10 images may not run correctly or at all.

4. NVIDIA documents Blackwell compatibility as requiring either native Blackwell cubins or PTX for forward compatibility. GB10 is compute capability 12.1. CUDA 13 / ARM64 / GB10 is a new stack, and many CUDA libraries/packages still need explicit `sm_121`/Blackwell support.

5. OpenCL is a plausible BigDFT path, but it requires a real OpenCL development/runtime stack. Our current machine lacks the needed headers and `libOpenCL.so`, so `OCLGPU` cannot be fixed by an input option alone.

## Fix paths, ranked

### Path 1 - Try a newer upstream BigDFT/SYCL/OpenCL branch in a separate environment

Rationale: upstream commits mention added SYCL/CUDA loader support and oneMath upgrades. This is the most plausible source-level fix for a modern CUDA 13 / Blackwell / ARM64 stack.

Gate before production:

- N2 finite energy and charge matching CPU.
- W atom finite energy and charge matching CPU.
- Small W slab finite charge/energy, no zero-electron collapse.

### Path 2 - Install OpenCL development/runtime support, then rebuild with `--enable-opencl`

Rationale: BigDFT historically supports OpenCL, and `OCLGPU` is an accepted input keyword. Current binary stops with fake OCL initialization only because OpenCL was not built in.

Risks:

- Need NVIDIA OpenCL ICD/runtime support on DGX Spark.
- OpenCL may still not be performant or compatible on GB10.

### Path 3 - Test the NGC BigDFT CUDA container only as a smoke comparison

Rationale: official docs cite `nvcr.io/hpc/bigdft:cuda10-ubuntu1804-ompi4-mkl`.

Risks:

- Very old CUDA 10 base.
- Blackwell/GB10 may require PTX compatibility or rebuild. This is likely a diagnostic comparison, not a production answer.

### Path 4 - Rebuild current CUDA BigDFT with different architecture flags

Rationale: current build used `sm_121`. NVIDIA Blackwell compatibility guidance emphasizes native cubins or PTX. A fat/forward-compatible build might require explicit `-gencode` PTX, for example Blackwell-compatible compute targets rather than only a single `-arch`.

Risks:

- If the BigDFT CUDA code itself is not CUDA-13/GB10 safe, flags will not fix correctness.

## Production decision

Do not use GPU BigDFT for Sunday production yet. Keep CPU BigDFT running. Treat all CUDA/OpenCL/SYCL BigDFT work as isolated experimental validation until the gates pass.

## Key sources

- BigDFT source install docs: https://l_sim.gitlab.io/bigdft-suite/users/source-install.html
- BigDFT CUDA container example: https://documentation.sigma2.no/code_development/guides/containers/bigdft.html
- BigDFT GPU acceleration/HPC paper: https://prace-ri.eu/wp-content/uploads/Improvements_of_BigDFT_code_in_modern_HPC_architectures.pdf
- NVIDIA Blackwell compatibility guide: https://docs.nvidia.com/cuda/archive/13.1.2/blackwell-compatibility-guide/index.html
- NVIDIA compute capability table: https://developer.nvidia.com/cuda/gpus
