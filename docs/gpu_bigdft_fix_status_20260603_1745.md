# GPU BigDFT Fix Status - 2026-06-03 17:45 EDT

## Production jobs

- Local CPU BigDFT follow-up is still running on `spark-5da5`.
- Peer CPU BigDFT follow-up is still running on `spark-0808`.
- Both active `bigdft -l yes` processes are CPU-bound and isolated from the GPU-fix experiments.

## Four GPU BigDFT fix paths

1. **Newer upstream BigDFT/SYCL/CUDA-loader**
   - Checked `origin/devel`; local source is already current at `aee86bd`.
   - No newer upstream commit available to test from GitLab at this time.

2. **OpenCL-enabled rebuild**
   - OpenCL runtime/ICD exists: `/etc/OpenCL/vendors/nvidia.icd` points to `libnvidia-opencl.so.1`.
   - Development headers/unversioned library were missing from the OS, so the isolated build installs `opencl-headers` and `ocl-icd` into `.local-bigdft-opencl`.
   - Build completed using `try_bigdft_gpu_fix_build.sh opencl` after patching the missing `stdlib.h` declarations in `liborbs`.
   - The binary links against `libOpenCL.so.1`, `libcudart.so.13`, `libcublas.so.13`, and `libcufft.so.12`.
   - Result: **failed validation**. CPU/no-acceleration controls passed, but `perf.blas: true` and/or `perf.accel: CUDAGPU` still produced NaNs/`rc=14` for N2 and NaN W atom energy.
   - Evidence: `runs/gpu_bigdft_fix_opencl_20260603_175916/smoke_matrix.log`.
   - An explicit `perf.accel: OCLGPU` N2 probe also failed before SCF: BigDFT reported `No OpenCL platform available!` and segfaulted before writing finite energy/charge.
   - Evidence: `runs/cuda_bigdft_debug_ocl_probe/n2_oclgpu/stdout.txt`.

3. **NGC/official BigDFT CUDA container**
   - Docker client is installed, but the user cannot access `/var/run/docker.sock`.
   - This path is blocked unless Docker permissions or an Apptainer/Singularity runtime become available.

4. **Explicit Blackwell/PTX CUDA rebuild**
   - CUDA 13 supports `compute_121` and `sm_121`.
   - Build completed with `try_bigdft_gpu_fix_build.sh gencode`.
   - The build used explicit `-gencode=arch=compute_121,code=sm_121` and PTX `-gencode=arch=compute_121,code=compute_121`.
   - Result: **failed validation**. No-`perf` controls passed, but `perf.blas: true` and/or `perf.accel: CUDAGPU` still produced NaNs/`rc=14` for N2 and NaN W atom energy.
   - Evidence: `runs/gpu_bigdft_fix_gencode_20260603_174721/smoke_matrix.log`.

## Validation gate

No GPU BigDFT build is production-safe unless it passes the same smoke matrix:

- N2 finite energy near CPU reference and charge near 10.
- W atom finite energy near CPU reference and charge near 6.
- No NaN, zero-electron collapse, or unphysical ion-only energy, even if BigDFT exits with code 0.

## Production decision

As of this validation pass, CPU BigDFT remains the production path for Sunday deliverables. The GPU-linked BigDFT builds compile and link, but the accelerated BigDFT paths do not pass scientific correctness checks on the current CUDA 13 / GB10 / ARM64 stack.
