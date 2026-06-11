#!/usr/bin/env bash
set -euo pipefail

# Clean, containerized BigDFT GPU rebuild and validation.
#
# Host usage:
#   DOCKER="sudo docker" bash scripts/29_bigdft_container_gpu_rebuild.sh
#
# The script intentionally installs into a new tree:
#   /home/ricfulop/Desktop/Cursor/.local-bigdft-container-gpu
#
# It does not modify the known-working CPU BigDFT environment or the older
# failed GPU build attempts.

HOST_BASE="${BIGDFT_BASE:-/home/ricfulop/Desktop/Cursor}"
FLASH_REPO="$HOST_BASE/flash-modelling"
IMAGE="${BIGDFT_GPU_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}"
CONTAINER_NAME="${BIGDFT_GPU_CONTAINER_NAME:-bigdft-gpu-clean}"
BUILD_JOBS="${BIGDFT_BUILD_JOBS:-4}"
CLEAN_REBUILD="${BIGDFT_CLEAN_REBUILD:-1}"
CUDA_ARCH_FLAGS="${BIGDFT_CUDA_ARCH_FLAGS:--gencode=arch=compute_121,code=sm_121 -gencode=arch=compute_121,code=compute_121}"

run_host() {
  mkdir -p "$HOST_BASE" "$FLASH_REPO/runs"
  local -a docker_cmd
  read -r -a docker_cmd <<<"${DOCKER:-docker}"

  echo "Starting clean BigDFT GPU rebuild in container image: $IMAGE"
  echo "If Docker permission is denied, rerun with:"
  echo "  DOCKER=\"sudo docker\" bash scripts/29_bigdft_container_gpu_rebuild.sh"
  echo

  "${docker_cmd[@]}" run --rm -i --gpus all \
    --name "$CONTAINER_NAME" \
    -v "$HOST_BASE:$HOST_BASE" \
    -w "$FLASH_REPO" \
    -e BIGDFT_BASE="$HOST_BASE" \
    -e BIGDFT_BUILD_JOBS="$BUILD_JOBS" \
    -e BIGDFT_CLEAN_REBUILD="$CLEAN_REBUILD" \
    -e BIGDFT_CUDA_ARCH_FLAGS="$CUDA_ARCH_FLAGS" \
    "$IMAGE" \
    bash "$FLASH_REPO/scripts/29_bigdft_container_gpu_rebuild.sh" --inside
}

run_inside() {
  local base="$HOST_BASE"
  local root="$base/.local-bigdft-container-gpu"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_dir="$base/flash-modelling/runs/bigdft_container_gpu_rebuild_${stamp}"
  local env_prefix="$root/envs/bigdft-container-gpu"
  local micromamba="$root/bin/micromamba"
  local source_dir="$root/src/bigdft-suite"
  local build_dir="$root/build"
  local rcfile="$root/bigdft-container-gpu.rc"
  local use_script="$base/use_bigdft_container_gpu.sh"
  local cuda_root="${CUDA_ROOT:-/usr/local/cuda}"
  local cuda_lib_dir="${CUDA_LIB_DIR:-$cuda_root/lib64}"

  mkdir -p "$base/flash-modelling/runs"
  if [[ "$CLEAN_REBUILD" == "1" && -e "$root/.build_attempt_started" ]]; then
    mv "$root" "${root}.bak.${stamp}"
  fi
  mkdir -p "$root/bin" "$root/pkgs" "$root/cache" "$root/src" "$log_dir"
  touch "$root/.build_attempt_started"

  {
    echo "BigDFT container GPU rebuild started: $(date --iso-8601=seconds)"
    echo "base=$base"
    echo "root=$root"
    echo "source_dir=$source_dir"
    echo "build_dir=$build_dir"
    echo "env_prefix=$env_prefix"
    echo "build_jobs=$BUILD_JOBS"
    echo "cuda_root=$cuda_root"
    echo "cuda_lib_dir=$cuda_lib_dir"
    echo "cuda_arch_flags=$CUDA_ARCH_FLAGS"
    uname -a
  } | tee "$log_dir/00_context.log"

  if command -v apt-get >/dev/null 2>&1; then
    apt-get update 2>&1 | tee "$log_dir/01_apt_update.log"
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      ca-certificates curl bzip2 git make patch pkg-config file \
      2>&1 | tee "$log_dir/02_apt_install.log"
  fi

  nvcc --version 2>&1 | tee "$log_dir/03_nvcc.log"
  nvidia-smi 2>&1 | tee "$log_dir/04_nvidia_smi.log"

  local mamba_platform
  case "$(uname -m)" in
    aarch64|arm64) mamba_platform="linux-aarch64" ;;
    x86_64|amd64) mamba_platform="linux-64" ;;
    *) echo "Unsupported architecture: $(uname -m)" >&2; exit 2 ;;
  esac

  export MAMBA_ROOT_PREFIX="$root"
  export MAMBA_PKGS_DIRS="$root/pkgs"
  export XDG_CACHE_HOME="$root/cache"

  if [[ ! -x "$micromamba" ]]; then
    curl -L "https://micro.mamba.pm/api/micromamba/${mamba_platform}/latest" \
      -o "$root/micromamba.tar.bz2" \
      2>&1 | tee "$log_dir/05_micromamba_download.log"
    tar -xjf "$root/micromamba.tar.bz2" -C "$root"
  fi

  "$micromamba" create -y -p "$env_prefix" -c conda-forge \
    python=3.11 pip setuptools wheel \
    c-compiler cxx-compiler fortran-compiler \
    openmpi mpi4py \
    libblas liblapack fftw gsl \
    yaml pyyaml spglib ntpoly \
    cmake make autoconf automake libtool pkg-config git curl cython \
    numpy scipy \
    2>&1 | tee "$log_dir/06_micromamba_create.log"

  "$micromamba" install -y -p "$env_prefix" -c conda-forge 'libxc=5.2.3' \
    2>&1 | tee "$log_dir/07_libxc_install.log"

  rm -rf "$source_dir" "$build_dir"
  if ! git clone --depth 1 https://gitlab.com/l_sim/bigdft-suite.git "$source_dir" \
      2>&1 | tee "$log_dir/08_git_clone.log"; then
    echo "Fresh clone failed; falling back to local source copy from $base/bigdft-suite" | tee -a "$log_dir/08_git_clone.log"
    cp -a "$base/bigdft-suite" "$source_dir"
  fi

  cat > "$rcfile" <<RCFILE
modules = ['bigdft']
skip = ["spglib", "PyYAML", "libyaml", "ntpoly"]

from os import environ
prefix = environ["PREFIX"]
cuda = environ.get("CUDA_ROOT", "$cuda_root")
cuda_lib = environ.get("CUDA_LIB_DIR", "$cuda_lib_dir")

def env_configuration():
    env = {}
    env["FC"] = "mpifort"
    env["CC"] = environ["CC_FOR_BUILD"]
    env["CXX"] = environ["CXX_FOR_BUILD"]
    env["CFLAGS"] = "-O2 -DCONDA " + environ["CFLAGS"] + " -ldl"
    env["CPPFLAGS"] = environ["CPPFLAGS"]
    env["FCFLAGS"] = "-O2 -fopenmp -fallow-argument-mismatch " + environ["FORTRANFLAGS"] + " -ldl"
    env["--with-ext-linalg"] = "-llapack -lblas"
    env["--with-yaml-path"] = prefix
    env["--with-libxc-libs"] = prefix + "/lib/libxcf90.a " + prefix + "/lib/libxc.a"
    env["--with-libxc-incs"] = "-I" + prefix + "/include"
    env["ac_cv_lib_xcf90_xc_f90_lda_vxc"] = "yes"
    env["--enable-cuda-gpu"] = ""
    env["--with-cuda-path"] = cuda
    env["LDFLAGS"] = "-L" + cuda_lib + " -Wl,-rpath," + cuda_lib
    env["LIBRARY_PATH"] = cuda_lib
    env["LD_LIBRARY_PATH"] = cuda_lib + ":" + environ.get("LD_LIBRARY_PATH", "")
    env["NVCC_FLAGS"] = "$CUDA_ARCH_FLAGS --compiler-options -fPIC -O3"
    return " ".join(['"' + x + (('=' + y) if y else '') + '"' for x, y in env.items()])

autogenargs = env_configuration()
RCFILE

  mkdir -p "$build_dir"
  export MAKEFLAGS="-j${BUILD_JOBS}"
  nice -n 15 "$micromamba" run -p "$env_prefix" bash -lc "
    set -euo pipefail
    export PREFIX=\"\$CONDA_PREFIX\"
    export CUDA_ROOT=\"$cuda_root\"
    export CUDA_LIB_DIR=\"$cuda_lib_dir\"
    export CUDA_HOME=\"$cuda_root\"
    export PATH=\"$cuda_root/bin:\$PATH\"
    export LD_LIBRARY_PATH=\"$cuda_lib_dir:\${LD_LIBRARY_PATH:-}\"
    export LIBRARY_PATH=\"$cuda_lib_dir:\${LIBRARY_PATH:-}\"
    export BIGDFT_PATH=\"$build_dir/install\"
    export JHBUILD_RUN_AS_ROOT=1
    export OMP_NUM_THREADS=1
    cd \"$build_dir\"
    python \"$source_dir/Installer.py\" -y autogen
    python \"$source_dir/Installer.py\" -y -f \"$rcfile\" build
  " 2>&1 | tee "$log_dir/09_build.log"

  if [[ -d "$build_dir/bigdft" ]]; then
    nice -n 15 "$micromamba" run -p "$env_prefix" bash -lc "cd \"$build_dir/bigdft\" && make install" \
      2>&1 | tee "$log_dir/10_install.log"
  fi

  "$micromamba" run -p "$env_prefix" python -m pip install PyBigDFT PyFutile \
    2>&1 | tee "$log_dir/11_python_packages.log"

  cat > "$use_script" <<USESCRIPT
#!/usr/bin/env bash
export MAMBA_ROOT_PREFIX="$root"
export MAMBA_PKGS_DIRS="$root/pkgs"
export XDG_CACHE_HOME="$root/cache"
export PATH="$cuda_root/bin:$root/bin:$env_prefix/bin:\$PATH"
export LD_LIBRARY_PATH="$cuda_lib_dir:\${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$cuda_lib_dir:\${LIBRARY_PATH:-}"
export CUDA_HOME="$cuda_root"
export CONDA_PREFIX="$env_prefix"
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-1}"
export BIGDFT_ROOT="$env_prefix/bin"
export BIGDFT_SOURCES="$source_dir/bigdft"
export GI_TYPELIB_PATH="$env_prefix/lib/girepository-1.0:\${GI_TYPELIB_PATH:-}"
echo "Container BigDFT GPU environment ready."
USESCRIPT
  chmod +x "$use_script"

  bash -lc "source \"$use_script\" >/tmp/use-bigdft-container-gpu.out && which bigdft && ldd \$(command -v bigdft)" \
    2>&1 | tee "$log_dir/12_liveness.log"

  BIGDFT_CUDA_ACTIVATE="$use_script" BIGDFT_SOURCE="$source_dir" \
    "$micromamba" run -p "$env_prefix" python "$base/flash-modelling/scripts/13_cuda_bigdft_debug_matrix.py" \
    2>&1 | tee "$log_dir/13_smoke_matrix.log"

  "$micromamba" run -p "$env_prefix" python - <<'PY' "$base/flash-modelling/runs" "$log_dir"
import json
import math
import sys
from pathlib import Path

runs = Path(sys.argv[1])
log_dir = Path(sys.argv[2])
latest = max(runs.glob("cuda_bigdft_debug_*"), key=lambda p: p.stat().st_mtime)
results_path = latest / "results.json"
data = json.loads(results_path.read_text())
cases = {row["case"]: row for row in data["cases"]}

def finite_case(name):
    row = cases.get(name)
    if not row:
        return False, f"missing {name}"
    energy = row.get("energy_Ha")
    charge = row.get("charge_e")
    if row.get("returncode") != 0:
        return False, f"{name} returncode={row.get('returncode')}"
    if row.get("has_nan"):
        return False, f"{name} contains NaN"
    if not isinstance(energy, (int, float)) or not math.isfinite(energy):
        return False, f"{name} non-finite energy={energy}"
    if not isinstance(charge, (int, float)) or not math.isfinite(charge):
        return False, f"{name} non-finite charge={charge}"
    return True, "ok"

checks = {
    "n2_no_perf_control": finite_case("n2_no_perf_control"),
    "n2_cuda_no_blas": finite_case("n2_cuda_no_blas"),
    "n2_cuda_blas": finite_case("n2_cuda_blas"),
    "w_cuda_blas": finite_case("w_cuda_blas"),
}

passed = all(ok for ok, _ in checks.values())
summary = {
    "passed": passed,
    "debug_run": str(latest),
    "results": str(results_path),
    "checks": {k: {"ok": v[0], "message": v[1]} for k, v in checks.items()},
    "cases": data["cases"],
}
(log_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if not passed:
    raise SystemExit("BigDFT container GPU validation failed")
PY

  echo
  echo "Container BigDFT GPU rebuild passed validation."
  echo "Activate with: source $use_script"
  echo "Logs: $log_dir"
}

case "${1:-}" in
  --inside) run_inside ;;
  ""|--host) run_host ;;
  *)
    echo "usage: $0 [--host|--inside]" >&2
    exit 2
    ;;
esac
