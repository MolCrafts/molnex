#!/bin/bash
#SBATCH --job-name=mace-nve-prec
#SBATCH --account=naiss2026-4-715-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# MACE-MatPES NVE precision matrix on GH200: wat64_h3o+ (193 atoms), three
# arms of the inference x MD split — (MD fp64, pot fp64), (MD fp64, pot fp32)
# and (MD fp32, pot fp32) — through molnex's own MD engine with the
# list-owned rebuild policy. Submit from the repo root:
#
#   sbatch benchmarks/run_gh200_mace_nve_precision.sbatch
#
# Extra args are forwarded to every arm (e.g. --steps 10000).

set -euo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date) ==="

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0
module load GPU/Python/3.13.5-bare-gcc-2025b-eb

REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
PY=/nobackup/proj/disk/teoroo/personal/jicli594/work/.aarch64/bin/python
WD=/nobackup/proj/disk/teoroo/personal/jicli594/work/mace-nve/precision-matrix
WEIGHTS=/home/jicli594/work/mace_models
cd "$REPO"

# The op .so is linked against the buildenv GCC's libstdc++ (CXXABI newer than
# /lib64's); make the runtime loader resolve the matching one.
GCC_LIBDIR=$(dirname "$(g++ -print-file-name=libstdc++.so)")
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}"

"$PY" -c "import torch, platform; print('arch', platform.machine(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cmake -S src/molix/op -B src/molix/op/build-gh200 -DMOLNEX_OP_ENABLE_CUDA=ON \
  -DCMAKE_PREFIX_PATH="$("$PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build src/molix/op/build-gh200 -j

EXTRA=("$@")
COMMON=(--system "$WD/system.pt" --weights-dir "$WEIGHTS"
        --steps 2000 --dt 0.5 --temperature 300 --stride 100
        --skin 0.2 --checkpoint-every 0 --compile --device cuda)

run_arm () {
  local name="$1"; shift
  echo "=== arm: $name ==="
  PYTHONPATH=src:. "$PY" scripts/matpes_port/run_nve.py \
      "${COMMON[@]}" --out "$WD/nve_${name}.pt" "${EXTRA[@]:-}" "$@" \
      2>&1 | grep -vE "UserWarning|warnings.warn|cuequivariance_ops|W[0-9]+ "
}

run_arm md64_pot64 --precision fp64
run_arm md64_pot32 --precision fp64 --potential-precision fp32
run_arm md32_pot32 --precision fp32

echo "=== DONE $(date) ==="
