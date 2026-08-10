#!/bin/bash
#SBATCH --job-name=ljcut-nve-prec
#SBATCH --account=naiss2026-4-715-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# One precision-matrix arm per job (submit 3 in parallel). dtype is the only
# free variable; all MD knobs are pinned. Submit from the repo root:
#
#   sbatch --export=ALL,ARM=md64_pot64 benchmarks/run_gh200_ljcut_nve.sbatch
#   sbatch --export=ALL,ARM=md64_pot32 benchmarks/run_gh200_ljcut_nve.sbatch
#   sbatch --export=ALL,ARM=md32_pot32 benchmarks/run_gh200_ljcut_nve.sbatch
#
# Or omit ARM to run all three arms sequentially in one job.
# Extra CLI args after the script name are forwarded to the verify script.

set -euo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date) ==="
echo "=== ARM=${ARM:-ALL} ==="

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0
module load GPU/Python/3.13.5-bare-gcc-2025b-eb

REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
PY=/nobackup/proj/disk/teoroo/personal/jicli594/work/.aarch64/bin/python
cd "$REPO"

GCC_LIBDIR=$(dirname "$(g++ -print-file-name=libstdc++.so)")
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}"

"$PY" -c "import torch, platform; print('arch', platform.machine(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cmake -S src/molix/op -B src/molix/op/build-gh200 -DMOLNEX_OP_ENABLE_CUDA=ON \
  -DCMAKE_PREFIX_PATH="$("$PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build src/molix/op/build-gh200 -j

EXTRA=("$@")
COMMON=(
  --ps 5000
  --n 5
  --dt 4.0
  --skin 0
  --every 1
  --delay 0
  --sample-every 1000
  --seed 1
  --device cuda
)

run_arm () {
  local name="$1"; shift
  echo "=== arm: $name ==="
  PYTHONPATH=src:. "$PY" benchmarks/verify_md_ljcut_nve.py \
      "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} "$@" \
      2>&1 | grep -vE "UserWarning|warnings.warn|W[0-9]+ "
}

case "${ARM:-ALL}" in
  md64_pot64) run_arm md64_pot64 --precision fp64 ;;
  md64_pot32) run_arm md64_pot32 --precision fp64 --potential-precision fp32 ;;
  md32_pot32) run_arm md32_pot32 --precision fp32 ;;
  ALL)
    run_arm md64_pot64 --precision fp64
    run_arm md64_pot32 --precision fp64 --potential-precision fp32
    run_arm md32_pot32 --precision fp32
    ;;
  *)
    echo "unknown ARM=${ARM} (want md64_pot64|md64_pot32|md32_pot32|ALL)" >&2
    exit 2
    ;;
esac

echo "=== DONE $(date) ==="
