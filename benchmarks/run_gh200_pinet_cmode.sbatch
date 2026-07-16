#!/bin/bash
#SBATCH --job-name=pinet-cmode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date) ==="

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0
module load GPU/Python/3.13.5-bare-gcc-2025b-eb

WORKTREE=/home/jicli594/work/molcrafts/molnex/.claude/worktrees/trainer-dataset-fixes
VENV="$WORKTREE/.venv-gpu"
cd "$WORKTREE"

if [ ! -x "$VENV/bin/python" ]; then
  echo "=== creating aarch64 venv ==="
  python -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "torch>=2.10" "scikit-build-core>=0.10" "cmake>=4.0.0" ninja
  MOLNEX_OP_ENABLE_CUDA=OFF "$VENV/bin/pip" install -q --no-build-isolation -e ".[dev]"
else
  # The C++ ext .so lives in the shared source tree and may be x86 from a CPU
  # build — recompile it for aarch64 before importing molix on the GPU node.
  echo "=== rebuilding aarch64 ext ==="
  MOLNEX_OP_ENABLE_CUDA=OFF "$VENV/bin/pip" install -q --no-build-isolation -e . --force-reinstall --no-deps
fi

"$VENV/bin/python" - <<'PY'
import torch, platform, molix
print("arch", platform.machine(), "torch", torch.__version__, "cuda", torch.cuda.is_available())
print("molix import OK")
PY

OMP_NUM_THREADS=16 "$VENV/bin/python" benchmarks/bench_pinet_compile_modes.py \
    2>&1 | grep -vE "UserWarning|warnings.warn|cuequivariance_ops|W[0-9]+ "
echo "=== DONE $(date) ==="
