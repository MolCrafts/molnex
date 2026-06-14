#!/bin/bash
#SBATCH --job-name=molnex-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

set -euo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date) ==="

# Batch shells are non-interactive — bring up the lmod `module` function.
source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"

module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
echo "python: $(which python) $(python --version)"
echo "gcc:    $(which g++) $(g++ --version | head -1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

WORKTREE=/home/jicli594/work/molcrafts/molnex/.claude/worktrees/trainer-dataset-fixes
VENV="$WORKTREE/.venv-gpu"
cd "$WORKTREE"

if [ ! -x "$VENV/bin/python" ]; then
  echo "=== creating aarch64 venv ==="
  python -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  # aarch64 Linux torch wheels are CUDA-enabled (sbsa); bundles its own runtime.
  "$VENV/bin/pip" install -q "torch>=2.10" "scikit-build-core>=0.10" "cmake>=4.0.0" ninja
  echo "=== building molnex ext (CPU-only) + installing editable ==="
  MOLNEX_OP_ENABLE_CUDA=OFF "$VENV/bin/pip" install -q --no-build-isolation -e ".[dev]"
fi

echo "=== torch / cuda sanity ==="
"$VENV/bin/python" - <<'PY'
import torch, platform
print("arch", platform.machine(), "torch", torch.__version__, "cuda_build", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
import molix
print("molix import OK")
PY

echo "=== BENCHMARK: cpu (Grace ARM) ==="
OMP_NUM_THREADS=16 "$VENV/bin/python" benchmarks/bench_trainer_throughput.py --device cpu --steps 50 \
  2>&1 | grep -vE "UserWarning|warnings.warn|cuequivariance_ops"

echo "=== BENCHMARK: cuda (Hopper) ==="
"$VENV/bin/python" benchmarks/bench_trainer_throughput.py --device cuda --steps 50 \
  2>&1 | grep -vE "UserWarning|warnings.warn|cuequivariance_ops"

echo "=== DONE $(date) ==="
