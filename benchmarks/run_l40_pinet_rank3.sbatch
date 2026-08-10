#!/bin/bash
#SBATCH --job-name=pinet-r3-speed
#SBATCH --account=naiss2026-4-715-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=benchmarks/pinet-r3-speed-%j.out
#SBATCH --error=benchmarks/pinet-r3-speed-%j.err
#
# Note: Arrhenius ``gpu`` partition is GH200 (aarch64). Login-node L40 is
# not a compute target. Prefer ``run_gh200_pinet_rank3_speed.sbatch``.
# This script auto-picks the arch-matching venv if both exist.

set -euo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date) ==="
nvidia-smi -L || true

ROOT=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex
cd "$ROOT"

ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ]; then
  # GH200 stack
  source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
  module purge
  module load GPU/buildenv-gcccuda/2026.03-cu13.0 2>/dev/null || true
  module load GPU/Python/3.13.5-bare-gcc-2025b-eb 2>/dev/null || true
  PY="$ROOT/.venv-gh200/bin/python"
else
  PY="$ROOT/.venv-l40/bin/python"
fi

if [ ! -x "$PY" ]; then
  echo "ERROR: missing python at $PY (arch=$ARCH)" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

"$PY" - <<'PY'
import torch, platform
print("arch", platform.machine())
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    x = torch.randn(4, device="cuda")
    print("cuda smoke", float(x.sum()))
PY

echo "=== small batch atoms=512 graphs=16 ==="
"$PY" benchmarks/bench_pinet_rank3_speed.py --device cuda --steps 50 --warmup 15 \
    --atoms 512 --edges 4096 --graphs 16

echo "=== large batch atoms=2048 graphs=64 ==="
"$PY" benchmarks/bench_pinet_rank3_speed.py --device cuda --steps 30 --warmup 10 \
    --atoms 2048 --edges 16384 --graphs 64

echo "=== DONE $(date) ==="
