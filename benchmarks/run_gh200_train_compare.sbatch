#!/bin/bash
#SBATCH --job-name=pinet-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
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
  python -m venv "$VENV"; "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "torch>=2.10" "scikit-build-core>=0.10" "cmake>=4.0.0" ninja
  MOLNEX_OP_ENABLE_CUDA=OFF "$VENV/bin/pip" install -q --no-build-isolation -e ".[dev]"
else
  MOLNEX_OP_ENABLE_CUDA=OFF "$VENV/bin/pip" install -q --no-build-isolation -e . --force-reinstall --no-deps
fi
"$VENV/bin/python" -c "import molix,torch;print('molix OK',torch.__version__,torch.cuda.is_available())"
OMP_NUM_THREADS=16 "$VENV/bin/python" benchmarks/train_pinet_ef_compare.py \
    --epochs 300 --n-total 1000 --lambda-f 100 \
    --cache-dir /dev/shm/pinet_ef_train --out pinet_ef_train_compare.json \
    2>&1 | grep -vaE "it/s|UserWarning|warnings.warn|cuequivariance_ops|W[0-9]+ "
echo "=== DONE $(date) ==="
