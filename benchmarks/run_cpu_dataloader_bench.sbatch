#!/bin/bash
#SBATCH --job-name=molnex-dlbench
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date) ==="

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load buildenv-gcc/2026.03 2>/dev/null || module load buildenv-gcc/recommendation
GCC_LIB=$(dirname "$(which g++)")/../lib64
export LD_LIBRARY_PATH="$GCC_LIB:${LD_LIBRARY_PATH:-}"

WORKTREE=/home/jicli594/work/molcrafts/molnex/.claude/worktrees/trainer-dataset-fixes
cd "$WORKTREE"
VENV=.venv

"$VENV/bin/python" -c "import molix, torch; print('molix OK, torch', torch.__version__, 'arch ok')"

# Clean per-job cache so the build cost is excluded from the profile.
CACHE=/dev/shm/molnex_dlbench_$$
OMP_NUM_THREADS=8 "$VENV/bin/python" benchmarks/bench_dataloader_datasets.py \
    --n 4000 --n-batches 60 --cache-dir "$CACHE" \
    2>&1 | grep -vE "it/s|UserWarning|warnings.warn|cuequivariance_ops|Loading|Fitting|rank|Caching"
rm -rf "$CACHE"
echo "=== DONE $(date) ==="
