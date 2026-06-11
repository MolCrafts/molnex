#!/bin/bash
#SBATCH --job-name=pinet-pad
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
# Use the prebuilt aarch64 venv (molnex editable on the MAIN checkout, with its
# own aarch64 ext) — no install, no rebuild, and it never touches the worktree .so.
VENV=/home/jicli594/work/molcrafts/.aarch64
cd /home/jicli594/work/molcrafts/molnex/.claude/worktrees/trainer-dataset-fixes
"$VENV/bin/python" -c "import molix,torch;print('molix OK',torch.__version__,torch.cuda.is_available())"
OMP_NUM_THREADS=16 "$VENV/bin/python" benchmarks/train_pinet_ef_padded.py \
    --epochs 300 --n-total 1000 --lambda-f 100 --e-pad 14000 \
    --cache-dir /dev/shm/pinet_ef_pad --out pinet_ef_padded_compare.json \
    2>&1 | grep -vaE "it/s|UserWarning|warnings.warn|cuequivariance_ops|W[0-9]+ "
echo "=== DONE $(date) ==="
