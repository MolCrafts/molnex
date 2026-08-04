#!/bin/bash
#SBATCH --job-name=pinet-r3-speed
#SBATCH --account=naiss2026-4-715-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/pinet-r3-speed-%j.out
#SBATCH --error=benchmarks/pinet-r3-speed-%j.err

set -euo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date) ==="
nvidia-smi -L || true

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0
module load GPU/Python/3.13.5-bare-gcc-2025b-eb

ROOT=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex
cd "$ROOT"
VENV="$ROOT/.venv-gh200"
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "=== creating aarch64 venv ==="
  python -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "torch>=2.10" "scikit-build-core>=0.10" "cmake>=4.0.0" ninja \
    numpy tensordict pydantic "molcrafts-molcfg>=1.0" "molcrafts-mollog>=1.0" \
    "molcrafts-molpy>=0.3" cuequivariance cuequivariance-torch opt_einsum_fx tqdm typer zarr numcodecs
  # Native op optional for PiNet pure-torch path; soft-skip in the bench if ABI mismatch.
  MOLNEX_OP_ENABLE_CUDA=OFF "$VENV/bin/pip" install -q --no-build-isolation -e . --no-deps || true
else
  echo "=== reusing venv $VENV ==="
fi

"$VENV/bin/python" - <<'PY'
import torch, platform
print("arch", platform.machine())
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    x = torch.randn(4, device="cuda")
    print("cuda smoke", float(x.sum()))
PY

export PYTHONUNBUFFERED=1

"$VENV/bin/python" benchmarks/bench_pinet_rank3_speed.py --device cuda --steps 50 --warmup 15 \
    --atoms 512 --edges 4096 --graphs 16

"$VENV/bin/python" benchmarks/bench_pinet_rank3_speed.py --device cuda --steps 30 --warmup 10 \
    --atoms 2048 --edges 16384 --graphs 64

echo "=== DONE $(date) ==="
