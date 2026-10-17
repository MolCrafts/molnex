#!/bin/bash
#SBATCH --job-name=mace-nve-prec
#SBATCH --account=naiss2025-5-447-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-12:00:00
#SBATCH --output=%x-%j.out

# molnex MACE-MatPES NVE — precision × seed matrix (5 ns @ dt=0.5 fs).
# Account: Chao Zhang — naiss2025-5-447-gpu
# Workdir: mace-nve/scratch/precision-matrix
#
# Full 2×2 MD-state × Pot precision (no autocast bf16 here):
#   md64_pot64  — MD fp64, Pot fp64
#   md64_pot32  — MD fp64, Pot fp32
#   md32_pot64  — MD fp32, Pot fp64   <-- required control vs md32_pot32
#   md32_pot32  — MD fp32, Pot fp32
#
# Each arm × SEED ∈ {0,1,2} (3 replicates). Example:
#   sbatch --export=ALL,ARM=md32_pot64,SEED=0,RESUME=0 --job-name=nve-md32p64-s0 \
#     benchmarks/run_gh200_mace_nve_precision.sbatch
#
# Outputs: nve5ns_${ARM}_s${SEED}.pt  (+ .ckpt.pt shards)

set -euo pipefail
ARM="${ARM:?set ARM=md64_pot64|md64_pot32|md32_pot64|md32_pot32|md64_bf16|md32_bf16}"
SEED="${SEED:-0}"
RESUME="${RESUME:-0}"

echo "=== node $(hostname) arch $(uname -m) $(date) ==="
echo "=== ARM=$ARM SEED=$SEED RESUME=$RESUME account=${SLURM_JOB_ACCOUNT:-naiss2025-5-447-gpu} ==="

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0
module load GPU/Python/3.13.5-bare-gcc-2025b-eb

REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
PY=/nobackup/proj/disk/teoroo/personal/jicli594/work/.aarch64/bin/python
WD=/nobackup/proj/disk/teoroo/personal/jicli594/work/mace-nve/scratch/precision-matrix
WEIGHTS=/home/jicli594/work/mace_models
BUILD="build/csrc-gh200-${SLURM_JOB_ID:-$$}"
cd "$REPO"
mkdir -p "$WD"

if [[ ! -f "$WD/system.pt" ]]; then
  echo "missing $WD/system.pt" >&2
  exit 2
fi

# Prefer precomputed IC for this seed (same MaxwellBoltzmann formula as molix).
IC="$WD/ic_seed${SEED}_T300.pt"
if [[ ! -f "$IC" ]]; then
  echo "missing IC $IC — generate with MaxwellBoltzmann seed=$SEED" >&2
  exit 2
fi

GCC_LIBDIR=$(dirname "$(g++ -print-file-name=libstdc++.so)")
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}"
export PATH="/software/sse2/el9_gh200/manual/CUDA/13.0.3_580.126.20/bin:${PATH:-}"

"$PY" -c "import torch, platform; print('arch', platform.machine(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== building molnex_op in $BUILD ==="
cmake -S csrc -B "$BUILD" -DMOLNEX_OP_ENABLE_CUDA=ON \
  -DCMAKE_PREFIX_PATH="$("$PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build "$BUILD" -j
export MOLNEX_OP_LIBRARY_PATH="$PWD/$BUILD"

# Map ARM → run_nve precision flags
case "$ARM" in
  md64_pot64) PREC=(--precision fp64) ;;
  md64_pot32) PREC=(--precision fp64 --potential-precision fp32) ;;
  md32_pot64) PREC=(--precision fp32 --potential-precision fp64) ;;
  md32_pot32) PREC=(--precision fp32) ;;
  md64_bf16)  PREC=(--precision fp64 --autocast-bf16) ;;
  md32_bf16)  PREC=(--precision fp32 --autocast-bf16) ;;
  *)
    echo "unknown ARM=$ARM" >&2
    exit 2
    ;;
esac

TAG="${ARM}_s${SEED}"
OUT="$WD/nve5ns_${TAG}.pt"
CKPT="$WD/nve5ns_${TAG}.ckpt.pt"

RESUME_ARGS=()
if [[ "$RESUME" != "0" && -f "$CKPT" ]]; then
  RESUME_ARGS=(--resume "$CKPT")
  echo "=== resuming from $CKPT ==="
fi

# Load IC positions+velocities into a one-shot torch file that run_nve understands
# via --system (pos only) + --seed (vel). We instead inject vel by writing a
# system snapshot that still only has pos, and use --seed for MB sample — but
# that regenerates vel. Prefer: dump system with pos from IC, and pass seed so
# MB matches precomputed IC only if formula+seed match.
# Precomputed IC vel is loaded by temporarily using a small wrapper: run_nve
# does not take vel file. So we use --seed=SEED which matches our IC generation
# (same formula as MaxwellBoltzmann) AND --system with IC pos.

echo "=== arm=$TAG PREC=${PREC[*]} out=$OUT seed=$SEED ==="

# system.pt positions are fixed lattice; IC only changes vel. run_nve samples
# vel with MaxwellBoltzmann(mass).sample(T, seed=SEED) — same seed/formula as
# ic_seed${SEED}_T300.pt generation → identical ICs.
PYTHONPATH=src:. "$PY" scripts/matpes_port/run_nve.py \
  --system "$WD/system.pt" \
  --weights-dir "$WEIGHTS" \
  --out "$OUT" \
  --steps 10000000 --dt 0.5 --temperature 300 --stride 10000 \
  --skin 0 --every 1 --delay 0 --seed "$SEED" \
  --checkpoint-every 1000000 --flush-every 200 \
  --compile --device cuda \
  "${PREC[@]}" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  2>&1 | grep -vE "UserWarning|warnings.warn|cuequivariance_ops|W[0-9]+ "

echo "=== DONE $(date) ARM=$TAG ==="
