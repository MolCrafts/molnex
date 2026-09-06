#!/bin/bash
#SBATCH --job-name=polar-ours
#SBATCH --account=naiss2026-1-53-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex/.scratch/polar_npt/npt-%j.out
#SBATCH --error=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex/.scratch/polar_npt/npt-%j.err

set -euo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date -Iseconds) ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} ==="
nvidia-smi --query-gpu=name,compute_mode,memory.total --format=csv || true

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0

MOLNEX=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex
SCRATCH="${MOLNEX}/.scratch/polar_npt"
SHIM="${MOLNEX}/.scratch/convfusion/aarch64site"
# One env: .aarch64 (GH200). Local molnex via PYTHONPATH. Do not create venvs.
PY="${PY:-/nobackup/proj/disk/teoroo/personal/jicli594/work/.aarch64/bin/python}"
# Never swap the editable molrs binary in its shared source tree: doing so
# breaks concurrent x86 tests on the login node. The architecture-specific
# import shim is immutable for the lifetime of this job.
file "${SHIM}/molrs/_lib.abi3.so" | grep -q aarch64 || {
  echo "FAIL: missing aarch64 molrs shim at ${SHIM}" >&2
  exit 1
}
GCC_LIB="$(dirname "$(readlink -f "$("${CXX:-g++}" -print-file-name=libstdc++.so.6)")")"
TORCH_LIB=$("${PY}" -c "import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')")
CUDA_LIB=/software/sse2/el9_gh200/manual/CUDA/13.0.3_580.126.20/lib64
export LD_LIBRARY_PATH="${GCC_LIB}:${TORCH_LIB}:${CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${SHIM}:${MOLNEX}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${MOLNEX}"

echo "=== python ${PY} ==="
"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

POLAR_SYSTEM="${POLAR_SYSTEM:-${SCRATCH}/peo_box.pt}"
POLAR_STEPS="${POLAR_STEPS:-100}"
CUEQ_TARGET_BLOCK_SIZE="${CUEQ_TARGET_BLOCK_SIZE:-4096}"
EXTRA_ARGS=(--benchmark-repeats "${POLAR_BENCHMARK_REPEATS:-0}" \
  --benchmark-output "${SCRATCH}/benchmark-${SLURM_JOB_ID}.pt")
if [[ -n "${POLAR_NEIGHBOR_BIN:-}" ]]; then
  EXTRA_ARGS+=(--neighbor-bin "${POLAR_NEIGHBOR_BIN}")
fi
if [[ "${POLAR_COMPILE_CORE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--compile-core)
fi
"${PY}" scripts/polar_port/run_npt.py \
  --system "${POLAR_SYSTEM}" \
  --dump "${SCRATCH}/MACE-POLAR-1-M.model" \
  --cueq-state "${SCRATCH}/polar_cueq_state.pt" \
  --trust-checkpoint \
  --steps "${POLAR_STEPS}" \
  --cueq-target-block-size "${CUEQ_TARGET_BLOCK_SIZE}" \
  --device cuda \
  --precision "${POLAR_PRECISION:-fp32}" \
  --potential-precision "${POLAR_POTENTIAL_PRECISION:-${POLAR_PRECISION:-fp32}}" \
  --energy-precision "${POLAR_ENERGY_PRECISION:-fp64}" \
  --matmul-precision "${POLAR_MATMUL_PRECISION:-highest}" \
  "${EXTRA_ARGS[@]}"

echo "=== DONE $(date -Iseconds) ==="
