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
# One env: .aarch64 (GH200). Local molnex via PYTHONPATH. Do not create venvs.
PY="${PY:-/nobackup/proj/disk/teoroo/personal/jicli594/work/.aarch64/bin/python}"
MRS=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molrs/molrs-python/python/molrs/_lib.abi3.so
if ! file "${MRS}" | grep -q aarch64; then
  if [[ -f "${MRS}.aarch64" ]]; then
    echo "restoring aarch64 molrs .so (login rustc overwrote the in-tree file)"
    cp -a "${MRS}.aarch64" "${MRS}"
  else
    echo "FAIL: ${MRS} is not aarch64 and no .aarch64 backup" >&2
    file "${MRS}" >&2
    exit 1
  fi
fi
GCC_LIB="$(dirname "$(readlink -f "$("${CXX:-g++}" -print-file-name=libstdc++.so.6)")")"
TORCH_LIB=$("${PY}" -c "import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')")
CUDA_LIB=/software/sse2/el9_gh200/manual/CUDA/13.0.3_580.126.20/lib64
export LD_LIBRARY_PATH="${GCC_LIB}:${TORCH_LIB}:${CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${MOLNEX}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${MOLNEX}"

echo "=== python ${PY} ==="
"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

"${PY}" scripts/polar_port/run_npt.py \
  --system "${SCRATCH}/peo_box.pt" \
  --dump "${SCRATCH}/MACE-POLAR-1-M.model" \
  --cueq-state "${SCRATCH}/polar_cueq_state.pt" \
  --steps 100 \
  --device cuda \
  --precision fp32

echo "=== DONE $(date -Iseconds) ==="
