#!/bin/bash
# One-day Class-I run for one model size (small|medium|large).
# Wall clock 24h is the stop. Checkpoints under runs/prod-day1-*.
#
#   molq submit slurm --name class-i-day-small --partition gpu \
#     --account naiss2026-4-715-gpu --gpus 1 --cpus 8 --time 1d \
#     --workdir /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex \
#     bash scripts/mm_param_learning/run_class_i_prod_day_gh200.sbatch small

set -euo pipefail

SIZE="${1:?usage: $0 small|medium|large|smoke}"

echo "=== node $(hostname) arch $(uname -m) $(date -Iseconds) size=${SIZE} ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} ==="

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0

MOLCRAFTS="${MOLCRAFTS:-/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts}"
MOLNEX="${MOLCRAFTS}/molnex"
PY="${PY:-/home/jicli594/work/.aarch64/bin/python}"
export RUSTUP_HOME="${MOLNEX}/.rustup-gh200"
export CARGO_HOME="${MOLNEX}/.cargo-gh200"
export CARGO_TARGET_DIR="${MOLCRAFTS}/molrs/molrs-python/target-aarch64"
export PATH="${CARGO_HOME}/bin:${PATH}"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${MOLNEX}:${MOLCRAFTS}/molhub/src:${MOLCRAFTS}/molpy/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# 4m extract is ~20 GiB; home is 30 GiB. Keep the cache on nobackup.
export MOLHUB_HOME="${MOLHUB_HOME:-${MOLCRAFTS}/workspaces/molhub-home}"
mkdir -p "${MOLHUB_HOME}"
echo "=== MOLHUB_HOME=${MOLHUB_HOME} ==="

cd "${MOLNEX}"

echo "=== python ${PY} ==="
"${PY}" -c "import platform, torch; print('arch', platform.machine(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

WHEEL_DIR="${MOLNEX}/.wheels-gh200"
mkdir -p "${WHEEL_DIR}"
WHEEL="$(ls -1t "${WHEEL_DIR}"/molcrafts_molrs-*.whl 2>/dev/null | head -1 || true)"
if [ -z "${WHEEL}" ]; then
  echo "=== compile current molrs into .wheels-gh200 ==="
  (
    cd "${MOLCRAFTS}/molrs/molrs-python"
    "${PY}" -m maturin build --release --out "${WHEEL_DIR}"
  )
  WHEEL="$(ls -1t "${WHEEL_DIR}"/molcrafts_molrs-*.whl | head -1)"
else
  echo "=== reuse ${WHEEL} ==="
fi
"${PY}" -m pip install --force-reinstall --no-deps "${WHEEL}"
"${PY}" -c "import lmdb" 2>/dev/null || "${PY}" -m pip install -q lmdb

echo "=== Class-I prod-day1 size=${SIZE} account=${SLURM_JOB_ACCOUNT:-} ==="
"${PY}" scripts/mm_param_learning/prod_day.py "${SIZE}"

echo "=== DONE $(date -Iseconds) ==="
