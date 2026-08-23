#!/bin/bash
# OMol25 self-split calibration: align + scale audit + timed pack sweep.
# Not a production train (checklist §6 stays NO-GO).
#
#   molq submit slurm --name class-i-calibrate --partition gpu \
#     --account naiss2026-1-53-gpu --gpus 1 --cpus 8 --time 1h \
#     --workdir /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex \
#     bash scripts/mm_param_learning/run_class_i_calibrate_gh200.sbatch

set -euo pipefail

echo "=== node $(hostname) arch $(uname -m) $(date -Iseconds) ==="
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
MOLPY_SRC="${MOLCRAFTS}/.worktrees/molpy-0.13.1/src"
if [ ! -d "${MOLPY_SRC}" ]; then
  echo "=== create molpy-0.13.1 worktree (molpy dev checkout is stale at 0.7.0) ==="
  git -C "${MOLCRAFTS}/molpy" worktree add --detach "${MOLCRAFTS}/.worktrees/molpy-0.13.1" master
fi
export PYTHONPATH="${MOLNEX}:${MOLCRAFTS}/molhub/src:${MOLPY_SRC}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

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
WHEEL_VERSION="$(basename "${WHEEL}" | sed -E 's/^molcrafts_molrs-([0-9.]+)-.*/\1/')"
INSTALLED_VERSION="$("${PY}" -c 'from importlib.metadata import version; print(version("molcrafts-molrs"))' 2>/dev/null || echo none)"
if [ "${INSTALLED_VERSION}" != "${WHEEL_VERSION}" ]; then
  echo "=== install molrs ${WHEEL_VERSION} (installed ${INSTALLED_VERSION}) ==="
  "${PY}" -m pip install --force-reinstall --no-deps "${WHEEL}"
else
  echo "=== molrs ${WHEEL_VERSION} already installed; skip reinstall ==="
fi
"${PY}" -c "import lmdb" 2>/dev/null || "${PY}" -m pip install -q lmdb

echo "=== Class-I calibrate: OMol25 8:2 align + inspect + pack ==="
"${PY}" -c "from scripts.mm_param_learning.quasi_prod import calibrate; calibrate()"

echo "=== DONE $(date -Iseconds) ==="
