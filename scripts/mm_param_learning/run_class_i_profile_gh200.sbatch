#!/bin/bash
# Class-I GPU smoke + profile on GH200. Submit with molq, not raw sbatch:
#
#   molq submit slurm --name class-i-profile --partition gpu \
#     --account naiss2026-1-53-gpu --gpus 1 --cpus 8 --time 40m \
#     --workdir /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex \
#     bash scripts/mm_param_learning/run_class_i_profile_gh200.sbatch
#
# Runs the compute smoke (gpu_bench.py) then the LazyOMol25 data+train
# profile (profile_class_i.py). Not a production train.

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
export PYTHONPATH="${MOLNEX}:${MOLCRAFTS}/molhub/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# The profile script reads the raw LMDB, so it must see the molhub cache.
export MOLHUB_HOME="${MOLHUB_HOME:-${MOLCRAFTS}/workspaces/molhub-home}"

cd "${MOLNEX}"

echo "=== python ${PY} ==="
"${PY}" -c "import platform, torch; print('arch', platform.machine(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

WHEEL_DIR="${MOLNEX}/.wheels-gh200"
mkdir -p "${WHEEL_DIR}"
WHEEL="$(ls -1t "${WHEEL_DIR}"/molcrafts_molrs-*.whl 2>/dev/null | head -1 || true)"
if [ -z "${WHEEL}" ]; then
  echo "=== compile current molrs into .aarch64 ==="
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
"${PY}" - <<'PY'
import molrs
print("molrs", molrs.__file__, "NeighborList", hasattr(molrs, "NeighborList"))
PY

echo "=== SMOKE: compute timings (gpu_bench.py) ==="
"${PY}" scripts/mm_param_learning/gpu_bench.py 2>&1 | grep -vE "UserWarning|_load_ops_library|libmolnex|warnings.warn|cuequivariance_ops"

echo "=== PROFILE: LazyOMol25 data + train path ==="
"${PY}" scripts/mm_param_learning/profile_class_i.py 2>&1 | grep -vE "UserWarning|_load_ops_library|libmolnex|warnings.warn|cuequivariance_ops"

echo "=== DONE $(date -Iseconds) ==="
