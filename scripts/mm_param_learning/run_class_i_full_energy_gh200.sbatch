#!/bin/bash
#SBATCH --job-name=class-i-full-e
#SBATCH --account=naiss2025-5-447-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=class-i-full-energy-%j.out

# Class-I full-energy mini → molexp run on GH200 (aarch64).
# Do NOT use login-node x86 cargo/maturin. Build molrs from source with
# aarch64 rustup (PyPI 0.13.0 wheel lacks Neighbors required by molpy 0.13).
#
#   sbatch scripts/mm_param_learning/run_class_i_full_energy_gh200.sbatch

set -euo pipefail

echo "=== node $(hostname) arch $(uname -m) $(date -Iseconds) ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} ==="

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0
module load GPU/Python/3.13.5-bare-gcc-2025b-eb

MOLCRAFTS="${MOLCRAFTS:-/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts}"
MOLNEX="${MOLCRAFTS}/molnex"
VENV="${MOLNEX}/.venv-gh200"
# aarch64-only toolchains (never x86 login cargo)
export RUSTUP_HOME="${MOLNEX}/.rustup-gh200"
export CARGO_HOME="${MOLNEX}/.cargo-gh200"
# strip x86 cargo from PATH if present
export PATH="$(echo "${PATH}" | tr ':' '\n' | grep -v 'x86_64/cargo' | paste -sd: -)"

cd "${MOLNEX}"

if [ ! -x "${VENV}/bin/python" ]; then
  echo "=== creating GH200 venv ==="
  python -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -q --upgrade pip

echo "=== torch + deps ==="
python -m pip install -q "torch>=2.10" tensordict tqdm pydantic pyyaml rich typer fastapi uvicorn maturin

# aarch64 rust (idempotent)
if [ ! -x "${CARGO_HOME}/bin/cargo" ]; then
  echo "=== rustup aarch64 into ${CARGO_HOME} ==="
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain stable \
      --default-host aarch64-unknown-linux-gnu
fi
export PATH="${CARGO_HOME}/bin:${PATH}"
echo "cargo: $(command -v cargo) $(cargo -vV | head -1)"

echo "=== maturin develop local molrs (needs Neighbors) ==="
(
  cd "${MOLCRAFTS}/molrs/molrs-python"
  # force reinstall over any incomplete PyPI wheel
  python -m pip uninstall -y molcrafts-molrs 2>/dev/null || true
  maturin develop --release
)
python -c "import molrs; print('molrs', getattr(molrs,'__version__','?'), 'Neighbors', hasattr(molrs,'Neighbors'))"

python -m pip install -q -e "${MOLCRAFTS}/molpy"
python -m pip install -q -e "${MOLCRAFTS}/mollog" -e "${MOLCRAFTS}/molcfg" -e "${MOLCRAFTS}/molq" 2>/dev/null || true

export PYTHONPATH="${MOLNEX}:${MOLNEX}/src:${MOLCRAFTS}/molhub/src:${MOLCRAFTS}/molexp/src:${MOLCRAFTS}/molpy/src:${MOLCRAFTS}/molq/src:${MOLCRAFTS}/mollog/src:${MOLCRAFTS}/molcfg/src:${PYTHONPATH:-}"

# molnex pure-Python (C++ op optional)
python -m pip install -q --no-build-isolation -e "${MOLNEX}" --no-deps 2>/dev/null || true

python - <<'PY'
import platform
import torch
print("arch", platform.machine(), "torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
from molpot.forcefield import Parameterizer  # noqa: F401
from molhub.dataset import Class1Universal2026  # noqa: F401
import molexp  # noqa: F401
print("imports OK", Class1Universal2026().load().source_coordinates)
PY

WS="${MOLCRAFTS}/workspaces/mm-param-learning"

echo "=== molexp Class-I v0 profile=gh200 ==="

OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" \
python -m molexp.cli run scripts/mm_param_learning/train.py \
  --config scripts/mm_param_learning/molcfg.yaml \
  --profile gh200 \
  -ws "${WS}"

echo "=== DONE $(date -Iseconds) ==="
