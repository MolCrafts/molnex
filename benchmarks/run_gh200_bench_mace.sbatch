#!/bin/bash
#SBATCH --job-name=gh200-bench-mace
#SBATCH --account=naiss2026-4-715-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex/%x-%j.out

# MACE-MatPES eager / CUDA-graph timing on GH200 (benchmarks/bench_mace_matpes.py),
# N=192 and N=1024, fp32 and fp64: the before/after record of the ElementLinear
# layout change (plan B4/B6).
#   sbatch benchmarks/run_gh200_bench_mace.sbatch
# Uses .venv-aarch64 through scripts/env/activate.sh and the native op
# libraries already under build/csrc-aarch64; nothing is rebuilt here — an
# import failure is reported and the job stops.
set -uo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date -Iseconds) job ${SLURM_JOB_ID:-?} ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv || true

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0

MOLNEX=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex
cd "${MOLNEX}" || exit 2
# molnex's own aarch64 env (scripts/env/build_env.sh); sets PY, PYTHONPATH and
# MOLNEX_OP_LIBRARY_PATH.
source scripts/env/activate.sh || exit 2
GCC_LIB="$(dirname "$(readlink -f "$("${CXX:-g++}" -print-file-name=libstdc++.so.6)")")"
# torch/lib located without importing torch (GCC 14 lib64 is not on the path yet).
TORCH_LIB=$("${PY}" -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('torch').origin).resolve().parent / 'lib')")
CUDA_LIB=/software/sse2/el9_gh200/manual/CUDA/13.0.3_580.126.20/lib64
export LD_LIBRARY_PATH="${GCC_LIB}:${TORCH_LIB}:${CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MOLNEX_ALLEGRO_WEIGHTS_DIR=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/.build-alloc/allegro-oracle/assets
echo "=== PY=${PY}"
echo "=== MOLNEX_OP_LIBRARY_PATH=${MOLNEX_OP_LIBRARY_PATH}"
cat .venv-aarch64/BUILT_FROM build/csrc-aarch64/BUILT_FROM 2>/dev/null || true

echo "=== preflight: import molix / molpy on this interpreter, native op on CUDA ==="
"${PY}" - <<'PY'
import sys, warnings
warnings.simplefilter("error", UserWarning)  # a failed native-op load is a UserWarning
import torch
import molix, molpy, molhub
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0))
print("molix", molix.__file__)
print("molpy", molpy.__file__, "molhub", molhub.__file__)
op = getattr(torch.ops.molix, "scatter_sum", None)
if op is None:
    sys.exit("molix::scatter_sum is not registered: native op library did not load")
src = torch.randn(10, 4, device="cuda", dtype=torch.float64)
idx = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 3], device="cuda")
ref = torch.zeros(4, 4, device="cuda", dtype=torch.float64).index_add_(0, idx, src)
print("native scatter_sum on cuda max|diff|", float((op(src, idx, 4) - ref).abs().max()))
import cuequivariance_torch, cuequivariance_ops_torch
print("cuequivariance_torch", cuequivariance_torch.__version__)
PY
rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "!!! preflight failed (rc=${rc}) — not rebuilding the env or the native libs; see the error above"
  exit 3
fi

echo
# benchmarks/bench_mace_matpes.py: eager vs Compiler(cuda_graphs=True) energy+force
# op of MaceMatpes on a periodic random-dense box, N=192 and N=1024, fp32 and fp64.
STATUS=0
for n in 192 1024; do
  for prec in "" "--fp64"; do
    echo "=== bench_mace_matpes --n-atoms ${n} ${prec} $(date -Iseconds) ==="
    "${PY}" benchmarks/bench_mace_matpes.py --n-atoms "${n}" --steps 30 ${prec}
    rc=$?
    echo "=== rc=${rc} ==="
    [ "${rc}" -eq 0 ] || STATUS=1
  done
done
echo "=== DONE status=${STATUS} $(date -Iseconds) ==="
exit "${STATUS}"
