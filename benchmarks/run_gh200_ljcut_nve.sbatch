#!/bin/bash
#SBATCH --job-name=ljcut-nve
#SBATCH --account=naiss2026-4-715-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Bulk lj/cut NVE on GH200: verify_md_ljcut_nve.py, pure-GPU (CUDA neighbour
# rebuild + compiled force field). Submit from the repo root:
#
#   sbatch benchmarks/run_gh200_ljcut_nve.sbatch            # 100 ps, N=500
#   sbatch benchmarks/run_gh200_ljcut_nve.sbatch --n 10     # N=4000
#
# Extra args are forwarded to the verify script.

set -euo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date) ==="

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0
module load GPU/Python/3.13.5-bare-gcc-2025b-eb

REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
PY=/nobackup/proj/disk/teoroo/personal/jicli594/work/.aarch64/bin/python
cd "$REPO"

# The op .so is linked against the buildenv GCC's libstdc++ (CXXABI newer than
# /lib64's); make the runtime loader resolve the matching one.
GCC_LIBDIR=$(dirname "$(g++ -print-file-name=libstdc++.so)")
export LD_LIBRARY_PATH="$GCC_LIBDIR:${LD_LIBRARY_PATH:-}"

"$PY" -c "import torch, platform; print('arch', platform.machine(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Refresh the aarch64 op library with CUDA kernels — the on-device neighbour
# rebuild (torch.ops.molix.get_neighbor_pairs) dispatches to AutogradCUDA.
cmake -S src/molix/op -B src/molix/op/build-gh200 -DMOLNEX_OP_ENABLE_CUDA=ON \
  -DCMAKE_PREFIX_PATH="$("$PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build src/molix/op/build-gh200 -j

echo "=== run 1: inductor fullgraph ==="
PYTHONPATH=src:. "$PY" benchmarks/verify_md_ljcut_nve.py "$@" \
    2>&1 | grep -vE "UserWarning|warnings.warn|W[0-9]+ "

echo "=== run 2: cuda-graphs (reduce-overhead) preset ==="
PYTHONPATH=src:. "$PY" benchmarks/verify_md_ljcut_nve.py --cuda-graphs --no-save "$@" \
    2>&1 | grep -vE "UserWarning|warnings.warn|W[0-9]+ "

echo "=== DONE $(date) ==="
