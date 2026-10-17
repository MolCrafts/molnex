# molnex's native operator libraries — LibTorch only, no Python, built out of
# source:
#
#   molix/  molnex_opLib: molnex's own operators (`TORCH_LIBRARY(molix)` —
#           neighbour pairs, PME, scatter). molix (Python training) and
#           molnex_runtime (C++ deployment) load the same built file; a wheel
#           installs it as molix/lib/.
#   cueq/   molnex_cueq_ops: cuEquivariance's operators registered in C++, for a
#           process without Python (upstream registers them in Python only).
#           Built only with -DCUEQ_OPS_LIBRARY=<.../cuequivariance_ops/lib/libcue_ops.so>;
#           never loaded next to cuequivariance_ops_torch (both define the schemas).
#
#   cmake -S csrc -B build/csrc-$(uname -m) -DMOLNEX_OP_ENABLE_CUDA=ON \
#         -DTorch_DIR=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')/Torch \
#         [-DCUEQ_OPS_LIBRARY=...]
#   cmake --build build/csrc-$(uname -m) -j
#
# Each library lands in its own binary directory, build/csrc-<arch>/{molix,cueq}/,
# named for the target architecture (lib<name>.<arch>.so); both loaders — molix
# and molnex_runtime's OpLibraries — search MOLNEX_OP_LIBRARY_PATH, which
# scripts/env/activate.sh points at those two directories.

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(molnex_csrc LANGUAGES CXX)

if(NOT TORCH_FOUND)
  find_package(Torch REQUIRED)
endif()

add_subdirectory(molix)

set(CUEQ_OPS_LIBRARY "" CACHE FILEPATH "cuEquivariance's libcue_ops.so; empty skips csrc/cueq")
if(CUEQ_OPS_LIBRARY)
  add_subdirectory(cueq)
endif()
