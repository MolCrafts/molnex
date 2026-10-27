# molnex's native operator libraries — LibTorch only, no Python, built out of
# source:
#
#   molix/  molnex_opLib: molnex's own operators (`TORCH_LIBRARY(molix)` —
#           neighbour pairs, PME, scatter). molix (Python training) and
#           molnex_runtime (C++ deployment) load the same built file; a wheel
#           installs it as molix/lib/.
#   equivariance/  molnex_equivariance: molnex's equivariant operators
#           (`TORCH_LIBRARY(molnex_equivariance)` — MACE's indexed channelwise
#           product and symmetric contraction, the spherical harmonics, the
#           Allegro contracter and weighted channels), CUDA kernels with
#           Meta kernels, every reduction in a fixed order (no atomics). CUDA
#           only: it is built with -DMOLNEX_OP_ENABLE_CUDA=ON; CPU tensors go
#           through molrep.equivariance.reference in Python.
#
#   cmake -S csrc -B build/csrc-$(uname -m) -DMOLNEX_OP_ENABLE_CUDA=ON \
#         -DTorch_DIR=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')/Torch
#   cmake --build build/csrc-$(uname -m) -j
#
# Each library lands in its own binary directory,
# build/csrc-<arch>/{molix,equivariance}/, named for the target architecture
# (lib<name>.<arch>.so); the loaders — molix, molrep.equivariance and
# molnex_runtime's OpLibraries — search MOLNEX_OP_LIBRARY_PATH, which
# scripts/env/activate.sh points at those directories.

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(molnex_csrc LANGUAGES CXX)

if(NOT TORCH_FOUND)
  find_package(Torch REQUIRED)
endif()

add_subdirectory(molix)

# The equivariance library is CUDA kernels only; it follows the CUDA switch.
if(MOLNEX_OP_ENABLE_CUDA)
  add_subdirectory(equivariance)
endif()
