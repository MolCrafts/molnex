cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(molnex LANGUAGES CXX)

# molnex's native operator libraries (csrc/: molix ops, and cueq with
# -DCUEQ_OPS_LIBRARY). LibTorch only; molix and the C++ runtime both load them.
# A wheel installs molnex_opLib as molix/lib/.
add_subdirectory(csrc)

# C++ deployment side: the engine-neutral molnex_runtime and the LAMMPS-free
# layers of the LAMMPS adapter, with their ctest suite. Off by default so the
# wheel build via scikit-build-core stays slim; manual builds and the C++ test
# suite turn it on with -DMOLNEX_BUILD_INTERFACE=ON.
option(MOLNEX_BUILD_INTERFACE "Build interface/ (molnex_runtime + LAMMPS adapter layers)" OFF)
if(MOLNEX_BUILD_INTERFACE)
  add_subdirectory(interface)
endif()
