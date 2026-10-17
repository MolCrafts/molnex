# Build the generic molnex LAMMPS pair-style plugin (pair_style molnex) against:
#   - LibTorch (find_package(Torch); from the active venv's torch)
#   - molnex_runtime (../runtime: metadata, model artifact, site-energy exec)
#   - liblammps.so (a LAMMPS source build with PKG_PLUGIN) + LAMMPS headers
#
# Pass on the command line:
#   -DLAMMPS_HEADER_DIR=<.../lammps-src/src>
#   -DLAMMPS_LIB=<.../site-packages/lammps/liblammps.so>
#   -DCMAKE_PREFIX_PATH=<.../site-packages/torch/share/cmake>
#   [-DLAMMPS_MPI_INCLUDE=<.../lammps-src/src/STUBS>]   # serial LAMMPS (BUILD_MPI=off)
#
# molnex_runtime and the two LAMMPS-free layers (bridge/, impls/) are added
# from their own directories unless an enclosing build already declared them.
# Configure standalone, e.g.:
#   cmake -S interface/lammps -B build/lammps \
#         -DCMAKE_PREFIX_PATH=$TORCH_CMAKE \
#         -DLAMMPS_HEADER_DIR=$LMP_SRC -DLAMMPS_LIB=$LMP_LIB

cmake_minimum_required(VERSION 3.20)
project(molnexplugin LANGUAGES CXX)

# C++20, not 17: the ATen headers of the pinned torch (>=2.14, pyproject)
# hard-`#error` below C++20 and use std::strong_ordering, string_view's
# starts_with/ends_with and constrained members. Matches
# csrc/molix/CMakeLists.txt, which is already on 20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT LAMMPS_HEADER_DIR)
  message(FATAL_ERROR "set -DLAMMPS_HEADER_DIR (path to LAMMPS src/ with pair.h, lammpsplugin.h)")
endif()
if(NOT LAMMPS_LIB)
  message(FATAL_ERROR "set -DLAMMPS_LIB (path to liblammps.so)")
endif()

find_package(Torch REQUIRED)

if(NOT TARGET molnex_runtime)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../runtime molnex_runtime_build)
endif()
if(NOT TARGET molnex_lammps_bridge)
  add_subdirectory(bridge)
endif()
if(NOT TARGET molnex_lammps_impls)
  add_subdirectory(impls)
endif()

add_library(molnexplugin MODULE molnexplugin.cpp pair_molnex.cpp)
# Written to the build directory, which is per architecture
# (build/plugin-<arch>/, see README): `plugin load` takes an explicit path, so
# x86_64 and aarch64 plugins never share a file. Override with
# -DMOLNEXPLUGIN_OUTPUT_DIR=... for a fixed path.
if(NOT MOLNEXPLUGIN_OUTPUT_DIR)
  set(MOLNEXPLUGIN_OUTPUT_DIR "${CMAKE_BINARY_DIR}")
endif()
set_target_properties(molnexplugin PROPERTIES
  PREFIX "" SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${MOLNEXPLUGIN_OUTPUT_DIR}"
)
message(STATUS "molnexplugin → ${MOLNEXPLUGIN_OUTPUT_DIR}/molnexplugin.so (${CMAKE_SYSTEM_PROCESSOR})")

target_include_directories(molnexplugin PRIVATE
  ${LAMMPS_HEADER_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}                 # pair_molnex.h
)
# when LAMMPS is built serial (BUILD_MPI=off) the plugin still needs the stub mpi.h
if(LAMMPS_MPI_INCLUDE)
  target_include_directories(molnexplugin PRIVATE ${LAMMPS_MPI_INCLUDE})
endif()
target_link_libraries(molnexplugin PRIVATE
  # CutoffGraphImpl; carries molnex_lammps_bridge and molnex_runtime (their
  # PUBLIC include dirs give "bridge/xxx.h", "impls/xxx.h" and
  # "molnex/runtime/xxx.h").
  molnex_lammps_impls
  ${TORCH_LIBRARIES}
  ${LAMMPS_LIB}
)
# match torch's libstdc++ ABI flags
if(TORCH_CXX_FLAGS)
  separate_arguments(_TF NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(molnexplugin PRIVATE ${_TF})
endif()
target_link_options(molnexplugin PRIVATE -rdynamic)
