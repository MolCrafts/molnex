cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(molnex_interface LANGUAGES CXX)

# C++20, not 17: the ATen headers of the pinned torch (>=2.14, pyproject)
# hard-`#error` below C++20 and use std::strong_ordering, string_view's
# starts_with/ends_with and constrained members. Matches
# src/molix/op/CMakeLists.txt, which is already on 20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MOLNEX_INTERFACE_BUILD_TESTS "Build interface C++ tests" ON)

find_package(Torch REQUIRED)

# CUDA is optional: with a CUDA toolkit present we compile the async/CUDA AOTI
# runner path; on a CPU-only torch (no cuda_runtime_api.h) we build a CPU-only
# library that throws if asked to load a device=cuda export. Auto-detected so
# the same source serves both x86-CPU and GPU builds.
find_package(CUDAToolkit QUIET)

add_library(molnex_interface STATIC
  src/model_runner.cpp
  # Header-only; listed for IDE / install visibility only (no translation unit).
  include/molnex/interface/meta_schema.h
)

target_include_directories(molnex_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(molnex_interface PUBLIC ${TORCH_LIBRARIES})

if(CUDAToolkit_FOUND)
  target_compile_definitions(molnex_interface PUBLIC MOLNEX_INTERFACE_CUDA)
  message(STATUS "molnex_interface: CUDA toolkit found — building with CUDA runner")
else()
  message(STATUS "molnex_interface: no CUDA toolkit — CPU-only build")
endif()

if(TORCH_CXX_FLAGS)
  separate_arguments(_MOLNEX_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(molnex_interface PUBLIC ${_MOLNEX_TORCH_CXX_FLAGS})
endif()

set_target_properties(molnex_interface PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# The atom-side LAMMPS bridge (spec lammps-iface-02-view). Declared here, from
# the parent, because interface/lammps/CMakeLists.txt is a separate
# project(molnexplugin) that FATAL_ERRORs without -DLAMMPS_HEADER_DIR /
# -DLAMMPS_LIB — add_subdirectory(lammps) would make this whole configure fail
# on a machine with no LAMMPS and destroy the LAMMPS-free ctest path the bridge
# exists to provide. lammps/bridge references no LAMMPS variable, so it is safe
# to add unconditionally. It must come BEFORE tests/, whose bridge_* cases are
# guarded on `if(TARGET molnex_lammps_bridge)`.
add_subdirectory(lammps/bridge)

# The contract-implementation layer (spec lammps-iface-04-facade), added from
# the parent for the same reason and under the same rule as lammps/bridge:
# NEVER add_subdirectory(lammps). It must come AFTER lammps/bridge (it links
# that target) and BEFORE tests/, whose impl_* / registry_* cases are guarded on
# `if(TARGET molnex_lammps_impls)`.
add_subdirectory(lammps/impls)

if(MOLNEX_INTERFACE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
