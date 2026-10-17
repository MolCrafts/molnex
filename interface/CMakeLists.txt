cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Aggregator for the C++ side of molnex deployment, and the home of its ctest
# suite. Two layers, one-way:
#
#   runtime/  molnex_runtime — engine-neutral: metadata, model artifact (by
#             format), native op libraries, atom-energy execution. LibTorch
#             only.
#   lammps/   the LAMMPS adapter: bridge/ and impls/ (LAMMPS-free, ctest-able)
#             and the pair_style plugin (its own project; needs LAMMPS).
#
# The native operator libraries a package may call (molnex's own, and the C++
# registration of cuEquivariance's) are built from csrc/, not here; runtime's
# OpLibraries loads them by name from MOLNEX_OP_LIBRARY_PATH.
#
# runtime never references lammps. Configuring this directory needs no LAMMPS:
# the plugin is built from interface/lammps, which adds what it needs from
# here.

project(molnex_interface LANGUAGES CXX)

# C++20: the ATen headers of the pinned torch (>=2.14) require it.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MOLNEX_INTERFACE_BUILD_TESTS "Build interface C++ tests" ON)

find_package(Torch REQUIRED)

add_subdirectory(runtime)

# NOT add_subdirectory(lammps): that is the plugin's project(molnexplugin),
# which FATAL_ERRORs without -DLAMMPS_HEADER_DIR / -DLAMMPS_LIB. Its two
# LAMMPS-free layers are added directly so their cases run without LAMMPS.
add_subdirectory(lammps/bridge)
add_subdirectory(lammps/impls)

if(MOLNEX_INTERFACE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
