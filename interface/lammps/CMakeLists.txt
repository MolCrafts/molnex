# Build the generic molnex LAMMPS pair-style plugin (pair_style molnex) against:
#   - LibTorch (find_package(Torch); from the active venv's torch)
#   - molnex_interface (the AOTI C++ runner; the parent interface/ library)
#   - liblammps.so (from the pip `lammps` wheel or a source build) + LAMMPS headers
#
# Pass on the command line:
#   -DLAMMPS_HEADER_DIR=<.../lammps-src/src>
#   -DLAMMPS_LIB=<.../site-packages/lammps/liblammps.so>
#   -DCMAKE_PREFIX_PATH=<.../site-packages/torch/share/cmake>
#   [-DLAMMPS_MPI_INCLUDE=<.../lammps-src/src/STUBS>]   # serial LAMMPS (BUILD_MPI=off)
#
# The molnex_interface library is picked up from the parent directory
# (../  == interface/), so no -DMOLNEX_INTERFACE_DIR is needed when configuring
# from here. Configure standalone, e.g.:
#   cmake -S interface/lammps -B build/lammps \
#         -DCMAKE_PREFIX_PATH=$TORCH_CMAKE \
#         -DLAMMPS_HEADER_DIR=$LMP_SRC -DLAMMPS_LIB=$LMP_LIB

cmake_minimum_required(VERSION 3.20)
project(molnexplugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT LAMMPS_HEADER_DIR)
  message(FATAL_ERROR "set -DLAMMPS_HEADER_DIR (path to LAMMPS src/ with pair.h, lammpsplugin.h)")
endif()
if(NOT LAMMPS_LIB)
  message(FATAL_ERROR "set -DLAMMPS_LIB (path to liblammps.so)")
endif()

find_package(Torch REQUIRED)

# The AOTI runner library lives one level up (interface/). Build it here as a
# subdirectory (static, PIC) unless an enclosing build already provides the
# `molnex_interface` target.
if(NOT TARGET molnex_interface)
  set(MOLNEX_INTERFACE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. molnex_interface_build)
endif()

add_library(molnexplugin MODULE molnexplugin.cpp pair_molnex.cpp)
# Arch-tagged output dir so x86_64 and aarch64 plugin builds coexist without
# clobbering each other (mirrors the arch-suffixed op lib in src/molix/op).
# `plugin load` takes an explicit path, so a per-arch directory is the cleanest
# separation. Override with -DMOLNEXPLUGIN_OUTPUT_DIR=... if you want a fixed path.
if(NOT MOLNEXPLUGIN_OUTPUT_DIR)
  set(MOLNEXPLUGIN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}")
endif()
set_target_properties(molnexplugin PROPERTIES
  PREFIX "" SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${MOLNEXPLUGIN_OUTPUT_DIR}"
)
message(STATUS "molnexplugin → ${MOLNEXPLUGIN_OUTPUT_DIR}/molnexplugin.so (${CMAKE_SYSTEM_PROCESSOR})")

target_include_directories(molnexplugin PRIVATE
  ${LAMMPS_HEADER_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}                 # lammps_meta.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../include      # molnex/interface/model_runner.h
)
# when LAMMPS is built serial (BUILD_MPI=off) the plugin still needs the stub mpi.h
if(LAMMPS_MPI_INCLUDE)
  target_include_directories(molnexplugin PRIVATE ${LAMMPS_MPI_INCLUDE})
endif()
target_link_libraries(molnexplugin PRIVATE
  molnex_interface
  ${TORCH_LIBRARIES}
  ${LAMMPS_LIB}
)
# match torch's libstdc++ ABI flags
if(TORCH_CXX_FLAGS)
  separate_arguments(_TF NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(molnexplugin PRIVATE ${_TF})
endif()
target_link_options(molnexplugin PRIVATE -rdynamic)
