# molnex_runtime: load a deployed model and run the site-energy convention.
#
# Engine-neutral by construction: depends on LibTorch and nothing else — no
# engine header, no Python. An engine adapter (interface/lammps, ...) links
# this target; this target never references one.
#
# Usable from an enclosing build (interface/CMakeLists.txt, an engine plugin)
# or standalone (`cmake -S interface/runtime`).

cmake_minimum_required(VERSION 3.20)
project(molnex_runtime LANGUAGES CXX)

# C++20: the ATen headers of the pinned torch (>=2.14) require it.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT TORCH_FOUND)
  find_package(Torch REQUIRED)
endif()

# CUDA is optional: with a CUDA toolkit the runtime loads device=cuda packages;
# without one it refuses them at load with a message saying why.
find_package(CUDAToolkit QUIET)

add_library(molnex_runtime STATIC
  src/metadata.cpp
  src/op_libraries.cpp
  src/model_artifact.cpp
  src/aoti_artifact.cpp
  src/site_energy_exec.cpp
)

set_target_properties(molnex_runtime PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_include_directories(molnex_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(molnex_runtime PUBLIC ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})

if(CUDAToolkit_FOUND)
  target_compile_definitions(molnex_runtime PRIVATE MOLNEX_RUNTIME_CUDA)
  message(STATUS "molnex_runtime: CUDA toolkit found — device=cuda packages load")
else()
  message(STATUS "molnex_runtime: no CUDA toolkit — CPU-only build")
endif()

if(TORCH_CXX_FLAGS)
  separate_arguments(_MOLNEX_RUNTIME_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(molnex_runtime PUBLIC ${_MOLNEX_RUNTIME_TORCH_CXX_FLAGS})
endif()
