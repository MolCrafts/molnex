cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(molnex_interface LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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

if(MOLNEX_INTERFACE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
