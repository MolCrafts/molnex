# molnex_opLib: molnex's native operators (`TORCH_LIBRARY(molix)` — neighbour
# pairs, PME, scatter) as a standalone LibTorch library, no Python. Built from
# csrc/ (see csrc/CMakeLists.txt for the command); usable on its own too.

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(molnex_op LANGUAGES CXX)

set(MOLNEX_OP_LIBRARY_NAME molnex_opLib)

option(MOLNEX_OP_ENABLE_CUDA "Build CUDA kernels" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT TORCH_FOUND)
  find_package(Torch REQUIRED)
endif()

file(GLOB_RECURSE MOLNEX_OP_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if(MOLNEX_OP_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  file(GLOB_RECURSE MOLNEX_OP_CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
  )

  list(APPEND MOLNEX_OP_SOURCES ${MOLNEX_OP_CUDA_SOURCES})
endif()

message(STATUS "molnex_op: CUDA=${MOLNEX_OP_ENABLE_CUDA}")
message(STATUS "molnex_op: sources=${MOLNEX_OP_SOURCES}")

add_library(${MOLNEX_OP_LIBRARY_NAME} SHARED ${MOLNEX_OP_SOURCES})

target_include_directories(${MOLNEX_OP_LIBRARY_NAME} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# LibTorch only — no Python, no libtorch_python — so a C++ engine process can
# load this library (molnex's deployment runtime, interface/runtime OpLibraries).
# The C++ autograd Functions' torch::autograd::_wrap_outputs is in libtorch_cpu.
target_link_libraries(${MOLNEX_OP_LIBRARY_NAME} PRIVATE ${TORCH_LIBRARIES})

if(TORCH_CXX_FLAGS)
  separate_arguments(_MOLNEX_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(${MOLNEX_OP_LIBRARY_NAME} PRIVATE ${_MOLNEX_TORCH_CXX_FLAGS})
endif()

if(MOLNEX_OP_ENABLE_CUDA)
  target_compile_definitions(${MOLNEX_OP_LIBRARY_NAME} PRIVATE
    __CUDA_NO_HALF_OPERATORS__
  )
  target_compile_options(${MOLNEX_OP_LIBRARY_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
endif()

# Tag the artifact with the target architecture (libmolnex_opLib.x86_64.so,
# libmolnex_opLib.aarch64.so) so builds for both arches can sit in one search
# directory. Both loaders accept the tagged name: molix (Python) and
# molnex_runtime's OpLibraries (C++), each searching MOLNEX_OP_LIBRARY_PATH.
# The library is written to this build's binary directory, never into the
# source tree.
set_target_properties(${MOLNEX_OP_LIBRARY_NAME} PROPERTIES
  OUTPUT_NAME "molnex_opLib.${CMAKE_SYSTEM_PROCESSOR}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# A wheel (scikit-build-core installs into the wheel root) carries it as
# molix/lib/, where molix looks when MOLNEX_OP_LIBRARY_PATH is unset.
install(TARGETS ${MOLNEX_OP_LIBRARY_NAME} LIBRARY DESTINATION molix/lib)
