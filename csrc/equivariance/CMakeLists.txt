cmake_minimum_required(VERSION 3.24)
project(molnex_equivariance LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
option(MOLNEX_EQUIVARIANCE_LINEINFO "Emit CUDA source locations for Nsight" OFF)
option(MOLNEX_EQUIVARIANCE_PTXAS_VERBOSE "Report registers and spills per kernel" OFF)

# One translation unit per (L1, L2) configuration so the unrolled kernels
# compile in parallel; the list mirrors MOLNEX_EQUIVARIANCE_CONFIGS in dispatch.cuh.
set(_molnex_configs "0 2;1 2;2 2;3 2;0 3;1 3;2 3;3 3")
set(_molnex_instances)
foreach(_pair IN LISTS _molnex_configs)
  separate_arguments(_pair)
  list(GET _pair 0 _l1)
  list(GET _pair 1 _l2)
  set(_file "${CMAKE_CURRENT_BINARY_DIR}/instance_${_l1}_${_l2}.cu")
  file(WRITE "${_file}"
    "#include \"dispatch.cuh\"\n"
    "namespace molnex::equivariance {\n"
    "template void run_product<${_l1}, ${_l2}>(const ProductArgs&);\n"
    "template void run_adjoint<${_l1}, ${_l2}>(const AdjointArgs&);\n"
    "}\n")
  list(APPEND _molnex_instances "${_file}")
endforeach()

add_library(molnex_equivariance SHARED operators.cpp channelwise.cu schedule.cu ${_molnex_instances})
target_include_directories(molnex_equivariance PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(molnex_equivariance PROPERTIES
  CXX_STANDARD 20 CUDA_STANDARD 20
  CXX_STANDARD_REQUIRED ON CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES 90 POSITION_INDEPENDENT_CODE ON)
target_link_libraries(molnex_equivariance PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
target_compile_options(molnex_equivariance PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
if(MOLNEX_EQUIVARIANCE_LINEINFO)
  target_compile_options(molnex_equivariance PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()
if(MOLNEX_EQUIVARIANCE_PTXAS_VERBOSE)
  target_compile_options(molnex_equivariance PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif()
if(TORCH_CXX_FLAGS)
  separate_arguments(TORCH_OPTIONS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(molnex_equivariance PRIVATE ${TORCH_OPTIONS})
endif()
install(TARGETS molnex_equivariance LIBRARY DESTINATION molrep/lib)
