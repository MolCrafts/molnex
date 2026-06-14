cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(molnex_interface LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MOLNEX_INTERFACE_BUILD_TESTS "Build interface C++ tests" ON)

find_package(Torch REQUIRED)

add_library(molnex_interface STATIC
  src/model_runner.cpp
)

target_include_directories(molnex_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(molnex_interface PUBLIC ${TORCH_LIBRARIES})

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
