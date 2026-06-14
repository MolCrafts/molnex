cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(molnex LANGUAGES CXX)

# Build C++ extensions from src/molix/op
add_subdirectory(src/molix/op)

# Pure-C++ runtime library for AOT-Inductor exported models
# (libmolnex_interface). Off by default so the wheel build via
# scikit-build-core stays slim; manual builds and the C++ test suite
# turn it on with -DMOLNEX_BUILD_INTERFACE=ON.
option(MOLNEX_BUILD_INTERFACE "Build interface/ (libmolnex_interface)" OFF)
if(MOLNEX_BUILD_INTERFACE)
  add_subdirectory(interface)
endif()
