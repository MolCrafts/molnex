# The contract-implementation layer: LAMMPS-free, meta-schema-aware.
#
# Third of the three layers under interface/lammps/, and the dependency
# gradient is the reason each is its own target:
#
#   bridge/  — primitives that do not know the meta schema exists
#   impls/   — strategy that DOES know the contract (meta_schema.h /
#              lammps_contract.h enter the C++ here and nowhere below)
#   facade   — pair_molnex.cpp, the only file that includes a LAMMPS header
#
# Like bridge/, this target references neither LAMMPS_HEADER_DIR nor LAMMPS_LIB
# and no file under impls/ includes a LAMMPS header, so the whole layer is
# drivable from ctest (interface/tests) on a machine with no LAMMPS build, no
# CUDA and no exported package — the Exec is injected.
#
# Added from interface/CMakeLists.txt — NOT via add_subdirectory(lammps), which
# is a separate project(molnexplugin) that FATAL_ERRORs without LAMMPS and would
# take the LAMMPS-free configure of interface/ down with it. Torch was already
# found by that parent, so TORCH_LIBRARIES / TORCH_CXX_FLAGS are in scope here.

add_library(molnex_lammps_impls STATIC
  cutoff_graph_impl.cpp
  registry.cpp
  # Header-only; listed for IDE / install visibility only (no translation unit).
  contract_impl.h
)

set_target_properties(molnex_lammps_impls PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# The PUBLIC include dir is impls/'s PARENT (interface/lammps), so consumers
# reach these headers as "impls/xxx.h" — saying which layer they are reaching
# into — and lammps_contract.h resolves from the same root. The test target
# needs nothing added on its side.
target_include_directories(molnex_lammps_impls PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

# The bridge carries molnex_interface (ModelRunner, meta_schema.h) and
# ${TORCH_LIBRARIES} transitively, plus molnex_interface's PUBLIC
# MOLNEX_INTERFACE_CUDA definition — one build-time answer to "is there a CUDA
# toolkit", shared by GraphedExec's guard and the runtime's.
target_link_libraries(molnex_lammps_impls PUBLIC molnex_lammps_bridge)

if(TORCH_CXX_FLAGS)
  separate_arguments(_MOLNEX_IMPLS_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(molnex_lammps_impls PUBLIC ${_MOLNEX_IMPLS_TORCH_CXX_FLAGS})
endif()
