# The contract-implementation layer: LAMMPS-free, over the runtime.
#
#   bridge/  — primitives that do not know the model exists
#   impls/   — strategy over molnex_runtime (Metadata, SiteEnergyExec)
#   facade   — pair_molnex.cpp, the only file that includes a LAMMPS header
#
# Like bridge/, this target references neither LAMMPS_HEADER_DIR nor LAMMPS_LIB
# and no file under impls/ includes a LAMMPS header, so the layer is drivable
# from ctest (interface/tests) with no LAMMPS build, no CUDA and no package —
# the execution is injected.

add_library(molnex_lammps_impls STATIC
  cutoff_graph_impl.cpp
  # Header-only; listed for IDE / install visibility only (no translation unit).
  contract_impl.h
)

set_target_properties(molnex_lammps_impls PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# PUBLIC include dir is impls/'s PARENT (interface/lammps): "impls/xxx.h".
target_include_directories(molnex_lammps_impls PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_link_libraries(molnex_lammps_impls PUBLIC molnex_lammps_bridge molnex_runtime)
