# The atom-side LAMMPS bridge: LAMMPS-free by construction.
#
# This target references neither LAMMPS_HEADER_DIR nor LAMMPS_LIB, and no
# source or header under bridge/ includes a LAMMPS header. That is the whole
# point: the components are drivable from ctest (interface/tests) on a machine
# with no LAMMPS build, over a synthetic LammpsView.
#
# Added from interface/CMakeLists.txt — NOT via add_subdirectory(lammps), which
# is a separate project(molnexplugin) that FATAL_ERRORs without LAMMPS and
# would take the LAMMPS-free configure of interface/ down with it. Torch was
# already found by that parent, so TORCH_LIBRARIES / TORCH_CXX_FLAGS are in
# scope here and the bridge compiles against the same ABI surface as
# molnex_interface rather than a second one of its own.

add_library(molnex_lammps_bridge STATIC
  species_map.cpp
  node_staging.cpp
  force_scatter.cpp
  graph_bridge.cpp
  comm_reverse.cpp
  exec.cpp
  dynamic_exec.cpp
  graphed_exec.cpp
  # Header-only; listed for IDE / install visibility only (no translation unit).
  lammps_view.h
)

set_target_properties(molnex_lammps_bridge PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# The PUBLIC include dir is bridge/'s PARENT, so every include reads
# "bridge/xxx.h" — consumers say which layer they are reaching into, and the
# test target needs nothing added on its side.
target_include_directories(molnex_lammps_bridge PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

# molnex_interface, not just torch: the Exec implementations delegate to
# molnex::interface::ModelRunner, so its header is in the bridge's PUBLIC
# interface (dynamic_exec.h / graphed_exec.h both name the type). This REVISES
# lammps-iface-02-view's "bridge depends only on torch" wording without touching
# what that spec actually gated: molnex_interface is itself LAMMPS-free, so
# configuring interface/ with neither LAMMPS_HEADER_DIR nor LAMMPS_LIB still
# builds this target and runs its ctest cases.
#
# The link also carries molnex_interface's PUBLIC MOLNEX_INTERFACE_CUDA
# definition, which is what keeps GraphedExec's guard in step with the runtime's
# — one build-time answer to "is there a CUDA toolkit", not two.
#
# What does NOT come with it: meta_schema.h and lammps_contract.h stay out of
# every file under bridge/. The bridge talks to the runtime; the export meta is
# the facade's business.
target_link_libraries(molnex_lammps_bridge PUBLIC ${TORCH_LIBRARIES} molnex_interface)

if(TORCH_CXX_FLAGS)
  separate_arguments(_MOLNEX_BRIDGE_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(molnex_lammps_bridge PUBLIC ${_MOLNEX_BRIDGE_TORCH_CXX_FLAGS})
endif()
