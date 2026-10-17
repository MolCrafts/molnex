# The atom-side LAMMPS bridge: LAMMPS-free by construction.
#
# This target references neither LAMMPS_HEADER_DIR nor LAMMPS_LIB, and no
# source or header under bridge/ includes a LAMMPS header, so the components
# are drivable from ctest (interface/tests) over a synthetic LammpsView on a
# machine with no LAMMPS build. They know nothing of the model either: no
# runtime header is included here.
#
# Added from interface/CMakeLists.txt and from interface/lammps/CMakeLists.txt
# (whichever configures first); Torch is already found by either.

add_library(molnex_lammps_bridge STATIC
  species_map.cpp
  node_staging.cpp
  force_scatter.cpp
  graph_bridge.cpp
  comm_reverse.cpp
  # Header-only; listed for IDE / install visibility only (no translation unit).
  lammps_view.h
)

set_target_properties(molnex_lammps_bridge PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# The PUBLIC include dir is bridge/'s PARENT, so every include reads
# "bridge/xxx.h" — consumers say which layer they are reaching into.
target_include_directories(molnex_lammps_bridge PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_link_libraries(molnex_lammps_bridge PUBLIC ${TORCH_LIBRARIES})

if(TORCH_CXX_FLAGS)
  separate_arguments(_MOLNEX_BRIDGE_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(molnex_lammps_bridge PUBLIC ${_MOLNEX_BRIDGE_TORCH_CXX_FLAGS})
endif()
