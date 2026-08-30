find_package(Python COMPONENTS Interpreter REQUIRED)
find_package(Threads REQUIRED)

set(_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
set(_FIXTURE_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/generate_fixtures.py")

add_executable(test_model_runner test_model_runner.cpp)
target_link_libraries(test_model_runner PRIVATE molnex_interface)
target_include_directories(test_model_runner PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(test_model_runner PRIVATE ${TORCH_LIBRARIES} Threads::Threads)
if(TORCH_CXX_FLAGS)
  separate_arguments(_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(test_model_runner PRIVATE ${_TORCH_CXX_FLAGS})
endif()

# Fixture: run Python script to produce exported models + reference tensors.
add_test(
  NAME fixture_setup
  COMMAND "${Python_EXECUTABLE}" "${_FIXTURE_SCRIPT}" "${_FIXTURE_DIR}"
)
set_tests_properties(fixture_setup PROPERTIES
  FIXTURES_SETUP molnex_interface_fixtures
)

set(_TESTS
  cpu_run_matches_reference
  cuda_run_matches_reference
  update_weights_from_path
  update_weights_from_map
  parameter_info
  concurrent_run_and_update
  load_bad_path_throws
)

foreach(_t IN LISTS _TESTS)
  add_test(
    NAME ${_t}
    COMMAND test_model_runner ${_t}
  )
  set_tests_properties(${_t} PROPERTIES
    FIXTURES_REQUIRED molnex_interface_fixtures
    ENVIRONMENT "MOLNEX_INTERFACE_FIXTURE_DIR=${_FIXTURE_DIR}"
  )
endforeach()


# Fixture-backed contract case: parses the schema-v2 meta emitted by
# generate_fixtures.py via the production _lammps_meta_block — ties the Python
# LAMMPS_META_SCHEMA to the C++ expected version at test time (spec 01 ac-013).
add_test(NAME contract_reads_generated_v2_fixture
         COMMAND test_lammps_contract contract_reads_generated_v2_fixture)
set_tests_properties(contract_reads_generated_v2_fixture PROPERTIES
  FIXTURES_REQUIRED molnex_interface_fixtures
  ENVIRONMENT "MOLNEX_INTERFACE_FIXTURE_DIR=${_FIXTURE_DIR}"
)

# ---------------------------------------------------------------------------
# NO-FIXTURE tests
#
# Header-reader cases driven entirely by in-test JSON string literals: no
# exported model, no Python fixture generation, no LAMMPS build. They are
# deliberately registered WITHOUT FIXTURES_REQUIRED so a CPU-only CI job can
# select them with an explicit `ctest -R` filter and never pull in
# fixture_setup (a full AOTI export). Same dispatcher convention as above: the
# executable takes the case name as argv[1].
# ---------------------------------------------------------------------------

add_executable(test_meta_schema test_meta_schema.cpp)
target_link_libraries(test_meta_schema PRIVATE molnex_interface)

set(_META_SCHEMA_TESTS
  meta_device_from_bare_export
  meta_scoped_block_shadowing
  meta_string_array
  meta_block_missing_returns_absent
)

foreach(_t IN LISTS _META_SCHEMA_TESTS)
  add_test(
    NAME ${_t}
    COMMAND test_meta_schema ${_t}
  )
endforeach()

add_executable(test_lammps_contract test_lammps_contract.cpp)
target_link_libraries(test_lammps_contract PRIVATE molnex_interface)
target_include_directories(test_lammps_contract PRIVATE
  # lammps_contract.h (header-only; NOT add_subdirectory(lammps) — that
  # project() FATAL_ERRORs without LAMMPS)
  ${CMAKE_CURRENT_SOURCE_DIR}/../lammps
)

set(_LAMMPS_CONTRACT_TESTS
  contract_schema_version_mismatch_throws
  contract_unknown_contract_throws
  contract_unknown_execution_throws
  contract_unsupported_unit_style_throws
  contract_length_unit_not_angstrom_throws
  contract_input_arity_mismatch_throws
  contract_species_int_array
  contract_species_rejects_non_integer
)

foreach(_t IN LISTS _LAMMPS_CONTRACT_TESTS)
  add_test(
    NAME ${_t}
    COMMAND test_lammps_contract ${_t}
  )
endforeach()

# ---------------------------------------------------------------------------
# Atom-side bridge cases (spec lammps-iface-02-view).
#
# Guarded on the target, not on a file glob: molnex_lammps_bridge does not
# exist yet, so this whole block is skipped and `ctest -N` simply does not list
# any bridge_* case. THAT ABSENCE IS THE RED EVIDENCE — test_bridge_atoms.cpp
# is written against the API the components task must satisfy, and the suite
# stays configurable and fully runnable in the meantime (registering it
# unconditionally would break `cmake` itself and take every other case down
# with it).
#
# The components task removes nothing here: declaring molnex_lammps_bridge in
# interface/lammps/bridge/CMakeLists.txt and adding it from
# interface/CMakeLists.txt is what makes this guard true, at which point the
# ten cases below appear in the ctest list and must go green.
#
# No FIXTURES_REQUIRED: the cases drive a synthetic LammpsView out of their own
# std::vector storage — no exported model, no Python generation, no LAMMPS.
# ---------------------------------------------------------------------------

if(TARGET molnex_lammps_bridge)
  add_executable(test_bridge_atoms test_bridge_atoms.cpp)
  # The bridge target carries its own PUBLIC include directory, so the cases
  # reach the headers as "bridge/xxx.h" with nothing added here.
  target_link_libraries(test_bridge_atoms PRIVATE molnex_lammps_bridge ${TORCH_LIBRARIES})
  if(TORCH_CXX_FLAGS)
    separate_arguments(_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(test_bridge_atoms PRIVATE ${_TORCH_CXX_FLAGS})
  endif()

  set(_BRIDGE_TESTS
    bridge_species_map_maps_types
    bridge_stage_nodes_shapes_and_values
    bridge_force_scatter_accumulates
    bridge_species_map_rejects_unknown_z
    bridge_species_map_rejects_bad_table_size
    bridge_species_map_rejects_nonpositive_z
    bridge_species_map_z_of_rejects_out_of_range_type
    bridge_force_scatter_rejects_wrong_row_count
    bridge_stage_nodes_rejects_n_graph_over_nall
    bridge_force_scatter_rejects_n_graph_over_nall
  )

  foreach(_t IN LISTS _BRIDGE_TESTS)
    add_test(
      NAME ${_t}
      COMMAND test_bridge_atoms ${_t}
    )
  endforeach()

  # -------------------------------------------------------------------------
  # Graph-side bridge cases (spec lammps-iface-03-graph): GraphBridge, the
  # reverse-comm free functions and the execution strategy, over a synthetic
  # LammpsView carrying neighbour spans. Same no-fixture, no-LAMMPS, no-CUDA
  # footing as the atom-side cases above, and registered inside the same
  # `if(TARGET molnex_lammps_bridge)` block because they link the same target.
  #
  # The two fixture-backed execution cases are registered separately below:
  # abstracting padding and output conversion into free functions is exactly
  # what keeps the rest of this list runnable with no GPU and no exported model.
  # -------------------------------------------------------------------------

  add_executable(test_bridge_graph test_bridge_graph.cpp)
  # molnex_lammps_bridge carries molnex_interface (for ModelRunner) and its
  # PUBLIC MOLNEX_INTERFACE_CUDA definition transitively, which is what lets
  # bridge_graphed_exec_cpu_only_ctor_throws know at compile time whether the
  # CPU-only diagnostic it asserts is reachable in this build.
  target_link_libraries(test_bridge_graph PRIVATE molnex_lammps_bridge ${TORCH_LIBRARIES})
  if(TORCH_CXX_FLAGS)
    separate_arguments(_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(test_bridge_graph PRIVATE ${_TORCH_CXX_FLAGS})
  endif()

  set(_BRIDGE_GRAPH_TESTS
    bridge_graph_edge_convention
    bridge_graph_cutoff_refilter
    bridge_graph_sorted_by_source_target
    bridge_graph_drops_beyond_n_graph
    bridge_graph_applies_neigh_mask
    bridge_graph_empty_neighbor_list
    bridge_graph_reuses_buffers_across_calls
    bridge_graph_rejects_n_graph_over_nall
    bridge_comm_pack_reverse_layout
    bridge_comm_unpack_accumulates
    bridge_pad_edges
    bridge_read_model_outputs_rejects_short_virial
    bridge_read_model_outputs_absent_virial_undefined
    bridge_read_model_outputs_ignores_virial_when_unsupported
    bridge_graphed_exec_cpu_only_ctor_throws
  )

  foreach(_t IN LISTS _BRIDGE_GRAPH_TESTS)
    add_test(
      NAME ${_t}
      COMMAND test_bridge_graph ${_t}
    )
  endforeach()

  # Fixture-backed execution cases: they need the three-input graph-potential
  # package generate_fixtures.py exports, so they carry FIXTURES_REQUIRED and
  # the fixture-dir environment variable — same wiring as
  # contract_reads_generated_v2_fixture. bridge_dynamic_exec_matches_reference
  # is CPU-only and belongs in CI; bridge_graphed_exec_matches_dynamic needs a
  # GPU and is excluded there by an explicit -E filter rather than by a runtime
  # skip.
  set(_BRIDGE_GRAPH_FIXTURE_TESTS
    bridge_dynamic_exec_matches_reference
    bridge_graphed_exec_matches_dynamic
  )

  foreach(_t IN LISTS _BRIDGE_GRAPH_FIXTURE_TESTS)
    add_test(
      NAME ${_t}
      COMMAND test_bridge_graph ${_t}
    )
    set_tests_properties(${_t} PROPERTIES
      FIXTURES_REQUIRED molnex_interface_fixtures
      ENVIRONMENT "MOLNEX_INTERFACE_FIXTURE_DIR=${_FIXTURE_DIR}"
    )
  endforeach()
else()
  message(STATUS
    "interface/tests: molnex_lammps_bridge absent — bridge_* cases unlisted "
    "(expected until lammps-iface-02-view lands the bridge target)")
endif()
