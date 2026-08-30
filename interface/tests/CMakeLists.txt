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


# Fixture-backed contract case: parses the schema-v3 meta emitted by
# generate_fixtures.py via the production _lammps_meta_block — ties the Python
# LAMMPS_META_SCHEMA to the C++ expected version at test time (spec 01 ac-013,
# spec 05 ac-015). The fixture FILENAME carries the version too, so a one-sided
# bump fails to open the file rather than parsing a stale one.
add_test(NAME contract_reads_generated_v3_fixture
         COMMAND test_lammps_contract contract_reads_generated_v3_fixture)
set_tests_properties(contract_reads_generated_v3_fixture PROPERTIES
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
  contract_requires_receptive_field_layers
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
    bridge_graph_ghost_closure
    bridge_graph_order_shuffle_invariance
    bridge_graph_sorts_only_when_contract_requires
    bridge_comm_pack_reverse_layout
    bridge_comm_unpack_accumulates
    bridge_pad_edges
    bridge_read_model_outputs_rejects_short_virial
    bridge_read_model_outputs_absent_virial_undefined
    bridge_read_model_outputs_ignores_virial_when_unsupported
    bridge_read_model_outputs_requires_atomic_energy
    bridge_read_model_outputs_six_atoms_no_transpose
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
  # contract_reads_generated_v3_fixture. bridge_dynamic_exec_matches_reference
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

# ---------------------------------------------------------------------------
# Contract-implementation cases (spec lammps-iface-04-facade): the contract
# types, CutoffGraphImpl as the composition point of every 02-view / 03-graph
# component, and the closed-set dispatch.
#
# Guarded on the target for the same reason the bridge block is: before
# interface/lammps/impls/CMakeLists.txt declares molnex_lammps_impls, this whole
# block is skipped, `ctest -N` lists no impl_* case, and THAT ABSENCE IS THE RED
# EVIDENCE — test_contract_impl.cpp is written against the API the impl task
# must satisfy, while the suite stays configurable and fully runnable.
#
# The impl_* cases need NO fixtures: the Exec is injected, so they drive a stub
# implementation over a synthetic LammpsView with no exported model, no CUDA and
# no LAMMPS. The two registry_* cases are the exception and are registered
# separately below — ContractImpl::for_contract takes a ModelRunner&, which
# cannot exist without a real package.
# ---------------------------------------------------------------------------

if(TARGET molnex_lammps_impls)
  add_executable(test_contract_impl test_contract_impl.cpp)
  # The impls target carries its own PUBLIC include directory (interface/lammps)
  # plus molnex_lammps_bridge and molnex_interface transitively, so the cases
  # reach "impls/xxx.h", "bridge/xxx.h", "lammps_contract.h" and
  # "molnex/interface/model_runner.h" with nothing added here.
  target_link_libraries(test_contract_impl PRIVATE molnex_lammps_impls ${TORCH_LIBRARIES})
  if(TORCH_CXX_FLAGS)
    separate_arguments(_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(test_contract_impl PRIVATE ${_TORCH_CXX_FLAGS})
  endif()

  set(_CONTRACT_IMPL_TESTS
    impl_needs_matches_legacy_request
    impl_compute_composes_pipeline
    impl_compute_n_graph_local_only
    impl_compute_n_graph_with_ghost
    impl_compute_rejects_ghost_without_pbc
    impl_virial_present_and_absent
    impl_reports_no_pair_reverse_comm
    impl_energy_sums_locals_only
    impl_e_atom_locals_only
    impl_compute_reasserts_newton_invariant
    impl_coeff_token_count
    impl_coeff_rejects_trailing_garbage
    impl_coeff_rejects_empty_token
    impl_coeff_rejects_out_of_range
    impl_init_rejects_special_bonds_on_molecular
    impl_init_single_domain_condition
    impl_init_rejects_newton_off_with_pbc
    impl_init_rejects_insufficient_ghost_cutoff
    impl_init_accepts_strictly_local_default_cutoff
    impl_init_accepts_nonpbc_multilayer_default_cutoff
    impl_reverse_comm_forwards
  )

  foreach(_t IN LISTS _CONTRACT_IMPL_TESTS)
    add_test(
      NAME ${_t}
      COMMAND test_contract_impl ${_t}
    )
  endforeach()

  # Fixture-backed dispatch cases: ContractImpl::for_contract takes a
  # ModelRunner&, so both need a real exported package — the TinyLinear CPU
  # fixture, which is only held and never run (the dispatch's job is to
  # construct). Same wiring as contract_reads_generated_v3_fixture.
  set(_CONTRACT_IMPL_FIXTURE_TESTS
    registry_returns_cutoff_graph_impl
    registry_rejects_fixed_graph
  )

  foreach(_t IN LISTS _CONTRACT_IMPL_FIXTURE_TESTS)
    add_test(
      NAME ${_t}
      COMMAND test_contract_impl ${_t}
    )
    set_tests_properties(${_t} PROPERTIES
      FIXTURES_REQUIRED molnex_interface_fixtures
      ENVIRONMENT "MOLNEX_INTERFACE_FIXTURE_DIR=${_FIXTURE_DIR}"
    )
  endforeach()
else()
  message(STATUS
    "interface/tests: molnex_lammps_impls absent — impl_* / registry_* cases "
    "unlisted (expected until lammps-iface-04-facade lands the impls target)")
endif()
