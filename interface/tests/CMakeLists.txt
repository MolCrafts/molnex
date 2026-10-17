find_package(Python COMPONENTS Interpreter REQUIRED)
find_package(Threads REQUIRED)

set(_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
set(_FIXTURE_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/generate_fixtures.py")

# One executable per source; the case name is argv[1]. `_molnex_cases(exe
# FIXTURE|NONE case...)` registers one ctest per case.
function(_molnex_cases exe fixture)
  foreach(_t IN LISTS ARGN)
    add_test(NAME ${_t} COMMAND ${exe} ${_t})
    if(fixture STREQUAL "FIXTURE")
      set_tests_properties(${_t} PROPERTIES
        FIXTURES_REQUIRED molnex_interface_fixtures
        ENVIRONMENT "MOLNEX_INTERFACE_FIXTURE_DIR=${_FIXTURE_DIR}")
    endif()
  endforeach()
endfunction()

# Fixture: generate_fixtures.py packages the toy models with molix.export.aoti
# and writes their molix.deploy metadata and reference tensors.
add_test(
  NAME fixture_setup
  COMMAND "${Python_EXECUTABLE}" "${_FIXTURE_SCRIPT}" "${_FIXTURE_DIR}"
)
# molix from this checkout's src/, so the fixture does not depend on the
# caller having sourced scripts/env/activate.sh.
set_tests_properties(fixture_setup PROPERTIES
  FIXTURES_SETUP molnex_interface_fixtures
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../../src"
)

# ---------------------------------------------------------------------------
# runtime/ — molnex_runtime, engine-neutral.
# ---------------------------------------------------------------------------

foreach(_exe test_metadata test_op_libraries test_model_artifact
             test_aoti_artifact test_site_energy_exec)
  add_executable(${_exe} ${_exe}.cpp)
  target_link_libraries(${_exe} PRIVATE molnex_runtime Threads::Threads)
endforeach()

_molnex_cases(test_metadata NONE
  metadata_parses_every_field
  metadata_refuses_other_schema_version
  metadata_refuses_bad_species
  metadata_refuses_bad_receptive_field
  metadata_refuses_bad_dtype_and_cutoff
  metadata_nested_key_is_not_top_level
)
_molnex_cases(test_metadata FIXTURE metadata_reads_python_writer)

# The library OpLibraries loads in its cases: no operator, one requirement.
add_library(molnex_stub_ops SHARED op_library_stub.cpp)
# The same library under molnex's arch-tagged name (libmolnex_opLib.<machine>.so).
add_library(molnex_tagged_stub_ops SHARED op_library_stub.cpp)
set_target_properties(molnex_tagged_stub_ops PROPERTIES
  OUTPUT_NAME "molnex_tagged_stub_ops.${CMAKE_SYSTEM_PROCESSOR}")
set(_OP_CASES
  op_library_loads_matching_requirement
  op_library_refuses_other_requirement
  op_library_missing_is_refused
  op_library_path_from_environment
  op_library_finds_arch_tagged_name
)
_molnex_cases(test_op_libraries NONE ${_OP_CASES})
set_tests_properties(${_OP_CASES} PROPERTIES
  ENVIRONMENT "MOLNEX_STUB_OPS_DIR=$<TARGET_FILE_DIR:molnex_stub_ops>")
add_dependencies(test_op_libraries molnex_stub_ops molnex_tagged_stub_ops)

_molnex_cases(test_model_artifact FIXTURE
  open_aoti_from_metadata
  open_refuses_unknown_format
  open_loads_op_libraries_first
)

_molnex_cases(test_aoti_artifact FIXTURE
  cpu_run_matches_reference
  cuda_run_matches_reference
  update_weights_from_path
  update_weights_from_map
  constant_names
  concurrent_run_and_update
  load_bad_path_throws
)

_molnex_cases(test_site_energy_exec NONE
  exec_requires_four_outputs
  exec_requires_one_site_energy_per_atom
  exec_requires_shaped_forces_and_virial
  exec_casts_pos_to_model_dtype
)
_molnex_cases(test_site_energy_exec FIXTURE exec_matches_reference)

# ---------------------------------------------------------------------------
# Classical Class-I export parity (spec composable-a0-05-export).
#
# Not a C++ case: parity_classical.py writes a LAMMPS deck with
# molix.ff_export.LammpsWriter and drives a real `lmp` binary in a subprocess,
# which is what confines it to interface/ (CLAUDE.md third-party surface rule —
# no `import lammps` anywhere, this tool included). It needs neither the plugin
# nor an exported model, so it carries no FIXTURES_REQUIRED and never pulls in
# fixture_setup.
#
# SKIP_RETURN_CODE 77: unlike the neural gate, which must never skip because a
# missing plugin is the very thing it exists to police, this one is a plain
# LAMMPS run — a machine without LAMMPS has nothing to say about it and says so
# by skipping visibly rather than by passing.
#
# LMP is forwarded from the configuring environment so a site with lmp under a
# non-standard prefix needs no cmake edit; the tool falls back to $LMP and then
# to lmp on PATH.
# ---------------------------------------------------------------------------

set(_PARITY_CLASSICAL_CASES butane periodic)

foreach(_case IN LISTS _PARITY_CLASSICAL_CASES)
  add_test(
    NAME parity_classical_${_case}
    COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/parity_classical.py"
            --case ${_case}
  )
  set_tests_properties(parity_classical_${_case} PROPERTIES
    SKIP_RETURN_CODE 77
    ENVIRONMENT "LMP=$ENV{LMP}"
  )
endforeach()


# ---------------------------------------------------------------------------
# lammps/ — the LAMMPS-free layers of the LAMMPS adapter (bridge/, impls/),
# over a synthetic LammpsView: no LAMMPS build, no package, no CUDA.
# ---------------------------------------------------------------------------

add_executable(test_bridge_atoms test_bridge_atoms.cpp)
target_link_libraries(test_bridge_atoms PRIVATE molnex_lammps_bridge)
_molnex_cases(test_bridge_atoms NONE
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

add_executable(test_bridge_graph test_bridge_graph.cpp)
target_link_libraries(test_bridge_graph PRIVATE molnex_lammps_bridge)
_molnex_cases(test_bridge_graph NONE
  bridge_graph_edge_convention
  bridge_graph_cutoff_refilter
  bridge_graph_drops_beyond_n_graph
  bridge_graph_applies_neigh_mask
  bridge_graph_empty_neighbor_list
  bridge_graph_reuses_buffers_across_calls
  bridge_graph_rejects_n_graph_over_nall
  bridge_graph_folds_self_images
  bridge_graph_ghost_closure
  bridge_graph_order_shuffle_invariance
  bridge_comm_pack_reverse_layout
  bridge_comm_unpack_accumulates
)

# CutoffGraphImpl, the one pair-style implementation, over a stub artifact.
add_executable(test_contract_impl test_contract_impl.cpp)
target_link_libraries(test_contract_impl PRIVATE molnex_lammps_impls)
_molnex_cases(test_contract_impl NONE
  impl_refuses_non_angstrom_length
  impl_needs_matches_legacy_request
  impl_compute_composes_pipeline
  impl_compute_n_graph_with_ghost
  impl_reports_no_pair_reverse_comm
  impl_energy_sums_locals_only
  impl_e_atom_locals_only
  impl_compute_reasserts_newton_invariant
  impl_coeff_token_count
  impl_coeff_rejects_trailing_garbage
  impl_coeff_rejects_empty_token
  impl_coeff_rejects_out_of_range
  impl_init_rejects_special_bonds_on_molecular
  impl_init_rejects_newton_off
  impl_init_rejects_insufficient_ghost_cutoff
  impl_reverse_comm_forwards
  impl_fold_requests_no_ghost_centres
  impl_fold_refuses_several_ranks
  impl_fold_compute_folds_images_onto_owners
  impl_fold_refuses_a_foreign_ghost
)
