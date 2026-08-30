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
)

foreach(_t IN LISTS _LAMMPS_CONTRACT_TESTS)
  add_test(
    NAME ${_t}
    COMMAND test_lammps_contract ${_t}
  )
endforeach()
