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
