// A minimal molnex op library for test_op_libraries.cpp: it registers no
// operator and states the requirement it was "built" against.

extern "C" const char* molnex_op_library_requirement() {
  return "molnex_stub_ops==1.0";
}
