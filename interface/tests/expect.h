// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// Shared assertion helpers for the interface ctest cases.
//
// The house style for interface/tests is deliberately framework-free (see
// test_aoti_artifact.cpp): one function per case, a function-local static map
// dispatcher, and the [pass]/[fail]/[skip] stderr protocol, with a failed
// expectation signalled by throwing std::runtime_error out of the case. These
// six helpers are that protocol's vocabulary.
//
// Test-only, and scoped as such: this header sits in interface/tests/, not
// under a library's include/, so it is not part of the installed surface.
//
// Dependencies: pure std. Nothing here knows about torch, LAMMPS, or the
// export JSON, so it can be included by a bridge case that links no
// library at all as readily as by a contract case.

#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

namespace molnex::testing {

/// Require a condition, reporting `tag` when it does not hold.
///
/// \param cond  Condition that must be true.
/// \param tag   Message identifying the expectation.
/// \throws std::runtime_error  If `cond` is false.
inline void expect_true(bool cond, const std::string& tag) {
  if (!cond) throw std::runtime_error(tag);
}

/// Require string equality, quoting both sides on failure.
///
/// \param got   Observed value.
/// \param want  Expected value.
/// \param tag   Message identifying the expectation.
/// \throws std::runtime_error  If the two differ.
inline void expect_string(const std::string& got, const std::string& want,
                          const std::string& tag) {
  if (got != want) {
    throw std::runtime_error(tag + ": expected '" + want + "', got '" + got +
                             "'");
  }
}

/// Require integer equality.
///
/// Takes `long long` so any integral width (including a tensor's `int64_t`
/// element or a `std::size_t` count, cast at the call site) compares without a
/// second overload.
///
/// \param got   Observed value.
/// \param want  Expected value.
/// \param tag   Message identifying the expectation.
/// \throws std::runtime_error  If the two differ.
inline void expect_int(long long got, long long want, const std::string& tag) {
  if (got != want) {
    throw std::runtime_error(tag + ": expected " + std::to_string(want) +
                             ", got " + std::to_string(got));
  }
}

/// Require two doubles to agree within an absolute tolerance.
///
/// \param got   Observed value, in whatever unit its own field carries.
/// \param want  Expected value, same unit.
/// \param tol   Absolute tolerance; 0.0 demands exact equality.
/// \param tag   Message identifying the expectation.
/// \throws std::runtime_error  If |got - want| > tol.
inline void expect_near(double got, double want, double tol,
                        const std::string& tag) {
  if (std::fabs(got - want) > tol) {
    throw std::runtime_error(tag + ": expected " + std::to_string(want) +
                             ", got " + std::to_string(got));
  }
}

/// Run `fn`, require it to throw a std::exception, and hand back its `what()`.
///
/// Returning the message rather than swallowing it is what lets a case assert
/// on the diagnostic text as well as on the fact of the throw.
///
/// \param fn   Callable expected to throw.
/// \param tag  Message identifying the expectation.
/// \returns    The caught exception's `what()`.
/// \throws std::runtime_error  If `fn` returns without throwing.
inline std::string expect_throws(const std::function<void()>& fn,
                                 const std::string& tag) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  throw std::runtime_error(tag + ": expected an exception, none thrown");
}

/// Require an error message to mention `needle`.
///
/// \param message  Text produced by the code under test.
/// \param needle   Substring the diagnostic must contain.
/// \param tag      Message identifying the expectation.
/// \throws std::runtime_error  If `needle` does not occur in `message`.
inline void expect_contains(const std::string& message,
                            const std::string& needle, const std::string& tag) {
  if (message.find(needle) == std::string::npos) {
    throw std::runtime_error(tag + ": message \"" + message +
                             "\" does not mention '" + needle + "'");
  }
}

}  // namespace molnex::testing
