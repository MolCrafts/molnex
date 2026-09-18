// C++ acceptance tests for the LAMMPS vocabulary layer of the meta.json v3
// contract (specs lammps-iface-01-meta and lammps-iface-05-pbc).
//
// House style follows test_model_runner.cpp: no test framework, one function
// per case, a function-local static map dispatcher, and the [pass]/[fail]/
// [skip] stderr protocol. Every document is an in-test raw string literal, so
// these cases need NO fixtures and — crucially — no LAMMPS build: the header
// under test carries LAMMPS *vocabulary* (block name, unit styles, pair-style
// contract) but includes no LAMMPS header, and this target reaches it through a
// plain include path into interface/lammps/.
//
// API contract under test. The headers arrive with the next task and MUST
// satisfy exactly this:
//
//   #include "molnex/interface/meta_schema.h"   // neutral layer, pure std
//   #include "lammps_contract.h"                // via -I interface/lammps
//
//   namespace molnex::lammps {
//   enum class Contract { cutoff_graph, fixed_graph };     // "cutoff-graph" /
//                                                          // "fixed-graph"
//   enum class Execution { dynamic, cuda_graph };          // "dynamic" /
//                                                          // "cuda-graph"
//   struct UnitConversion { double energy; double force; };
//
//   class LammpsContract {
//    public:
//     // reads doc.block("lammps"); hard-validates schema_version == 3 and the
//     // input arity rule, throwing std::runtime_error otherwise.
//     explicit LammpsContract(const MetaSchema& doc);
//     double cutoff() const;
//     std::string units() const;                    // "real" | "metal"
//     std::string model_dtype() const;
//     Contract contract() const;                    // unknown → throw, listing
//                                                   // every valid name
//     Execution execution() const;                  // unknown → throw, listing
//                                                   // every valid name
//     UnitConversion units_for(const std::string& lammps_unit_style) const;
//     std::vector<std::string> inputs() const;
//     std::vector<std::string> outputs() const;
//     std::vector<int> species() const;             // whole array; throws on a
//                                                   // non-integer element, on
//                                                   a
//                                                   // Z <= 0, and on an empty
//                                                   // list
//     bool supports_pbc() const;
//     bool supports_virial() const;
//     int receptive_field_layers() const;           // REQUIRED, positive;
//     throws
//                                                   // naming the field
//                                                   otherwise
//     bool requires_sorted_edges() const;           // optional, default false
//     int n_atoms() const;                          // cuda-graph execution
//     only long e_max() const;                           // cuda-graph
//     execution only
//   };
//   }  // namespace molnex::lammps
//
// units_for() looks the style up in the `energy_conv` table (unknown style →
// throw listing every table key), asserts length_unit == "angstrom" (else
// throw), and only then sets force = energy. Input arity is validated at
// construction: execution "dynamic" wants 4 inputs (Z, pos, pair_index,
// local_mask), "cuda-graph" wants 5 (pair_mask inserted before local_mask).
//
// The two v3 fields differ deliberately in their defaulting.
// `receptive_field_layers` (T) has NO default: every model has a receptive
// field, and defaulting it to 1 would declare a 3-layer MACE strictly local and
// let the impl's fat-ghost check pass against a threshold three times too
// small — the exact fail-open the field exists to prevent.
// `requires_sorted_edges` defaults to false: scatter-class reductions are
// permutation-invariant, so an export that never mentions the field is saying
// "any order", which is true of every model in the tree.
//
// The contract owns a copy of the block text, so the cases below deliberately
// construct from temporary MetaSchema values that die before the accessors run.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "expect.h"
#include "lammps_contract.h"
#include "molnex/interface/meta_schema.h"

using molnex::interface::MetaSchema;
using molnex::lammps::Contract;
using molnex::lammps::Execution;
using molnex::lammps::LammpsContract;
using molnex::lammps::UnitConversion;
using molnex::testing::expect_contains;
using molnex::testing::expect_int;
using molnex::testing::expect_near;
using molnex::testing::expect_string;
using molnex::testing::expect_throws;
using molnex::testing::expect_true;

namespace {

// ---- documents ------------------------------------------------------------

// The schema-v3 `lammps` block, field by field, so one negative case can
// perturb exactly one field of an otherwise well-formed document.
struct MetaFields {
  std::string schema_version = "3";
  std::string contract = "cutoff-graph";
  std::string execution = "dynamic";
  std::string cutoff = "5.0";
  std::string units = "real";
  std::string energy_unit = "kcal/mol";
  std::string length_unit = "angstrom";
  // 1 model energy unit (kcal/mol, because units == "real") expressed in each
  // style. The metal value is the SI-exact kcal/mol → eV ratio; see the spec's
  // Domain basis. Hard-coded here on purpose: a fixture may not be computed by
  // the code under test.
  std::string energy_conv = R"({"real": 1.0, "metal": 0.0433641043})";
  std::string species = "[1, 6, 8]";
  std::string model_dtype = "float32";
  std::string inputs = R"(["Z", "pos", "pair_index", "local_mask"])";
  std::string outputs = R"(["energy", "forces", "atomic_energy", "virial"])";
  std::string adapter = "molnex-tensordict";
  std::string supports_pbc = "true";
  std::string supports_virial = "true";
  // T, the message-passing depth. EMPTY omits the key entirely, which is the
  // only way to write the "author never declared it" document — there is no
  // default to fall back on.
  std::string receptive_field_layers = "2";
  // Empty omits the key, i.e. an export that predates / never mentions the
  // flag. That document must still load, reading false.
  std::string requires_sorted_edges = "false";
  // n_atoms / e_max, emitted only when execution == "cuda-graph".
  std::string extra = "";
};

/// Render one optional key, or nothing at all when `value` is empty.
///
/// The omission has to be expressible: `receptive_field_layers` is required and
/// `requires_sorted_edges` is optional, and neither statement can be tested
/// against a document that always carries the key.
std::string optional_key(const std::string& name, const std::string& value) {
  if (value.empty()) return "";
  return ",\n    \"" + name + "\": " + value;
}

// One well-formed v3 document: the spec's block, plus the two optional keys and
// the execution-conditional ones carried in `extra`.
std::string v3_meta(const MetaFields& f) {
  return std::string(R"json({
  "device": "cuda",
  "model_class": "Sonata",
  "lammps": {
    "schema_version": )json") +
         f.schema_version + R"json(,
    "contract": ")json" +
         f.contract + R"json(",
    "execution": ")json" +
         f.execution + R"json(",
    "cutoff": )json" +
         f.cutoff + R"json(,
    "units": ")json" +
         f.units + R"json(",
    "energy_unit": ")json" +
         f.energy_unit + R"json(",
    "length_unit": ")json" +
         f.length_unit + R"json(",
    "energy_conv": )json" +
         f.energy_conv + R"json(,
    "species": )json" +
         f.species + R"json(,
    "model_dtype": ")json" +
         f.model_dtype + R"json(",
    "inputs": )json" +
         f.inputs + R"json(,
    "outputs": )json" +
         f.outputs + R"json(,
    "adapter": ")json" +
         f.adapter + R"json(",
    "supports_pbc": )json" +
         f.supports_pbc + R"json(,
    "supports_virial": )json" +
         f.supports_virial +
         optional_key("receptive_field_layers", f.receptive_field_layers) +
         optional_key("requires_sorted_edges", f.requires_sorted_edges) +
         f.extra + R"json(
  }
})json";
}

// The cuda-graph variant of the same document: 5 inputs and the fixed shapes.
MetaFields cuda_graph_fields() {
  MetaFields f;
  f.execution = "cuda-graph";
  f.inputs = R"(["Z", "pos", "pair_index", "pair_mask", "local_mask"])";
  f.supports_pbc = "false";  // ghost count varies; the pair style refuses both
  f.extra = R"json(,
    "n_atoms": 64,
    "e_max": 4096)json";
  return f;
}

// ---- tests ----------------------------------------------------------------

void test_contract_schema_version_mismatch_throws() {
  // Happy path first: the canonical document constructs, and the contract
  // outlives the MetaSchema temporary it was built from.
  const LammpsContract ok{MetaSchema{v3_meta(MetaFields{})}};
  expect_near(ok.cutoff(), 5.0, 1e-12, "cutoff()");
  expect_string(ok.units(), "real", "units()");
  expect_string(ok.model_dtype(), "float32", "model_dtype()");

  // A v1 export must be refused loudly: v1's `cuda_graph` key is gone in v2, so
  // a silent accept would run a CUDA-graph export down the dynamic path.
  MetaFields v1;
  v1.schema_version = "1";
  const std::string msg = expect_throws(
      [&] {
        const LammpsContract stale{MetaSchema{v3_meta(v1)}};
        (void)stale;
      },
      "schema_version mismatch");
  expect_contains(msg, "schema", "schema_version mismatch");
  expect_contains(msg, "3", "schema_version mismatch (expected version)");
  expect_contains(msg, "1", "schema_version mismatch (actual version)");

  // v2 is refused just as loudly, and that is the v2 -> v3 break made
  // executable: v3 renamed one input, added another and added an output, so a
  // v2 artifact accepted here would be fed a graph whose tensor list no longer
  // matches its signature. There is no migration but re-exporting.
  MetaFields v2;
  v2.schema_version = "2";
  const std::string stale_v2 = expect_throws(
      [&] {
        const LammpsContract stale{MetaSchema{v3_meta(v2)}};
        (void)stale;
      },
      "schema_version 2 is refused");
  expect_contains(stale_v2, "re-export",
                  "the v2 refusal must name the way out");
}

void test_contract_unknown_contract_throws() {
  const LammpsContract ok{MetaSchema{v3_meta(MetaFields{})}};
  expect_true(ok.contract() == Contract::cutoff_graph,
              "well-formed document must read contract == cutoff-graph");

  MetaFields bad;
  bad.contract = "banana";
  const std::string msg = expect_throws(
      [&] {
        const LammpsContract c{MetaSchema{v3_meta(bad)}};
        (void)c.contract();
      },
      "unknown contract");
  expect_contains(msg, "banana", "unknown contract");
  expect_contains(msg, "cutoff-graph",
                  "unknown contract must list every valid name");
  expect_contains(msg, "fixed-graph",
                  "unknown contract must list every valid name");
}

void test_contract_unknown_execution_throws() {
  const LammpsContract ok{MetaSchema{v3_meta(MetaFields{})}};
  expect_true(ok.execution() == Execution::dynamic,
              "well-formed document must read execution == dynamic");
  const std::vector<std::string> inputs = ok.inputs();
  expect_int(static_cast<long long>(inputs.size()), 4, "dynamic inputs()");
  expect_string(inputs[2], "pair_index", "dynamic inputs()[2]");
  // local_mask is LAST and is what makes the dynamic arity 4 under v3: it marks
  // the atoms this rank owns, and the exported energy is the site-energy sum
  // over exactly those.
  expect_string(inputs[3], "local_mask", "dynamic inputs()[3]");

  MetaFields bad;
  bad.execution = "banana";
  const std::string msg = expect_throws(
      [&] {
        const LammpsContract c{MetaSchema{v3_meta(bad)}};
        (void)c.execution();
      },
      "unknown execution");
  expect_contains(msg, "banana", "unknown execution");
  expect_contains(msg, "dynamic",
                  "unknown execution must list every valid name");
  expect_contains(msg, "cuda-graph",
                  "unknown execution must list every valid name");
}

void test_contract_unsupported_unit_style_throws() {
  const LammpsContract c{MetaSchema{v3_meta(MetaFields{})}};

  // The model is exported in `real`, so one model energy unit is 1.0 in real
  // and 0.0433641043 in metal (kcal/mol → eV, SI-exact; spec Domain basis).
  const UnitConversion self = c.units_for("real");
  expect_near(self.energy, 1.0, 1e-12, "units_for('real').energy");
  expect_near(self.force, 1.0, 1e-12, "units_for('real').force");

  const UnitConversion metal = c.units_for("metal");
  expect_near(metal.energy, 0.0433641043, 1e-6, "units_for('metal').energy");
  // [F] = [E]/[L]; both styles measure length in Å (asserted via length_unit),
  // so the force factor is the energy factor — exactly, not approximately.
  expect_near(metal.force, metal.energy, 0.0,
              "units_for('metal').force must equal .energy");

  const std::string msg =
      expect_throws([&] { (void)c.units_for("lj"); }, "unsupported unit style");
  expect_contains(msg, "lj", "unsupported unit style");
  expect_contains(msg, "real",
                  "unsupported unit style must list every table key");
  expect_contains(msg, "metal",
                  "unsupported unit style must list every table key");
}

void test_contract_length_unit_not_angstrom_throws() {
  const LammpsContract ok{MetaSchema{v3_meta(MetaFields{})}};
  expect_true(ok.supports_pbc(), "supports_pbc()");
  expect_true(ok.supports_virial(), "supports_virial()");
  const std::vector<std::string> outputs = ok.outputs();
  expect_int(static_cast<long long>(outputs.size()), 4, "outputs()");
  // The v3 output order, and the reason the decoder's virial index moved from 2
  // to 3: per-atom energy now sits between forces and virial.
  expect_string(outputs[2], "atomic_energy", "outputs()[2]");
  expect_string(outputs[3], "virial", "outputs()[3]");

  // force = energy holds only while the model's length unit matches LAMMPS'.
  // A Bohr export would otherwise be silently mis-scaled by a0 = 0.529177 Å.
  MetaFields bohr;
  bohr.length_unit = "bohr";
  const std::string msg = expect_throws(
      [&] {
        const LammpsContract c{MetaSchema{v3_meta(bohr)}};
        (void)c.units_for("metal");
      },
      "non-angstrom length unit");
  expect_contains(msg, "angstrom", "non-angstrom length unit");
}

void test_contract_input_arity_mismatch_throws() {
  // Happy path for the cuda-graph variant: 5 inputs plus the fixed shapes.
  const LammpsContract graphed{MetaSchema{v3_meta(cuda_graph_fields())}};
  expect_true(graphed.execution() == Execution::cuda_graph,
              "cuda-graph document must read execution == cuda_graph");
  const std::vector<std::string> inputs = graphed.inputs();
  expect_int(static_cast<long long>(inputs.size()), 5, "cuda-graph inputs()");
  // The two masks are adjacent and mean entirely different things —
  // `pair_mask (E_max,)` marks the real rows of a padded edge tensor,
  // `local_mask (N,)` marks the atoms this rank owns. v3 renamed the former
  // from the bare `mask` precisely because two differently-shaped booleans
  // called `mask` and `local_mask` in one list is a mis-wiring waiting to
  // happen; the order here is what the C++ buffer order must match.
  expect_string(inputs[3], "pair_mask", "cuda-graph inputs()[3]");
  expect_string(inputs[4], "local_mask", "cuda-graph inputs()[4]");
  expect_int(graphed.n_atoms(), 64, "n_atoms()");
  expect_int(graphed.e_max(), 4096, "e_max()");

  // A cuda-graph export whose inputs list lost `pair_mask` would feed the
  // static wrapper four tensors and only fail deep inside AOTI, so reject at
  // load.
  MetaFields bad = cuda_graph_fields();
  bad.inputs = R"(["Z", "pos", "pair_index", "local_mask"])";
  const std::string msg = expect_throws(
      [&] {
        const LammpsContract c{MetaSchema{v3_meta(bad)}};
        (void)c;
      },
      "input arity mismatch");
  expect_contains(msg, "input", "input arity mismatch");
  expect_contains(msg, "5", "input arity mismatch (expected count)");
  expect_contains(msg, "4", "input arity mismatch (actual count)");

  // And the dynamic side of the same rule: a v3 dynamic export that never grew
  // `local_mask` is a THREE-input model, which is exactly a stale v2 artifact
  // whose schema_version was hand-edited. Refused at load, not at the first
  // step.
  MetaFields short_dynamic;
  short_dynamic.inputs = R"(["Z", "pos", "pair_index"])";
  const std::string dyn = expect_throws(
      [&] {
        const LammpsContract c{MetaSchema{v3_meta(short_dynamic)}};
        (void)c;
      },
      "dynamic input arity mismatch");
  expect_contains(dyn, "4", "dynamic arity mismatch (expected count)");
  expect_contains(dyn, "3", "dynamic arity mismatch (actual count)");
}

void test_contract_requires_receptive_field_layers() {
  // T is READ, and it is read as an int: the fat-ghost check multiplies the
  // cutoff by it, so a document declaring 2 must not come back as 1.
  const LammpsContract ok{MetaSchema{v3_meta(MetaFields{})}};
  expect_int(ok.receptive_field_layers(), 2, "receptive_field_layers()");

  MetaFields deep;
  deep.receptive_field_layers = "3";
  expect_int(LammpsContract{MetaSchema{v3_meta(deep)}}.receptive_field_layers(),
             3, "receptive_field_layers() follows the document");

  // ABSENT IS A REFUSAL, NOT A DEFAULT. This is the whole point of the field:
  // a 3-layer MACE whose export forgot to declare T would, under any default,
  // be checked against a receptive field it does not have — the fat-ghost
  // condition would pass with a threshold up to three times too small and the
  // model would be evaluated on a truncated graph, silently.
  MetaFields undeclared;
  undeclared.receptive_field_layers = "";
  const std::string missing = expect_throws(
      [&] {
        const LammpsContract c{MetaSchema{v3_meta(undeclared)}};
        (void)c.receptive_field_layers();
      },
      "receptive_field_layers absent");
  expect_contains(missing, "receptive_field_layers",
                  "the diagnostic must name the field");

  // Zero and negative are refused for the same reason, and by this layer rather
  // than by the impl: T < 1 makes the required ghost cutoff SMALLER than the
  // model cutoff, which is a threshold that can only pass by accident.
  for (const std::string& bad : {std::string("0"), std::string("-2")}) {
    MetaFields nonpositive;
    nonpositive.receptive_field_layers = bad;
    const std::string msg = expect_throws(
        [&] {
          const LammpsContract c{MetaSchema{v3_meta(nonpositive)}};
          (void)c.receptive_field_layers();
        },
        "receptive_field_layers " + bad);
    expect_contains(msg, "receptive_field_layers",
                    "the diagnostic must name the field");
    expect_contains(msg, bad, "the diagnostic must quote the value");
  }

  // A fractional T is not a layer count. Refused rather than truncated to 1.
  MetaFields fractional;
  fractional.receptive_field_layers = "2.5";
  const std::string frac = expect_throws(
      [&] {
        const LammpsContract c{MetaSchema{v3_meta(fractional)}};
        (void)c.receptive_field_layers();
      },
      "fractional receptive_field_layers");
  expect_contains(frac, "receptive_field_layers",
                  "the diagnostic must name the field");

  // requires_sorted_edges is the OTHER half of the v3 field increment, and its
  // defaulting is deliberately the opposite: absent means false, because
  // scatter-class reductions are permutation-invariant and every model in the
  // tree is one. Only a segmented / CSR-aggregating export opts in.
  expect_true(!ok.requires_sorted_edges(),
              "an export declaring false must read false");

  MetaFields sorted;
  sorted.requires_sorted_edges = "true";
  expect_true(
      LammpsContract{MetaSchema{v3_meta(sorted)}}.requires_sorted_edges(),
      "an export declaring true must read true");

  MetaFields silent;
  silent.requires_sorted_edges = "";
  expect_true(
      !LammpsContract{MetaSchema{v3_meta(silent)}}.requires_sorted_edges(),
      "an export that never mentions the flag must read false");
}

void test_contract_species_int_array() {
  // `species` is the atomic numbers the export was built for, and it is the
  // validation source SpeciesMap checks every `pair_coeff` Z against. Reading
  // only its first element — what get_number would do when handed an array —
  // would silently accept every element past the first, which is the whole
  // reason the integer-array reader waited for this consumer.
  //
  // Five entries, deliberately not the three-element default, so a reader that
  // truncates cannot pass by coincidence; order is asserted element by element
  // because SpeciesMap's diagnostics quote the list back to the user.
  MetaFields many;
  many.species = "[1, 6, 7, 8, 16]";
  const LammpsContract c{MetaSchema{v3_meta(many)}};

  const std::vector<int> species = c.species();
  const std::vector<int> want = {1, 6, 7, 8, 16};
  expect_int(static_cast<long long>(species.size()),
             static_cast<long long>(want.size()), "species() size");
  for (std::size_t i = 0; i < want.size(); ++i) {
    expect_int(species[i], want[i], "species()[" + std::to_string(i) + "]");
  }
}

void test_contract_species_rejects_non_integer() {
  // molix writes `species` as a sorted, de-duplicated list of positive atomic
  // numbers. Anything else is a malformed export and must be refused where it
  // is read, rather than reaching SpeciesMap as a plausible-looking Z and from
  // there the model as garbage. The rejection surface is split by layer: a
  // non-integer token is the array reader's business, while "positive" and
  // "non-empty" are properties of this contract's own field.
  MetaFields fractional;
  fractional.species = "[1, 1.5, 8]";
  const std::string frac = expect_throws(
      [&] {
        const LammpsContract c{MetaSchema{v3_meta(fractional)}};
        (void)c.species();
      },
      "fractional species element");
  expect_contains(frac, "species", "fractional species element");
  expect_contains(frac, "1.5",
                  "fractional species element must quote the token");

  MetaFields alphabetic;
  alphabetic.species = R"([1, "x", 8])";
  const std::string alpha = expect_throws(
      [&] {
        const LammpsContract c{MetaSchema{v3_meta(alphabetic)}};
        (void)c.species();
      },
      "string species element");
  expect_contains(alpha, "species", "string species element");

  // A negative Z is syntactically an integer, so it passes the reader and is
  // caught by the contract: there is no element 0 and no element -6.
  MetaFields negative;
  negative.species = "[1, -6, 8]";
  const std::string neg = expect_throws(
      [&] {
        const LammpsContract c{MetaSchema{v3_meta(negative)}};
        (void)c.species();
      },
      "negative species element");
  expect_contains(neg, "species", "negative species element");
  expect_contains(neg, "-6", "negative species element must quote the value");

  // An empty list would let SpeciesMap reject every Z the user configures with
  // a message listing no valid species at all — refuse it at the source.
  MetaFields empty;
  empty.species = "[]";
  const std::string none = expect_throws(
      [&] {
        const LammpsContract c{MetaSchema{v3_meta(empty)}};
        (void)c.species();
      },
      "empty species list");
  expect_contains(none, "species", "empty species list");
}

// ---- dispatcher -----------------------------------------------------------
//
// Fixture-backed case: parses the schema-v3 meta that generate_fixtures.py
// emits through the production _lammps_meta_block, tying the Python-side
// LAMMPS_META_SCHEMA to the C++ expected version at test time. Bumping either
// constant alone turns this red. Requires MOLNEX_INTERFACE_FIXTURE_DIR.
//
// The tie is now doubled: the fixture FILENAME carries the schema version, so a
// C++ side still expecting v2 fails to open `lammps_v3.meta.json` at all, and a
// Python side still writing v2 leaves this case with no file. Only a matched
// pair goes green.
void test_contract_reads_generated_v3_fixture() {
  const char* env = std::getenv("MOLNEX_INTERFACE_FIXTURE_DIR");
  if (env == nullptr || *env == '\0') {
    throw std::runtime_error("MOLNEX_INTERFACE_FIXTURE_DIR not set");
  }
  const std::string path = std::string(env) + "/lammps_v3.meta.json";
  const auto doc = molnex::interface::MetaSchema::from_file(path);
  expect_true(doc.device() == "cpu", "fixture device must be cpu");
  const molnex::lammps::LammpsContract contract(
      doc);  // schema_version == 3 validated here
  expect_near(contract.cutoff(), 4.5, 1e-12, "fixture cutoff");
  expect_true(contract.units() == "real", "fixture units must be real");
  expect_true(contract.contract() == molnex::lammps::Contract::cutoff_graph,
              "fixture contract must be cutoff-graph");
  expect_true(contract.execution() == molnex::lammps::Execution::dynamic,
              "fixture execution must be dynamic");
  const auto conv = contract.units_for("real");
  expect_near(conv.energy, 1.0, 1e-12, "self-conversion is exactly 1");
  expect_near(conv.force, conv.energy, 0.0, "force == energy under angstrom");
  const auto outs = contract.outputs();
  expect_true(
      outs.size() == 4 && outs[2] == "atomic_energy" && outs[3] == "virial",
      "the v3 fixture must list atomic_energy before the virial");
  const auto ins = contract.inputs();
  expect_true(
      ins.size() == 4 && ins[3] == "local_mask",
      "the v3 dynamic fixture must carry local_mask as its fourth input");

  // The two v3 fields, read off the document the PRODUCTION writer emitted —
  // which is the only place the two spellings are checked against each other.
  // The fixture declares T = 2 on purpose: a fixture written with T = 1 would
  // let an impl that ignores T pass the fat-ghost check anyway, leaving this
  // tie blind to exactly the bug the field exists to prevent.
  expect_int(contract.receptive_field_layers(), 2,
             "fixture receptive_field_layers");
  expect_true(!contract.requires_sorted_edges(),
              "the fixture exports the default: no sorted-edge requirement");

  // The writer pretty-prints `species` one element per line, while every
  // in-test document above writes it on one. This is the only case that reads
  // the integer array as the exporter actually formats it.
  const std::vector<int> species = contract.species();
  expect_int(static_cast<long long>(species.size()), 3, "fixture species size");
  expect_int(species[0], 1, "fixture species[0]");
  expect_int(species[1], 6, "fixture species[1]");
  expect_int(species[2], 8, "fixture species[2]");
}

using TestFn = std::function<void()>;
const std::map<std::string, TestFn>& registry() {
  static const std::map<std::string, TestFn> r = {
      {"contract_schema_version_mismatch_throws",
       test_contract_schema_version_mismatch_throws},
      {"contract_unknown_contract_throws",
       test_contract_unknown_contract_throws},
      {"contract_unknown_execution_throws",
       test_contract_unknown_execution_throws},
      {"contract_unsupported_unit_style_throws",
       test_contract_unsupported_unit_style_throws},
      {"contract_length_unit_not_angstrom_throws",
       test_contract_length_unit_not_angstrom_throws},
      {"contract_input_arity_mismatch_throws",
       test_contract_input_arity_mismatch_throws},
      {"contract_species_int_array", test_contract_species_int_array},
      {"contract_species_rejects_non_integer",
       test_contract_species_rejects_non_integer},
      {"contract_requires_receptive_field_layers",
       test_contract_requires_receptive_field_layers},
      {"contract_reads_generated_v3_fixture",
       test_contract_reads_generated_v3_fixture},
  };
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <test_name>\n", argv[0]);
    std::fprintf(stderr, "available:\n");
    for (const auto& [k, _] : registry())
      std::fprintf(stderr, "  %s\n", k.c_str());
    return 2;
  }
  std::string name = argv[1];
  auto it = registry().find(name);
  if (it == registry().end()) {
    std::fprintf(stderr, "unknown test: %s\n", name.c_str());
    return 2;
  }
  try {
    it->second();
    std::fprintf(stderr, "[pass] %s\n", name.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[fail] %s: %s\n", name.c_str(), e.what());
    return 1;
  }
}
