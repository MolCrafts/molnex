// Copyright (c) MolNex contributors. SPDX-License-Identifier: MIT
//
// LAMMPS vocabulary layer over the meta.json schema-v3 contract.
//
// Everything LAMMPS-flavoured about a molnex export lives here: the `lammps`
// block name, the unit styles, the pair-style contract and its execution
// strategy. The layer below (molnex/interface/meta_schema.h) stays engine
// neutral and knows none of it.
//
// This header includes **no LAMMPS header** — not `pair.h`, not `update.h` —
// and no LibTorch. It carries LAMMPS *concepts*, not LAMMPS *types*, so the
// whole contract is exercisable by ctest on a machine with no LAMMPS build.
// The pair style keeps the two dialects apart as before: this layer throws
// std::runtime_error, and pair_molnex.cpp translates at the seam into
// error->all(FLERR, ...).
//
// Design notes that outlive any one call site:
//
//   * `Execution` selects a *strategy*, never a backend. The closed spec
//     lammps-pair-molnex-01-aoti-force-export-wall established that AOTI
//     cannot export cuEquivariance force graphs and that the recorded route
//     is a TorchScript + libtorch runtime backward, which will arrive as a
//     third `torchscript` value. Nothing in this header — no field name, no
//     enumerator, no error text — may hardwire an AOTI assumption; `.pt2` and
//     AOTI belong in provenance comments only. Adding that value is a
//     schema_version bump to 3, not a silent widening of the enum.
//
//   * `Contract` selects *where the edge set comes from* (recomputed from a
//     cutoff each step, versus handed over as a fixed topology), which is why
//     it is not a model name and does not breach "no per-model C++". Its
//     binding consumer is the impl dispatch of lammps-iface-04-facade; this
//     header is only that pattern's first use. `fixed-graph`'s future payload
//     is `bond_index`, a `(2, N)` COO covalent topology — never `pair_index`,
//     the `(E, 2)` cutoff neighbour graph. CLAUDE.md gives those two objects
//     deliberately different shapes precisely so neither can ever be silently
//     used as the other; this header must preserve that anti-alias guard when
//     the payload lands. Until an impl exists, reading `fixed-graph` must fail
//     loudly upstream rather than fall back to the cutoff path.
//
//   * `force = energy` in `UnitConversion` is licensed by exactly one thing:
//     the `length_unit == "angstrom"` assertion in `units_for`. [F] = [E]/[L],
//     so the identity holds only while the model's length unit equals LAMMPS'.
//     A Bohr export would otherwise be silently mis-scaled by a0 = 0.529177 Å.
//     Never return a UnitConversion that skipped that assertion.
//
// SCHEMA v3, AND v2 IS A HARD BREAK. The field increment over v2 is:
//
//   * input `local_mask (N,) bool` — which atoms this rank owns. The exported
//     energy is `(atomic_energy * local_mask).sum()`, so the dynamic arity goes
//     3 -> 4.
//   * the cuda-graph edge-validity mask is renamed `mask` -> `pair_mask`, and
//     `local_mask` follows it, so that arity goes 4 -> 5. Two differently
//     shaped booleans in one list, one called `mask` and one `local_mask`, is a
//     mis-wiring waiting to happen.
//   * output `atomic_energy (N,)` — the per-site energies, inserted BEFORE the
//     virial. `virial` therefore moves from output index 2 to index 3.
//   * `receptive_field_layers` (T) — required, positive, no default.
//   * `requires_sorted_edges` — optional, default false.
//
// Every export produced before v3 is invalid: its tensor count, names and
// output order all differ, so a silent accept would feed a graph whose
// signature no longer matches. The version check refuses it and names the
// re-export, which is the only migration there is.
//
// Header-only, so the plugin links nothing beyond libmolnex_interface.

#ifndef MOLNEX_LAMMPS_CONTRACT_H
#define MOLNEX_LAMMPS_CONTRACT_H

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "molnex/interface/meta_schema.h"

namespace molnex::lammps {

/// Where the edge set the model consumes comes from.
///
/// An *edge* is one ordered pair of atoms the network exchanges messages along.
/// Spelled `"cutoff-graph"` / `"fixed-graph"` in meta.json. As in the header
/// notes: the binding consumer of this dispatch pattern is the impl selection
/// of lammps-iface-04-facade, and `fixed_graph` will carry `bond_index`, a
/// `(2, N)` COO covalent topology — never `pair_index`, the `(E, 2)` cutoff
/// neighbour graph.
enum class Contract {
  cutoff_graph,  ///< Neighbour graph rebuilt every step from a cutoff radius:
                 ///< pairs farther apart than `cutoff()` angstrom are dropped.
  fixed_graph,   ///< Edges given by a fixed covalent topology. No impl yet.
};

/// How the model is executed once its inputs are staged.
///
/// Spelled `"dynamic"` / `"cuda-graph"` in meta.json. A strategy selector, not
/// a backend selector: the anticipated third value is `torchscript` (see the
/// header notes for why that route is on record), and adding it is a
/// schema_version bump to 3, never a silent widening of this enum.
enum class Execution {
  dynamic,     ///< Variable shapes, one call per step.
  cuda_graph,  ///< Fixed shapes captured into a CUDA graph — a recorded,
               ///< replayable sequence of GPU kernels, which can only be taken
               ///< over static shapes at fixed device addresses.
};

/// Multiplicative factors taking model units to the active LAMMPS unit style.
///
/// Attribute-only value type (same shape as molix.md.types.ForceOutput): the
/// factors are the whole content, there is no behaviour to attach. `force`
/// equals `energy` only because `units_for` asserted a matching length unit.
struct UnitConversion {
  double energy;  ///< Dimensionless: multiply a model energy by this to get
                  ///< the same energy counted in the LAMMPS unit style.
  double force;   ///< Dimensionless: same, for a force. Equal to `energy`
                  ///< only under the length_unit == "angstrom" assertion.
};

namespace detail {

/// Join names for a "valid ..." error message.
inline std::string join(const std::vector<std::string>& items) {
  std::string joined;
  for (const std::string& item : items) {
    if (!joined.empty()) joined += ", ";
    joined += item;
  }
  return joined;
}

/// Map a meta.json spelling onto a closed enum, or throw listing every name.
///
/// Explicit table lookup, deliberately without static self-registration: the
/// set of valid names is readable in one place and the failure message can
/// enumerate all of it (error-text style follows molix.engine.EngineAdapter's
/// "valid adapters: ..." message).
///
/// \param value  Spelling read from meta.json.
/// \param table  Every valid spelling.
/// \param what   Singular noun for the message ("contract", "execution").
/// \returns      The mapped enumerator.
/// \throws std::runtime_error  If `value` is not in `table`.
template <typename E>
E parse_enum(const std::string& value, const std::map<std::string, E>& table,
             const std::string& what) {
  const auto it = table.find(value);
  if (it != table.end()) return it->second;
  std::vector<std::string> names;
  names.reserve(table.size());
  for (const auto& entry : table) names.push_back(entry.first);
  throw std::runtime_error("meta.json: unknown " + what + " '" + value +
                           "'; valid " + what + "s: " + detail::join(names));
}

}  // namespace detail

/// The `lammps` block of a schema-v3 meta.json, validated and typed.
///
/// Construction hard-validates the two invariants that cannot be recovered
/// from later:
///
///  1. `schema_version == 3`. Older exports must be refused loudly: v1's
///     discriminator (`cuda_graph`, a bool) no longer exists, and v2's tensor
///     lists are a different length with a different output order, so a silent
///     accept would run a mis-wired graph.
///  2. Input arity against `execution` — 4 for `dynamic`, 5 for `cuda-graph`
///     (the static wrapper's extra `pair_mask`). An export whose `inputs` list
///     lost one would otherwise fail deep inside the runtime instead of at
///     load.
///
/// The contract keeps its **own copy** of the block text, so it does not
/// dangle when the `MetaSchema` it was built from goes away. meta.json is a
/// few hundred bytes; the copy is cheaper than the lifetime coupling.
///
/// Example:
/// \code
///   const LammpsContract contract{MetaSchema::from_file(meta_path)};
///   const UnitConversion conv = contract.units_for(update->unit_style);
/// \endcode
class LammpsContract {
 public:
  /// Read and validate the `lammps` block of `doc`.
  ///
  /// \param doc  A whole meta.json document. Only its `lammps` block is read;
  ///             `doc` need not outlive this object.
  /// \throws std::runtime_error  If the block is absent, its `schema_version`
  ///                             is not 3, its `execution` is unknown, or its
  ///                             input arity disagrees with that execution.
  explicit LammpsContract(const molnex::interface::MetaSchema& doc)
      : block_(doc.block("lammps")) {
    const long long version =
        static_cast<long long>(block_.get_number("schema_version"));
    if (version != kSchemaVersion) {
      throw std::runtime_error(
          "meta.json: lammps block has schema_version " +
          std::to_string(version) + ", this plugin requires schema_version " +
          std::to_string(kSchemaVersion) +
          " (re-export the model with molix.engine.export_for_lammps)");
    }

    const std::size_t declared = inputs().size();
    const std::size_t required = execution() == Execution::cuda_graph
                                     ? kCudaGraphInputs
                                     : kDynamicInputs;
    if (declared != required) {
      throw std::runtime_error(
          "meta.json: lammps block lists " + std::to_string(declared) +
          " input(s), but execution '" + block_.get_string("execution") +
          "' requires " + std::to_string(required) +
          " (a wrong input arity only surfaces deep inside the runtime, so it "
          "is refused at load)");
    }
  }

  /// Neighbour-list cutoff the model was exported with, in Å.
  ///
  /// Atom pairs farther apart than this contribute nothing, so the pair style
  /// builds `pair_index` from exactly the pairs inside it.
  double cutoff() const { return block_.get_number("cutoff"); }

  /// LAMMPS unit style the model's own energies are expressed in.
  ///
  /// A *unit style* is LAMMPS' name for a whole system of units chosen at the
  /// start of a run; the model declares its own, and the pair style converts
  /// between it and the one the running simulation asked for.
  ///
  /// \returns `"real"` (kcal/mol, Å, fs) or `"metal"` (eV, Å, ps).
  std::string units() const { return block_.get_string("units"); }

  /// Floating-point type the exported model computes in (e.g. `"float32"`).
  std::string model_dtype() const { return block_.get_string("model_dtype"); }

  /// Where the model's edge set comes from.
  ///
  /// A caller that reads `Contract::fixed_graph` must refuse loudly: no impl
  /// exists, and running such an export down the cutoff path would silently
  /// substitute a different edge set — and so different physics.
  ///
  /// \returns The parsed contract.
  /// \throws std::runtime_error  If the spelling is not a known contract; the
  ///                             message lists every valid name.
  Contract contract() const {
    static const std::map<std::string, Contract> kContracts = {
        {"cutoff-graph", Contract::cutoff_graph},
        {"fixed-graph", Contract::fixed_graph},
    };
    return detail::parse_enum(block_.get_string("contract"), kContracts,
                              "contract");
  }

  /// Execution strategy for the exported model.
  ///
  /// \returns The parsed strategy.
  /// \throws std::runtime_error  If the spelling is not a known execution; the
  ///                             message lists every valid name.
  Execution execution() const {
    static const std::map<std::string, Execution> kExecutions = {
        {"cuda-graph", Execution::cuda_graph},
        {"dynamic", Execution::dynamic},
    };
    return detail::parse_enum(block_.get_string("execution"), kExecutions,
                              "execution");
  }

  /// Conversion factors from model units to the given LAMMPS unit style.
  ///
  /// The factor is *read*, never computed here: `energy_conv` maps each unit
  /// style onto the value of one model energy unit in that style, and the
  /// export side fills it with molix.units (pint). No conversion constant is
  /// restated in C++.
  ///
  /// Then, and only then, `force` is set equal to `energy`: [F] = [E]/[L], so
  /// the identity needs the model's length unit to match LAMMPS'. It is
  /// asserted rather than assumed — a Bohr export must fail, not be mis-scaled
  /// by a0 = 0.529177 Å. A non-Å export needs schema v3 and a length table,
  /// which is deliberately not pre-built here.
  ///
  /// \param lammps_unit_style  LAMMPS' active unit style
  /// (`update->unit_style`).
  /// \returns Energy and force factors for that style.
  /// \throws std::runtime_error  If the style has no usable entry in the
  ///                             `energy_conv` table (the message lists every
  ///                             style this contract serves), or if the model's
  ///                             `length_unit` is not `"angstrom"`.
  UnitConversion units_for(const std::string& lammps_unit_style) const {
    const molnex::interface::MetaSchema table = block_.block("energy_conv");
    double energy = 0.0;
    try {
      energy = table.get_number(lammps_unit_style);
    } catch (const std::exception&) {
      throw std::runtime_error(
          "meta.json: no energy conversion for LAMMPS unit style '" +
          lammps_unit_style + "'; valid styles: " + unit_style_names());
    }

    const std::string length_unit = block_.get_string("length_unit");
    if (length_unit != kLengthUnit) {
      throw std::runtime_error(
          "meta.json: model length_unit '" + length_unit + "' is not '" +
          kLengthUnit +
          "'; force and energy share a conversion factor only "
          "while the model measures length in angstrom (LAMMPS 'real' and "
          "'metal' both do), so this export cannot be run safely");
    }
    return UnitConversion{energy, energy};
  }

  /// Names of the tensors the exported model takes, in call order.
  ///
  /// For N atoms and E edges: `Z (N,)` int64 atomic numbers, `pos (N, 3)`
  /// positions in angstrom, `pair_index (E, 2)` int64 with column 0 the source
  /// atom and column 1 the target, then — under `Execution::cuda_graph` only —
  /// `pair_mask (E_max,)` bool marking which of the padded edge slots are real,
  /// and finally `local_mask (N,)` bool marking the atoms this rank owns.
  ///
  /// `local_mask` is last under both strategies, so its position differs
  /// between them; that is why the arity is validated at construction rather
  /// than left to a positional read downstream.
  ///
  /// \returns The names as written at export; the shapes above are the contract
  ///          the caller must honour when it stages them.
  std::vector<std::string> inputs() const {
    return block_.get_string_array("inputs");
  }

  /// Names of the tensors the exported model returns, in call order.
  ///
  /// `energy ()` a scalar, `forces (N, 3)` and `atomic_energy (N,)`, all in the
  /// model's own units (see `units()`), plus `virial (6,)` ordered
  /// (xx, yy, zz, xy, xz, yz) when `supports_virial()`. Multiply energies by
  /// `UnitConversion::energy` and forces by `UnitConversion::force` to hand
  /// them to LAMMPS.
  ///
  /// THE VIRIAL IS OUTPUT THREE, not two: v3 inserted `atomic_energy` ahead of
  /// it. A reader still using the v2 index would take an `(N,)` per-atom energy
  /// for a `(6,)` virial, which the length guards catch at every N except 6.
  std::vector<std::string> outputs() const {
    return block_.get_string_array("outputs");
  }

  /// Atomic numbers the exported model was built for (Z, dimensionless).
  ///
  /// The whole list, in the order molix wrote it — which is sorted and
  /// de-duplicated on the export side, though nothing here depends on that.
  /// Its consumer is the construction of the pair style's type-to-Z table,
  /// which checks every `pair_coeff` element against this set: a Z outside it
  /// is an element the model was never trained on, and feeding it forward
  /// produces a confident, wrong energy with no diagnostic anywhere.
  ///
  /// Reading the array whole matters for the same reason. Truncating it to the
  /// first element would leave the table rejecting every legal type but one.
  ///
  /// \returns The atomic numbers, in document order; never empty.
  /// \throws std::runtime_error  If `species` is absent, is not an array, holds
  ///                             an element that is not a whole integer, holds
  ///                             a Z <= 0, or is empty.
  std::vector<int> species() const {
    const std::vector<int> values = block_.get_int_array("species");
    if (values.empty()) {
      throw std::runtime_error(
          "meta.json: lammps block has an empty 'species' list; the export "
          "declares no element it can be run on (re-export the model with "
          "molix.engine.export_for_lammps)");
    }
    for (const int z : values) {
      if (z <= 0) {
        throw std::runtime_error(
            "meta.json: lammps block 'species' contains " + std::to_string(z) +
            ", which is not an atomic number (every entry must be > 0)");
      }
    }
    return values;
  }

  /// Whether the model accepts ghost atoms / periodic images.
  ///
  /// *Ghost atoms* are LAMMPS' local copies of atoms owned by a periodic image
  /// or by a neighbouring MPI rank; a model that accepts them can be run with
  /// periodic boundaries and across several ranks, with the forces landing on
  /// ghosts reverse-communicated to their owners.
  ///
  /// \returns False when the flag is absent — an export that never declared
  ///          the capability does not have it.
  bool supports_pbc() const { return block_.get_bool("supports_pbc", false); }

  /// Whether the model returns a virial output alongside energy and forces.
  ///
  /// The *virial* is the 6-component stress-like tensor LAMMPS accumulates into
  /// the system pressure. When the model does not provide it, LAMMPS can still
  /// fall back to its own force-times-distance sum.
  ///
  /// \returns False when the flag is absent.
  bool supports_virial() const {
    return block_.get_bool("supports_virial", false);
  }

  /// Number of message-passing layers T the model was built with (a count).
  ///
  /// A site's energy in a T-layer network depends on every atom within
  /// `T * cutoff()`, so a single-shot evaluation needs a local copy of all of
  /// them — the *fat-ghost* condition
  /// `ghost cutoff >= T * cutoff + skin`, checked by the impl before any run.
  ///
  /// REQUIRED, AND WITH NO DEFAULT, which is the one design decision worth
  /// stating here. Defaulting an absent field to 1 would declare a 3-layer MACE
  /// strictly local; the fat-ghost check would then pass against a threshold
  /// three times too small, and the model would be evaluated every step on a
  /// graph whose outer sites are missing neighbours — no error, no warning,
  /// plausible-looking forces. Refusing to guess is the whole value of the
  /// field, so absence is a hard failure here and the export side likewise
  /// refuses to write one without it.
  ///
  /// \returns T, always >= 1.
  /// \throws std::runtime_error  If the field is absent, is not a number, is
  ///                             not a whole number, or is less than 1; the
  ///                             message names the field and the re-export.
  int receptive_field_layers() const {
    double declared = 0.0;
    try {
      declared = block_.get_number(kReceptiveFieldLayers);
    } catch (const std::exception&) {
      throw std::runtime_error(
          std::string("meta.json: lammps block has no '") +
          kReceptiveFieldLayers +
          "'; it is the number of message-passing layers T, and the pair style "
          "needs it to check that the ghost cutoff covers T * cutoff + skin "
          "(there is deliberately no default: guessing 1 would declare a "
          "multi-layer model strictly local and let that check pass against a "
          "threshold several times too small). Re-export with "
          "molix.engine.export_for_lammps(..., receptive_field_layers=T)");
    }
    const int layers = static_cast<int>(declared);
    if (static_cast<double>(layers) != declared || layers < 1) {
      throw std::runtime_error(
          std::string("meta.json: lammps block '") + kReceptiveFieldLayers +
          "' is " + std::to_string(declared) +
          "; it must be a whole number of message-passing layers, at least 1 "
          "(re-export with molix.engine.export_for_lammps(..., "
          "receptive_field_layers=T))");
    }
    return layers;
  }

  /// Whether the model needs `pair_index` sorted by `(source, target)`.
  ///
  /// Scatter-class reductions (`index_add_` / `scatter_add_`, which is every
  /// aggregation in this tree today) are permutation-invariant, so the builder
  /// pays no per-step sort by default. Segmented / CSR aggregation is the
  /// exception: it requires rows grouped by index, and a model using it is
  /// simply WRONG on unordered edges — quietly, since the arithmetic still
  /// completes.
  ///
  /// \returns False when the flag is absent — an export that never declared the
  ///          requirement does not have it, which is true of every model in the
  ///          tree. Opting in is the segmented-aggregation author's job.
  bool requires_sorted_edges() const {
    return block_.get_bool("requires_sorted_edges", false);
  }

  /// Fixed atom count N the graph was captured for (a count, no units).
  ///
  /// Emitted only when `execution() == Execution::cuda_graph`.
  ///
  /// \throws std::runtime_error  If the field is absent (a dynamic export).
  int n_atoms() const { return static_cast<int>(block_.get_number("n_atoms")); }

  /// Fixed edge capacity E the graph was captured for (a count, no units).
  ///
  /// The caller must always pass exactly this many edge rows: the real edges
  /// first with `pair_mask` true, the remaining slots padded and masked off
  /// (the export makes them inert by giving them an over-cutoff displacement),
  /// and it must fail loudly when the real edge count exceeds this capacity.
  ///
  /// Emitted only when `execution() == Execution::cuda_graph`.
  ///
  /// \throws std::runtime_error  If the field is absent (a dynamic export).
  long e_max() const { return static_cast<long>(block_.get_number("e_max")); }

 private:
  /// LAMMPS unit styles this contract serves, for the `units_for` error.
  ///
  /// Names only — every value still comes from the meta's `energy_conv` table.
  /// The set mirrors the export side's unit labels; a new style is added there
  /// (so the table gains a key) and here (so the message stays truthful).
  static std::string unit_style_names() {
    static const std::vector<std::string> kStyles = {"real", "metal"};
    return detail::join(kStyles);
  }

  /// Must equal molix.engine.export.LAMMPS_META_SCHEMA on the writer side; the
  /// ctest that parses a generated fixture is what keeps the two in step.
  static constexpr long long kSchemaVersion = 3;
  /// Z (N,), pos (N, 3) in angstrom, pair_index (E, 2), local_mask (N,) bool.
  static constexpr std::size_t kDynamicInputs = 4;
  /// ... with pair_mask (E_max,) bool, marking the real edges among the padded
  /// slots, inserted before local_mask.
  static constexpr std::size_t kCudaGraphInputs = 5;
  static constexpr const char* kLengthUnit = "angstrom";
  /// Spelled once: the name appears in three diagnostics that must agree.
  static constexpr const char* kReceptiveFieldLayers = "receptive_field_layers";

  molnex::interface::MetaSchema block_;
};

}  // namespace molnex::lammps

#endif  // MOLNEX_LAMMPS_CONTRACT_H
