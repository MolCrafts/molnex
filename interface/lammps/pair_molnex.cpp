/* PairMolnex implementation — see pair_molnex.h.

   Protocol mechanics only. Each callback below does one of four things: pipe a
   command-line argument, hand LAMMPS a number the impl produced, forward a call
   to the impl, or translate a std::runtime_error into error->all. There is no
   branch here on what the model can do — every such branch moved into
   ContractImpl, and what is left branches only on what LAMMPS asked for
   (eflag/vflag) or on a value the impl returned.

   Error dialect: the impls/ and bridge/ layers own the user-facing text, and
   most of their messages already name the pair style or the component that
   refused. This file therefore forwards `e.what()` unprefixed rather than
   pasting a second "pair_style molnex:" in front of a message that already
   carries one; FLERR still points at the callback that translated it. The
   runtime's messages ("metadata: ...", a missing op library, a missing
   package) are prefixed with "pair_style molnex:".
*/

#include "pair_molnex.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <c10/core/ScalarType.h>

#include "bridge/lammps_view.h"
#include "impls/cutoff_graph_impl.h"
#include "molnex/runtime/metadata.h"
#include "molnex/runtime/model_artifact.h"
#include "molnex/runtime/atom_energy_exec.h"

namespace fs = std::filesystem;

using namespace LAMMPS_NS;

PairMolnex::PairMolnex(LAMMPS* lmp) : Pair(lmp) {
  single_enable = 0;  // no pairwise single() — this is a many-body model
  restartinfo = 0;
  one_coeff = 1;  // requires a single `pair_coeff * *`
  manybody_flag = 1;
  model_name = "model";
  // Placeholder sizes only. The real ones arrive with the impl's NeighborNeeds
  // in settings(); LAMMPS reads these members directly, so they must hold
  // something valid from construction on.
  comm_reverse = 3;
  comm_reverse_off = 3;
}

PairMolnex::~PairMolnex() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairMolnex::allocate() {
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;
  // LAMMPS' own per-type cutoff table, not a molnex value: Pair::init() writes
  // cutsq[i][j] = init_one(i,j)^2 into it unconditionally (pair.cpp), so every
  // pair style must provide the storage. Nothing in this plugin ever reads it.
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

/* pair_style molnex <model_dir> [name] [mode fold|ghost] */
void PairMolnex::settings(int narg, char** arg) {
  constexpr const char* kUsage =
      "Illegal pair_style molnex command: pair_style molnex <model_dir> "
      "[name] [mode fold|ghost]";
  if (narg < 1) error->all(FLERR, kUsage);
  model_dir = arg[0];
  int iarg = 1;
  if (iarg < narg && strcmp(arg[iarg], "mode") != 0) model_name = arg[iarg++];
  // The default: fold on one rank, where every ghost is a periodic image of an
  // owned atom; ghost staging otherwise.
  fold_ = comm->nprocs == 1;
  if (iarg < narg) {
    if (strcmp(arg[iarg], "mode") != 0 || iarg + 2 != narg)
      error->all(FLERR, kUsage);
    if (strcmp(arg[iarg + 1], "fold") == 0) {
      fold_ = true;
    } else if (strcmp(arg[iarg + 1], "ghost") == 0) {
      fold_ = false;
    } else {
      error->all(FLERR, "pair_style molnex: mode must be fold or ghost, got {}",
                 arg[iarg + 1]);
    }
  }
  if (fold_ && atom->tag_enable == 0)
    error->all(FLERR,
               "pair_style molnex: mode fold maps each periodic image to the "
               "atom it copies by atom ID, and this system has none "
               "(atom_modify id no); use mode ghost");
  // Folded forces sit on owned atoms only, so LAMMPS' force-times-position
  // virial would miss every periodic image: the model's virial is tallied in
  // compute() instead.
  no_virial_fdotr_compute = fold_ ? 1 : 0;

  // Read <model_dir>/<name>.json, then open the package in the format it
  // names, now, so a bad export fails at pair_style rather than at step 1.
  // `artifact_` is assigned before `impl_`, which borrows it.
  molnex::runtime::Metadata meta;
  try {
    meta = molnex::runtime::Metadata::read(
        (fs::path(model_dir) / (model_name + ".json")).string());
    artifact_ =
        molnex::runtime::ModelArtifact::open(model_dir, model_name, meta);
  } catch (const std::exception& e) {
    error->all(FLERR, "pair_style molnex: {}", e.what());
  }

  const double factor = energy_factor(meta.energy_unit);
  try {
    impl_ = std::make_unique<molnex::lammps::CutoffGraphImpl>(
        meta, factor, molnex::runtime::AtomEnergyExec(*artifact_, meta.dtype),
        fold_ ? molnex::lammps::GraphMode::kFold
              : molnex::lammps::GraphMode::kGhost);
  } catch (const std::exception& e) {
    error->all(FLERR, "{}", e.what());
  }

  // The request table, asked once and executed verbatim from here on.
  needs_ = impl_->needs();
  comm_reverse = needs_.comm_reverse;
  comm_reverse_off = needs_.comm_reverse_off;

  if (comm->me == 0)
    utils::logmesg(lmp,
                   "pair_molnex: loaded '{}' from {} (format={}, device={}, "
                   "dtype={}, model energy={}, LAMMPS units={}, factor={}, "
                   "cutoff={:.3f} A, layers={}, mode={})\n",
                   model_name, model_dir, meta.format,
                   artifact_->device().str(), c10::toString(meta.dtype),
                   meta.energy_unit, update->unit_style, factor, needs_.cutoff,
                   meta.receptive_field_layers, fold_ ? "fold" : "ghost");
}

double PairMolnex::energy_factor(const std::string& model_energy_unit) const {
  // The LAMMPS unit style whose energy unit the model uses. LAMMPS converts
  // between real and metal only (utils::get_conversion_factor), so those are
  // the two the model can be run in.
  std::string model_style;
  if (model_energy_unit == "eV") model_style = "metal";
  if (model_energy_unit == "kcal/mol") model_style = "real";
  if (model_style.empty())
    error->all(FLERR,
               "pair_style molnex: model energy unit '{}' has no LAMMPS unit "
               "style; eV (metal) and kcal/mol (real) are supported",
               model_energy_unit);

  const std::string style = update->unit_style;
  if (style == model_style) return 1.0;
  if (model_style == "metal" && style == "real")
    return utils::get_conversion_factor(utils::ENERGY, utils::METAL2REAL);
  if (model_style == "real" && style == "metal")
    return utils::get_conversion_factor(utils::ENERGY, utils::REAL2METAL);
  error->all(FLERR,
             "pair_style molnex: the model's energy is in {} ({} units), and "
             "LAMMPS converts it only to real or metal; this run uses units {}",
             model_energy_unit, model_style, style);
  return 0.0;
}

/* pair_coeff * * <Z_type1> <Z_type2> ... <Z_typeN> */
void PairMolnex::coeff(int narg, char** arg) {
  if (!allocated) allocate();
  if (narg < 2 || strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff for molnex must be: pair_coeff * * ...");
  if (!impl_)
    error->all(
        FLERR,
        "pair_coeff for molnex must come after pair_style molnex: the elements "
        "the model was exported for are read from its metadata, and every Z "
        "here is checked against them");

  // The two literal `*` are the whole of what this file understands about a
  // pair_coeff line. What the rest of the tokens MEAN — how many there must be,
  // how each is parsed, which values the export accepts — is the contract's
  // business, so they travel over the seam exactly as the input script wrote
  // them.
  std::vector<std::string> tokens;
  tokens.reserve(static_cast<std::size_t>(narg - 2));
  for (int i = 2; i < narg; i++) tokens.emplace_back(arg[i]);

  try {
    impl_->coeff(tokens, atom->ntypes);
  } catch (const std::exception& e) {
    error->all(FLERR, "{}", e.what());
  }

  const int ntypes = atom->ntypes;
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 1;
}

double PairMolnex::init_one(int /*i*/, int /*j*/) { return needs_.cutoff; }

void PairMolnex::init_style() {
  // The request table, executed verbatim. Ghosts are always requested.
  int flags = needs_.full ? NeighConst::REQ_FULL : NeighConst::REQ_DEFAULT;
  if (needs_.ghost) flags |= NeighConst::REQ_GHOST;
  neighbor->add_request(this, flags);

  // The globals the impl's checks read, snapshotted at the seam so nothing
  // under impls/ has to include a LAMMPS header to see them. `atom->molecular`
  // is LAMMPS' int (0 = atom_style atomic), copied by value; `special_lj` /
  // `special_coul` are borrowed for the duration of this call only.
  //
  // `comm->cutghostuser` and `neighbor->skin` are transcribed RAW, with no
  // defaulting: LAMMPS holds 0.0 in the former when no `comm_modify cutoff`
  // ran, and what that zero MEANS (the effective ghost cutoff is the model
  // cutoff plus the skin) is decided inside CutoffGraphImpl::init. Interpreting
  // it here as well would give one rule two implementations, and would grow a
  // semantic `if` back into the layer that exists to have none.
  //
  // `impl_` is live here: LAMMPS refuses `pair_coeff` and any run before
  // `pair_style` has succeeded, and init_style() is reached only through
  // Pair::init().
  const molnex::lammps::LammpsRunEnv env{
      force->newton_pair, comm->nprocs,        atom->molecular,
      force->special_lj,  force->special_coul, comm->cutghostuser,
      neighbor->skin};
  try {
    impl_->init(env);
  } catch (const std::exception& e) {
    error->all(FLERR, "{}", e.what());
  }
}

void PairMolnex::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);

  // One guard for the whole method: empty means the input script reached a run
  // without a pair_style that loaded. (A pair_style without its pair_coeff is
  // the impl's own guard — the type -> Z table is its state, not this one's.)
  if (!impl_)
    error->all(FLERR,
               "pair_style molnex: pair_style/pair_coeff not initialized");

  if (fold_) fold_owners();
  molnex::lammps::EvalResult r{};
  try {
    r = impl_->compute(current_view());
  } catch (const std::exception& e) {
    error->all(FLERR, "{}", e.what());
  }

  // --- tally ---
  // Three branches on what LAMMPS ASKED FOR (the ev_init flags), not on what
  // the model can do: "can it" is answered by which fields of EvalResult exist
  // and are defined.
  if (eflag_global) eng_vdwl += r.e_total;

  // Per-atom energy: `EvalResult::e_atom` is this rank's local slice.
  //
  // THE BOUND IS THE RETURNED TENSOR'S OWN EXTENT, NEVER `atom->nlocal`. Both
  // are meant to be the local count, which is exactly why reading the second
  // one here would be a defect: the facade would then hold an independent
  // opinion about which atoms are local, and the first time the two disagreed
  // this loop would overrun `eatom` or leave owned atoms un-tallied — silently,
  // in a per-atom field nobody validates. `e_atom` is documented as length
  // nlocal (impls/contract_impl.h); trusting that number rather than
  // re-deriving it is what keeps a single definition.
  //
  // The bound is safe against `eatom`'s allocation, which `ev_init` sized at
  // `atom->nmax >= nlocal + nghost` and zeroed over at least `[0, nlocal)`
  // (pair.cpp: the ghost range is zeroed too when newton is on). `+=` is
  // LAMMPS' accumulation convention, and the zeroing is what makes it right.
  //
  if (eflag_atom) {
    const auto e_atom = r.e_atom.accessor<double, 1>();
    for (int64_t i = 0; i < e_atom.size(0); i++) eatom[i] += e_atom[i];
  }

  // The model returns no per-atom virial. A silent zero would look legal.
  if (vflag_atom)
    error->all(FLERR,
               "pair_molnex: per-atom virial requested (e.g. compute "
               "stress/atom); the model returns no per-atom virial. Remove "
               "the per-atom compute");

  // The global virial. Folded: the model's strain derivative, the only source
  // that sees the periodic images (fdotr is disabled for this mode in
  // settings). Ghost: LAMMPS' own force-times-distance sum over local and
  // ghost forces, run when LAMMPS asked for that form.
  if (vflag_global && r.virial_from_model)
    for (int k = 0; k < 6; ++k) virial[k] += r.virial[k];
  if (vflag_fdotr) virial_fdotr_compute();

  // The reduction-ownership seam. Whether the pair style itself must reduce is
  // the impl's answer, per step; this side is the mechanical branch on it and
  // reads no capability flag of its own.
  if (r.needs_pair_reverse_comm) comm->reverse_comm(this);
}

/* ----------------------------------------------------------------------
   the LAMMPS-side arrays, as the bridge sees them
------------------------------------------------------------------------- */

molnex::lammps_bridge::LammpsView PairMolnex::current_view() const {
  // `double**` -> `const double* const*` and `int**` -> `const int* const*` are
  // qualification conversions, so nothing here needs a const_cast at the seam.
  // NEIGHMASK travels as data: it is LAMMPS' constant, applied by the view's
  // accessor, which is what keeps every file under bridge/ and impls/ free of
  // LAMMPS vocabulary.
  //
  // `list` is live at all three call sites — compute() runs after init_list(),
  // and the two comm callbacks run only inside the reverse_comm that compute()
  // itself issues.
  //
  // `list->gnum` is 0 unless the impl asked for REQ_GHOST, in which case LAMMPS
  // has appended that many ghost centres to `ilist` behind the `inum` local
  // ones. Transcribed like every other field: how far to walk is the bridge's
  // decision, not this file's.
  return molnex::lammps_bridge::LammpsView{
      atom->x,        atom->f,          atom->type,
      atom->nlocal,   atom->nghost,     list->ilist,
      list->numneigh, list->firstneigh, list->inum,
      list->gnum,     NEIGHMASK,        fold_ ? owner_.data() : nullptr};
}

void PairMolnex::fold_owners() {
  // owner_[i]: the owned atom atom i is a copy of. On one rank every ghost is
  // a periodic image of an owned atom, and the owned copy is the one local
  // index carrying its ID — read off a table of the owned IDs built here, so
  // this needs no `atom_modify map` (atom->map would give the same answer: it
  // gives owned atoms precedence over images, atom_map.cpp).
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const tagint* tag = atom->tag;
  tagint max_tag = 0;
  for (int i = 0; i < nlocal; ++i) max_tag = std::max(max_tag, tag[i]);
  local_of_tag_.assign(static_cast<std::size_t>(max_tag) + 1, -1);
  for (int i = 0; i < nlocal; ++i)
    local_of_tag_[static_cast<std::size_t>(tag[i])] = i;
  owner_.resize(static_cast<std::size_t>(nall));
  for (int i = 0; i < nlocal; ++i) owner_[static_cast<std::size_t>(i)] = i;
  for (int j = nlocal; j < nall; ++j) {
    // An ID no owned atom carries maps to -1, which the bridge refuses with a
    // message naming the ghost: it can only be another rank's atom.
    const tagint t = tag[j];
    owner_[static_cast<std::size_t>(j)] =
        (t > 0 && t <= max_tag) ? local_of_tag_[static_cast<std::size_t>(t)]
                                : -1;
  }
}

/* ----------------------------------------------------------------------
   reverse communication of ghost forces → owned atoms
------------------------------------------------------------------------- */

int PairMolnex::pack_reverse_comm(int n, int first, double* buf) {
  return impl_->pack_reverse(current_view(), n, first, buf);
}

void PairMolnex::unpack_reverse_comm(int n, int* list, double* buf) {
  impl_->unpack_reverse(current_view(), n, list, buf);
}
