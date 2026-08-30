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
   carries one; FLERR still points at the callback that translated it. The one
   exception is the meta reader, whose messages start with "meta.json:" and read
   correctly behind the command that was running.
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

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "bridge/lammps_view.h"
#include "impls/contract_impl.h"
#include "impls/registry.h"
#include "lammps_contract.h"
#include "molnex/interface/meta_schema.h"
#include "molnex/interface/model_runner.h"

namespace fs = std::filesystem;

using namespace LAMMPS_NS;

namespace {

/* Where this export's meta.json is, from the two command-line arguments.
   Three call sites want it: the reader in settings(), and the two per-atom
   refusals in compute(), which name the contract by pointing at the one file
   that declares it. */
std::string meta_path_of(const std::string &model_dir, const std::string &model_name)
{
  return (fs::path(model_dir) / (model_name + ".meta.json")).string();
}

}    // namespace

PairMolnex::PairMolnex(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;        // no pairwise single() — this is a many-body model
  restartinfo = 0;
  one_coeff = 1;            // requires a single `pair_coeff * *`
  manybody_flag = 1;
  model_name = "model";
  // Placeholder sizes only. The real ones arrive with the impl's NeighborNeeds
  // in settings(); LAMMPS reads these members directly, so they must hold
  // something valid from construction on.
  comm_reverse = 3;
  comm_reverse_off = 3;
}

PairMolnex::~PairMolnex()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairMolnex::allocate()
{
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

/* pair_style molnex <model_dir> [name] */
void PairMolnex::settings(int narg, char **arg)
{
  if (narg < 1 || narg > 2)
    error->all(FLERR, "Illegal pair_style molnex command: pair_style molnex <model_dir> [name]");
  model_dir = arg[0];
  if (narg == 2) model_name = arg[1];

  // --- read the model-agnostic contract from <model_dir>/<name>.meta.json ---
  const std::string meta_path = meta_path_of(model_dir, model_name);
  std::optional<molnex::interface::MetaSchema> doc;
  try {
    doc = molnex::interface::MetaSchema::from_file(meta_path);
  } catch (const std::exception &e) {
    error->all(FLERR, "pair_style molnex: {}", e.what());
  }
  if (!doc->has_block("lammps"))
    error->all(FLERR, "pair_style molnex: {} has no 'lammps' block — export with "
                      "molix.engine.export_for_lammps, not the bare Exporter",
               meta_path);

  // Constructing the contract hard-validates schema_version == 2: a v1 meta is
  // refused here rather than silently losing its cuda_graph discriminator. It
  // is kept because the startup message quotes it — nothing else in this file
  // reads it, and no capability is copied out of it.
  try {
    contract_.emplace(*doc);
  } catch (const std::exception &e) {
    error->all(FLERR, "pair_style molnex: malformed meta.json: {}", e.what());
  }

  // Load the AOTI model now (so errors surface at setup, not the first step).
  // num_models = 1: the runner's multi-slot constructor is the seam a future
  // hot-reload consumer will use (README, "Weight hot-reload"), and until such
  // a consumer exists there is no pair_style grammar for it to reach.
  runner = std::make_unique<molnex::interface::ModelRunner>(model_dir, /*num_models=*/1, model_name);

  // The dispatch does the rest: it reads the contract's own key, builds the
  // execution strategy the meta asks for, and returns the implementation of
  // that contract — or refuses, listing the contracts that do dispatch. The
  // runner is borrowed, not transferred: this pair style outlives the impl and
  // keeps the model alive for the whole of it.
  try {
    impl_ = molnex::lammps::ContractImpl::for_contract(contract_->contract(), *contract_,
                                                       update->unit_style, *runner);
  } catch (const std::exception &e) {
    error->all(FLERR, "{}", e.what());
  }

  // The request table, asked once and executed verbatim from here on.
  needs_ = impl_->needs();
  comm_reverse = needs_.comm_reverse;
  comm_reverse_off = needs_.comm_reverse_off;

  if (comm->me == 0)
    utils::logmesg(lmp,
                   "pair_molnex: loaded '{}' from {} (device={}, model units={}, "
                   "LAMMPS units={}, dtype={}, cutoff={:.3f} A)\n",
                   model_name, model_dir, runner->device(), contract_->units(),
                   update->unit_style, contract_->model_dtype(), needs_.cutoff);
}

/* pair_coeff * * <Z_type1> <Z_type2> ... <Z_typeN> */
void PairMolnex::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg < 2 || strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff for molnex must be: pair_coeff * * ...");
  if (!impl_)
    error->all(FLERR, "pair_coeff for molnex must come after pair_style molnex: the elements "
                      "the model was exported for are read from its meta.json, and every Z "
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
  } catch (const std::exception &e) {
    error->all(FLERR, "{}", e.what());
  }

  const int ntypes = atom->ntypes;
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 1;
}

double PairMolnex::init_one(int /*i*/, int /*j*/)
{
  return needs_.cutoff;
}

void PairMolnex::init_style()
{
  // Two mechanical translations of the request table into LAMMPS' vocabulary.
  // `ghost` is false for every contract that exists today; the `|=` is here so
  // that flipping it in an impl is the whole change (lammps-iface-05-pbc), with
  // nothing to add on this side.
  int flags = needs_.full ? NeighConst::REQ_FULL : NeighConst::REQ_DEFAULT;
  if (needs_.ghost) flags |= NeighConst::REQ_GHOST;
  neighbor->add_request(this, flags);

  // The globals the impl's checks read, snapshotted at the seam so nothing
  // under impls/ has to include a LAMMPS header to see them. `atom->molecular`
  // is LAMMPS' int (0 = atom_style atomic), copied by value; `special_lj` /
  // `special_coul` are borrowed for the duration of this call only.
  //
  // `impl_` is live here: LAMMPS refuses `pair_coeff` and any run before
  // `pair_style` has succeeded, and init_style() is reached only through
  // Pair::init().
  const molnex::lammps::LammpsRunEnv env{force->newton_pair, comm->nprocs, atom->molecular,
                                         force->special_lj, force->special_coul};
  try {
    impl_->init(env);
  } catch (const std::exception &e) {
    error->all(FLERR, "{}", e.what());
  }
}

void PairMolnex::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // One guard for the whole method: empty means the input script reached a run
  // without a pair_style that loaded. (A pair_style without its pair_coeff is
  // the impl's own guard — the type -> Z table is its state, not this one's.)
  if (!impl_)
    error->all(FLERR, "pair_style molnex: pair_style/pair_coeff not initialized");

  molnex::lammps::EvalResult r{};
  try {
    r = impl_->compute(current_view());
  } catch (const std::exception &e) {
    error->all(FLERR, "{}", e.what());
  }

  // --- tally ---
  // Three branches on what LAMMPS ASKED FOR (the ev_init flags), not on what
  // the model can do: "can it" is answered by which fields of EvalResult exist
  // and are defined.
  if (eflag_global) eng_vdwl += r.e_total;

  // Not a missing feature — a refusal. Before this spec these two requests were
  // met with a silently all-zero eatom / vatom: `ev_init` allocates them, this
  // pair style never wrote them, and the user got a legal-looking zero field.
  // Nothing in the current contract can produce either (a model returns energy,
  // forces and an optional global virial), so the honest answer is loud.
  if (eflag_atom)
    error->all(FLERR, "pair_molnex: per-atom energy requested (e.g. compute pe/atom) but the "
                      "contract declared in {} provides none — the model returns a total "
                      "energy only; remove the per-atom compute",
               meta_path_of(model_dir, model_name));
  if (vflag_atom)
    error->all(FLERR, "pair_molnex: per-atom virial requested (e.g. compute stress/atom) but "
                      "the contract declared in {} provides none — the model returns at most a "
                      "global virial; remove the per-atom compute",
               meta_path_of(model_dir, model_name));

  // DO NOT REORDER THESE TWO CONDITIONS. Written the other way round —
  // `vflag_global` first, then presence — the third case changes behaviour:
  // vflag_global set, no model virial, vflag_fdotr unset must do NOTHING, and
  // that is what happens today. `defined()` is the whole presence test:
  // read_model_outputs leaves the virial undefined unless it holds exactly six
  // numbers, and the impl has already applied the unit factor.
  if (r.virial.defined() && vflag_global) {
    // model virial: (6,) xx,yy,zz,xy,xz,yz, already in LAMMPS energy units
    auto va = r.virial.accessor<double, 1>();
    virial[0] += va[0];
    virial[1] += va[1];
    virial[2] += va[2];
    virial[3] += va[3];
    virial[4] += va[4];
    virial[5] += va[5];
  } else if (vflag_fdotr) {
    virial_fdotr_compute();
  }

  // The reduction-ownership seam. Whether the pair style itself must reduce is
  // the impl's answer, per step; this side is the mechanical branch on it and
  // reads no capability flag of its own.
  if (r.needs_pair_reverse_comm) comm->reverse_comm(this);
}

/* ----------------------------------------------------------------------
   the LAMMPS-side arrays, as the bridge sees them
------------------------------------------------------------------------- */

molnex::lammps_bridge::LammpsView PairMolnex::current_view() const
{
  // `double**` -> `const double* const*` and `int**` -> `const int* const*` are
  // qualification conversions, so nothing here needs a const_cast at the seam.
  // NEIGHMASK travels as data: it is LAMMPS' constant, applied by the view's
  // accessor, which is what keeps every file under bridge/ and impls/ free of
  // LAMMPS vocabulary.
  //
  // `list` is live at all three call sites — compute() runs after init_list(),
  // and the two comm callbacks run only inside the reverse_comm that compute()
  // itself issues.
  return molnex::lammps_bridge::LammpsView{
      atom->x, atom->f, atom->type, atom->nlocal, atom->nghost,
      list->ilist, list->numneigh, list->firstneigh, list->inum, NEIGHMASK};
}

/* ----------------------------------------------------------------------
   reverse communication of ghost forces → owned atoms
------------------------------------------------------------------------- */

int PairMolnex::pack_reverse_comm(int n, int first, double *buf)
{
  return impl_->pack_reverse(current_view(), n, first, buf);
}

void PairMolnex::unpack_reverse_comm(int n, int *list, double *buf)
{
  impl_->unpack_reverse(current_view(), n, list, buf);
}
