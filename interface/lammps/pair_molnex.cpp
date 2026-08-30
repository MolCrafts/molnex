/* PairMolnex implementation — see pair_molnex.h.

   Generalized from the PiNet-specific pair_pinet.cpp: model-specific knobs
   (cutoff, native units, compute dtype, PBC/virial capabilities) are read from
   the export's meta.json rather than hardcoded, so the same pair style serves
   any molnex potential exported via molix.engine.export_for_lammps.
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

#include <cstring>
#include <filesystem>
#include <optional>
#include <utility>
#include <vector>

#include <ATen/ATen.h>

#include "bridge/comm_reverse.h"
#include "bridge/dynamic_exec.h"
#include "bridge/exec.h"
#include "bridge/force_scatter.h"
#include "bridge/graph_bridge.h"
#include "bridge/graphed_exec.h"
#include "bridge/lammps_view.h"
#include "bridge/node_staging.h"
#include "bridge/species_map.h"
#include "lammps_contract.h"
#include "molnex/interface/meta_schema.h"
#include "molnex/interface/model_runner.h"

namespace fs = std::filesystem;

using namespace LAMMPS_NS;

PairMolnex::PairMolnex(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;        // no pairwise single() — this is a many-body model
  restartinfo = 0;
  one_coeff = 1;            // requires a single `pair_coeff * *`
  manybody_flag = 1;
  cut_global = 0.0;
  model_name = "model";
  energy_conv = 1.0;
  // ghost force reverse-comm size (fx,fy,fz). Needed when supports_pbc and
  // the model writes forces on ghosts; also used with newton off.
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
  // Two dialects meet here: the reader layers below throw std::runtime_error,
  // LAMMPS wants error->all(FLERR, ...). Every error->all stays OUTSIDE the
  // try blocks — it raises a LAMMPSException, which our own catch would
  // otherwise swallow and re-label as a malformed meta.
  const std::string meta_path = (fs::path(model_dir) / (model_name + ".meta.json")).string();
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

  // What the export declares about its own execution. Parsed inside the try
  // (an unknown spelling, or a cuda-graph meta missing its fixed shapes,
  // throws), acted on after it — the refusals below are LAMMPS errors, not
  // malformed metas, and the runner they need does not exist yet. Locals, not
  // members: their only reader is the Exec constructed at the end of this
  // method, which then owns them.
  auto model_contract = molnex::lammps::Contract::cutoff_graph;
  auto model_execution = molnex::lammps::Execution::dynamic;
  at::ScalarType model_dtype = at::kFloat;
  int64_t n_atoms = 0;
  int64_t e_max_edges = 0;
  try {
    // Constructing the contract hard-validates schema_version == 2: a v1 meta
    // is refused here rather than silently losing its cuda_graph discriminator.
    const molnex::lammps::LammpsContract contract(*doc);

    cut_global = contract.cutoff();
    model_units = contract.units();
    model_dtype = (contract.model_dtype() == "float64") ? at::kDouble : at::kFloat;

    // units: the export stamped an energy-conversion table (computed with
    // molix.units / pint) keyed by LAMMPS unit style; look up the active one.
    // No conversion constant is restated in C++. When the model's own units are
    // the active style the table holds exactly 1.0 by construction, and force
    // may share the factor only because units_for asserted length_unit == Å.
    const molnex::lammps::UnitConversion conv = contract.units_for(update->unit_style);
    energy_conv = conv.energy;
    // The force factor is handed to ForceScatter and kept nowhere else: the
    // scatter loop it fed moved into that component, so a `force_conv` member
    // here would be a second copy with nothing to keep it in step.
    force_scatter_.emplace(conv.force);
    // Same for the cutoff: GraphBridge squares it once here and refilters every
    // candidate pair against it each step, so the neighbour loop's copy is
    // gone. `cut_global` itself stays only because init_one() still reports it
    // to LAMMPS as this pair style's cutoff.
    graph_bridge_.emplace(cut_global);

    model_contract = contract.contract();

    supports_pbc = contract.supports_pbc();
    supports_virial = contract.supports_virial();

    // CUDA-graph fast path: fixed N + padded E_max, captured once & replayed.
    model_execution = contract.execution();
    if (model_execution == molnex::lammps::Execution::cuda_graph) {
      n_atoms = contract.n_atoms();
      e_max_edges = contract.e_max();
    }

    // Kept, not just read: `pair_coeff` runs after `pair_style` and needs the
    // export's species list to reject a Z the model was never built for. The
    // contract owns its copy of the meta text, so nothing here dangles.
    contract_ = contract;
  } catch (const std::exception &e) {
    error->all(FLERR, "pair_style molnex: malformed meta.json: {}", e.what());
  }

  if (model_contract != molnex::lammps::Contract::cutoff_graph)
    error->all(FLERR, "pair_style molnex: contract 'fixed-graph' has no implementation yet "
                      "(this pair style builds its edge set from a cutoff; a fixed covalent "
                      "topology needs an impl that does not exist)");

  // Load the AOTI model now (so errors surface at setup, not first step).
  // run_single_threaded (the ModelRunner default) is required for graph capture.
  runner = std::make_unique<molnex::interface::ModelRunner>(model_dir, /*num_models=*/1, model_name);
  device = runner->device();

  // Two preconditions the CUDA-graph strategy cannot check for itself: they are
  // statements about this export's meta against this machine, and they are
  // LAMMPS errors, so they stay here and stay outside any try.
  if (model_execution == molnex::lammps::Execution::cuda_graph) {
    if (device != "cuda")
      error->all(FLERR, "pair_style molnex: meta execution=cuda-graph but device={} (need cuda)",
                 device);
    if (supports_pbc)
      error->all(FLERR, "pair_style molnex: execution=cuda-graph is incompatible with "
                        "supports_pbc (ghost count varies; re-export with cuda_graph=False "
                        "for PBC MD)");
  }

  // Pick the execution strategy once, here. Which object this constructs IS the
  // choice — there is no boolean left over for compute() to re-read, and the
  // step path calls exec_->run() without knowing which one it got. The runner is
  // borrowed, not transferred: this pair style outlives its Exec and keeps the
  // model alive for the whole of it.
  //
  // Same two dialects as above (GraphedExec refuses a CPU-only build by
  // throwing), so the error->all is issued from the handler and never from
  // inside the try — a LAMMPSException is a std::exception and would otherwise
  // be caught here and re-labelled.
  const auto dev = (device == "cuda") ? at::Device(at::kCUDA) : at::Device(at::kCPU);
  try {
    if (model_execution == molnex::lammps::Execution::cuda_graph)
      exec_ = std::make_unique<molnex::lammps_bridge::GraphedExec>(
          runner.get(), dev, model_dtype, supports_virial, n_atoms, e_max_edges);
    else
      exec_ = std::make_unique<molnex::lammps_bridge::DynamicExec>(runner.get(), dev, model_dtype,
                                                                  supports_virial);
  } catch (const std::exception &e) {
    error->all(FLERR, "pair_style molnex: {}", e.what());
  }

  if (comm->me == 0)
    utils::logmesg(lmp,
                   "pair_molnex: loaded '{}' from {} (device={}, model units={}, "
                   "LAMMPS units={}, dtype={}, cutoff={:.3f} A, energy*={:.6g}, "
                   "supports_pbc={}, supports_virial={})\n",
                   model_name, model_dir, device, model_units, update->unit_style,
                   // Transcribed from the strategy that actually casts `pos`,
                   // rather than judged a second time here.
                   exec_->model_dtype_name(), cut_global, energy_conv,
                   supports_pbc ? "true" : "false",
                   supports_virial ? "true" : "false");
}

/* pair_coeff * * <Z_type1> <Z_type2> ... <Z_typeN> */
void PairMolnex::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  int ntypes = atom->ntypes;
  if (narg != 2 + ntypes)
    error->all(FLERR, "pair_coeff for molnex needs: * * <Z1> ... <Zntypes> (one Z per atom type)");
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff for molnex must be: pair_coeff * * ...");
  if (!contract_)
    error->all(FLERR, "pair_coeff for molnex must come after pair_style molnex: the elements "
                      "the model was exported for are read from its meta.json, and every Z "
                      "here is checked against them");

  // 0-based and dense: entry t-1 is the Z of LAMMPS type t, and the length is
  // ntypes. No placeholder slot at index 0 — that only made the table's length
  // something a reader had to guess at.
  std::vector<int> z_by_type(ntypes);
  for (int t = 1; t <= ntypes; t++)
    z_by_type[t - 1] = utils::inumeric(FLERR, arg[1 + t], false, lmp);

  // The same two dialects as settings(): SpeciesMap throws (an element the
  // model was never exported for used to reach the network as an embedding
  // index it has no row for, and produced a confident wrong energy), LAMMPS
  // wants error->all. It is issued from the handler, never from inside the
  // try — a LAMMPSException is a std::exception and would be self-caught.
  try {
    species_map_.emplace(std::move(z_by_type), contract_->species());
  } catch (const std::exception &e) {
    error->all(FLERR, "pair_coeff for molnex: {}", e.what());
  }

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 1;
}

double PairMolnex::init_one(int /*i*/, int /*j*/)
{
  return cut_global;
}

void PairMolnex::init_style()
{
  if (!supports_pbc && force->newton_pair == 1 && comm->nprocs > 1)
    error->all(FLERR, "pair_molnex without supports_pbc assumes a single domain; "
                      "run on 1 MPI rank or re-export with supports_pbc=true");
  // full neighbour list within cut_global (needed for many-body aggregation)
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

void PairMolnex::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // One guard establishes the invariant for the whole method: past here all
  // four components hold a value, so every use below dereferences directly
  // rather than re-asking. Empty means the input script reached a run without a
  // complete pair_style / pair_coeff pair.
  if (!species_map_ || !force_scatter_ || !graph_bridge_ || !exec_)
    error->all(FLERR, "pair_style molnex: pair_style/pair_coeff not initialized");

  const int nlocal = atom->nlocal;
  const int nghost = atom->nghost;
  const int nall = nlocal + nghost;

  if (!supports_pbc && nghost != 0)
    error->all(FLERR, "pair_molnex (supports_pbc=false) requires no ghost atoms "
                      "(single molecule, 1 rank, non-periodic). nghost={}. "
                      "Re-export with supports_pbc=true for periodic / multi-rank MD.",
               nghost);

  const molnex::lammps_bridge::LammpsView view = current_view();

  // Graph atom count: local-only when supports_pbc is false; local+ghost otherwise.
  const int n_graph = supports_pbc ? nall : nlocal;

  // --- stage, build the graph, run, scatter ---
  // Four bridge calls, one seam. Each throws std::runtime_error where this used
  // to either error->all inline (the CUDA-graph shape checks) or read past the
  // end of a LAMMPS array without looking (the neighbour loop), so one handler
  // translates the lot into LAMMPS' dialect. It is issued from the handler,
  // never from inside the try — a LAMMPSException is a std::exception and would
  // be self-caught.
  //
  // What each contributes, and what this method therefore no longer knows:
  // stage_nodes copies Z / pos (fp64; the model-dtype cast is the Exec's);
  // GraphBridge collects, refilters against the cutoff and sorts by
  // (source, target); the Exec moves to the device, pads and masks if it is the
  // graphed one, and hands back CPU float64 outputs this method owns;
  // ForceScatter accumulates them onto atom->f in LAMMPS units.
  molnex::lammps_bridge::ModelOutputs out;
  try {
    const molnex::lammps_bridge::NodeTensors nodes =
        molnex::lammps_bridge::stage_nodes(view, *species_map_, n_graph);
    const at::Tensor edge_index = graph_bridge_->build(view, n_graph);
    out = exec_->run(nodes.Z, nodes.pos, edge_index);
    // Write onto local (+ ghost when supports_pbc); reverse_comm pulls ghost
    // force contributions onto owners for multi-rank / periodic images.
    force_scatter_->scatter(view, out.forces, n_graph);
  } catch (const std::exception &e) {
    error->all(FLERR, "pair_molnex: {}", e.what());
  }

  double e_model = out.energy.item<double>() * energy_conv;
  if (eflag_global) eng_vdwl += e_model;

  // `defined()` is the whole presence test: read_model_outputs leaves the
  // virial undefined unless it holds exactly six numbers, so there is nothing
  // left to check here before indexing it.
  if (out.virial.defined() && vflag_global) {
    // model virial: (6,) xx,yy,zz,xy,xz,yz in model energy units
    auto va = out.virial.accessor<double, 1>();
    virial[0] += energy_conv * va[0];
    virial[1] += energy_conv * va[1];
    virial[2] += energy_conv * va[2];
    virial[3] += energy_conv * va[3];
    virial[4] += energy_conv * va[4];
    virial[5] += energy_conv * va[5];
  } else if (vflag_fdotr) {
    virial_fdotr_compute();
  }

  if (supports_pbc && nghost > 0)
    comm->reverse_comm(this);
}

/* ----------------------------------------------------------------------
   the LAMMPS-side arrays, as the bridge sees them
------------------------------------------------------------------------- */

molnex::lammps_bridge::LammpsView PairMolnex::current_view() const
{
  // `double**` -> `const double* const*` and `int**` -> `const int* const*` are
  // qualification conversions, so nothing here needs a const_cast at the seam.
  // NEIGHMASK travels as data: it is LAMMPS' constant, applied by the view's
  // accessor, which is what keeps every file under bridge/ free of LAMMPS
  // vocabulary.
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
  return molnex::lammps_bridge::pack_reverse_forces(current_view(), n, first, buf);
}

void PairMolnex::unpack_reverse_comm(int n, int *list, double *buf)
{
  molnex::lammps_bridge::unpack_reverse_forces(current_view(), n, list, buf);
}
