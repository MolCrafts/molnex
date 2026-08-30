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

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <optional>

#include <ATen/ATen.h>

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
  fp64_model = false;
  energy_conv = 1.0;
  force_conv = 1.0;
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

  // Which contract the export declares. Parsed inside the try (an unknown
  // spelling throws), acted on after it (the refusal is a LAMMPS error, not a
  // malformed meta).
  auto model_contract = molnex::lammps::Contract::cutoff_graph;
  try {
    // Constructing the contract hard-validates schema_version == 2: a v1 meta
    // is refused here rather than silently losing its cuda_graph discriminator.
    const molnex::lammps::LammpsContract contract(*doc);

    cut_global = contract.cutoff();
    model_units = contract.units();
    fp64_model = (contract.model_dtype() == "float64");

    // units: the export stamped an energy-conversion table (computed with
    // molix.units / pint) keyed by LAMMPS unit style; look up the active one.
    // No conversion constant is restated in C++. When the model's own units are
    // the active style the table holds exactly 1.0 by construction, and force
    // may share the factor only because units_for asserted length_unit == Å.
    const molnex::lammps::UnitConversion conv = contract.units_for(update->unit_style);
    energy_conv = conv.energy;
    force_conv = conv.force;

    model_contract = contract.contract();

    supports_pbc = contract.supports_pbc();
    supports_virial = contract.supports_virial();

    // CUDA-graph fast path: fixed N + padded E_max, captured once & replayed.
    use_cuda_graph = (contract.execution() == molnex::lammps::Execution::cuda_graph);
    if (use_cuda_graph) {
      n_fixed = contract.n_atoms();
      e_max = contract.e_max();
    }
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

  if (use_cuda_graph) {
    if (device != "cuda")
      error->all(FLERR, "pair_style molnex: meta execution=cuda-graph but device={} (need cuda)",
                 device);
    if (supports_pbc)
      error->all(FLERR, "pair_style molnex: execution=cuda-graph is incompatible with "
                        "supports_pbc (ghost count varies; re-export with cuda_graph=False "
                        "for PBC MD)");
    auto dev = at::Device(at::kCUDA);
    at::ScalarType mdt = fp64_model ? at::kDouble : at::kFloat;
    Zg_ = at::zeros({n_fixed}, at::TensorOptions().dtype(at::kLong).device(dev));
    posg_ = at::zeros({n_fixed, 3}, at::TensorOptions().dtype(mdt).device(dev));
    edgeg_ = at::zeros({e_max, 2}, at::TensorOptions().dtype(at::kLong).device(dev));
    maskg_ = at::zeros({e_max}, at::TensorOptions().dtype(at::kBool).device(dev));
  }
  if (comm->me == 0)
    utils::logmesg(lmp,
                   "pair_molnex: loaded '{}' from {} (device={}, model units={}, "
                   "LAMMPS units={}, dtype={}, cutoff={:.3f} A, energy*={:.6g}, "
                   "supports_pbc={}, supports_virial={})\n",
                   model_name, model_dir, device, model_units, update->unit_style,
                   fp64_model ? "float64" : "float32", cut_global, energy_conv,
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

  type2z.assign(ntypes + 1, 0);
  for (int t = 1; t <= ntypes; t++)
    type2z[t] = utils::inumeric(FLERR, arg[1 + t], false, lmp);

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

  const int nlocal = atom->nlocal;
  const int nghost = atom->nghost;
  const int nall = nlocal + nghost;

  if (!supports_pbc && nghost != 0)
    error->all(FLERR, "pair_molnex (supports_pbc=false) requires no ghost atoms "
                      "(single molecule, 1 rank, non-periodic). nghost={}. "
                      "Re-export with supports_pbc=true for periodic / multi-rank MD.",
               nghost);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;

  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  const double cutsq_g = cut_global * cut_global;

  // Graph atom count: local-only when supports_pbc is false; local+ghost otherwise.
  const int n_graph = supports_pbc ? nall : nlocal;

  // --- node tensors: Z (N,), pos (N,3) ---
  auto opt_f = at::TensorOptions().dtype(at::kDouble);     // build in fp64 then cast
  auto opt_l = at::TensorOptions().dtype(at::kLong);
  at::Tensor Z = at::empty({n_graph}, opt_l);
  at::Tensor pos = at::empty({n_graph, 3}, opt_f);
  auto Za = Z.accessor<int64_t, 1>();
  auto pa = pos.accessor<double, 2>();
  for (int i = 0; i < n_graph; i++) {
    Za[i] = type2z[type[i]];
    pa[i][0] = x[i][0];
    pa[i][1] = x[i][1];
    pa[i][2] = x[i][2];
  }

  // --- edges: full bidirectional within cutoff ---
  // molnex aggregation requires edge_index SORTED by (source,target) — exactly
  // what the reference data pipeline's nonzero() yields. The raw neighbour-list
  // order gives the same pairs but a different order, which silently produces
  // wrong energy/forces. So collect then sort.
  //
  // With supports_pbc, j may be a ghost: its coordinates already encode the
  // minimum-image placement, so edge_diff = pos[j]-pos[i] needs no separate
  // shifts tensor for models that recompute geometry from positions.
  edge_pairs_.clear();
  edge_pairs_.reserve(static_cast<size_t>(inum) * 16);
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    double xi = x[i][0], yi = x[i][1], zi = x[i][2];
    int *jl = firstneigh[i];
    int jn = numneigh[i];
    for (int jj = 0; jj < jn; jj++) {
      int j = jl[jj] & NEIGHMASK;
      if (!supports_pbc && j >= nlocal) continue;
      if (j >= n_graph) continue;
      double dx = x[j][0] - xi, dy = x[j][1] - yi, dz = x[j][2] - zi;
      if (dx * dx + dy * dy + dz * dz < cutsq_g) edge_pairs_.emplace_back(i, j);
    }
  }
  std::sort(edge_pairs_.begin(), edge_pairs_.end());
  const int64_t ne = static_cast<int64_t>(edge_pairs_.size());
  edge_flat_.resize(static_cast<size_t>(2 * ne));
  for (int64_t k = 0; k < ne; k++) {
    edge_flat_[static_cast<size_t>(2 * k)] = edge_pairs_[static_cast<size_t>(k)].first;
    edge_flat_[static_cast<size_t>(2 * k + 1)] = edge_pairs_[static_cast<size_t>(k)].second;
  }
  at::Tensor edge_index = at::from_blob(edge_flat_.data(), {ne, 2}, opt_l).clone();

  // --- run: CUDA-graph fast path (fixed N + padded E_max) or dynamic run ---
  at::ScalarType mdt = fp64_model ? at::kDouble : at::kFloat;   // from meta.json
  at::Tensor e_t, f_t, v_t;
  bool have_virial_out = false;
#ifdef MOLNEX_INTERFACE_CUDA
  if (use_cuda_graph) {
    if (nghost != 0)
      error->all(FLERR, "pair_molnex cuda_graph: nghost={} (need 0)", nghost);
    if (nlocal != n_fixed)
      error->all(FLERR, "pair_molnex cuda_graph: nlocal={} != fixed N={} "
                        "(export n_atoms must match the system)", nlocal, n_fixed);
    if (ne > e_max)
      error->all(FLERR, "pair_molnex cuda_graph: {} edges > e_max {} "
                        "(re-export with a larger e_max)", ne, e_max);
    // pad edges to e_max (extra rows = (0,0), inert via mask) + valid-edge mask
    edge_pad_flat_.assign(static_cast<size_t>(2 * e_max), 0);
    for (int64_t k = 0; k < ne; k++) {
      edge_pad_flat_[static_cast<size_t>(2 * k)] = edge_pairs_[static_cast<size_t>(k)].first;
      edge_pad_flat_[static_cast<size_t>(2 * k + 1)] = edge_pairs_[static_cast<size_t>(k)].second;
    }
    at::Tensor mask_cpu = at::zeros({e_max}, at::TensorOptions().dtype(at::kBool));
    auto ma = mask_cpu.accessor<bool, 1>();
    for (int64_t k = 0; k < ne; k++) ma[k] = true;
    at::Tensor edges_cpu = at::from_blob(edge_pad_flat_.data(), {e_max, 2}, opt_l).clone();
    // update the persistent device buffers in place, then capture/replay
    Zg_.copy_(Z);
    posg_.copy_(pos.to(mdt));
    edgeg_.copy_(edges_cpu);
    maskg_.copy_(mask_cpu);
    std::vector<at::Tensor> inputs = {Zg_, posg_, edgeg_, maskg_};
    std::vector<at::Tensor> out = runner->run_graphed(inputs);   // [energy, forces(N,3), ?virial]
    e_t = out[0].to(at::kDouble).to(at::kCPU);
    f_t = out[1].to(at::kDouble).to(at::kCPU).contiguous();
    if (supports_virial && out.size() >= 3) {
      v_t = out[2].to(at::kDouble).to(at::kCPU).contiguous();
      have_virial_out = true;
    }
  } else
#endif
  {
    auto dev = (device == "cuda") ? at::Device(at::kCUDA) : at::Device(at::kCPU);
    std::vector<at::Tensor> inputs = {Z.to(dev), pos.to(mdt).to(dev), edge_index.to(dev)};
    std::vector<at::Tensor> out = runner->run(inputs);   // [energy_total, forces(N,3), ?virial]
    e_t = out[0].to(at::kDouble).to(at::kCPU);
    f_t = out[1].to(at::kDouble).to(at::kCPU).contiguous();
    if (supports_virial && out.size() >= 3) {
      v_t = out[2].to(at::kDouble).to(at::kCPU).contiguous();
      have_virial_out = true;
    }
  }

  if (f_t.size(0) != n_graph)
    error->all(FLERR, "pair_molnex: model returned forces for {} atoms, expected {}",
               f_t.size(0), n_graph);

  // --- scatter forces + energy back to LAMMPS (converted to LAMMPS units) ---
  // Write onto local (+ ghost when supports_pbc); reverse_comm pulls ghost
  // force contributions onto owners for multi-rank / periodic images.
  auto fa = f_t.accessor<double, 2>();
  for (int i = 0; i < n_graph; i++) {
    f[i][0] += force_conv * fa[i][0];
    f[i][1] += force_conv * fa[i][1];
    f[i][2] += force_conv * fa[i][2];
  }
  double e_model = e_t.item<double>() * energy_conv;
  if (eflag_global) eng_vdwl += e_model;

  if (have_virial_out && vflag_global) {
    // model virial: (6,) xx,yy,zz,xy,xz,yz in model energy units
    auto va = v_t.accessor<double, 1>();
    if (v_t.numel() < 6)
      error->all(FLERR, "pair_molnex: supports_virial but virial output has {} elements (need 6)",
                 v_t.numel());
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
   reverse communication of ghost forces → owned atoms
------------------------------------------------------------------------- */

int PairMolnex::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;
  double **f = atom->f;
  for (int i = first; i < last; i++) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
  }
  return m;
}

void PairMolnex::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  double **f = atom->f;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
  }
}
