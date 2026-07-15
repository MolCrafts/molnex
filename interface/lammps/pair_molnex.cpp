/* PairMolnex implementation — see pair_molnex.h.

   Generalized from the PiNet-specific pair_pinet.cpp: model-specific knobs
   (cutoff, native units, compute dtype) are read from the export's meta.json
   rather than hardcoded, so the same pair style serves any molnex potential
   exported via molix.engine.export_for_lammps.
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
#include <cmath>
#include <cstring>
#include <filesystem>

#include <ATen/ATen.h>

#include "lammps_meta.h"
#include "molnex/interface/model_runner.h"

namespace fs = std::filesystem;

using namespace LAMMPS_NS;

namespace {

/* Energy of one unit of the given LAMMPS unit system, expressed in eV.
   Distance is Å in both `real` and `metal`, so a single scalar converts both
   energy (kcal/mol or eV) and force (per-Å of the same energy). */
double energy_unit_in_eV(const std::string &units) {
  if (units == "metal") return 1.0;                 // eV
  if (units == "real") return 0.0433641043;         // kcal/mol -> eV
  throw std::runtime_error(
      "pair_style molnex: unsupported unit system '" + units +
      "' (only 'real' and 'metal' carry an energy<->eV conversion)");
}

}  // namespace

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
  const std::string meta_path = (fs::path(model_dir) / (model_name + ".meta.json")).string();
  std::string meta;
  try {
    meta = molnex::lammps_meta::read_text(meta_path);
  } catch (const std::exception &e) {
    error->all(FLERR, "pair_style molnex: {}", e.what());
  }
  if (!molnex::lammps_meta::has_field(meta, "lammps"))
    error->all(FLERR, "pair_style molnex: {} has no 'lammps' block — export with "
                      "molix.engine.export_for_lammps, not the bare Exporter",
               meta_path);

  try {
    cut_global = molnex::lammps_meta::get_number(meta, "cutoff");
    model_units = molnex::lammps_meta::get_string(meta, "units");
    const std::string model_dtype = molnex::lammps_meta::get_string(meta, "model_dtype");
    fp64_model = (model_dtype == "float64");

    // units: convert model-native energy/forces to LAMMPS' active unit system.
    const std::string lmp_units = update->unit_style;
    if (model_units == lmp_units) {
      energy_conv = 1.0;
    } else {
      energy_conv = energy_unit_in_eV(model_units) / energy_unit_in_eV(lmp_units);
    }
    force_conv = energy_conv;   // force = energy / Å, distance unchanged

    // CUDA-graph fast path: fixed N + padded E_max, captured once & replayed.
    use_cuda_graph = molnex::lammps_meta::get_bool(meta, "cuda_graph", false);
    if (use_cuda_graph) {
      n_fixed = static_cast<int>(molnex::lammps_meta::get_number(meta, "n_atoms"));
      e_max = static_cast<int64_t>(molnex::lammps_meta::get_number(meta, "e_max"));
    }
  } catch (const std::exception &e) {
    error->all(FLERR, "pair_style molnex: malformed meta.json: {}", e.what());
  }

  // Load the AOTI model now (so errors surface at setup, not first step).
  // run_single_threaded (the ModelRunner default) is required for graph capture.
  runner = std::make_unique<molnex::interface::ModelRunner>(model_dir, /*num_models=*/1, model_name);
  device = runner->device();

  if (use_cuda_graph) {
    if (device != "cuda")
      error->all(FLERR, "pair_style molnex: meta cuda_graph=true but device={} (need cuda)", device);
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
                   "LAMMPS units={}, dtype={}, cutoff={:.3f} A, energy*={:.6g})\n",
                   model_name, model_dir, device, model_units, update->unit_style,
                   fp64_model ? "float64" : "float32", cut_global, energy_conv);
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
  if (force->newton_pair == 1 && comm->nprocs > 1)
    error->all(FLERR, "pair_molnex (this version) assumes a single domain; run on 1 MPI rank");
  // full neighbour list within cut_global
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

void PairMolnex::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const int nlocal = atom->nlocal;
  const int nghost = atom->nghost;
  if (nghost != 0)
    error->all(FLERR, "pair_molnex (this version) requires no ghost atoms "
                      "(single molecule, 1 rank, non-periodic). nghost != 0.");

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;

  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  const double cutsq_g = cut_global * cut_global;

  // --- node tensors: Z (N,), pos (N,3) ---
  auto opt_f = at::TensorOptions().dtype(at::kDouble);     // build in fp64 then cast
  auto opt_l = at::TensorOptions().dtype(at::kLong);
  at::Tensor Z = at::empty({nlocal}, opt_l);
  at::Tensor pos = at::empty({nlocal, 3}, opt_f);
  auto Za = Z.accessor<int64_t, 1>();
  auto pa = pos.accessor<double, 2>();
  for (int i = 0; i < nlocal; i++) {
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
  std::vector<std::pair<int, int>> ep;
  ep.reserve(inum * 16);
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    double xi = x[i][0], yi = x[i][1], zi = x[i][2];
    int *jl = firstneigh[i];
    int jn = numneigh[i];
    for (int jj = 0; jj < jn; jj++) {
      int j = jl[jj] & NEIGHMASK;
      if (j >= nlocal) continue;   // no ghosts in this regime
      double dx = x[j][0] - xi, dy = x[j][1] - yi, dz = x[j][2] - zi;
      if (dx * dx + dy * dy + dz * dz < cutsq_g) ep.emplace_back(i, j);
    }
  }
  std::sort(ep.begin(), ep.end());
  int64_t ne = static_cast<int64_t>(ep.size());
  std::vector<int64_t> edges(2 * ne);
  for (int64_t k = 0; k < ne; k++) {
    edges[2 * k] = ep[k].first;       // source
    edges[2 * k + 1] = ep[k].second;  // target
  }
  at::Tensor edge_index = at::from_blob(edges.data(), {ne, 2}, opt_l).clone();

  // --- run: CUDA-graph fast path (fixed N + padded E_max) or dynamic run ---
  at::ScalarType mdt = fp64_model ? at::kDouble : at::kFloat;   // from meta.json
  at::Tensor e_t, f_t;
#ifdef MOLNEX_INTERFACE_CUDA
  if (use_cuda_graph) {
    if (nlocal != n_fixed)
      error->all(FLERR, "pair_molnex cuda_graph: nlocal={} != fixed N={} "
                        "(export n_atoms must match the system)", nlocal, n_fixed);
    if (ne > e_max)
      error->all(FLERR, "pair_molnex cuda_graph: {} edges > e_max {} "
                        "(re-export with a larger e_max)", ne, e_max);
    // pad edges to e_max (extra rows = (0,0), inert via mask) + valid-edge mask
    std::vector<int64_t> pedges(2 * e_max, 0);
    for (int64_t k = 0; k < ne; k++) { pedges[2*k] = ep[k].first; pedges[2*k+1] = ep[k].second; }
    at::Tensor mask_cpu = at::zeros({e_max}, at::TensorOptions().dtype(at::kBool));
    auto ma = mask_cpu.accessor<bool, 1>();
    for (int64_t k = 0; k < ne; k++) ma[k] = true;
    at::Tensor edges_cpu = at::from_blob(pedges.data(), {e_max, 2}, opt_l).clone();
    // update the persistent device buffers in place, then capture/replay
    Zg_.copy_(Z);
    posg_.copy_(pos.to(mdt));
    edgeg_.copy_(edges_cpu);
    maskg_.copy_(mask_cpu);
    std::vector<at::Tensor> inputs = {Zg_, posg_, edgeg_, maskg_};
    std::vector<at::Tensor> out = runner->run_graphed(inputs);   // [energy, forces(N,3)]
    e_t = out[0].to(at::kDouble).to(at::kCPU);
    f_t = out[1].to(at::kDouble).to(at::kCPU).contiguous();
  } else
#endif
  {
    auto dev = (device == "cuda") ? at::Device(at::kCUDA) : at::Device(at::kCPU);
    std::vector<at::Tensor> inputs = {Z.to(dev), pos.to(mdt).to(dev), edge_index.to(dev)};
    std::vector<at::Tensor> out = runner->run(inputs);   // [energy_total, forces(N,3)]
    e_t = out[0].to(at::kDouble).to(at::kCPU);
    f_t = out[1].to(at::kDouble).to(at::kCPU).contiguous();
  }

  // --- scatter forces + energy back to LAMMPS (converted to LAMMPS units) ---
  auto fa = f_t.accessor<double, 2>();
  for (int i = 0; i < nlocal; i++) {
    f[i][0] += force_conv * fa[i][0];
    f[i][1] += force_conv * fa[i][1];
    f[i][2] += force_conv * fa[i][2];
  }
  double e_model = e_t.item<double>() * energy_conv;
  if (eflag_global) eng_vdwl += e_model;

  static bool dbg_once = true;
  if (dbg_once) {
    dbg_once = false;
    utils::logmesg(lmp, "pair_molnex[dbg] nlocal={} nedges={} energy={:.6f} ({} units)\n",
                   nlocal, ne, e_model, update->unit_style);
  }

  if (vflag_fdotr) virial_fdotr_compute();
}
