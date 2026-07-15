/* PairMolnex — generic LAMMPS pair_style for any molnex potential exported via
   molix.engine.export_for_lammps. Evaluates the AOT-Inductor model through the
   molnex C++ runtime (molnex::interface::ModelRunner).

   One pair style, any model. Everything model-specific is read from the export
   directory's `<name>.meta.json` `lammps` block (cutoff, native units, compute
   dtype, capabilities) — there is no per-model C++. Model input/output glue lives
   on the Python export side as a molix.engine.EngineAdapter.

   Command:
       pair_style molnex <model_dir> [name]
       pair_coeff * * <Z_type1> <Z_type2> ... <Z_typeN>

   `name` is the artifact basename inside <model_dir> (default "model").

   Units: the model's native unit system is declared in meta.json (`units`:
   "real" = kcal/mol, "metal" = eV). The pair style reads LAMMPS' active `units`
   and converts energy/forces to it (distance is Å in both real and metal).

   Scope (this version): single MPI rank, non-periodic (`boundary f f f`) => no
   ghost atoms; edges from a FULL neighbour list within the cutoff. Domain
   decomposition / ghost reverse-communication / virial output are follow-ups
   (the meta.json `supports_pbc` / `supports_virial` flags are reserved for them).
*/

#ifdef PAIR_CLASS
// clang-format off
PairStyle(molnex,PairMolnex);
// clang-format on
#else

#ifndef LMP_PAIR_MOLNEX_H
#define LMP_PAIR_MOLNEX_H

#include "pair.h"

#include <memory>
#include <string>
#include <vector>

#include <ATen/core/Tensor.h>

namespace molnex::interface {
class ModelRunner;
}

namespace LAMMPS_NS {

class PairMolnex : public Pair {
 public:
  PairMolnex(class LAMMPS *);
  ~PairMolnex() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

 protected:
  double cut_global;                 // cutoff (Å), from meta.json
  std::string model_dir;
  std::string model_name;            // artifact basename inside model_dir
  std::vector<int> type2z;           // LAMMPS type (1-based) -> atomic number Z
  std::unique_ptr<molnex::interface::ModelRunner> runner;
  std::string device;                // "cpu" or "cuda", from the export meta
  std::string model_units;           // "real" or "metal", from meta.json
  bool fp64_model;                   // true if meta model_dtype == float64
  double energy_conv;                // model-energy-unit -> LAMMPS-energy-unit factor
  double force_conv;                 // model-force-unit  -> LAMMPS-force-unit  factor

  // CUDA-graph fast path (meta `cuda_graph`): fixed N + padded E_max, captured
  // once and replayed. Persistent device buffers are updated in place each step.
  bool use_cuda_graph = false;
  int n_fixed = 0;                   // fixed atom count N (meta `n_atoms`)
  int64_t e_max = 0;                 // fixed padded edge count (meta `e_max`)
  at::Tensor Zg_, posg_, edgeg_, maskg_;   // persistent device input buffers

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
