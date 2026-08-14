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

   Ghost / PBC:
       When meta `supports_pbc` is true, the pair style builds the graph over
       local + ghost atoms (ghost positions already carry the minimum-image
       placement), scatters forces onto ghosts, and reverse-communicates them
       to owners. Multi-rank domain decomposition is allowed in that mode.
       When `supports_pbc` is false, the pair style requires a single rank
       and no ghost atoms (open-boundary, one domain).

   Virial:
       When meta `supports_virial` is true and the model returns a third
       output of shape (6,) (xx,yy,zz,xy,xz,yz in model energy units), it is
       accumulated into the global virial. Otherwise `virial_fdotr_compute`
       is used when LAMMPS requests virial via fdotr.
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
#include <utility>
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

  // ghost force reverse-comm (used when supports_pbc)
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

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

  // capability flags from meta.json
  bool supports_pbc = false;         // ghost-aware graph + reverse_comm
  bool supports_virial = false;      // model emits a 3rd virial output

  // CUDA-graph fast path (meta `cuda_graph`): fixed N + padded E_max, captured
  // once and replayed. Persistent device buffers are updated in place each step.
  // Requires no ghosts (supports_pbc models with ghosts use the dynamic path).
  bool use_cuda_graph = false;
  int n_fixed = 0;                   // fixed atom count N (meta `n_atoms`)
  int64_t e_max = 0;                 // fixed padded edge count (meta `e_max`)
  at::Tensor Zg_, posg_, edgeg_, maskg_;   // persistent device input buffers

  // host staging reused across steps (avoids per-step vector realloc)
  std::vector<std::pair<int, int>> edge_pairs_;
  std::vector<int64_t> edge_flat_;
  std::vector<int64_t> edge_pad_flat_;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
