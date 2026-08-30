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
#include <optional>
#include <string>

#include "bridge/exec.h"
#include "bridge/force_scatter.h"
#include "bridge/graph_bridge.h"
#include "bridge/lammps_view.h"
#include "bridge/species_map.h"
#include "lammps_contract.h"

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
  std::unique_ptr<molnex::interface::ModelRunner> runner;
  std::string device;                // "cpu" or "cuda", from the export meta
  std::string model_units;           // "real" or "metal", from meta.json
  double energy_conv;                // model-energy-unit -> LAMMPS-energy-unit factor

  // Bridge components. Each is empty until the command that configures it has
  // run — the contract, the scatter, the graph and the execution strategy at
  // `pair_style` (settings), the species map at `pair_coeff` (coeff) — and
  // `compute()` establishes with one guard at its top that all four hold a
  // value, so nothing below it re-asks.
  //
  // std::optional for the first three: cheap, complete types, so a heap
  // allocation would buy nothing and "not configured yet" is a state of the
  // pair style rather than an ownership question. std::unique_ptr for `exec_`
  // only because `Exec` is polymorphic — which of DynamicExec / GraphedExec was
  // constructed IS the execution choice, so there is no flag beside it to
  // consult and no branch on execution anywhere in compute().
  //
  // Two values deliberately absent beside them, on the same rule: the last
  // reader leaves, the value leaves. `force_conv`'s reader was the scatter
  // loop, now ForceScatter's; the model-dtype flag's was the `pos` cast at the
  // execution site, now each Exec's — which reports that dtype's name back for
  // the startup message. A second copy here could only drift from the one that
  // acts.
  std::optional<molnex::lammps::LammpsContract> contract_;    // meta, cached for coeff()
  std::optional<molnex::lammps_bridge::SpeciesMap> species_map_;      // type -> Z, validated
  std::optional<molnex::lammps_bridge::ForceScatter> force_scatter_;  // holds force_conv
  std::optional<molnex::lammps_bridge::GraphBridge> graph_bridge_;    // holds cutoff^2
  std::unique_ptr<molnex::lammps_bridge::Exec> exec_;                 // dynamic or graphed

  // capability flags from meta.json
  bool supports_pbc = false;         // ghost-aware graph + reverse_comm
  bool supports_virial = false;      // model emits a 3rd virial output

  // The bridge's window onto LAMMPS' arrays for the current step, assembled in
  // one place because three call sites need it: compute() reads every field,
  // and the two reverse-comm callbacks read only x / f / type / nlocal /
  // nghost. The view borrows and owns nothing, so it is rebuilt per call rather
  // than cached across a reneighbouring (which moves the arrays it points at).
  molnex::lammps_bridge::LammpsView current_view() const;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
