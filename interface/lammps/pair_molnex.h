/* PairMolnex — one LAMMPS pair_style for any deployed model that follows the
   atom-energy convention (molix.deploy.AtomEnergyModel). The model is opened
   through the engine-neutral runtime (molnex::runtime::ModelArtifact), in
   whatever package format its metadata names.

   One pair style, any model. The model directory holds `<name>.json`
   (molix.deploy.DeployMetadata: format, cutoff, species, dtype, depth, the
   model's own units, op libraries) and the package itself (`<name>.pt2` for
   the aoti format). There is no per-model C++: a third party writes a module
   whose forward returns (energy, forces, atom_energy), packages it with
   molix.export, and writes its metadata.

   THIS FILE IS THE PROTOCOL LAYER, AND NOTHING ELSE. It is the only file in the
   plugin that includes a LAMMPS header, and in exchange it holds no capability
   flag, no neighbour loop and no branch on what a model can do. The one model
   fact it computes is the energy factor, because LAMMPS' own conversion table
   (utils::get_conversion_factor) is reachable only from here.
   Three layers meet under it:

       bridge/  LAMMPS-free primitives (SpeciesMap, stage_nodes, GraphBridge,
                ForceScatter) that do not know the model exists;
       impls/   LAMMPS-free strategy over the runtime — ContractImpl and,
                today, CutoffGraphImpl;
       here     parameter pipe, unit factor, allocate, neighbour request,
                forwarding, tally.

   Everything this class used to decide for itself — how many atoms are graph
   nodes, whether ghosts are allowed, whether a reverse comm is owed — is now a
   `ContractImpl` answer, and what remains here is the mechanical execution of
   those answers. A new capability therefore lands as a new impl, not as another
   `if` in this file. Two dialects meet at that seam: everything below throws
   std::runtime_error and every callback here translates into
   `error->all(FLERR, ...)`, always from OUTSIDE the try (a LAMMPSException is a
   std::exception and would otherwise be self-caught and re-labelled).

   Command:
       pair_style molnex <model_dir> [name] [mode fold|ghost]
       pair_coeff * * <Z_type1> <Z_type2> ... <Z_typeN>

   `name` is the artifact basename inside <model_dir> (default "model").
   `mode` (default: fold on one rank, ghost otherwise) is how periodic
   neighbours reach the model: `fold` evaluates the owned atoms only, each
   periodic image folded onto its owner with the lattice translation as a pair
   shift, and takes the virial from the model; `ghost` stages every ghost
   within T * cutoff + skin as a node and takes the virial from LAMMPS'
   virial_fdotr_compute. Edges are never sorted. The model returns energy,
   forces, per-atom energy and its strain derivative. See
   interface/lammps/README.md.
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

#include "bridge/lammps_view.h"
#include "impls/contract_impl.h"
#include "molnex/runtime/model_artifact.h"

namespace LAMMPS_NS {

class PairMolnex : public Pair {
 public:
  PairMolnex(class LAMMPS*);
  ~PairMolnex() override;
  void compute(int, int) override;
  void settings(int, char**) override;
  void coeff(int, char**) override;
  double init_one(int, int) override;
  void init_style() override;

  // ghost force reverse-comm; whether it runs at all is the impl's answer,
  // returned per step as EvalResult::needs_pair_reverse_comm.
  int pack_reverse_comm(int, int, double*) override;
  void unpack_reverse_comm(int, int*, double*) override;

 protected:
  // What the input script said, kept because the diagnostics name it: the
  // export directory and the artifact basename inside it.
  std::string model_dir;
  std::string model_name;

  // The whole state of this pair style. Declaration order is destruction order
  // reversed, and one pair depends on it: `impl_` borrows `artifact_`, so
  // `artifact_` is declared BEFORE `impl_` and therefore destroyed AFTER it.
  // Swapping the two would free the model out from under the execution still
  // holding it. The impl and the cached needs are the two things every
  // callback below reads.
  std::unique_ptr<molnex::runtime::ModelArtifact> artifact_;
  std::unique_ptr<molnex::lammps::ContractImpl> impl_;

  // The impl's request table, asked ONCE at `pair_style` time and executed
  // verbatim afterwards: `init_style` turns `full` / `ghost` into NeighConst
  // flags, `init_one` returns `cutoff`, and `settings` copies the two comm
  // sizes into LAMMPS' own `comm_reverse` / `comm_reverse_off`. Cached rather
  // than re-asked because LAMMPS reads `comm_reverse` between callbacks, so the
  // answer must be one fixed table rather than a per-call question.
  molnex::lammps::NeighborNeeds needs_{};

  // The bridge's window onto LAMMPS' arrays for the current step, assembled in
  // one place because three call sites need it: compute() passes it to the
  // impl, and the two reverse-comm callbacks pass it to the impl's forwarders.
  // The view borrows and owns nothing, so it is rebuilt per call rather than
  // cached across a reneighbouring (which moves the arrays it points at).
  // Assembling this aggregate is the whole of the facade's data handling: the
  // pointers go over the seam untouched, and no tensor is built on this side.
  molnex::lammps_bridge::LammpsView current_view() const;

  // The folded mode (`pair_style ... mode fold`): fixed at pair_style.
  bool fold_ = false;
  // Rebuilt per step in the folded mode by fold_owners(): the owned atom each
  // local or ghost atom is a copy of (the view's `owner`), and the scratch
  // ID -> local table it is read from. Members so the storage is reused.
  std::vector<int> owner_;
  std::vector<int> local_of_tag_;
  void fold_owners();

  virtual void allocate();

  // One model energy unit in the active LAMMPS unit style. The model's unit
  // is mapped onto the LAMMPS style that uses it (eV: metal, kcal/mol: real)
  // and converted with LAMMPS' own table; anything else is refused.
  double energy_factor(const std::string& model_energy_unit) const;
};

}  // namespace LAMMPS_NS

#endif
#endif
