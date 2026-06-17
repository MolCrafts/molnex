/* LAMMPS plugin registration for pair_style molnex. Loaded at runtime via
   `plugin load molnexplugin.so`. Generic over any molnex potential exported
   with molix.lammps.export_for_lammps. */

#include "lammpsplugin.h"
#include "version.h"

#include "pair_molnex.h"

using namespace LAMMPS_NS;

static Pair *molnexcreator(LAMMPS *lmp)
{
  return new PairMolnex(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "molnex";
  plugin.info = "Generic molnex AOTInductor pair style (molix.lammps)";
  plugin.author = "molnex";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &molnexcreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
