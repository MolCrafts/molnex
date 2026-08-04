# MACE-OMOL port — verification scripts

Goal: extend MolNex (molrep/molpot/molzoo, cuEquivariance) so it can load the
official **MACE-OMOL** foundation model (`MACE-omol-0-extra-large-1024.model`,
`ScaleShiftMACE`, 1024 ch, r_max=6.0, 3 interactions, correlation=3, 83 elements)
and reproduce its energy/forces.

These scripts check our ported blocks **bit-for-bit against the official
`mace-torch`** on CPU (float64). They load the plain-torch molrep/molpot modules
in isolation (stubbing `molix.config`) so no cuequivariance is required.

## Reference env
- CPU venv with official mace: `work/.mace-ref` (mace-torch 0.3.16, e3nn 0.4.4).
- OMOL checkpoint + dumped inventory: `work/mace_models/`
  (`omol_inventory.json`, full block reference `OMOL_REFERENCE.md`).

## Run
```bash
cd /nobackup/proj/disk/teoroo/personal/jicli594/work
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 .mace-ref/bin/python \
    molcrafts/molnex/scripts/omol_port/verify_radial.py
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 .mace-ref/bin/python \
    molcrafts/molnex/scripts/omol_port/verify_e0_scaleshift.py
```

## Status (CPU-verified, max|diff| ~ machine eps)
- `verify_radial.py` — `molrep.embedding.BesselRBF` (normalize=False, eps=0,
  trainable=True) vs `mace BesselBasis`; `molrep.embedding.PolynomialCutoff` vs
  `mace PolynomialCutoff`. PASS.
- `verify_e0_scaleshift.py` — `molpot.heads.AtomicReferenceEnergy` vs
  `mace AtomicEnergiesBlock`; `molpot.heads.GlobalRescale` vs
  `mace ScaleShiftBlock` (single head). PASS.

## Remaining (equivariant — verify on aarch64 GPU, needs cuequivariance)
charge/spin joint embedding (project layer), RealAgnosticResidual**NonLinear**
InteractionBlock (linear_up/conv_tp/skip_tp/gate/linear_1/2/res/density_fn),
SymmetricContraction product, NonLinearBiasReadout; then the e3nn(mul_ir) →
cuEq(ir_mul) weight converter and full end-to-end energy/force comparison.
