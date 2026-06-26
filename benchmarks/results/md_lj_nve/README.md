# LJ₁₃ NVE energy-conservation artifact

Gold-standard integrator test for the compiled `molix.md` engine (spec
`md-component-engine-01`, acceptance criterion ac-008). A 13-atom argon
Lennard-Jones cluster (icosahedral minimum) integrated under **NVE** (γ=0,
BAOAB → velocity-Verlet) for **5 ns** with a `torch.compile(fullgraph=True)`
single step.

## Result

| quantity | value |
|---|---|
| steps | 1 000 000 (dt = 5 fs) |
| rel. energy drift `|slope·dur|/|E₀|` | **4.31e-07** (bound 1e-3) |
| rel. RMS energy fluctuation | 3.05e-06 |
| total energy bound | ±15 ppm over 5 ns |

No systematic drift — the integrator is symplectic and conserves energy.

## Files

- `lj13_nve.npz` — `time_ps (500,)`, `e_total/e_pot/e_kin (500,)`,
  `positions (500, 13, 3)`, `meta` (provenance string). Units (amu, Å, fs);
  energy in amu·Å²/fs².
- `lj13_nve_energy.png` — top: relative total-energy drift (ppm); bottom:
  PE / KE / total energy vs time.

## Reproduce

```bash
PYTHONPATH=src:. python benchmarks/verify_md_lj_nve.py        # full 5 ns
PYTHONPATH=src:. python benchmarks/verify_md_lj_nve.py --ps 100  # short smoke
```

Parameters: argon (ε = 0.0103 eV, σ = 3.4 Å, m = 39.95 amu), T₀ = 20 K,
seed = 1. All-pairs interaction (no cutoff) so the PES is exact for any
displacement.
