# Data acquisition prerequisites — sonata-05-data sources

This file documents how to obtain the raw extended-XYZ files consumed by
`WaterLESSource`. The class ships with `download=False` by default; users
either fetch the files manually following the recipe below or pass
`download=True` to use the built-in `urllib.request`-based fetcher.

The SHA-256 digests below are pinned by the source class; if the
downloaded file does not match, `WaterLESSource.__init__` raises
`ValueError(f"checksum mismatch: expected <expected>, got <actual>")`.
Override by passing `WaterLESSource(..., verify_checksum=False)` (the
default) when knowingly shipping a different upstream snapshot, or by
leaving the placeholder digests in place until a contributor lands real
values.

Parsing is done by the in-tree `_extxyz.parse_extxyz_frames` helper.
There is no `ase` dependency anywhere in `src/molix/datasets/`.

## 1. WaterLESSource — bulk liquid water (RPBE-D3)

- Reference: Cheng B., *Latent Ewald summation for machine-learning
  potentials*, npj Comput. Mater. **11**:80 (2025),
  doi:10.1038/s41524-025-01577-7.
- Upstream code + data: <https://github.com/ChengUCB/les_fit>.
- Raw extxyz files (under
  `https://github.com/ChengUCB/les_fit/tree/main/data-benchmark`):
  - `train-H2O_RPBE-D3.xyz` — split internally as 0.95 / 0.05 train / val
    by deterministic tail-slice (matches upstream `lr_r45_nlayer3_lmax2.yaml`).
  - `test-H2O_RPBE-D3.xyz` — independent test file.
- Per-config layout: 64 H₂O molecules, 192 atoms, periodic cubic cell
  (~12 Å edge per Cheng 2025 §III.2). Energies in eV, forces in eV·Å⁻¹.

### Manual fetch

```bash
mkdir -p ~/datasets/water_les
cd ~/datasets/water_les
curl -L -O https://raw.githubusercontent.com/ChengUCB/les_fit/main/data-benchmark/train-H2O_RPBE-D3.xyz
curl -L -O https://raw.githubusercontent.com/ChengUCB/les_fit/main/data-benchmark/test-H2O_RPBE-D3.xyz
```

Then construct the source pointing at that directory:

```python
from molix.datasets import WaterLESSource

train = WaterLESSource("~/datasets/water_les", split="train")
val   = WaterLESSource("~/datasets/water_les", split="val")
test  = WaterLESSource("~/datasets/water_les", split="test")
```

### Auto-download

`WaterLESSource(..., download=True)` fetches both files via
`urllib.request.urlretrieve` from the raw URLs above into `root` if
missing.

### SHA-256 digests

The upstream `les_fit` repo is a moving target; the digests here record
the file as of the spec's creation date (2026-05-10). Bump them in lockstep
with `WaterLESSource.SOURCE_VERSION` whenever the upstream snapshot
changes. The placeholder strings below are intentionally invalid — the
contributor running the first manual fetch MUST replace them with the
real digests via `sha256sum train-H2O_RPBE-D3.xyz` and commit the
result. Until then, instantiate with the default `verify_checksum=False`.

| File | Expected SHA-256 |
|------|------------------|
| `train-H2O_RPBE-D3.xyz` | `0000000000000000000000000000000000000000000000000000000000000000` |
| `test-H2O_RPBE-D3.xyz`  | `0000000000000000000000000000000000000000000000000000000000000000` |

## 2. Charged dimers — deferred

Charged molecular dimers data are deferred to a future OOD-only spec
(provisional slug `sonata-06-ood-dimers`); they were temporarily exposed
during an earlier draft of `sonata-05-data` but are out of scope here.
Do not re-introduce a corresponding source class without a new spec.

## Why this file lives in `src/`

Co-locating data-acquisition recipes with the source classes that
consume them keeps the spec/source/recipe triple within one `git mv`
when the dataset is renamed. The file is markdown rather than Python
because it is read by humans, not imported.
