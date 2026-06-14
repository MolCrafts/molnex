# <Encoder> Specification

This page is the implementation contract for `molzoo.<Encoder>`. It is not a
tutorial; use the MolZoo user guide for theory narrative and worked examples.

| Field | Value |
|-------|-------|
| Module | `molzoo.<encoder>` |
| Entry point | `<Encoder>` (config `<Encoder>Spec`) |
| Paper | <citation> |
| arXiv | <url or not applicable> |
| DOI | <url or not applicable> |
| Reference implementation | `<org>/<repo>@<sha>` |
| Spec status | draft / partial / stable |

## 1. Scope

<What this encoder owns and does not own.>

## 2. Public Contract

### 2.1 Required Inputs

| Direction | TensorDict path | Shape | Dtype | Contract |
|-----------|------------------|-------|-------|----------|
| In | <...> | <...> | <...> | <...> |

### 2.2 Outputs

| Direction | TensorDict path | Shape | Written when | Contract |
|-----------|------------------|-------|--------------|----------|
| Out | <...> | <...> | <...> | <...> |

## 3. Forward Contract

### 3.1 Notation

| Symbol | Meaning | Code anchor |
|--------|---------|-------------|
| <...> | <...> | <...> |

### 3.2 <Pipeline Step>

$$
<equation>
$$

| Quantity | Shape | Code anchor |
|----------|-------|-------------|
| <...> | <...> | <...> |

## 4. Configuration Contract

| `<Encoder>Spec` field | Meaning | Default | Constraint / note |
|-----------------------|---------|---------|-------------------|
| <...> | <...> | <...> | <...> |

## 5. Reference Crosswalk

| Concept | Reference implementation | MolNex anchor | Status |
|---------|--------------------------|---------------|--------|
| <...> | <...> | <...> | matched / adapted / missing / unknown |

## 6. MolNex Adaptations

| ID | Adaptation | Reason | Risk | Validation |
|----|------------|--------|------|------------|
| A1 | <...> | <...> | low / medium / high | <test path or run id> |

## 7. Validation Contract

### 7.1 Research Reproduction

<Claim supported metrics, or say not claimed and why.>

### 7.2 Symmetry and Shape Tests

| Claim | Test path | Tolerance |
|-------|-----------|-----------|
| <...> | <...> | <...> |

### 7.3 Engineering Benchmark

<Driver and benchmark table.>

### 7.4 Run Log

| run_id | date | commit | dirty | dataset | config | steps | train_mae | val_mae | fwd_ms | bwd_ms | compiled | note |
|--------|------|--------|-------|---------|--------|-------|-----------|---------|--------|--------|----------|------|

## 8. System Boundary

| Concern | Owner | Contract |
|---------|-------|----------|
| <...> | <...> | <...> |

## 9. Version Pinning

| Item | Value |
|------|-------|
| Paper | <...> |
| Reference repository | <...> |
| Reference commit | <...> |
| Dependencies | <...> |
| Public docs mirror | `docs/molzoo/specs/<encoder>.md` |

## 10. Drift Policy

<Triggers and enforcement.>

## Appendix A. Maintenance Log

- <YYYY-MM-DD>: <short entry>
