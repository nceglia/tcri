# Metrics: behavioural test plan and refactor

Working document. Deleted before merge; contents move to the PR body.

## 1. What each axis is supposed to mean

| axis | meaning | shape effect |
|---|---|---|
| `covariate=<v>` | restrict to one covariate level | one table |
| `covariate=None` | **all covariate levels, unified** | see §2 — currently inconsistent |
| `groupby=<col>` | compute the metric separately per group; requires clones disjoint across groups | tidy frame, one row per group |
| `splitby=<col>` | attach a group-level label for downstream comparison; only meaningful with `groupby` | adds a label column |
| `clones=[...]` | restrict the clone set | fewer rows |
| `weighted` | clone marginal P(c): `False` = 1/C, `True` ∝ n_c | changes the estimand |
| `normalize_mode` | `min` = I/min(H(c),H(φ)); `average` = eq 6 | changes the normalizer |
| `n_samples` | 0 = plug-in; >0 = posterior draws + HDI | scalar vs summary dict |
| `temperature` | analysis-time temper of the mean | changes the table |

## 2. Measured behaviour of `covariate=None` — three different answers

Fixture: 10 clones × 2 covariates × 4 phenotypes, fitted.

| call | returns | covariate axis |
|---|---|---|
| `joint_distribution(covariate=None)` | `DataFrame(39, 4)`, MultiIndex `(covariate, clonotype)` | **kept**, stacked |
| `mutual_information(covariate=None)` | `float` | **stacked** into one table, then scored |
| `clonotypic_entropy(covariate=None)` | `Series[4]` indexed by **phenotype** | **collapsed** |
| `phenotypic_entropy(covariate=None)` | `Series[39]` indexed by `(covariate, clonotype)` | **kept** |

Three unifications for one argument value.

### 2.1 The stacking defect (release-blocking)

`tcri/tools/_common.py:74` concatenates per-covariate blocks row-wise:

```python
arr = np.concatenate([J[s] for _m, _ci, J in blocks], axis=0)
```

So a clone present in *k* covariates becomes *k* rows. The row axis of the joint is
`(covariate, clone)`, not `clone`. `H(c)` is then the entropy over pseudo-clones.

Measured, same fitted model:

| | stacked (current) | marginalised over covariate | diff |
|---|---|---|---|
| `NMI min` (package default) | 0.265389 | 0.265385 | 0.000003 |
| `NMI average` (note's eq 6) | 0.141953 | 0.164115 | **0.022161** |

`min` picks `H(φ)` as the denominator here, which row-splitting does not touch — so the
defect is **invisible at the default and material in the mode the note's benchmark
requires**. This is the same shape as DE-2/DE-3: latent at defaults, wrong when it matters.

**What `covariate=None` should mean.** `I(c;φ)` with the covariate marginalised out:
sum the per-covariate blocks over the covariate axis into one clone × phenotype table,
then score. Not a stack, and not a mean of per-covariate scores.

### 2.2 The entropy inconsistency (release-blocking)

`clonotypic_entropy(covariate=None)` collapses to a per-phenotype Series;
`phenotypic_entropy(covariate=None)` keeps a `(covariate, clonotype)` index. Both are
"all covariates" but only one is unified. They must agree.

### 2.3 `joint_distribution(groupby=...)` raises (release-blocking)

Declared in `_contract.pyi` and in the signature; the body raises `NotImplementedError`.
This is a sixth instance of the class in issue #64 — declared, shaped correctly, not live —
and the worst kind, because it fails loudly only when someone uses it.

## 3. Synthetic design — concatenated blocks with closed-form truth

Requirement: one AnnData combining independent synthetic sets under different labels, such
that every axis has a *known expected value*, not merely "it runs".

**Construction.** `B` independent `simulate_tcri` blocks with deliberately different
coupling strength (via `omega_concentration`), each carrying:

- its own `patient` label — `P0 … P{B-1}`
- **disjoint clone ids** (`clone_i@P{b}`), which is what makes `groupby` legal
- 2 covariate levels, with different coupling in each
- a `response` label constant within patient (`R` / `NR`), so `splitby` has something to carry

Concatenated with `anndata.concat`, then `setup_anndata` + fit + `to_anndata` once.

**Why this gives expected values.** Clone ids are disjoint across patients, so the cells of
patient *b* are exactly block *b*. The metric at `groupby="patient"` restricted to block *b*'s
clones must equal the metric computed directly on block *b*'s own crosstab — computable
without the model, from `obs` alone, via `mi_from_joint_oracle`. Every group assertion is
therefore an equality against an independently computed number.

**Blocks differ on purpose.** If all blocks had the same truth, a `groupby` that silently
ignored its argument would still pass. Different per-block MI means an inert `groupby` is
detectable — the same trap that let `permutation_null`'s `groupby` survive.

**Feasibility: yes.** No new generator machinery is needed; `simulate_tcri` plus
`anndata.concat` plus a clone-id suffix is sufficient. Nothing here requires a fitted model
to produce the expected value, so the assertions do not depend on fit quality.

## 4. Test plan

### 4.1 Axis liveness — every argument changes the answer

For each of `mutual_information`, `clonotypic_entropy`, `phenotypic_entropy`,
`joint_distribution`, `phenotypic_flux`:

- `covariate="cov_0"` ≠ `covariate="cov_1"` when the blocks differ
- `groupby="patient"` returns one row per patient, and the values are **not** all equal
- `splitby="response"` adds the label column, correct value per patient
- `clones=[subset]` changes the result and returns only that subset
- `weighted=True` ≠ `weighted=False`
- `normalize_mode="min"` ≠ `"average"`
- `n_samples>0` returns mean/sd/HDI with `hdi_low ≤ mean ≤ hdi_high`
- `temperature≠1` changes the result

### 4.2 Correctness against closed-form truth

- `groupby="patient"` value for block *b* == metric on block *b*'s own crosstab
- `covariate=None` == metric on the covariate-marginalised table
- `covariate=None` ≠ the stacked table (pins the fix; fails on `main` at `average`)
- `clonotypic_entropy` and `phenotypic_entropy` agree on what `covariate=None` unifies
- MI keystone identity `I(c;φ) = H(c) − Σ_φ P(φ)·H[P(c|φ)]` holds under every axis setting

### 4.3 Degenerate and edge cases

- single covariate: `covariate=None` == `covariate=<the only one>`
- a clone absent from one covariate (ragged blocks) does not produce NaN
- `groupby` on a non-disjoint clone column raises (existing guard)
- empty `clones=[]` raises rather than returning an empty frame silently

## 5. Proposed implementation changes

1. **`_common.joint_draws`** — add a `collapse_covariate` path that sums blocks over the
   covariate axis instead of stacking, and use it for the scalar metrics at `covariate=None`.
   `joint_distribution` keeps the labelled `(covariate, clonotype)` frame, which is its
   documented job.
2. **`clonotypic_entropy` / `phenotypic_entropy`** — one shared rule for `covariate=None`.
3. **`joint_distribution(groupby=...)`** — implement, or remove from the contract. Implementing
   is preferred and mirrors what `permutation_null` just did.
4. **Metrics contract** — declare the `covariate=None` semantics so it cannot drift again.

## 6. Open questions for the audit

- Does collapsing change `weighted=True`? The clone marginal is over the collapsed table, so
  `n_c` must be summed across covariates too — needs checking, not assuming.
- Does `phenotypic_flux` share the stacking path? It takes `cov_from`/`cov_to` explicitly, so
  probably not, but it must be verified rather than reasoned about.
