# Metrics contract

What the numbers mean. This file is the definition of every `tl` metric and of every `perturb`
query; the block at the end pins the constants, the defaults, and the values on a reference
joint. `tests/test_metrics_contract_conformance.py` enforces it: each equation below is
recomputed from a hand-written reference on that joint, the identities are asserted, the
defaults are read off the live signatures, and the pinned values are compared to a tolerance.
The perturbation is not a function of the joint, so it has no pinned value; its identities are
enforced by `tests/test_perturbation.py`. Nothing else binds the metrics.

Equation numbers are this document's; 2 to 6 keep the numbering of the manuscript's metrics
section, which is not kept in the repository.

## The substrate

Every metric is a function of a clone × phenotype joint at one covariate level, produced by
`joint_distribution` from the fitted clone × covariate distributions `p_ct` (see
`API_CONTRACT.md` for the engine's arguments). All entropies and mutual information are in
**bits**, log base 2.

## Which clones count

A clone is a label of the registered clonotype column. `pp.pool_rare_clones` gives each group's
rare clones one `pooled@{group}` label and lists the pool labels in the derivation record. A pool
is a mixture of many rare clones, not a clone, so at `exclude_pools=True` (the default) every
metric counts the registered labels minus the pools: exactly the clones `pool_rare_clones` kept.
An explicit `clones=` selects among them; `exclude_pools=False` counts each pool as one clone. A
column with no pooling step has nothing to exclude, whatever its labels look like. No tool applies
a second threshold, per covariate level or otherwise: the pooling rule is the only rule for which
clones count.

The counted clones are the rows of the joint, so they are the clones `weighted=False` gives equal
mass, the clones a normalizer counts in `log₂ C`, and the support of a flux or a delta.

## Definitions

### `clonotypic_entropy`, one value per phenotype (eq 3)

```
H[P(c|φ)] = − Σ_c P(c|φ) log₂ P(c|φ)
```

How spread a phenotype is across clones. **Support-only:** clones with zero mass in that
column are dropped before renormalizing; there is no epsilon clip, which would fabricate
uniform mass on absent clones. Normalizer `log₂(number of supported clones)`, or
`log₂(n_clones_ref)` when given, so that groups with different clone counts are comparable.
An empty column is **NaN**.

### `phenotypic_entropy`, one value per clone (eq 4)

```
H[P(φ|c)] = − Σ_φ P(φ|c) log₂ P(φ|c)
```

Plasticity versus commitment of a clone. All P phenotypes are in the sum with `0·log 0 = 0`.
Normalizer `log₂(P)`. A clone with zero mass is **NaN**, never reindexed to zeros (which would
report H = 1 for a clone that was never observed).

### `mutual_information`, one value per joint (eqs 5, 6)

```
I(c;φ) = Σ_{c,φ} P(c,φ) log₂ ( P(c,φ) / (P(c) P(φ)) )
```

The joint is renormalized to sum to 1; a numerical floor of 1e-15 guards `log 0`.
Normalized forms: `normalize_mode="min"` divides by `min(H(c), H(φ))` (the coefficient of
constraint) and is the **default**; `"average"` divides by `½(H(c) + H(φ))`, which is eq 6 of
the manuscript's numbering. `min` is the default because the mean denominator scales with
`log₂(C)` and so is not comparable across groups with different clone counts. Anything
reproducing the manuscript's benchmark passes `normalize_mode="average"` explicitly.

Joint entropy (eq 2, `H(c,φ) = −Σ P log₂ P`) is not exposed; the test reaches it through
`H(c,φ) = H(c) + H(φ) − I(c;φ)`.

### The replicate values

Mutual information is one value per replicate: the joint is the replicate's at the covariate.
The two entropies have a replicate value beside their item values, the item values weighted by
their shares of that joint:

```
H(P|C) = Σ_c P(c) H[P(φ|c)]       phenotypic entropy of a replicate
H(C|P) = Σ_φ P(φ) H[P(c|φ)]       clonotypic entropy of a replicate
```

`P(c)` is `1/C` at `weighted=False` and the clone's cell share at `weighted=True` (see "The
`weighted` axis"); `P(φ)` is the phenotype's share of the joint. In bits,
`I(c;φ) = H(φ) − H(P|C) = H(c) − H(C|P)`. A clone or phenotype with zero mass has zero weight
and does not make the value NaN; an empty joint, a replicate with no cells at the covariate or
no clone left by `clones=`, is NaN. Normalizers: `log₂(P)` for `H(P|C)`; `log₂ C` for `H(C|P)`,
C the clones the replicate has at the covariate, or `log₂(n_clones_ref)` when given.

A normalized replicate value whose normalizer is 0 raises `ValueError` naming the metric, the
replicate and the covariate level: one clone for clonotypic entropy, one phenotype for phenotypic
entropy, and for mutual information one clone or one phenotype at `normalize_mode="min"`, one
clone and one phenotype at `"average"`. The item values keep their own rules: a phenotype with
one supported clone has clonotypic entropy 0.

### `phenotypic_flux`, one value per clone (eq 7)

```
D( P(φ|c) at cov_from ‖ P(φ|c) at cov_to )
```

`distance_metric="kl"` (default) is `Σ_φ p log₂(p/q)` in bits with both rows floored at 1e-12
and renormalized; `"l1"` is `Σ_φ |p − q|` in [0, 2]. A clone absent from either level is
**NaN**. Not normalized: a divergence has no maximum-entropy reference.

### `delta_clonotypic_entropy`, `delta_phenotypic_entropy`

`value(cov_to) − value(cov_from)` per item, computed **within a posterior draw** so the
summary is of the difference's distribution, not a difference of summaries. Only the two
metrics with an item axis have a delta; mutual information has none, so its difference is a
subtraction the caller performs. The clone set is the intersection of clones present at both
levels within each replicate, which for the clonotypic form makes the normalizer identical on
both sides; the drop is warned about.

## The perturbation (`tcri.perturb`)

A query on the fitted model with its parameters held fixed. Both functions score cells with
the per-cell rule of `predict()` (`MODEL_CONTRACT.md`, "Prediction"): `μ_i` is the encoder's
posterior mean and no encoder sample is drawn, so the pass is deterministic given the
expression matrix. `use_gate=True` (default) is the model's own rule, the gate when the model
has one and the additive rule when it has none; `use_gate=False` is the head alone,
`softmax(f_cls(μ_i))`, which does not depend on the clone × covariate prior at all.

### `knockout`, one probability vector per cell

```
X^(J)       = X with the columns in J set to zero
φ_i(X^(J))  = softmax( π·f_cls(μ_i(X^(J))) + (1−π)·log p_ct[g(i)] )
```

Every gene in `J` is silenced together, so a gene program is one call. `knockout(genes=[])`
is `predict()` exactly, and `knockout(genes=[j])` equals `predict()` on the matrix with column
`j` zeroed.

### `gene_importance`, one value per gene

```
φ̄_C(X)     = (1/|C|) Σ_{i∈C} φ_i(X)
shift_j,p   = φ̄_C(X)_p − φ̄_C(X^(j))_p
I_j         = Σ_p | shift_j,p |                                in [0, 2]
```

`C` is every cell of a counted clone ("Which clones count": at `exclude_pools=True` the pools'
cells are left out), or those at one `covariate` level, partitioned by `groupby` (one `I_j` per
group, the replicate unit of the `splitby` contrast, which is **per gene**). `shift` sums
to zero over phenotypes; a positive entry means silencing the gene removed mass from that
phenotype. A gene whose column is already zero has `I_j = 0` exactly. At `n_samples > 0` the
prior `p_ct` is drawn from the guide's Dirichlet posterior once, shared by every gene, and
enters only through the gate; `I_j` and `shift_j,p` are computed per draw and summarized as in
"Posterior summaries". With `use_gate=False` there is nothing to draw, and `n_samples` is
ignored with a warning. The reduction is over cells, so `covariate=None` is allowed and clones
need not be disjoint across groups.

`I_j` is the replicate level of the stored result, one row per gene and group, and `shift_j,p`
its item level, one row per gene, group and phenotype; `I_j = Σ_p |shift_j,p|` on every draw.
A group with no cell in `C` keeps its rows, NaN. Both are computed per draw on the fit and on the
fits of the reference ("References"), and **only the adjusted values are stored**: on each level,
`adjusted = value − null_value` per paired draw, with its summary (`adjusted`, `adjusted_sd`,
`adjusted_hdi_low`, `adjusted_hdi_high`); `value`, `null_value` and the null runs' own results are
not stored, and `null_model=None` raises. `stats` tests the adjusted importance. The adjusted
shift is a per-phenotype difference, not a decomposition of the adjusted importance: `Σ_p
|adjusted shift_j,p|` over a gene's item rows is not its adjusted importance, because the L1 norm
of a difference is not the difference of the L1 norms.

**Pairing for gene importance (rotated).** With `K` fits in the reference: at `n_samples = 0` the
fit's one value is paired with each of the `K` nulls, so a row has `K` pairs, `adjusted` is their
mean, `adjusted_sd` the spread between the nulls and the HDI is NaN. At `n_samples > 0` fit draw
`d` is paired with null `d mod K` only, and null `i` draws only the `|{d < n_samples : d mod K =
i}|` draws it is paired with, `n_samples / K` each when `n_samples` is a multiple of `K`, which
gives each null an equal share; `0 < n_samples < K` raises `ValueError`. Each draws frame has
`max(n_samples, K)` rows per summary row. The other metrics pair every draw with every null.

## The `weighted` axis

`weighted=False` (default): every clone contributes equally, `P(c) = 1/C`; the metric
describes the repertoire, one vote per clone. `weighted=True`: `P(c) ∝ n_c`; the metric
describes the cell population. These are different estimands. The manuscript's benchmark and
the simulator oracle are abundance-weighted, so reproducing them passes `weighted=True`.

## Posterior summaries

At `n_samples=0` a metric is the plug-in at the posterior mean, `F(E[p_ct])`. At `n_samples>0`
it is `E_s[F(p_ct^(s))]`, the metric evaluated on each draw of the joint and then summarized
(mean, sd, highest-density interval). The two differ by a Jensen gap: plug-in entropy is ≥ the
posterior mean, plug-in flux is ≤ it, and the sign is indeterminate for mutual information. No
test may equate them. The return shape does not change with `n_samples`; `sd`, `hdi_low` and
`hdi_high` are NaN at `n_samples ≤ 1`.

## Identities the test enforces

| identity | what it catches |
|---|---|
| uniform over k → `log₂ k`, normalized 1 | a wrong base or normalizer |
| all mass on one outcome → 0 | sign or normalization errors |
| zero-mass clone or phenotype → NaN item value | the spurious H = 1 reindexing regression |
| zero-mass clone or phenotype → zero weight in a replicate value | a NaN item turning its replicate NaN |
| support-only normalization | an epsilon clip creeping back in |
| independent joint → I = 0 | a broken MI |
| `I(c;φ) = I(φ;c)`, `I ≥ 0` | transpose or sign errors |
| permutation joint → normalized I = 1 (`min`) | a wrong denominator |
| **`I(c;φ) = H(c) − Σ_φ P(φ) H[P(c\|φ)]`** | redefining either family alone |
| weighting by the marginal instead of the conditional exceeds `log₂ C` and gives I < 0 | the natural mis-transcription of eqs 3 and 4 |
| `H(P\|C) = Σ_c P(c) H[P(φ\|c)]`, `H(C\|P) = Σ_φ P(φ) H[P(c\|φ)]` | replicate values weighted by anything but the joint's shares |
| `I(c;φ) = H(φ) − H(P\|C) = H(c) − H(C\|P)` | a replicate value that disagrees with mutual information |
| row-normalized joint: `H(P\|C)` is the plain mean, normalized `H(C\|P) = 1 − I / log₂ C` | wrong `weighted=False` weights |
| a normalizer of 0 raises | a replicate value reported as 0 or NaN when its normalizer is 0 |

The decomposition is the keystone: it ties the entropy and MI families together so neither can
change alone.

## References

Every quantity here is positive for data with no structure in it. Mutual information is positive
for any clone x phenotype table, an entropy is a number whatever the table says, a flux between
two fits is non-zero whenever the two differ at all, and an in-silico knockout moves the
phenotype call for every gene. **So no model-based number is reported bare.** Each scored metric
is read against a permutation reference: the same model, the same knobs, the same seed, the same
training arguments, fitted `n_nulls` times (default 5), each on its own permuted label vector, and
`adjusted = value - null_value` is the part of the observed number the permuted structure
accounts for.

**One reported quantity per metric.** A scored metric reports `adjusted` and stores its two inputs,
`value` and `null_value`, beside it; they are what a reader checking the correction needs, not
alternative answers. `stats` contrasts `adjusted`, or `value` when no usable reference was computed
(`null_model=None`, or a reference that is non-finite everywhere), and names the column it
contrasted in `quantity`. A delta's `adjusted_from` and `adjusted_to` are its endpoints and carry no
contrast, as `value_from` and `value_to` do not. A denominator has a reference (`null_denom`) and no
adjusted column: a difference of two normalizers is not a quantity anyone reports.

| quantity | read against | what the reference destroys |
|---|---|---|
| clonotypic entropy, phenotypic entropy, mutual information, gene importance | the phenotype null | which cell carries which phenotype |
| phenotypic flux, both deltas | the covariate null | which covariate level a cell sits at |

The reference is **the caller's own call on the null** -- the fit, `null_model=None` and the
seed of its draws changed -- and every other argument forwarded verbatim. `groupby`, `splitby`,
`clones`, `exclude_pools`, `weighted`, `normalized`, `normalize_mode`, `n_clones_ref`,
`distance_metric`, `temperature` and `n_samples` each change the estimand, so a reference
computed at defaults is a different quantity subtracted from a different quantity.

**`adjusted` is computed per paired draw, against every fit of the reference.** The null is the
value expected under no association. The fit and each null are independent fits, so draw *d* of
the fit is paired with draw *d* of every one of the reference's `n_nulls` fits and
`adjusted_{i,d} = value_d - null_value_{i,d}` (gene importance pairs by rotation instead, see
its section); the per-draw frames (each level's `draws`, or
`table`) carry `null` (which fit), `null_value` and `adjusted` on every pair, and the summaries
(each level's `summary`, or `result`) give the adjusted value's mean, `adjusted_sd` and HDI
(`adjusted_hdi_low`, `adjusted_hdi_high`) over all the pairs, like any value, and `null_value` as
the mean over them. Pooling the pairs is what puts the null's fit-to-fit spread into the interval:
one null's draws do not carry it, and the mean of the nulls' draws would hide it. Each null's
draws are seeded from their own seed, derived from the caller's `random_state` and the null's
index, so no two samples share noise. At `n_samples=0` each side is one point value, so a row has
one pair per null: `adjusted` is their mean, `adjusted_sd` the sd between the nulls, and no HDI
is reported (NaN), since an interval read off that few points is not one; the HDI is reported
only when `n_samples > 0`.

**Both denominators are stored.** A normalized MI divides by a normalizer taken from the same
joint it normalizes: at `weighted=False` the engine row-normalizes, so `h_c` is exactly
`log2(C)` and `normalize_mode="min"` selects `h_p`, the fitted phenotype marginal entropy, which
a null does not share. Its replicate summary therefore carries `denom` and `null_denom`, so a
reader can recover both numbers in bits as `value*denom` and `null_value*null_denom`. At `n_samples=0` that
recovery is exact. At `n_samples>0` it is not: `build_result` reduces each column over draws
independently, so the product is `E[I/D]·E[D]` rather than `E[I]` -- measured, 0.4429194 against
a per-draw mean of 0.4429281. Take the bits per draw from the replicate draws
(`tcri.get.draws`) when the difference matters.
The null's MI is never computed with the parent's denominator: that would break the one thing
`null_value` means, which is the metric OF the null.

**A delta's adjusted value is closed within its own result.** `adjusted - (adjusted_to -
adjusted_from)` equals `value - (value_to - value_from)`: the reference introduces no gap that was
not already there.
It is not zero, because `value == value_to - value_from` already fails when a draw has a
non-finite endpoint, and the two sides use different masks.

## The machine-checked part

```python contract
LOG_BASE = 2

# every public tl function; each has a section above and a pinned default set below
METRICS = [
    "joint_distribution", "clonotypic_entropy", "phenotypic_entropy", "mutual_information",
    "phenotypic_flux", "delta_clonotypic_entropy", "delta_phenotypic_entropy",
]

# every public perturb function; each has a section above. Not functions of the joint, so
# nothing is pinned on the reference joint; tests/test_perturbation.py holds the identities
PERTURBATIONS = ["knockout", "gene_importance"]

# signature defaults that change what a number means; read off the live functions
DEFAULTS = {
    "weighted": False,
    "normalized": True,
    "normalize_mode": "min",
    "distance_metric": "kl",
    "n_samples": 0,
    "temperature": 1.0,
    "use_gate": True,
    "null_model": "auto",
    "exclude_pools": True,
}

# the permutation axes, and the random stream each one's permutation is drawn from. Keyed by
# NAME, never positional: a reordering here must not silently change every recorded permutation
NULLS = ["phenotype", "clonotype", "covariate"]
OFFSET = {"phenotype": 10_000, "clonotype": 20_000, "covariate": 30_000}

# which null each scored quantity is read against at null_model="auto". joint_distribution has
# no entry (it returns a matrix, not a scored quantity); knockout has none (it returns the
# per-cell frame). The conformance test asserts the KEY SET, so a metric added later cannot
# quietly acquire no default
DEFAULT_NULL = {
    "clonotypic_entropy":       "phenotype",
    "phenotypic_entropy":       "phenotype",
    "mutual_information":       "phenotype",
    "phenotypic_flux":          "covariate",
    "delta_clonotypic_entropy": "covariate",
    "delta_phenotypic_entropy": "covariate",
    "gene_importance":          "phenotype",
}

# the reference joint (3 clones x 2 phenotypes, deliberately asymmetric) and the values the
# kernels must return on it, to 1e-9. Changing a definition changes these numbers.
GOLDEN = {
    "joint": [[4.0, 1.0], [1.0, 1.0], [1.0, 6.0]],
    "clonotypic_entropy": {"raw": [1.251629167388, 1.061278124459], "normalized": [0.789690082143, 0.669591945536]},
    "phenotypic_entropy": {"raw": [0.721928094887, 1.0, 0.591672778582], "normalized": [0.721928094887, 1.0, 0.591672778582]},
    "mutual_information": {"raw": 0.288703141426, "min": 0.293031766823, "average": 0.238914701013},
    # flux between row 0 and row 2 of the joint, as P(phi|c) at two levels
    "phenotypic_flux": {"kl": 1.568434327026, "l1": 1.314285714286},
    # H(P|C) and H(C|P) of the joint as given, and of its rows each normalized to sum to 1
    "replicate": {
        "joint": {"phenotypic_entropy": 0.696524994608,
                  "clonotypic_entropy": {"raw": 1.142857142857, "normalized": 0.721062575510}},
        "row_normalized": {"phenotypic_entropy": 0.771200291157,
                           "clonotypic_entropy": {"raw": 1.357209898776,
                                                  "normalized": 0.856304106980}},
    },
}
```
