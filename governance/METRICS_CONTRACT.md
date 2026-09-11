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

## Definitions

### `clonotypic_entropy`, one value per phenotype (eq 3)

```
H[P(c|φ)] = − Σ_c P(c|φ) log₂ P(c|φ)
```

How spread a phenotype is across clones. **Support-only:** clones with zero mass in that
column are dropped before renormalising; there is no epsilon clip, which would fabricate
uniform mass on absent clones. Normaliser `log₂(number of supported clones)`, or
`log₂(n_clones_ref)` when given, so that groups with different clone counts are comparable.
An empty column is **NaN**.

### `phenotypic_entropy`, one value per clone (eq 4)

```
H[P(φ|c)] = − Σ_φ P(φ|c) log₂ P(φ|c)
```

Plasticity versus commitment of a clone. All P phenotypes are in the sum with `0·log 0 = 0`.
Normaliser `log₂(P)`. A clone with zero mass is **NaN**, never reindexed to zeros (which would
report H = 1 for a clone that was never observed).

### `mutual_information`, one value per joint (eqs 5, 6)

```
I(c;φ) = Σ_{c,φ} P(c,φ) log₂ ( P(c,φ) / (P(c) P(φ)) )
```

The joint is renormalised to sum to 1; a numerical floor of 1e-15 guards `log 0`.
Normalised forms: `normalize_mode="min"` divides by `min(H(c), H(φ))` (the coefficient of
constraint) and is the **default**; `"average"` divides by `½(H(c) + H(φ))`, which is eq 6 of
the manuscript's numbering. `min` is the default because the mean denominator scales with
`log₂(C)` and so is not comparable across groups with different clone counts. Anything
reproducing the manuscript's benchmark passes `normalize_mode="average"` explicitly.

Joint entropy (eq 2, `H(c,φ) = −Σ P log₂ P`) is not exposed; the test reaches it through
`H(c,φ) = H(c) + H(φ) − I(c;φ)`.

### `phenotypic_flux`, one value per clone (eq 7)

```
D( P(φ|c) at cov_from ‖ P(φ|c) at cov_to )
```

`distance_metric="kl"` (default) is `Σ_φ p log₂(p/q)` in bits with both rows floored at 1e-12
and renormalised; `"l1"` is `Σ_φ |p − q|` in [0, 2]; `"jsd"` is the Jensen–Shannon divergence
in [0, 1] bit. A clone absent from either level is **NaN**. Not normalised: a divergence has no
maximum-entropy reference.

### `delta_clonotypic_entropy`, `delta_phenotypic_entropy`

`value(cov_to) − value(cov_from)` per item, computed **within a posterior draw** so the
summary is of the difference's distribution, not a difference of summaries. Only the two
metrics with an item axis have a delta; mutual information has none, so its difference is a
subtraction the caller performs. The clone set is the intersection of clones present at both
levels within each replicate, which for the clonotypic form makes the normaliser identical on
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

`C` is every cell, or the cells at one `covariate` level, partitioned by `groupby` (one `I_j`
per group, the replicate unit of the `splitby` contrast, which is **per gene**). `shift` sums
to zero over phenotypes; a positive entry means silencing the gene removed mass from that
phenotype. A gene whose column is already zero has `I_j = 0` exactly. At `n_samples > 0` the
prior `p_ct` is drawn from the guide's Dirichlet posterior once, shared by every gene, and
enters only through the gate; `I_j` is computed per draw and summarised as in "Posterior
summaries". With `use_gate=False` there is nothing to draw, and `n_samples` is ignored with a
warning. The reduction is over cells, so `covariate=None` is allowed and clones need not be
disjoint across groups.

## The `weighted` axis

`weighted=False` (default): every clone contributes equally, `P(c) = 1/C`; the metric
describes the repertoire, one vote per clone. `weighted=True`: `P(c) ∝ n_c`; the metric
describes the cell population. These are different estimands. The manuscript's benchmark and
the simulator oracle are abundance-weighted, so reproducing them passes `weighted=True`.

## Posterior summaries

At `n_samples=0` a metric is the plug-in at the posterior mean, `F(E[p_ct])`. At `n_samples>0`
it is `E_s[F(p_ct^(s))]`, the metric evaluated on each draw of the joint and then summarised
(mean, sd, highest-density interval). The two differ by a Jensen gap: plug-in entropy is ≥ the
posterior mean, plug-in flux is ≤ it, and the sign is indeterminate for mutual information. No
test may equate them. The return shape does not change with `n_samples`; `sd`, `hdi_low` and
`hdi_high` are NaN at `n_samples ≤ 1`.

## Identities the test enforces

| identity | what it catches |
|---|---|
| uniform over k → `log₂ k`, normalised 1 | a wrong base or normaliser |
| all mass on one outcome → 0 | sign or normalisation errors |
| zero-mass clone or phenotype → NaN | the spurious H = 1 reindexing regression |
| support-only normalisation | an epsilon clip creeping back in |
| independent joint → I = 0 | a broken MI |
| `I(c;φ) = I(φ;c)`, `I ≥ 0` | transpose or sign errors |
| permutation joint → normalised I = 1 (`min`) | a wrong denominator |
| **`I(c;φ) = H(c) − Σ_φ P(φ) H[P(c\|φ)]`** | redefining either family alone |
| weighting by the marginal instead of the conditional exceeds `log₂ C` and gives I < 0 | the natural mis-transcription of eqs 3 and 4 |

The decomposition is the keystone: it ties the entropy and MI families together so neither can
change alone.

## References

Every quantity here is positive for data with no structure in it. Mutual information is positive
for any clone x phenotype table, an entropy is a number whatever the table says, a flux between
two fits is non-zero whenever the two differ at all, and an in-silico knockout moves the
phenotype call for every gene. **So no model-based number is reported bare.** Each scored metric
is read against a permutation reference: the same model, the same knobs, the same seed, the same
training arguments, fitted on one permuted label vector, and `excess = value - null_value` is the
part of the observed number the permuted structure accounts for.

| quantity | read against | what the reference destroys |
|---|---|---|
| clonotypic entropy, phenotypic entropy, mutual information, gene importance | the phenotype null | which cell carries which phenotype |
| phenotypic flux, both deltas | the condition null | which condition a cell sits at |

The reference is **the caller's own call with two arguments changed** -- the fit and
`null_model=None` -- and every other argument forwarded verbatim. `groupby`, `splitby`, `clones`,
`weighted`, `normalized`, `normalize_mode`, `n_clones_ref`, `distance_metric`, `temperature` and
`n_samples` each change the estimand, so a reference computed at defaults is a different quantity
subtracted from a different quantity.

**Both denominators are stored.** A normalised MI divides by a normaliser taken from the same
joint it normalises: at `weighted=False` the engine row-normalises, so `h_c` is exactly
`log2(C)` and `normalize_mode="min"` selects `h_p`, the fitted phenotype marginal entropy, which
a null does not share. `result` therefore carries `denom` and `null_denom`, so a reader can
recover both numbers in bits as `value*denom` and `null_value*null_denom`. At `n_samples=0` that
recovery is exact. At `n_samples>0` it is not: `build_result` reduces each column over draws
independently, so the product is `E[I/D]·E[D]` rather than `E[I]` -- measured, 0.4429194 against
a per-draw mean of 0.4429281. Take the bits per draw from `table` when the difference matters.
The null's MI is never computed with the parent's denominator: that would break the one thing
`null_value` means, which is the metric OF the null.

**A delta's excess is closed within its own result.** `excess - (excess_to - excess_from)` equals
`value - (value_to - value_from)`: the reference introduces no gap that was not already there.
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
}

# the permutation axes, and the random stream each one's permutation is drawn from. Keyed by
# NAME, never positional: a reordering here must not silently change every recorded permutation
NULLS = ["phenotype", "clonotype", "condition"]
OFFSET = {"phenotype": 10_000, "clonotype": 20_000, "condition": 30_000}

# which null each scored quantity is read against at null_model="auto". joint_distribution has
# no entry (it returns a matrix, not a scored quantity); knockout has none (it returns the
# per-cell frame). The conformance test asserts the KEY SET, so a metric added later cannot
# quietly acquire no default
DEFAULT_NULL = {
    "clonotypic_entropy":       "phenotype",
    "phenotypic_entropy":       "phenotype",
    "mutual_information":       "phenotype",
    "phenotypic_flux":          "condition",
    "delta_clonotypic_entropy": "condition",
    "delta_phenotypic_entropy": "condition",
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
    "phenotypic_flux": {"kl": 1.568434327026, "l1": 1.314285714286, "jsd": 0.340842859252},
}
```
