# Metrics contract

What the numbers mean. This file is the definition of every `tl` metric; the block at the end
pins the constants, the defaults, and the values on a reference joint.
`tests/test_metrics_contract_conformance.py` enforces it: each equation below is recomputed
from a hand-written reference on that joint, the identities are asserted, the defaults are read
off the live signatures, and the pinned values are compared to a tolerance. Nothing else binds
the metrics.

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

## The machine-checked part

```python contract
LOG_BASE = 2

# every public tl function; each has a section above and a pinned default set below
METRICS = [
    "joint_distribution", "clonotypic_entropy", "phenotypic_entropy", "mutual_information",
    "phenotypic_flux", "delta_clonotypic_entropy", "delta_phenotypic_entropy",
]

# signature defaults that change what a number means; read off the live functions
DEFAULTS = {
    "weighted": False,
    "normalized": True,
    "normalize_mode": "min",
    "distance_metric": "kl",
    "n_samples": 0,
    "temperature": 1.0,
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
