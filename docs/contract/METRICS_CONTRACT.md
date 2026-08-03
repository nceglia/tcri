# Metrics Contract — what the numbers mean

Freezes the **information-theoretic metrics**: the two entropies and mutual
information over a clone × phenotype joint.

| | freezes | manifest | prose | test |
|---|---|---|---|---|
| API contract | the public *interface* | `tcri/_contract.pyi` | `tcri_api_and_responsibilities.md` | `test_contract_conformance.py` |
| Model contract | the *generative mathematics* | `tcri/model/_model_contract.py` | `MODEL_CONTRACT.md` | `test_model_contract_conformance.py` |
| **Metrics contract** | **what the metrics compute** | `tcri/tools/_metrics_contract.py` | this file | `test_metrics_contract_conformance.py` |

**Why separate from the model contract.** The two are verified by different means. The
model contract *traces* `model()`/`guide()` and inspects sample sites, plates and
distribution families. Metrics are pure functions of a joint table, so they are pinned
by **numeric identities** — uniform → log₂(k), independent → MI 0, and the
entropy/MI decomposition. Folding them together would force one mechanism to do a job
it is bad at.

Source of truth: **Supplementary Note 1**, "Entropy" section (eqs 2–6). Each equation
is transcribed literally into the conformance test; the one deliberate divergence
(`normalize_mode`) is documented below.

**Governance: update this file and the manifest FIRST, then the code.** A failing
conformance test means the *meaning of a published number* changed. Never relax an
identity to make it pass.

## Definitions (all in **bits**, log base 2)

### `clonotypic_entropy` — one value per **phenotype**

```
H[P(c|φ)] = − Σ_c P(c|φ) log₂ P(c|φ)
```

How spread a phenotype is across clones. **Support-only**: clones with zero mass in
that column are dropped *before* renormalizing — no epsilon clip, which would fabricate
uniform mass on absent clones and inflate H toward 1. Normalizer `log₂(#supported
clones)`, or `log₂(n_clones_ref)` when supplied. Empty column → **NaN**.

### `phenotypic_entropy` — one value per **clone**

```
H[P(φ|c)] = − Σ_φ P(φ|c) log₂ P(φ|c)
```

Plasticity vs commitment of a clone. All P phenotypes are in the sum with `0·log0 := 0`.
Normalizer `log₂(P)`. A clone with zero mass → **NaN**, never reindexed to zeros (which
would report a spurious `H=1` for a clone that was never observed).

### `mutual_information` — one value per joint

```
I(c;φ) = Σ_{c,φ} P(c,φ) log₂( P(c,φ) / (P(c)·P(φ)) )
```

Default `normalize_mode="min"` → `I / min(H(c), H(φ))`, the coefficient of constraint.
`"average"` → `I / (½(H(c)+H(φ)))`. **`min` is the default because the `average`
denominator scales with `log₂(C)` and is therefore not comparable across groups with
different clone counts.**

## Enforced identities

| identity | what it catches |
|---|---|
| uniform over k → `log₂(k)`, normalized `1.0` | a wrong log base or normalizer |
| all mass on one outcome → `0` | sign/normalization errors |
| zero-mass clone/phenotype → **NaN** | the spurious-`H=1` reindexing regression |
| support-only normalization | an epsilon clip creeping back in |
| independent joint → `I = 0` | a broken MI |
| `I(c;φ) = I(φ;c)`, `I ≥ 0` | transpose/sign errors |
| permutation joint → normalized `I = 1` | a wrong denominator |
| **`I(c;φ) = H(c) − Σ_φ P(φ)·H[P(c|φ)]`** | **redefining either family alone** |

That last one is the keystone: it ties entropy and MI together, so you cannot change
one without breaking it.

## Conformance with the manuscript equations

The manuscript's Entropy section (eqs 2–6) and the implementation agree. Each equation
is transcribed **literally** from the note in
`tests/test_metrics_contract_conformance.py` and asserted against the code — a stronger
check than the identity tests below, which pin *consequences* of a formula
(uniform → log₂ k, degenerate → 0) rather than the formula itself.

| eq | manuscript | enforced by |
|---|---|---|
| 2 | `H(p(c,φ)) = −Σ p(c,φ) log p(c,φ)` | `test_eq2_joint_entropy` (via `H(c,φ) = H(c)+H(φ)−I`) |
| 3 | `H(p(c\|φ)) = −Σ_c p(c\|φ) log p(c\|φ)` | `test_eq3_clonotypic_entropy_matches_the_manuscript` |
| 4 | `H(p(φ\|c)) = −Σ_φ p(φ\|c) log p(φ\|c)` | `test_eq4_phenotypic_entropy_matches_the_manuscript` |
| 5 | `I(c,φ) = Σ p(φ,c) log( p(c,φ)/(p(φ)p(c)) )` | `test_eq5_mutual_information_matches_the_manuscript` |
| 6 | `NMI = I / (½(H(c)+H(φ)))` | `test_eq6_nmi_is_the_average_denominator` — **see the deviation below** |

> **Historical note.** An earlier revision of the note mistranscribed eqs 3–4: both
> weighted by the **marginal** while taking the log of the **conditional** (a
> cross-entropy, not an entropy), and eq 4's left-hand side read `H(p(c))` while its
> right-hand side summed over φ. The code was correct throughout and was left
> unchanged; **the manuscript has since been corrected** and the erratum is retired.
> `test_marginal_weighting_is_not_an_entropy` remains as a standing guard, because
> marginal-weighting is the natural way to mis-transcribe these equations: it fails
> two ways at once — the value can exceed `log₂|C|` (impossible for an entropy over
> `|C|` outcomes), and substituting it into the MI decomposition yields a **negative**
> mutual information, impossible for a KL divergence.

### The one live deviation: `normalize_mode`

Eq 6 specifies the **mean** denominator. tcri's default is **`min`**:

| | denominator | value on the contract's test joint |
|---|---|---|
| eq 6 / `normalize_mode="average"` | `½(H(c)+H(φ))` | **0.238915** |
| tcri default `normalize_mode="min"` | `min(H(c), H(φ))` | **0.293032** |

`min` is the default because the mean denominator scales with `log₂(C)`, making it
**not comparable across groups with different clone counts** — the blocking issue for
any per-group or per-patient comparison. The two differ materially, so **anything
reproducing the note's benchmark must pass `normalize_mode="average"` explicitly.**
`test_eq6_nmi_is_the_average_denominator` asserts both halves — that `"average"`
reproduces eq 6, and that the default does *not* — so the divergence can never become
silent.

## Sanctioned extensions (the note does not specify these)

- **bits / log₂** — the note writes an unspecified `log`.
- **`normalized=True`** — divide by the maximum-entropy value so results land in [0,1].
- **`n_clones_ref`** — fix the clonotypic normalizer across groups; without it each
  group normalizes by its own supported-clone count and the values are not comparable.
- **`n_samples>0`** — return mean/sd/HDI over posterior draws. The plug-in entropy is
  ≥ the posterior mean (Jensen), so the two are reported as distinct quantities.
