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

Source of truth: **Supplementary Note 1**, "Entropy" section (eqs 2–4) — with the
errata below.

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

## Errata in Supplementary Note 1 (the code is correct)

The note's eqs 3–4, read literally, do **not** match the implementation — and the note
is the one that is wrong. Recorded here so nobody "fixes" the code to match a typo.

1. **Eq 3** reads `H(p(c|φ)) = − Σ_c p(c) log p(c|φ)` — it weights by the **marginal**
   `p(c)` while taking the log of the **conditional**. That is a cross-entropy, not an
   entropy.
2. **Eq 4** is labelled `H(p(c))` but its right-hand side sums over φ and uses `p(φ|c)`,
   so the label is wrong; it also weights by the marginal.
3. The **prose** introduces both as "the entropy of the marginal distributions", but the
   equations are conditionals.

**Why the code is right.** Mutual information must satisfy
`I(c;φ) = H(c) − E_φ[H(c|φ)]`. On a test joint with true MI **0.288703**:

- the implemented conditional entropy reproduces it **exactly** (0.288703);
- the note's literal formula yields **−0.345883** — a *negative* mutual information,
  which is impossible.

The literal equations are inconsistent with the note's own MI, so they cannot be what
was intended. `test_note_literal_formula_would_break_the_decomposition` pins this.

## Sanctioned extensions (the note does not specify these)

- **bits / log₂** — the note writes an unspecified `log`.
- **`normalized=True`** — divide by the maximum-entropy value so results land in [0,1].
- **`n_clones_ref`** — fix the clonotypic normalizer across groups; without it each
  group normalizes by its own supported-clone count and the values are not comparable.
- **`n_samples>0`** — return mean/sd/HDI over posterior draws. The plug-in entropy is
  ≥ the posterior mean (Jensen), so the two are reported as distinct quantities.
