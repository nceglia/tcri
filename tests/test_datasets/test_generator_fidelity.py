"""What the synthetic generator guarantees about its own ground truth.

The generator defines the truth every benchmark number is scored against, so a wrong value
here does not make a test fail — it makes the answer wrong while the suite stays green. Three
properties are pinned: the declared phenotype label space survives into the categorical codes,
``temperature_scale`` refuses inputs it cannot compute, and the fuzziness knob blends the
expression programs through the concave mapping ``g(f) = sqrt(f)``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from tcri.datasets import simulate_tcri, temperature_scale


# ── the generator declares its label space ───────────────────────────────────

@pytest.mark.parametrize("n_phenotypes", [4, 12])
def test_category_codes_match_the_integer_labels(n_phenotypes):
    """Phenotype ``phen_k`` must carry category code ``k``, so a code can be mapped back to
    the phenotype index the cell was generated from.

    ``pd.Categorical`` without ``categories=`` infers levels from the observed values and sorts
    them LEXICOGRAPHICALLY, which at K >= 10 puts ``phen_11`` before ``phen_2`` and decouples a
    code from its integer. Anything round-tripping a code back to a phenotype index would then
    read a different phenotype. Both a one-digit and a two-digit K are covered because the two
    orderings agree below 10.
    """
    adata = simulate_tcri(n_clones=6, n_phenotypes=n_phenotypes, n_genes=20,
                          n_cells=400, seed=0)
    cats = list(adata.obs["phenotype"].cat.categories)

    assert len(cats) == n_phenotypes, (
        f"{len(cats)} phenotype levels for n_phenotypes={n_phenotypes}; a phenotype with no "
        f"sampled cells has vanished from the level set"
    )
    for p in range(n_phenotypes):
        assert cats.index(f"phen_{p}") == p, (
            f"'phen_{p}' has code {cats.index(f'phen_{p}')}, not {p} — the category order is "
            f"lexicographic rather than the declared label space (DE-13). Levels: {cats}"
        )


def test_true_and_observed_phenotype_share_a_level_set():
    """``label_error_rate`` can empty a phenotype in one column but not the other. With
    inferred levels the two columns then carry different category sets, and comparing them
    silently misaligns."""
    adata = simulate_tcri(n_clones=6, n_phenotypes=8, n_genes=20, n_cells=200,
                          label_error_rate=0.5, seed=0)
    assert (list(adata.obs["phenotype"].cat.categories)
            == list(adata.obs["true_phenotype"].cat.categories))


# ── temperature_scale refuses what it cannot compute ─────────────────────────

@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_temperature_scale_rejects_non_positive_or_non_finite(bad):
    """Temperature scaling is defined only for finite T > 0, so the function must raise
    instead of returning something a caller cannot recognise as wrong: T=0 divides by zero,
    T=nan gives an all-NaN matrix, and a negative T gives finite, plausible, row-stochastic
    output that inverts the distribution and would propagate into a benchmark silently."""
    with pytest.raises(ValueError, match="finite and > 0"):
        temperature_scale(np.array([[0.7, 0.2, 0.1]]), bad)


def test_temperature_scale_raises_on_float64_underflow():
    """Once (1/T)*log10(min(P)) < -308 every entry of a row underflows to exactly 0.0 in
    float64 and the renormalisation is 0/0. Raising names the cause; returning a NaN row leaves
    the caller to notice. The two temperatures here bracket that point for this row."""
    P = np.array([[0.7, 0.2, 0.1]])
    assert np.isfinite(temperature_scale(P, 1e-3)).all(), "1e-3 should still be computable"
    with pytest.raises(ValueError, match="underflow"):
        temperature_scale(P, 1e-4)


def test_temperature_scale_endpoints_are_sane():
    P = np.array([[0.7, 0.2, 0.1]])
    assert np.allclose(temperature_scale(P, 1.0), P), "T=1 must be the identity"
    hot = temperature_scale(P, 1e5)
    assert np.allclose(hot.sum(1), 1.0)
    assert hot.std() < P.std(), "large T must flatten"


# ── the concave mapping g(f) = sqrt(f) ───────────────────────────────────────

def test_fuzziness_uses_the_concave_mapping():
    """``fuzziness`` blends each phenotype's expression program toward the mean as
    ``theta'_k = (1 - g(f)) theta_k + g(f) theta_bar`` with the concave ``g(f) = sqrt(f)``.

    Interpolating with ``f`` itself under-mixes across the whole interior of a fuzziness sweep,
    so the difficulty axis a benchmark reports would not be the axis it swept. The endpoints
    agree under either mapping, so only an interior ``f`` can detect the difference.

    The blend is recomputed here independently from the same RNG draw, so this checks the
    quantity the mapping defines rather than matching the source line; the final assertion
    confirms the two mappings do differ on this fixture.
    """
    from tcri.datasets._simulate import _phenotype_programs

    n_phen, n_fac, f = 5, 4, 0.25
    got_alpha, got_beta = _phenotype_programs(np.random.default_rng(0), n_phen, n_fac, f)

    r = np.random.default_rng(0)
    alpha = r.uniform(1.5, 6.0, size=(n_phen, n_fac))
    beta = r.uniform(1.0, 3.0, size=(n_phen, n_fac))
    theta = np.concatenate([alpha - 1.0, -beta], axis=1)

    g = np.sqrt(f)
    blended = (1.0 - g) * theta + g * theta.mean(0, keepdims=True)
    want_alpha = np.clip(blended[:, :n_fac] + 1.0, 1e-3, None)
    want_beta = np.clip(-blended[:, n_fac:], 1e-3, None)

    assert np.allclose(got_alpha, want_alpha), "alpha is not blended by g(f)=sqrt(f) (DE-20)"
    assert np.allclose(got_beta, want_beta), "beta is not blended by g(f)=sqrt(f) (DE-20)"

    linear = (1.0 - f) * theta + f * theta.mean(0, keepdims=True)
    assert not np.allclose(want_alpha, np.clip(linear[:, :n_fac] + 1.0, 1e-3, None)), (
        "sqrt(f) and f coincide on this fixture, so it cannot detect the defect"
    )


@pytest.mark.parametrize("f", [0.0, 1.0])
def test_fuzziness_endpoints_are_unchanged_by_the_mapping(f):
    """g(0)=0 and g(1)=1, so the endpoints of a fuzziness sweep are the same under either
    mapping: the choice of g can only matter for interior f. At f=1 the programs must collapse
    onto one, which is what makes the phenotypes indistinguishable from expression."""
    from tcri.datasets._simulate import _phenotype_programs

    alpha, beta = _phenotype_programs(np.random.default_rng(0), 5, 4, f)
    assert np.isfinite(alpha).all() and np.isfinite(beta).all()
    if f == 1.0:
        assert np.allclose(alpha, alpha[0]), "f=1 must collapse phenotypes to a common program"
        assert np.allclose(beta, beta[0])
