"""Generator fidelity to Supplementary Note 1 (DE-13, DE-14, DE-20).

The synthetic generator defines the ground truth every benchmark number is scored against, so
a defect here does not make a test fail — it makes the answer wrong while everything stays
green. All three of these were silent.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from tcri.datasets import simulate_tcri, temperature_scale


# ── DE-13: the generator declares its label space ────────────────────────────

@pytest.mark.parametrize("n_phenotypes", [4, 12])
def test_category_codes_match_the_integer_labels(n_phenotypes):
    """``pd.Categorical`` without ``categories=`` infers levels from the observed values and
    sorts them LEXICOGRAPHICALLY.

    At K >= 10 that decouples a code from the integer it was built from: measured before the
    fix at K=12, ``phen_2`` got code 4 and ``phen_11`` got code 3. Anything round-tripping a
    code back to a phenotype index was reading a different phenotype. K=4 is here to show the
    defect is invisible below 10, which is why it survived.
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


# ── DE-14: temperature_scale refuses what it cannot compute ──────────────────

@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_temperature_scale_rejects_non_positive_or_non_finite(bad):
    """Note 1 specifies T > 0. Outside that the old code failed three different silent ways:
    ZeroDivisionError at T=0, an all-NaN matrix at T=nan, and — worst — finite, plausible,
    row-stochastic output at T=-1.0 that inverts the distribution and would have propagated
    into a benchmark as though it meant something."""
    with pytest.raises(ValueError, match="finite and > 0"):
        temperature_scale(np.array([[0.7, 0.2, 0.1]]), bad)


def test_temperature_scale_raises_on_float64_underflow():
    """Once (1/T)*log10(min(P)) < -308 every entry of a row underflows to exactly 0.0 and the
    renormalisation is 0/0. Measured on [0.7, 0.2, 0.1]: computable at T=1e-3, all-NaN at
    T=1e-4. Raising beats returning NaN that the caller has to notice."""
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


# ── DE-20: the concave mapping g(f) = sqrt(f) ────────────────────────────────

def test_fuzziness_uses_the_concave_mapping():
    """Note 1: ``theta'_k = (1 - g(f)) theta_k + g(f) theta_bar``, with "in the reported
    experiments, we use g(f) = sqrt(f)".

    The code interpolated with ``f`` itself, which under-mixes across the whole interior of
    the sweep — at f=0.1 the note blends 0.316 toward the mean where the old code blended
    0.100. The endpoints agree, so f=0 and f=1 could not detect it.

    The blend is recomputed here independently from the same RNG draw, so this checks the
    quantity the note defines rather than matching the source line.
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
    """g(0)=0 and g(1)=1, so the sweep endpoints are identical either way. Recorded because it
    bounds what the DE-20 correction invalidates: interior f only."""
    from tcri.datasets._simulate import _phenotype_programs

    alpha, beta = _phenotype_programs(np.random.default_rng(0), 5, 4, f)
    assert np.isfinite(alpha).all() and np.isfinite(beta).all()
    if f == 1.0:
        assert np.allclose(alpha, alpha[0]), "f=1 must collapse phenotypes to a common program"
        assert np.allclose(beta, beta[0])
