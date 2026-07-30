"""Conformance test for the METRICS contract (``tcri/tools/_metrics_contract.py``).

Companion to ``test_model_contract_conformance.py``. Where the model contract is
verified by tracing ``model()``/``guide()``, metrics are pure functions of a joint
table, so they are pinned by **numeric identities**: uniform -> log2(k),
independent -> MI 0, and the entropy/MI decomposition.

A failure here means the *meaning* of a published number changed. Update the manifest
and ``docs/contract/METRICS_CONTRACT.md`` first, deliberately — never relax an
identity to make this pass.
"""
from __future__ import annotations

import numpy as np
import pytest

from tcri.tools import _metrics_contract as MC
from tcri.tools._entropy import _clonotypic_one, _phenotypic_one
from tcri.tools._mutual_information import _mi_from_joint


# ── manifest hygiene ────────────────────────────────────────────────────────
def test_manifest_is_complete():
    """Every public metric is specified, and every spec field is filled in."""
    assert set(MC.METRIC_SPECS) == {
        "clonotypic_entropy", "phenotypic_entropy", "mutual_information"
    }
    for name, spec in MC.METRIC_SPECS.items():
        for field in ("formula", "per", "support", "normalizer", "empty", "note_eq"):
            assert getattr(spec, field), f"{name}.{field} is empty"
    assert MC.LOG_BASE == 2


def test_source_errata_are_documented():
    """The note's eq 3/4 errors must stay recorded, with the justification."""
    for key in ("eq3_weights_marginal", "eq4_label_and_weights", "why_the_code_is_right"):
        assert key in MC.SOURCE_ERRATA and len(MC.SOURCE_ERRATA[key]) > 40


# ── entropy identities ──────────────────────────────────────────────────────
def test_entropy_uniform_is_log2_k():
    """IDENTITIES['entropy_uniform_is_log2_k']."""
    for k in (2, 4, 8):
        J = np.ones((k, k))
        # clonotypic: each phenotype column is uniform over k clones
        raw = _clonotypic_one(J, [f"p{j}" for j in range(k)], normalized=False)
        for v in raw.values():
            assert v == pytest.approx(np.log2(k))
        nrm = _clonotypic_one(J, [f"p{j}" for j in range(k)], normalized=True)
        for v in nrm.values():
            assert v == pytest.approx(1.0)
        # phenotypic: each clone row is uniform over k phenotypes
        rawp = _phenotypic_one([f"c{i}" for i in range(k)], J,
                               [f"p{j}" for j in range(k)], normalized=False)
        for v in rawp.values():
            assert v == pytest.approx(np.log2(k))


def test_entropy_degenerate_is_zero():
    """IDENTITIES['entropy_degenerate_is_zero']."""
    J = np.array([[5.0, 0.0], [0.0, 3.0]])
    c = _clonotypic_one(J, ["a", "b"], normalized=False)
    assert c["a"] == pytest.approx(0.0) and c["b"] == pytest.approx(0.0)
    p = _phenotypic_one(["c0", "c1"], J, ["a", "b"], normalized=False)
    assert p["c0"] == pytest.approx(0.0) and p["c1"] == pytest.approx(0.0)


def test_entropy_zero_mass_is_nan_not_zero_or_one():
    """IDENTITIES['entropy_zero_mass_is_nan'] — the spurious-H=1 regression."""
    J = np.array([[1.0, 0.0], [1.0, 0.0]])       # phenotype 'b' has no mass
    c = _clonotypic_one(J, ["a", "b"], normalized=True)
    assert np.isnan(c["b"]), "empty phenotype column must be NaN"
    J2 = np.array([[1.0, 1.0], [0.0, 0.0]])      # clone 'c1' has no mass
    p = _phenotypic_one(["c0", "c1"], J2, ["a", "b"], normalized=True)
    assert np.isnan(p["c1"]), "zero-mass clone must be NaN, never a spurious 1.0"


def test_clonotypic_entropy_is_support_only():
    """Absent clones are dropped BEFORE normalizing (no epsilon-clip inflation).

    Two supported clones out of five: H must be log2(2), normalized 1.0 — not the
    log2(5)-normalized value an epsilon clip would produce.
    """
    J = np.zeros((5, 1))
    J[0, 0] = J[1, 0] = 1.0
    raw = _clonotypic_one(J, ["a"], normalized=False)["a"]
    nrm = _clonotypic_one(J, ["a"], normalized=True)["a"]
    assert raw == pytest.approx(np.log2(2))
    assert nrm == pytest.approx(1.0)
    # n_clones_ref fixes the normalizer instead, for cross-group comparability
    fixed = _clonotypic_one(J, ["a"], normalized=True, n_clones_ref=5)["a"]
    assert fixed == pytest.approx(np.log2(2) / np.log2(5))


# ── mutual information identities ───────────────────────────────────────────
def test_mi_independent_is_zero():
    """IDENTITIES['mi_independent_is_zero']."""
    px = np.array([0.2, 0.3, 0.5])[:, None]
    py = np.array([0.4, 0.6])[None, :]
    J = px @ py                       # exactly independent
    assert _mi_from_joint(J, normalized=False) == pytest.approx(0.0, abs=1e-9)


def test_mi_is_symmetric_and_nonnegative():
    """IDENTITIES['mi_is_symmetric'] + ['mi_is_nonnegative']."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        J = rng.random((rng.integers(2, 7), rng.integers(2, 7)))
        mi = _mi_from_joint(J, normalized=False)
        assert mi >= -1e-12
        assert mi == pytest.approx(_mi_from_joint(J.T, normalized=False))


def test_mi_perfect_coupling_is_one():
    """IDENTITIES['mi_perfect_coupling_is_one'] (normalize_mode='min')."""
    J = np.eye(4) * 7.0
    assert _mi_from_joint(J, normalized=True, mode="min") == pytest.approx(1.0)


def test_mi_normalize_modes_differ_as_specified():
    """'min' vs 'average' denominators — 'min' is the default for comparability."""
    J = np.array([[8.0, 1.0, 1.0], [1.0, 5.0, 1.0]])
    mi = _mi_from_joint(J, normalized=False)
    pxy = J / J.sum()
    px, py = pxy.sum(1), pxy.sum(0)
    h_c = -np.sum(px * np.log2(px))
    h_p = -np.sum(py * np.log2(py))
    assert _mi_from_joint(J, normalized=True, mode="min") == pytest.approx(mi / min(h_c, h_p), rel=1e-6)
    assert _mi_from_joint(J, normalized=True, mode="average") == pytest.approx(
        mi / (0.5 * (h_c + h_p)), rel=1e-6)


# ── the cross-metric identity that caught the note's erratum ────────────────
def test_mi_equals_marginal_minus_expected_conditional_entropy():
    """IDENTITIES['mi_entropy_decomposition'] — I(c;phi) = H(c) - E_phi[H(c|phi)].

    This is the identity that proves the implemented conditional entropy is the right
    one: weighting by the marginal (as the note's eqs 3-4 literally read) makes this
    yield a NEGATIVE mutual information. It ties the entropy and MI families together,
    so redefining either alone breaks it.
    """
    rng = np.random.default_rng(7)
    for _ in range(20):
        n_c, n_p = int(rng.integers(2, 8)), int(rng.integers(2, 6))
        J = rng.random((n_c, n_p)) + 0.05        # strictly positive: full support
        cols = [f"p{j}" for j in range(n_p)]

        pxy = J / J.sum()
        p_c, p_ph = pxy.sum(1), pxy.sum(0)
        H_c = -np.sum(p_c * np.log2(p_c))

        # UNNORMALIZED clonotypic entropy per phenotype, weighted by P(phi)
        H_cond = _clonotypic_one(J, cols, normalized=False)
        expected_cond = sum(p_ph[j] * H_cond[cols[j]] for j in range(n_p))

        mi = _mi_from_joint(J, normalized=False)
        assert mi == pytest.approx(H_c - expected_cond, abs=1e-9)


def test_note_literal_formula_would_break_the_decomposition():
    """Guards the erratum itself: the note's literal eq 3 gives a NEGATIVE MI.

    If someone 'fixes' the code to match the mistranscribed equation, this test
    documents exactly why that is wrong.
    """
    J = np.array([[4.0, 1.0], [1.0, 1.0], [1.0, 6.0]])
    pxy = J / J.sum()
    p_c, p_ph = pxy.sum(1), pxy.sum(0)
    H_c = -np.sum(p_c * np.log2(p_c))
    mi = _mi_from_joint(J, normalized=False)

    # the note as literally written: weight by the MARGINAL p(c)
    literal = sum(
        p_ph[j] * (-np.sum(p_c * np.log2(J[:, j] / J[:, j].sum())))
        for j in range(J.shape[1])
    )
    assert H_c - literal < 0, "expected the literal formula to give a negative MI"
    assert mi > 0
