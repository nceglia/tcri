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
        "clonotypic_entropy", "phenotypic_entropy", "mutual_information",
        "phenotypic_flux",
    }
    for name, spec in MC.METRIC_SPECS.items():
        for field in ("formula", "per", "support", "normalizer", "empty", "note_eq"):
            assert getattr(spec, field), f"{name}.{field} is empty"
    assert MC.LOG_BASE == 2


def test_every_spec_names_its_source_document():
    """Equation numbers COLLIDE between the two source documents — "eq 3" is the clonotypic
    entropy in one and the VampPrior in the other — so a bare "eq 3" is ambiguous. This makes
    that ambiguity un-shippable."""
    for name, spec in MC.METRIC_SPECS.items():
        assert any(src in spec.note_eq for src in MC.SOURCES), (
            f"{name}.note_eq must name its source document (one of {sorted(MC.SOURCES)}); "
            f"got {spec.note_eq!r}"
        )
    assert "NOTE_1" not in " ".join(s.note_eq for s in MC.METRIC_SPECS.values()), (
        "Supplementary Note 1 carries no entropy/MI definitions — a metric citing it is "
        "citing the wrong document."
    )


def test_sources_are_archived_with_a_hash():
    """The upstream documents live in the repo, not on someone's desktop, and the recorded
    hash turns a silent revision into a failing build."""
    import hashlib
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for key, src in MC.SOURCES.items():
        f = root / src["file"]
        assert f.exists(), f"SOURCES[{key!r}] missing from the repo: {src['file']}"
        got = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        assert got == src["sha256_16"], (
            f"{src['file']} changed (hash {got}, manifest says {src['sha256_16']}). The "
            f"source document is UPSTREAM of this contract — reconcile the manifest to the "
            f"new document; do not edit the hash to silence this."
        )


def test_open_questions_are_not_quietly_sanctioned():
    """A live disagreement with the source document must not be filed as an 'extension'.
    Extensions are things the document does not specify; these are things it does."""
    for key in ("flux_distance_default", "posterior_summary_of_a_nonlinear_metric"):
        assert key in MC.OPEN_QUESTIONS and len(MC.OPEN_QUESTIONS[key]) > 40
        assert key not in MC.SANCTIONED_EXTENSIONS


# ── the manuscript equations, transcribed literally ─────────────────────────
# These are the primary definitional tests: each transcribes an equation from
# Supplementary Note 1 straight from the manuscript and asserts the code computes
# exactly that. Stronger than the identity tests below, which pin *consequences* of
# the formula (uniform -> log2 k, degenerate -> 0) rather than the formula itself.

def _joint():
    """A small, deliberately asymmetric clone x phenotype count table."""
    return np.array([[4.0, 1.0], [1.0, 1.0], [1.0, 6.0]])


def test_eq2_joint_entropy():
    """Eq 2: H(p(c,phi)) = -sum_{c,phi} p(c,phi) log p(c,phi)."""
    J = _joint()
    P = J / J.sum()
    eq2 = -np.sum(P[P > 0] * np.log2(P[P > 0]))
    # the package does not expose joint entropy directly, so assert the identity that
    # ties it to the quantities that ARE exposed: H(c,phi) = H(c) + H(phi) - I(c;phi)
    p_c, p_ph = P.sum(1), P.sum(0)
    h_c = -np.sum(p_c * np.log2(p_c))
    h_ph = -np.sum(p_ph * np.log2(p_ph))
    mi = _mi_from_joint(J, normalized=False)
    assert eq2 == pytest.approx(h_c + h_ph - mi, abs=1e-12)


def test_eq3_clonotypic_entropy_matches_the_manuscript():
    """Eq 3: H(p(c|phi)) = -sum_{c in C} p(c|phi) log(p(c|phi)).

    Note the weight is the CONDITIONAL p(c|phi), matching the log — an earlier
    revision weighted by the marginal p(c), which is a cross-entropy and can exceed
    log2(|C|) (see ``test_marginal_weighting_is_not_an_entropy``).
    """
    J = _joint()
    cols = ["phen_A", "phen_B"]
    code = _clonotypic_one(J, cols, normalized=False)
    for j, ph in enumerate(cols):
        col = J[:, j]
        p_c_given_phi = col / col.sum()          # p(c|phi)
        eq3 = -np.sum(p_c_given_phi * np.log2(p_c_given_phi))
        assert code[ph] == pytest.approx(eq3, abs=1e-12), f"eq 3 mismatch for {ph}"


def test_eq4_phenotypic_entropy_matches_the_manuscript():
    """Eq 4: H(p(phi|c)) = -sum_{phi in Phi} p(phi|c) log(p(phi|c))."""
    J = _joint()
    ids, cols = ["c0", "c1", "c2"], ["phen_A", "phen_B"]
    code = _phenotypic_one(ids, J, cols, normalized=False)
    for i, c in enumerate(ids):
        row = J[i]
        p_phi_given_c = row / row.sum()          # p(phi|c)
        eq4 = -np.sum(p_phi_given_c * np.log2(p_phi_given_c))
        assert code[c] == pytest.approx(eq4, abs=1e-12), f"eq 4 mismatch for {c}"


def test_eq5_mutual_information_matches_the_manuscript():
    """Eq 5: I(c,phi) = sum_{c,phi} p(phi,c) log( p(c,phi) / (p(phi) p(c)) )."""
    J = _joint()
    P = J / J.sum()
    p_c, p_ph = P.sum(1), P.sum(0)
    eq5 = np.sum(P * np.log2(P / np.outer(p_c, p_ph)))
    assert _mi_from_joint(J, normalized=False) == pytest.approx(eq5, abs=1e-12)


def test_eq6_nmi_is_the_average_denominator():
    """Eq 6: NMI = I / ( (1/2)(H(c) + H(phi)) ) — the MEAN denominator.

    The package exposes this as ``normalize_mode="average"``. Its DEFAULT is
    ``"min"``, which is a deliberate deviation (the mean denominator scales with
    log2(C) and so is not comparable across groups with different clone counts) —
    recorded in ``SANCTIONED_EXTENSIONS['normalize_mode_default']``. This test pins
    both: that 'average' reproduces eq 6, and that the default does NOT, so the
    divergence can never become silent.
    """
    J = _joint()
    P = J / J.sum()
    p_c, p_ph = P.sum(1), P.sum(0)
    mi = np.sum(P * np.log2(P / np.outer(p_c, p_ph)))
    h_c = -np.sum(p_c * np.log2(p_c))
    h_ph = -np.sum(p_ph * np.log2(p_ph))
    eq6 = mi / (0.5 * (h_c + h_ph))

    assert _mi_from_joint(J, normalized=True, mode="average") == pytest.approx(eq6, abs=1e-12)
    assert _mi_from_joint(J, normalized=True) != pytest.approx(eq6, abs=1e-6), (
        "the default normalize_mode now equals eq 6 — update the contract deliberately"
    )


def test_marginal_weighting_is_not_an_entropy():
    """Weighting by the marginal instead of the conditional is provably wrong.

    Kept as a standing guard rather than an erratum: it is the natural way to
    mis-transcribe eqs 3-4, and it fails two ways at once — the value can exceed
    log2(|C|) (impossible for an entropy over |C| outcomes), and substituting it into
    the MI decomposition yields a NEGATIVE mutual information, which is impossible
    for a KL divergence.
    """
    J = _joint()
    P = J / J.sum()
    p_c, p_ph = P.sum(1), P.sum(0)
    n_clones = J.shape[0]

    # the marginal-weighted quantity for phenotype A
    col = J[:, 0]
    wrong = -np.sum(p_c * np.log2(col / col.sum()))
    assert wrong > np.log2(n_clones), "expected the bound violation that flags it"

    # and it breaks the decomposition
    h_c = -np.sum(p_c * np.log2(p_c))
    expected_wrong = sum(
        p_ph[j] * (-np.sum(p_c * np.log2(J[:, j] / J[:, j].sum())))
        for j in range(J.shape[1])
    )
    assert h_c - expected_wrong < 0, "expected a negative (impossible) MI"
    assert _mi_from_joint(J, normalized=False) > 0


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
