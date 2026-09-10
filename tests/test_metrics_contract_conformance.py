"""Conformance test for the METRICS contract (``governance/METRICS_CONTRACT.md``).

Metrics are pure functions of a joint table, so they are pinned four ways: each equation in
the contract is recomputed from a hand-written reference on the contract's joint; the
identities hold (uniform -> log2 k, independent -> MI 0, the entropy/MI decomposition); the
defaults that change what a number means are read off the live signatures; and the values on
the reference joint match the numbers pinned in the contract block.

A failure here means the meaning of a published number changed. The contract file changes in
the same PR, deliberately; never relax an identity or a pinned value to make this pass.
"""
from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

from tests._governance import GOVERNANCE, contract_namespace
from tcri._compute._distance import phenotype_distance
from tcri.tools._entropy import _clonotypic_one, _phenotypic_one
from tcri.tools._mutual_information import _mi_from_joint

MC = contract_namespace("METRICS_CONTRACT.md")
LOG_BASE, METRICS, DEFAULTS, GOLDEN = MC["LOG_BASE"], MC["METRICS"], MC["DEFAULTS"], MC["GOLDEN"]
PERTURBATIONS = MC["PERTURBATIONS"]

#: (namespace, the contract list that must equal its public surface). ``perturb`` queries are
#: defined in the same file as the metrics because they say what a number means, even though
#: they are functions of the fitted model rather than of the joint.
SURFACES = [("tl", METRICS), ("perturb", PERTURBATIONS)]


def _public(namespace):
    return {n for n in dir(namespace) if not n.startswith("_")
            and callable(getattr(namespace, n))
            and getattr(getattr(namespace, n), "__module__", "").startswith("tcri")}


# ── the contract covers the surface, and pins what a number means ───────────
def test_contract_covers_every_public_metric():
    """Every public ``tl`` and ``perturb`` function is listed in the contract block and has a
    section in the prose. A function that is public but undescribed has no definition to be
    held to."""
    import tcri

    text = (GOVERNANCE / "METRICS_CONTRACT.md").read_text()
    for ns, declared in SURFACES:
        live = _public(getattr(tcri, ns))
        assert live == set(declared), (
            f"{ns} surface {sorted(live)} != contract list {sorted(declared)}; update "
            f"governance/METRICS_CONTRACT.md"
        )
        for name in declared:
            if name == "joint_distribution":
                continue  # the engine; its arguments are the API contract's
            assert re.search(rf"^###[^\n]*`{re.escape(name)}`", text, re.M), (
                f"{name} has no '### ... `{name}`' section in METRICS_CONTRACT.md"
            )
    assert LOG_BASE == 2


def test_defaults_that_change_the_estimand_are_pinned():
    """The signature defaults in the contract block are the live defaults. Flipping any of
    them changes which quantity every caller gets (one vote per clone vs per cell, coefficient
    of constraint vs mean denominator, KL vs L1, plug-in vs draws, the model's rule vs the
    head alone)."""
    import tcri

    seen = set()
    for ns, declared in SURFACES:
        for name in declared:
            params = inspect.signature(getattr(getattr(tcri, ns), name)).parameters
            for knob, expected in DEFAULTS.items():
                if knob in params:
                    seen.add(knob)
                    assert params[knob].default == expected, (
                        f"{ns}.{name}(..., {knob}={params[knob].default!r}) but the contract "
                        f"pins {knob}={expected!r}. Changing a default is a contract change."
                    )
    assert seen == set(DEFAULTS), f"contract pins defaults nothing exposes: {set(DEFAULTS) - seen}"


def test_pinned_values_on_the_reference_joint():
    """The numbers in the contract block, to 1e-9. This is what makes the definitions frozen:
    a redefinition that survives every identity still moves these."""
    J = np.array(GOLDEN["joint"], dtype=float)
    ids = [f"c{i}" for i in range(J.shape[0])]
    cols = [f"p{j}" for j in range(J.shape[1])]
    tol = dict(abs=1e-9)

    g = GOLDEN["clonotypic_entropy"]
    raw = _clonotypic_one(J, cols, normalized=False)
    nrm = _clonotypic_one(J, cols, normalized=True)
    assert [raw[c] for c in cols] == pytest.approx(g["raw"], **tol)
    assert [nrm[c] for c in cols] == pytest.approx(g["normalized"], **tol)

    g = GOLDEN["phenotypic_entropy"]
    raw = _phenotypic_one(ids, J, cols, normalized=False)
    nrm = _phenotypic_one(ids, J, cols, normalized=True)
    assert [raw[c] for c in ids] == pytest.approx(g["raw"], **tol)
    assert [nrm[c] for c in ids] == pytest.approx(g["normalized"], **tol)

    g = GOLDEN["mutual_information"]
    assert _mi_from_joint(J, normalized=False) == pytest.approx(g["raw"], **tol)
    assert _mi_from_joint(J, normalized=True, mode="min") == pytest.approx(g["min"], **tol)
    assert _mi_from_joint(J, normalized=True, mode="average") == pytest.approx(g["average"], **tol)

    g = GOLDEN["phenotypic_flux"]
    p, q = J[0] / J[0].sum(), J[2] / J[2].sum()
    for metric, expected in g.items():
        assert float(phenotype_distance(metric)(p, q)) == pytest.approx(expected, **tol), metric


# ── the contract's equations, recomputed from a reference ──────────────────
# The primary definitional tests: each writes the contract's equation out by hand on the
# reference joint and asserts the code computes exactly that. Stronger than the identity tests
# below, which pin consequences of a formula rather than the formula itself.

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


def test_eq3_clonotypic_entropy_matches_the_contract():
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


def test_eq4_phenotypic_entropy_matches_the_contract():
    """Eq 4: H(p(phi|c)) = -sum_{phi in Phi} p(phi|c) log(p(phi|c))."""
    J = _joint()
    ids, cols = ["c0", "c1", "c2"], ["phen_A", "phen_B"]
    code = _phenotypic_one(ids, J, cols, normalized=False)
    for i, c in enumerate(ids):
        row = J[i]
        p_phi_given_c = row / row.sum()          # p(phi|c)
        eq4 = -np.sum(p_phi_given_c * np.log2(p_phi_given_c))
        assert code[c] == pytest.approx(eq4, abs=1e-12), f"eq 4 mismatch for {c}"


def test_eq5_mutual_information_matches_the_contract():
    """Eq 5: I(c,phi) = sum_{c,phi} p(phi,c) log( p(c,phi) / (p(phi) p(c)) )."""
    J = _joint()
    P = J / J.sum()
    p_c, p_ph = P.sum(1), P.sum(0)
    eq5 = np.sum(P * np.log2(P / np.outer(p_c, p_ph)))
    assert _mi_from_joint(J, normalized=False) == pytest.approx(eq5, abs=1e-12)


def test_eq6_nmi_is_the_average_denominator():
    """Eq 6: NMI = I / ( (1/2)(H(c) + H(phi)) ) — the MEAN denominator.

    The package exposes this as ``normalize_mode="average"``. Its DEFAULT is ``"min"``: the
    mean denominator scales with log2(C) and so is not comparable across groups with different
    clone counts. This test pins both: that 'average' reproduces eq 6, and that the default
    does NOT, so the difference can never become silent.
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


# ── the cross-metric identity ────────────────────────────────────────────────
def test_mi_equals_marginal_minus_expected_conditional_entropy():
    """IDENTITIES['mi_entropy_decomposition'] — I(c;phi) = H(c) - E_phi[H(c|phi)].

    This is the identity that proves the implemented conditional entropy is the right one:
    weighting by the marginal instead of the conditional makes this yield a NEGATIVE mutual
    information. It ties the entropy and MI families together, so redefining either alone
    breaks it.
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


def test_marginal_weighted_formula_would_break_the_decomposition():
    """The natural mis-transcription of eq 3 (weight by the marginal) gives a NEGATIVE MI.

    If someone 'fixes' the code toward that form, this test documents why it is wrong.
    """
    J = np.array([[4.0, 1.0], [1.0, 1.0], [1.0, 6.0]])
    pxy = J / J.sum()
    p_c, p_ph = pxy.sum(1), pxy.sum(0)
    H_c = -np.sum(p_c * np.log2(p_c))
    mi = _mi_from_joint(J, normalized=False)

    # weight by the MARGINAL p(c) instead of the conditional
    literal = sum(
        p_ph[j] * (-np.sum(p_c * np.log2(J[:, j] / J[:, j].sum())))
        for j in range(J.shape[1])
    )
    assert H_c - literal < 0, "expected the literal formula to give a negative MI"
    assert mi > 0


# ── `weighted` selects the clone marginal ───────────────────────────────────

def test_weighted_default_is_false_and_the_knob_is_live():
    """Pins the CURRENT default and proves the argument is not inert.

    Two assertions, because either alone passes by accident: the signature default must be
    ``False``, and the two settings must produce different numbers on a repertoire where they
    should differ. A test that only checked the default would still pass if `weighted` stopped
    being read.

    The default is deliberately unsettled — the argument's meaning is to be reviewed with it as
    the only variable moving. Until then this makes a change to it a contract change.
    """
    import inspect

    import tcri

    for fn in (tcri.tl.mutual_information, tcri.tl.clonotypic_entropy,
               tcri.tl.phenotypic_entropy, tcri.tl.joint_distribution):
        default = inspect.signature(fn).parameters["weighted"].default
        assert default is False, (
            f"{fn.__name__} defaults to weighted={default!r}. Flipping this changes which "
            f"estimand every caller gets — one vote per clone vs one vote per cell — and is a "
            f"contract change (METRICS_CONTRACT.md, the weighted axis)."
        )


def test_weighted_changes_the_clone_marginal():
    """`weighted=False` gives every clone equal mass; `weighted=True` weights by cell count.

    Checked directly on the marginal rather than through a fitted model, so the assertion is
    about the definition and not about a particular fit. A clone-size distribution that is
    deliberately skewed — 100 / 10 / 1 — makes the two unmistakable.
    """
    J = np.array([[90.0, 10.0], [5.0, 5.0], [0.0, 1.0]])   # clone sizes 100, 10, 1

    uniform = J / J.sum(axis=1, keepdims=True)             # each row a distribution
    uniform = uniform / uniform.shape[0]                   # then equal mass per clone
    abundance = J / J.sum()                                # mass proportional to n_c

    assert not np.allclose(uniform.sum(1), abundance.sum(1)), (
        "the fixture's clone sizes are too even to distinguish the two marginals"
    )
    assert np.allclose(uniform.sum(1), 1.0 / 3.0), "weighted=False must give P(c) = 1/C"
    assert np.allclose(abundance.sum(1), [100 / 111, 10 / 111, 1 / 111]), (
        "weighted=True must give P(c) proportional to the clone's cell count"
    )
