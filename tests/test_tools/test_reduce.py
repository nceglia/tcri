"""`_compute/_reduce` — the batched information-theoretic reductions default to
**bits (log2)**, the tcri convention."""
import numpy as np

from tcri._compute import _reduce


def test_entropy_uniform_is_log2_P_bits():
    """Entropy of the uniform distribution over P outcomes = log2(P) bits."""
    for P in (2, 3, 8):
        u = np.full(P, 1.0 / P)
        np.testing.assert_allclose(_reduce.entropy(u), np.log2(P), atol=1e-12)
    # a point mass has zero entropy
    assert _reduce.entropy(np.array([1.0, 0.0, 0.0])) == 0.0


def test_entropy_base_override_nats():
    u = np.full(4, 0.25)
    np.testing.assert_allclose(_reduce.entropy(u, base=None), np.log(4), atol=1e-12)  # nats


def test_entropy_batched_over_leading_axes():
    p = np.stack([np.full(4, 0.25), np.array([1.0, 0, 0, 0])])  # [2, 4]
    out = _reduce.entropy(p)  # -> [2]
    np.testing.assert_allclose(out, [2.0, 0.0], atol=1e-12)


def test_mutual_information_independent_is_zero():
    Px = np.array([0.5, 0.5]); Py = np.array([0.25, 0.75])
    joint = np.outer(Px, Py)  # independent
    np.testing.assert_allclose(_reduce.mutual_information(joint), 0.0, atol=1e-12)


def test_mutual_information_perfectly_coupled_equals_marginal_entropy():
    """A diagonal joint (X determines Y) has MI == H(X) == H(Y) (bits)."""
    joint = np.diag([0.5, 0.5])  # perfectly coupled 2x2
    np.testing.assert_allclose(_reduce.mutual_information(joint), 1.0, atol=1e-9)  # 1 bit


def test_mutual_information_renormalizes_unnormalized_joint():
    """An un-normalized (cell-weighted count) joint is accepted (renormalized)."""
    joint = np.diag([5.0, 5.0])  # counts, not probabilities
    np.testing.assert_allclose(_reduce.mutual_information(joint), 1.0, atol=1e-9)
