"""Batched information-theoretic reductions over the joint stack ``[S, n_clones, P]``.

**All quantities default to BITS (log2)** — the tcri convention (matches
:mod:`tcri._distance`); pass ``base=None`` for nats or ``base=b`` for an arbitrary
base. Reductions run in **float64** accumulators (§7.4 guardrail 6) so a future GPU
path matches CPU, and use the ``0·log0 := 0`` convention. Grafiti `joint.py::_entropy/_mi`
parity, generalized to batch over the leading sample/covariate/group axes.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _logb(x, base):
    if base == 2:
        return np.log2(x)
    if base is None:  # nats
        return np.log(x)
    return np.log(x) / np.log(base)


def entropy(p, *, axis=-1, base=2):
    """Shannon entropy of a distribution along ``axis`` (bits by default), batched
    over every other axis. ``0·log0 := 0``; ``p`` is used as-is (assumed normalized)."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * _logb(p, base), 0.0)
    return -np.sum(terms, axis=axis)


def mutual_information(joint, *, base=2):
    """MI(X;Y) (bits by default) of a joint ``P[..., X, Y]`` over the last two axes,
    batched over the leading axes. ``joint`` is renormalized per batch element, so
    an un-normalized (e.g. cell-weighted count) table is accepted."""
    P = np.asarray(joint, dtype=np.float64)
    Z = P.sum(axis=(-2, -1), keepdims=True)
    P = np.where(Z > 0, P / Z, 0.0)
    Px = P.sum(axis=-1, keepdims=True)  # [..., X, 1]
    Py = P.sum(axis=-2, keepdims=True)  # [..., 1, Y]
    indep = Px * Py
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(P > 0, P * _logb((P + _EPS) / (indep + _EPS), base), 0.0)
    return np.sum(terms, axis=(-2, -1))


def joint_from_conditional(cond, weights):
    """Assemble a normalized joint ``P[..., C, Y]`` from per-clone conditionals
    ``cond`` = P(φ|c) ``[..., C, Y]`` and clone ``weights`` ``[..., C]`` (any scale;
    normalized here). ``weighted=False`` passes uniform weights, ``weighted=True``
    passes clone cell counts — the clone-mass choice lives with the caller."""
    cond = np.asarray(cond, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.clip(w.sum(axis=-1, keepdims=True), _EPS, None)
    return cond * w[..., None]
