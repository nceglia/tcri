"""Distance / divergence kernels for phenotype distributions.

Single home for the KL kernel, plus L1. Both operate on 1-D probability vectors and share
one eps floor; KL is in bits (log base 2).
``phenotype_distance`` is the string→callable dispatcher used by
``phenotypic_flux(distance_metric=)``.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _normalize(p):
    p = np.clip(np.asarray(p, float), EPS, None)
    return p / p.sum()


def kl_divergence(p, q, *, base: float = 2.0, eps: float = EPS) -> float:
    """KL(p ‖ q) in bits (base 2). Asymmetric."""
    p = np.clip(np.asarray(p, float), eps, None); p = p / p.sum()
    q = np.clip(np.asarray(q, float), eps, None); q = q / q.sum()
    return float(np.sum(p * (np.log(p / q) / np.log(base))))


def l1_distance(p, q) -> float:
    """L1 (Manhattan) distance between two normalized distributions, in [0, 2]."""
    return float(np.abs(_normalize(p) - _normalize(q)).sum())


_REGISTRY = {
    "l1": l1_distance,
    "kl": kl_divergence,
    "dkl": kl_divergence,
}


def phenotype_distance(metric):
    """Resolve ``distance_metric`` (a name or a callable ``f(p, q) -> float``)."""
    if callable(metric):
        return metric
    key = str(metric).lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"distance_metric must be a callable or one of {sorted(_REGISTRY)}; got {metric!r}"
        )
    return _REGISTRY[key]
