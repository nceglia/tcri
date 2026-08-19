"""Distance / divergence kernels for phenotype distributions.

Single home for the KL kernel (was `metrics.dkl` + `flux.dkl_func`, two copies)
plus L1 and a symmetric Jensen–Shannon option. All operate on 1-D probability
vectors, use log base 2 (bits), and share one eps floor. ``phenotype_distance``
is the string→callable dispatcher used by ``phenotypic_flux(distance_metric=)``.
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


def jensen_shannon(p, q, *, base: float = 2.0, eps: float = EPS) -> float:
    """Jensen–Shannon divergence: symmetric, bounded [0, 1] bit — the recommended
    symmetric shift measure."""
    p = _normalize(p)
    q = _normalize(q)
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m, base=base, eps=eps)
                 + 0.5 * kl_divergence(q, m, base=base, eps=eps))


_REGISTRY = {
    "l1": l1_distance,
    "kl": kl_divergence,
    "dkl": kl_divergence,
    "js": jensen_shannon,
    "jsd": jensen_shannon,
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
