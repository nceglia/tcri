"""``tl.mutual_information`` — clone↔phenotype coupling I(c;φ|m) in **bits** (§7.4).

Engine-backed rewrite of the old ``metrics.mutual_information``. Default
``normalize_mode="min"`` (coefficient of constraint I/min(H_c,H_p)) — the ``"average"``
denominator throttles normalized MI by ~1/log2(C) and is non-comparable across groups with
different clone counts (the blocking fix). ``n_samples=0`` is the deterministic plug-in.
"""
from __future__ import annotations

import numpy as np

from ._common import grouped_scalar, is_precomputed_joint, joint_draws, summarize

__all__ = ["mutual_information"]

_EPS = 1e-15


def _mi_from_joint(J: np.ndarray, *, normalized: bool = True, mode: str = "min") -> float:
    """MI (bits) of a clone×phenotype table ``J`` (any scale — renormalized here)."""
    J = np.asarray(J, dtype=np.float64)
    total = J.sum()
    if total <= 0:
        return np.nan
    pxy = J / total
    px = pxy.sum(1, keepdims=True)   # P(clone)
    py = pxy.sum(0, keepdims=True)   # P(phenotype)
    mi = float(np.sum(pxy * np.log2((pxy + _EPS) / (px @ py + _EPS))))
    if not normalized:
        return mi
    h_c = float(-np.sum(px * np.log2(px + _EPS)))
    h_p = float(-np.sum(py * np.log2(py + _EPS)))
    denom = min(h_c, h_p) if mode == "min" else 0.5 * (h_c + h_p)
    return mi / denom if denom > 0 else 0.0


def mutual_information(
    adata_or_jd, *, covariate=None, groupby=None, splitby=None, n_samples=0,
    temperature=1.0, clones=None, weighted=False, normalized=True,
    normalize_mode="min", random_state=None, device=None,
):
    """Normalized mutual information ``I(c;φ)`` between clonotype and phenotype (bits).

    Parameters
    ----------
    adata_or_jd : AnnData | pandas.DataFrame
        A fitted AnnData (``to_anndata`` applied) or a precomputed joint from
        :func:`joint_distribution` (fast path; ``n_samples=0`` and no ``groupby``).
    covariate : str | None
        Covariate value to compute at; ``None`` uses the covariate-marginal joint.
    groupby : str | None
        Compute one value per group (e.g. ``"patient"``); returns a tidy DataFrame with
        an ``"MI"`` column instead of a scalar.
    splitby : str | None
        A per-group label (e.g. response) carried onto the grouped DataFrame for a
        downstream :func:`compare_groups`.
    n_samples : int
        ``0`` → point estimate; ``>0`` → posterior draws summarised as mean/sd/HDI.
    weighted : bool
        Weight each clone by its cell count rather than treating clones as equal units.
    normalized : bool
        Divide by the max-entropy denominator (see ``normalize_mode``) so the result is in
        ``[0, 1]``; ``False`` returns the raw MI in bits.
    normalize_mode : {"min", "average"}
        Denominator: ``"min"`` = ``min(H(c), H(φ))`` (default, comparable across groups
        with different clone counts); ``"average"`` = ``½(H(c)+H(φ))`` — the note's eq 6
        NMI. Pass ``"average"`` to reproduce the manuscript benchmark.

    Returns
    -------
    float | dict | pandas.DataFrame
        A scalar (``n_samples=0``, no ``groupby``), a mean/sd/HDI summary
        (``n_samples>0``), or a tidy DataFrame (``groupby``).
    """
    if groupby is not None:
        if is_precomputed_joint(adata_or_jd):
            raise ValueError("groupby requires an AnnData, not a precomputed joint (§7.9).")

        def _compute(cl):
            draws, cols = joint_draws(
                adata_or_jd, covariate, n_samples=n_samples, weighted=weighted, device=device,
                temperature=temperature, clones=cl, random_state=random_state,
            )
            vals = [_mi_from_joint(J, normalized=normalized, mode=normalize_mode) for _, J in draws]
            return (float(np.nanmean(vals)), vals if (n_samples and int(n_samples) > 0) else None)
        return grouped_scalar(adata_or_jd, groupby=groupby, splitby=splitby, value="MI", compute=_compute)

    if is_precomputed_joint(adata_or_jd):
        if n_samples and int(n_samples) > 0:
            raise ValueError("precomputed-joint fast path is valid only at n_samples=0 (§7.9).")
        return _mi_from_joint(adata_or_jd.values, normalized=normalized, mode=normalize_mode)

    draws, cols = joint_draws(
        adata_or_jd, covariate, n_samples=n_samples, weighted=weighted, device=device,
        temperature=temperature, clones=clones, random_state=random_state,
    )
    vals = [_mi_from_joint(J, normalized=normalized, mode=normalize_mode) for _, J in draws]
    if n_samples and int(n_samples) > 0:
        return summarize(vals)
    return vals[0]
