"""``tl.phenotypic_flux`` (renamed from ``flux``) — per-clonotype phenotype-distribution
shift between two covariate values (§7.5), engine-backed.

For each clone in the ``cov_from`` ∩ ``cov_to`` intersection, the distance between its
phenotype distribution at ``cov_from`` and ``cov_to``, via ``_distance`` (kl / l1 / jsd,
KL & JSD in **bits**). **KL is the default**: the metrics document's eq 7 defines
phenotypic flux AS the KL divergence, and l1/jsd are tcri extensions for when a bounded or
symmetric measure is wanted. ``n_samples=0`` is the plug-in (a clone with no real shift reads
exactly 0); ``n_samples>0`` redraws both sides coherently (same seed) and summarizes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._distance import phenotype_distance
from ._common import grouped_series, joint_draws, summarize

__all__ = ["phenotypic_flux"]


def _flux_once(adata, *, cov_from, cov_to, n_samples, weighted, temperature, clones,
               distance_metric, random_state, device=None):
    dist_fn = phenotype_distance(distance_metric)
    draws_from, _ = joint_draws(adata, cov_from, n_samples=n_samples, weighted=weighted, device=device,
                                temperature=temperature, clones=clones, random_state=random_state)
    draws_to, _ = joint_draws(adata, cov_to, n_samples=n_samples, weighted=weighted, device=device,
                              temperature=temperature, clones=clones, random_state=random_state)
    per = []
    for (ids_f, Jf), (ids_t, Jt) in zip(draws_from, draws_to):
        idx_t = {c: i for i, c in enumerate(ids_t)}
        d = {}
        for i, c in enumerate(ids_f):
            if c not in idx_t:
                continue
            p = np.asarray(Jf[i], dtype=np.float64); q = np.asarray(Jt[idx_t[c]], dtype=np.float64)
            ps, qs = p.sum(), q.sum()
            if ps <= 0 or qs <= 0:
                d[c] = np.nan
                continue
            d[c] = float(dist_fn(p / ps, q / qs))
        per.append(d)
    keys = sorted(set().union(*[set(p) for p in per])) if per else []
    point = {c: float(np.nanmean([p.get(c, np.nan) for p in per])) for c in keys}
    drawsd = ({c: [p.get(c, np.nan) for p in per] for c in keys}
              if (n_samples and int(n_samples) > 0) else None)
    return point, drawsd


def phenotypic_flux(adata, *, cov_from, cov_to, groupby=None, splitby=None, n_samples=0,
                    temperature=1.0, clones=None, weighted=False, distance_metric="kl",
                    random_state=None, device=None):
    """Phenotypic flux — per-clone shift in ``P(φ|c)`` from ``cov_from`` to ``cov_to``.

    Parameters
    ----------
    cov_from, cov_to : str
        The two covariate values to compare (e.g. ``"pre"`` → ``"post"``).
    distance_metric : {"kl", "jsd", "l1"}
        How the two per-clone distributions are compared. Default ``"kl"`` (bits) matches
        METRICS eq 7, which defines flux as the KL divergence; ``"jsd"`` (bits) and
        ``"l1"`` are also available.
    n_samples : int
        ``0`` → point estimate; ``>0`` → posterior mean/sd/HDI per clone.
    weighted : bool
        Weight clones by cell count rather than treating them as equal units.
    groupby, splitby : str | None
        Compute per group and carry a split label, as in :func:`mutual_information`.

    Returns
    -------
    pandas.Series | pandas.DataFrame
        Flux per clonotype (or a tidy DataFrame, one row per group×clonotype, with
        ``groupby``).
    """
    if groupby is not None:
        def _compute(cl):
            return _flux_once(adata, cov_from=cov_from, cov_to=cov_to, n_samples=n_samples,
                              weighted=weighted, temperature=temperature, clones=cl,
                              distance_metric=distance_metric, random_state=random_state,
                              device=device)
        return grouped_series(adata, groupby=groupby, splitby=splitby, item_name="clonotype",
                              value="phenotypic_flux", compute=_compute)

    point, drawsd = _flux_once(adata, cov_from=cov_from, cov_to=cov_to, n_samples=n_samples,
                               weighted=weighted, temperature=temperature, clones=clones,
                               distance_metric=distance_metric, random_state=random_state,
                              device=device)
    if n_samples and int(n_samples) > 0:
        return pd.DataFrame({c: summarize(drawsd[c]) for c in point}).T
    return pd.Series(point, name="phenotypic_flux")
