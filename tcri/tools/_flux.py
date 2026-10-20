"""``tl.phenotypic_flux`` — how far each clone's phenotype distribution moves between two
covariate levels, per replicate and per persistent clone, engine-backed.

For each clone at both ``cov_from`` and ``cov_to`` (a persistent clone), the distance between its
phenotype distribution at the two levels, via ``_distance`` (kl / l1, KL in **bits**). **KL is
the default** — flux is defined as the KL divergence; l1 is available for when a bounded or
symmetric measure is wanted. Per replicate, the flux is the persistent clones' distances weighted
by each clone's share of the replicate's ``cov_from`` joint, renormalized over them: ``1/C`` at
``weighted=False``, the cell share at ``weighted=True``. A clone whose distance is undefined has
no weight. ``n_samples=0`` is the plug-in (a clone with no real shift reads exactly 0);
``n_samples>0`` redraws both levels from one shared sample (common random numbers) and
summarizes.
"""
from __future__ import annotations

import numpy as np

from .._compute._distance import phenotype_distance
from .._state import keys as K
from .._state import schemas
from .._state.storage import tl_result, with_resolved_params
from .._compute._tables import (build_result, build_stats, change_draws, check_covariates,
                                metric_frames, resolve_groupby, resolve_seed, rows_of,
                                validate_splitby)

__all__ = ["phenotypic_flux"]


def _distances(P, Q, dist_fn):
    """The distance of each row of ``P`` to the same row of ``Q``, each row renormalized; NaN
    for a row with no mass on either side."""
    out = np.full(len(P), np.nan)
    for i, (p, q) in enumerate(zip(P, Q)):
        ps, qs = p.sum(), q.sum()
        if ps > 0 and qs > 0:
            out[i] = float(dist_fn(p / ps, q / qs))
    return out


def _weighted(d, mass):
    """The mean of ``d`` weighted by ``mass``, over the rows where both are finite and the mass
    is positive; NaN when there are none."""
    keep = np.isfinite(d) & np.isfinite(mass) & (mass > 0)
    if not keep.any():
        return np.nan
    return float(np.sum(d[keep] * mass[keep]) / np.sum(mass[keep]))


@tl_result(key=K.PHENOTYPIC_FLUX, version=4, schema=schemas.PhenotypicFlux,
           default_null="covariate", distance=True)
def phenotypic_flux(adata, *, cov_from, cov_to, groupby=None, splitby=None, n_samples=0,
                    temperature=1.0, clones=None, exclude_pools=True, weighted=False,
                    distance_metric="kl", random_state=None, device=None, fit=None):
    """Phenotype-distribution distance from ``cov_from`` to ``cov_to`` (bits for kl), per
    replicate and per persistent clone. See the module docstring.

    Stores ``{replicate, item, stats}``, each level holding ``draws`` and ``summary``, under
    ``uns[key_added or 'tcri_phenotypic_flux']`` and returns ``None``; with ``inplace=False``
    nothing is stored and the whole result is returned. The replicate level has one row per
    replicate, read with ``tcri.get.replicates``; the item level one row per replicate and
    persistent clone, read with ``tcri.get.items``. A clone at one level only has no item row; a
    replicate with no persistent clone has a NaN replicate row.

    Flux is defined as the KL divergence, which is why ``distance_metric`` defaults to
    ``"kl"``; ``"l1"`` is available for when a bounded or symmetric measure is wanted.

    ``stats`` holds the test of the adjusted flux against 0, one row per split level with
    ``splitby`` or one over every replicate without, beside the split rows. Flux is never negative, so the value itself is not tested against 0: without a
    usable reference that row has a NaN p and blank stars.

    ``null_model`` names the permutation reference; ``"auto"`` is the COVARIATE null, because
    what a flux measures is movement between covariate levels and the reference has to be a fit
    in which that movement is absent. ``fit`` selects which fit the number is computed on.

    ``exclude_pools`` leaves out the ``pooled@{group}`` labels ``tcri.pp.pool_rare_clones``
    writes. A pool is a mixture of many rare clones, not a clone, so by default it is left out
    and the flux counts exactly the clones ``pool_rare_clones`` kept, with ``clones``
    selecting among them. ``False`` counts each pool as one clone. A column with no pooling
    step has nothing to leave out.
    """
    if cov_from is None or cov_to is None:
        raise ValueError(
            f"phenotypic_flux: pass both cov_from= and cov_to=; flux is the distance from one "
            f"covariate level to another (got cov_from={cov_from!r}, cov_to={cov_to!r}).")
    check_covariates(adata, covariate=None, cov_from=cov_from, cov_to=cov_to,
                     metric="phenotypic_flux")
    gkey, resolved = resolve_groupby(adata, groupby)
    validate_splitby(adata.obs, gkey, splitby)
    dist_fn = phenotype_distance(distance_metric)
    seed = resolve_seed(random_state)

    def _compute(clone_subset, persistent):
        draws_from, draws_to, _cols = change_draws(
            adata, cov_from, cov_to, clones=clone_subset, seed=seed, n_samples=n_samples,
            weighted=weighted, temperature=temperature, device=device, fit=fit)
        per = []
        for (ids_f, Jf), (ids_t, Jt) in zip(draws_from, draws_to):
            P = rows_of(ids_f, Jf, persistent)
            d = _distances(P, rows_of(ids_t, Jt, persistent), dist_fn)
            per.append(({"value": _weighted(d, P.sum(axis=1))},
                        {c: {"value": float(v)} for c, v in zip(persistent, d)}))
        return per

    rep_draws, item_draws = metric_frames(
        adata, cov_from=cov_from, cov_to=cov_to, groupby=gkey, splitby=splitby, clones=clones,
        exclude_pools=exclude_pools, item_col="clonotype", compute=_compute,
        metric="phenotypic_flux", fit=fit)
    rep_summary = build_result(rep_draws)
    payload = {"replicate": {"draws": rep_draws, "summary": rep_summary},
               "item": {"draws": item_draws, "summary": build_result(item_draws)},
               "stats": build_stats(rep_summary, groupby=gkey, splitby=splitby,
                                    change={"cov_from": cov_from, "cov_to": cov_to,
                                            "distance": True})}
    return with_resolved_params(payload, groupby=gkey) if resolved else payload
