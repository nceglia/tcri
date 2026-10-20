"""``tl.mutual_information`` — clone↔phenotype coupling I(c;φ|m) in **bits**.

Default ``normalize_mode="min"`` — the coefficient of constraint, I/min(H_c,H_p). The
``"average"`` denominator (classical NMI) scales with log2(C) and is not comparable across
groups with different clone counts, which is why it is not the default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._state import keys as K
from .._state import schemas
from .._state.storage import tl_result, with_resolved_params
from .._compute._tables import (
    ZeroNormalizer,
    at_level,
    build_result,
    build_stats,
    change_draws,
    check_covariates,
    joint_draws,
    metric_frames,
    resolve_groupby,
    resolve_seed,
    validate_splitby,
)
from ._entropy import DEFAULT_NULL

__all__ = ["mutual_information"]

_EPS = 1e-15


def _mi_from_joint(J: np.ndarray, *, normalized: bool = True, mode: str = "min",
                   return_denom: bool = False):
    """MI (bits) of a clone×phenotype table ``J`` (any scale — renormalized here).

    ``return_denom`` also returns the normalizer, which a REFERENCE needs and cannot be given
    from outside. The normalizer comes from the same joint it normalizes, so a null does not
    share it: at the default ``weighted=False`` the engine row-normalizes, making ``h_c``
    exactly ``log2(C)``, and ``normalize_mode="min"`` therefore selects ``h_p``, the FITTED
    phenotype marginal entropy. Both denominators are stored so a reader recovers both bit
    values exactly, as ``value*denom`` and ``null_value*null_denom``.
    """
    J = np.asarray(J, dtype=np.float64)
    total = J.sum()
    if total <= 0:
        return (np.nan, np.nan) if return_denom else np.nan
    pxy = J / total
    px = pxy.sum(1, keepdims=True)   # P(clone)
    py = pxy.sum(0, keepdims=True)   # P(phenotype)
    mi = float(np.sum(pxy * np.log2((pxy + _EPS) / (px @ py + _EPS))))
    if not normalized:
        return (mi, 1.0) if return_denom else mi
    h_c = float(-np.sum(px * np.log2(px + _EPS)))
    h_p = float(-np.sum(py * np.log2(py + _EPS)))
    denom = min(h_c, h_p) if mode == "min" else 0.5 * (h_c + h_p)
    value = mi / denom if denom > 0 else 0.0
    return (value, denom) if return_denom else value


@tl_result(key=K.MUTUAL_INFORMATION, version=4, schema=schemas.MutualInformation,
           values=("value", "denom", "value_from", "value_to", "denom_from", "denom_to"),
           denominators=("denom", "denom_from", "denom_to"), default_null=DEFAULT_NULL)
def mutual_information(
    adata, *, covariate=None, cov_from=None, cov_to=None, groupby=None, splitby=None,
    n_samples=0, temperature=1.0, clones=None, exclude_pools=True, weighted=False,
    normalized=True, normalize_mode="min", random_state=None, device=None, fit=None,
):
    """I(c;φ|covariate) in bits, one value per replicate, or its change between two levels.

    Stores ``{replicate: {draws, summary}, stats}`` under ``uns[key_added or
    'tcri_mutual_information']`` and returns ``None``; with ``inplace=False`` nothing is stored
    and that whole result is returned. The replicate summary is read with
    ``tcri.get.replicates`` and the tests with ``tcri.get.stats``. Mutual information has one
    value per joint, so it has no item level. ``pl.mutual_information`` renders the stored
    result rather than recomputing, so the plot cannot disagree with it.

    ``groupby`` is the replicate; left implicit it resolves to the column registered as
    ``replicate`` at ``setup_anndata``, and the effective value is recorded in provenance.
    ``splitby`` requires ``groupby`` and must be constant within group; when set, the
    between-split contrast lands in ``stats`` with ``n`` counting GROUPS.

    Pass ``covariate`` for the value at one level, or ``cov_from`` with ``cov_to`` for the change:
    ``cov_to`` − ``cov_from`` within a draw, both levels drawn from one shared sample, each side
    counting every clone the replicate has at its level and divided by its own normalizer. The
    change carries its endpoints, ``value_from`` and ``value_to``, and ``stats`` also holds its
    paired test, one row per split level with ``splitby``, or one over every replicate without.

    ``null_model`` names the permutation reference (``"auto"`` is the phenotype null at one
    level and the covariate null on a change, ``None`` the bare value); ``fit`` selects which fit
    the number is computed on. The result carries ``denom``, the normalizer this MI was divided
    by (``denom_from`` and ``denom_to`` on a change), so the reference's own normalizer can be
    stored beside it and both bit values recovered exactly.

    ``exclude_pools`` leaves out the ``pooled@{group}`` labels ``tcri.pp.pool_rare_clones``
    writes. A pool is a mixture of many rare clones, not a clone, so by default it is left out
    and the MI counts exactly the clones ``pool_rare_clones`` kept, with ``clones``
    selecting among them. ``False`` counts each pool as one clone. A column with no pooling
    step has nothing to leave out.

    A replicate with no joint at a level has a NaN row. With ``normalized=True``, a replicate
    whose normalizer is 0 raises ``ValueError`` naming it and the level: one clone or one phenotype at
    ``normalize_mode="min"``, one clone and one phenotype at ``"average"``. ``normalized=False``
    reports its bits.
    """
    change = check_covariates(adata, covariate=covariate, cov_from=cov_from, cov_to=cov_to,
                              metric="mutual_information") == "change"
    gkey, resolved = resolve_groupby(adata, groupby)
    validate_splitby(adata.obs, gkey, splitby)

    def _mi(J):
        value, denom = _mi_from_joint(J, normalized=normalized, mode=normalize_mode,
                                      return_denom=True)
        if normalized and np.isfinite(denom) and not denom > 0:
            which = "or" if normalize_mode == "min" else "and"
            raise ZeroNormalizer(
                f"its normalizer ({normalize_mode!r} of the clone and phenotype entropies) "
                f"is 0: it has one clone {which} one phenotype. Pass normalized=False for "
                f"bits, or leave this replicate out with clones=.")
        return value, denom

    def _compute(clone_subset):
        draws, _cols = joint_draws(
            adata, covariate, n_samples=n_samples, weighted=weighted, device=device,
            temperature=temperature, clones=clone_subset, random_state=random_state, fit=fit,
        )
        out = []
        for _ids, J in draws:
            value, denom = _mi(J)
            out.append(({"value": value, "denom": denom}, None))
        return out

    seed = resolve_seed(random_state) if change else None

    def _compute_change(clone_subset, _persistent):
        draws_from, draws_to, _cols = change_draws(
            adata, cov_from, cov_to, clones=clone_subset, seed=seed, n_samples=n_samples,
            weighted=weighted, temperature=temperature, device=device, fit=fit)
        out = []
        for (_f, Jf), (_t, Jt) in zip(draws_from, draws_to):
            v_from, d_from = at_level(cov_from, _mi, Jf)
            v_to, d_to = at_level(cov_to, _mi, Jt)
            out.append(({"value": v_to - v_from, "value_from": v_from, "value_to": v_to,
                         "denom_from": d_from, "denom_to": d_to}, None))
        return out

    rep_draws, _ = metric_frames(adata, covariate=covariate, cov_from=cov_from, cov_to=cov_to,
                                 groupby=gkey, splitby=splitby, clones=clones,
                                 exclude_pools=exclude_pools, item_col=None,
                                 compute=_compute_change if change else _compute,
                                 metric="mutual_information", fit=fit)
    extras = (("value_from", "value_to", "denom_from", "denom_to") if change else ("denom",))
    rep_summary = build_result(rep_draws, extra_values=extras, one_mask=change)
    payload = {"replicate": {"draws": rep_draws, "summary": rep_summary},
               "stats": build_stats(rep_summary, groupby=gkey, splitby=splitby,
                                    change=({"cov_from": cov_from, "cov_to": cov_to}
                                            if change else None))}
    return with_resolved_params(payload, groupby=gkey) if resolved else payload
