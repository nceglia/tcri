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
    build_result,
    build_stats,
    joint_draws,
    metric_frames,
    resolve_groupby,
    validate_splitby,
)

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
           values=("value", "denom"), denominators=("denom",), default_null="phenotype")
def mutual_information(
    adata, *, covariate=None, groupby=None, splitby=None, n_samples=0,
    temperature=1.0, clones=None, exclude_pools=True, weighted=False, normalized=True,
    normalize_mode="min", random_state=None, device=None,
    null_model="auto", fit=None, key_added=None, inplace=True,
):
    """I(c;φ|covariate) in bits — computed once, cached, and returned.

    Returns the replicate summary, one row per replicate, and stores ``{replicate: {draws,
    summary}, stats}`` under ``uns[key_added or 'tcri_mutual_information']``; with
    ``inplace=False`` nothing is stored and that whole result is returned. Mutual information has
    one value per joint, so it has no item level. ``pl.mutual_information`` renders from the
    cache rather than recomputing, so the plot cannot disagree with the frame in your hand.

    ``groupby`` is the replicate; left implicit it resolves to the column registered as
    ``replicate`` at ``setup_anndata``, and the effective value is recorded in provenance.
    ``splitby`` requires ``groupby`` and must be constant within group; when set, the
    between-split contrast lands in ``stats`` with ``n`` counting GROUPS.

    ``null_model`` names the permutation reference (``"auto"`` is the phenotype null, ``None``
    the bare value); ``fit`` selects which fit the number is computed on. The result carries
    ``denom``, the normalizer this MI was divided by, so the reference's own normalizer can be
    stored beside it and both bit values recovered exactly.

    ``exclude_pools`` leaves out the ``pooled@{group}`` labels ``tcri.pp.pool_rare_clones``
    writes. A pool is a mixture of many rare clones, not a clone, so by default it is left out
    and the MI counts exactly the clones ``pool_rare_clones`` kept, with ``clones``
    selecting among them. ``False`` counts each pool as one clone. A column with no pooling
    step has nothing to leave out.

    A replicate with no joint at ``covariate`` has a NaN row. With ``normalized=True``, a
    replicate whose normalizer is 0 (one clone, or one phenotype) raises ``ValueError`` naming
    it; ``normalized=False`` reports its bits.
    """
    gkey, resolved = resolve_groupby(adata, groupby)
    validate_splitby(adata.obs, gkey, splitby)

    def _compute(clone_subset):
        draws, _cols = joint_draws(
            adata, covariate, n_samples=n_samples, weighted=weighted, device=device,
            temperature=temperature, clones=clone_subset, random_state=random_state, fit=fit,
        )
        out = []
        for _ids, J in draws:
            value, denom = _mi_from_joint(J, normalized=normalized, mode=normalize_mode,
                                          return_denom=True)
            if normalized and np.isfinite(denom) and not denom > 0:
                raise ZeroNormalizer(
                    f"its normalizer ({normalize_mode!r} of the clone and phenotype entropies) "
                    f"is 0: it has one clone or one phenotype. Pass normalized=False for bits, "
                    f"or leave this replicate out with clones=.")
            out.append(({"value": value, "denom": denom}, None))
        return out

    rep_draws, _ = metric_frames(adata, covariate=covariate, groupby=gkey, splitby=splitby,
                                 clones=clones, exclude_pools=exclude_pools, item_col=None,
                                 compute=_compute, metric="mutual_information", fit=fit)
    rep_summary = build_result(rep_draws, extra_values=("denom",))
    payload = {"replicate": {"draws": rep_draws, "summary": rep_summary},
               "stats": build_stats(rep_summary, groupby=gkey, splitby=splitby)}
    return with_resolved_params(payload, groupby=gkey) if resolved else payload
