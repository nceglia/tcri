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
    build_result,
    build_stats,
    joint_draws,
    metric_table,
    resolve_groupby,
    validate_splitby,
)

__all__ = ["mutual_information"]

_EPS = 1e-15


def _mi_from_joint(J: np.ndarray, *, normalized: bool = True, mode: str = "min",
                   return_denom: bool = False):
    """MI (bits) of a clone×phenotype table ``J`` (any scale — renormalized here).

    ``return_denom`` also returns the normaliser, which a REFERENCE needs and cannot be given
    from outside. The normaliser comes from the same joint it normalises, so a null does not
    share it: at the default ``weighted=False`` the engine row-normalises, making ``h_c``
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


@tl_result(key=K.MUTUAL_INFORMATION, version=1, schema=schemas.MutualInformation,
           values=("value", "denom"), denominators=("denom",), default_null="phenotype")
def mutual_information(
    adata, *, covariate=None, groupby=None, splitby=None, n_samples=0,
    temperature=1.0, clones=None, weighted=False, normalized=True,
    normalize_mode="min", random_state=None, device=None,
    null_model="auto", fit=None, key_added=None, inplace=True,
):
    """I(c;φ|covariate) in bits — computed once, cached, and returned.

    Returns ``{"table", "result", "stats"}`` and stores the same object under
    ``uns[key_added or 'tcri_mutual_information']``. ``pl.mutual_information`` renders from that
    cache rather than recomputing, so the plot cannot disagree with the frame in your hand.

    ``groupby`` is the replicate; left implicit it resolves to the column registered as
    ``replicate`` at ``setup_anndata``, and the effective value is recorded in provenance.
    ``splitby`` requires ``groupby`` and must be constant within group; when set, the
    between-split contrast lands in ``stats`` with ``n`` counting GROUPS.

    ``null_model`` names the permutation reference (``"auto"`` is the phenotype null, ``None``
    the bare value); ``fit`` selects which fit the number is computed on. The result carries
    ``denom``, the normaliser this MI was divided by, so the reference's own normaliser can be
    stored beside it and both bit values recovered exactly.
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
            out.append({"value": value, "denom": denom})
        return out

    table = metric_table(adata, covariate=covariate, groupby=gkey, splitby=splitby,
                         clones=clones, item_col=None, compute=_compute, fit=fit)
    result = build_result(table, extra_values=("denom",))
    stats = build_stats(result, groupby=gkey, splitby=splitby)

    payload = {"table": table, "result": result, "stats": stats}
    return with_resolved_params(payload, groupby=gkey) if resolved else payload
