"""The joint-distribution engine (``tl``) — a thin DataFrame wrapper over
:func:`tcri._compute._joint._joint_draws`. Re-exported top-level as
``tcri.joint_distribution``; unifies the old ``joint_distribution`` +
``joint_distribution_posterior``.

See §7.1 of ``governance/API_CONTRACT.md`` for the math. This
is the substrate every metric consumes (Phase 6 migrates them onto it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._state import keys as K
from .._state import schemas
from .._state.storage import tl_result, with_resolved_params
from .._compute._joint import _joint_draws

__all__ = ["joint_distribution"]


def _engine_blocks(
    adata,
    *,
    covariate,
    n_samples,
    use_logits,
    weighted,
    temperature,
    random_state,
    device,
):
    """Validate, then run the numeric core; return the raw per-covariate blocks.

    Shared by :func:`joint_distribution` (which formats them into a labelled
    DataFrame) and the metric fast path (which consumes the arrays directly), so
    both see identical validation, the same engine call, and one shared draw.

    Returns ``(blocks, n_draws, clonotype_cats, covariate_cats, phenotype_cats)``
    where ``blocks`` is a list of ``(covariate_index, clone_idx, J)`` and ``J`` has
    shape ``[S, n_rows, P]``.
    """
    phenotype_cats = list(adata.uns[K.PHENOTYPE_CATEGORIES])
    clonotype_cats = list(adata.uns[K.CLONOTYPE_CATEGORIES])
    covariate_cats = list(adata.uns[K.COVARIATE_CATEGORIES])

    logits = None
    if use_logits:
        if K.X_LOGITS not in adata.obsm:
            raise RuntimeError(
                f"obsm[{K.X_LOGITS!r}] missing — run model.to_anndata(...) or pass use_logits=False."
            )
        logits = adata.obsm[K.X_LOGITS]

    cov_idx = None
    if covariate is not None:
        try:
            cov_idx = covariate_cats.index(covariate)
        except ValueError:
            raise ValueError(f"covariate {covariate!r} not found among {covariate_cats}")

    # subset/filtered-AnnData guard: the per-cell uns arrays live in full-cell space and
    # are NOT sliced when adata is subset, whereas obsm/obs ARE — so a slice silently
    # misaligns cells. Fail loudly (mirrors the legacy joint_distribution_posterior guard).
    n_obs = adata.n_obs
    n_reg = len(np.asarray(adata.uns[K.CT_ARRAY]))
    if n_reg != n_obs or len(np.asarray(adata.uns[K.COV_ARRAY])) != n_obs:
        raise ValueError(
            f"joint_distribution received an AnnData whose per-cell registration arrays "
            f"(uns[{K.CT_ARRAY!r}], len {n_reg}) do not match adata.n_obs ({n_obs}). This "
            f"happens on a filtered/sliced AnnData: the full-space uns arrays misalign against "
            f"the subset obsm/obs. Re-run model.to_anndata(...) on the filtered object, or pass "
            f"the full object and filter with `clones=`."
        )

    # local_scale is required for the Dirichlet draw; refuse to silently fall back at n>0.
    local_scale = adata.uns.get(K.LOCAL_SCALE, None)
    if n_samples and int(n_samples) > 0 and local_scale is None:
        raise RuntimeError(
            f"n_samples>0 needs uns[{K.LOCAL_SCALE!r}] for the clamped-Dirichlet draw, but it "
            f"is missing; run model.to_anndata(...)."
        )
    local_scale = float(local_scale) if local_scale is not None else 1.0
    # DE-5b: present on anything written by a current to_anndata; absent on older objects,
    # which fall back to the local_scale reconstruction rather than failing.
    conc_ct = adata.uns.get(K.CONC_CT, None)

    blocks, n_draws = _joint_draws(
        adata.uns[K.P_CT],
        adata.uns[K.CT_TO_COV],
        adata.uns[K.CT_TO_C],
        adata.uns[K.CT_ARRAY],
        adata.uns[K.COV_ARRAY],
        local_scale=local_scale,
        conc_ct=conc_ct,
        n_samples=n_samples,
        temperature=temperature,
        use_logits=use_logits,
        covariate_idx=cov_idx,
        logits=logits,
        gate_prob=adata.uns.get(K.GATE_PROB, None),
        weighted=weighted,
        random_state=random_state,
        device=device,
    )
    return blocks, n_draws, clonotype_cats, covariate_cats, phenotype_cats


@tl_result(key=K.JOINT_DISTRIBUTION, version=1, schema=schemas.JointDistribution)
def joint_distribution(
    adata,
    *,
    covariate=None,
    n_samples=0,
    use_logits=True,
    weighted=False,
    clones=None,
    temperature=1.0,
    random_state=None,
    device=None,
    key_added=None,
    inplace=True,
) -> dict:
    """Clone×phenotype distribution from the learned posterior of ``p_ct``.

    Parameters
    ----------
    covariate : str | None
        A covariate value; ``None`` computes all covariate values in one shared-draw
        pass (adds a leading ``covariate`` index level).
    n_samples : int
        ``0`` → deterministic posterior-mean table; ``N`` → ``N`` clamped-Dirichlet
        draws (adds a ``sample_id`` index level). Only place ``local_scale`` enters.
    use_logits : bool
        ``True`` → fold per-cell classifier logits with ``log(base)`` (gate-aware) and
        aggregate per clone, matching :meth:`~tcri.model._model.TCRIModel.predict`;
        ``False`` → the ct-level base table. Neither touches the generative prior.
    weighted : bool
        ``False`` → each clone is one unit (per-clone simplex). ``True`` → each clone
        row is scaled by its (ct-keyed) cell count (cell-weighted).
    temperature : float
        Tempers the base once; ``T=1`` is the identity (and reproduces ``predict()``
        on the ``use_logits=True`` path).
    random_state : int | numpy.Generator | torch.Generator | None
        Seeds the torch Dirichlet generator for ``n_samples>0``; ignored at ``0``.
    device : str | None
        Routes the numeric core through ``_compute/_xp`` (CPU / torch-CUDA). Result is
        always a host DataFrame.

    Returns
    -------
    pandas.DataFrame
        Columns = phenotype categories. Index = clonotype (``+ sample_id`` for
        ``n_samples>0``, ``+ covariate`` leading level for ``covariate=None``).
        Provenance in ``df.attrs["params"]``.
    """
    blocks, n_draws, clonotype_cats, covariate_cats, phenotype_cats = _engine_blocks(
        adata,
        covariate=covariate,
        n_samples=n_samples,
        use_logits=use_logits,
        weighted=weighted,
        temperature=temperature,
        random_state=random_state,
        device=device,
    )

    sampling = bool(n_samples and int(n_samples) > 0)
    all_cov = covariate is None
    frames = []
    for m, clone_idx, J in blocks:                       # J: [S, n_rows, P]
        clone_ids = [clonotype_cats[i] for i in clone_idx]
        S = J.shape[0]
        if sampling:
            arr = J.transpose(1, 0, 2).reshape(-1, J.shape[2])   # [n_rows*S, P]
            idx_clone = np.repeat(clone_ids, S)
            idx_samp = np.tile(np.arange(S), len(clone_ids))
            cols = [idx_clone, idx_samp]
            names = ["clonotype", "sample_id"]
        else:
            arr = J[0]                                            # [n_rows, P]
            cols = [clone_ids]
            names = ["clonotype"]
        if all_cov:
            cols = [[covariate_cats[m]] * arr.shape[0], *cols]
            names = ["covariate", *names]
        index = (pd.MultiIndex.from_arrays(cols, names=names)
                 if len(cols) > 1 else pd.Index(cols[0], name=names[0]))
        frames.append(pd.DataFrame(arr, columns=phenotype_cats, index=index))

    df = pd.concat(frames) if len(frames) > 1 else frames[0]

    if clones is not None:
        clones = list(clones)
        if isinstance(df.index, pd.MultiIndex):
            # filter to the listed clones (absent dropped, not all-zero) then order by the
            # requested list — stable within the sample_id/covariate levels (matches the
            # single-index reindex; §7.1 "reindex to the exact list").
            keep = df.index.get_level_values("clonotype").isin(clones)
            df = df[keep]
            rank = pd.Index(df.index.get_level_values("clonotype")).map({c: i for i, c in enumerate(clones)})
            df = df.iloc[np.argsort(np.asarray(rank), kind="stable")]
        else:
            df = df.reindex([c for c in clones if c in df.index])

    # WIDE on purpose, unlike the four scalar metrics. A joint is a matrix -- clone x
    # phenotype, rows summing to 1 -- and forcing it into their long (item, draw, value) form
    # would make the common case worse to read. The long convention applies where each item
    # reduces to ONE number; this does not.
    #
    # `result` is the posterior mean over draws (identical to `table` at n_samples=0), so the
    # caller always has one table to reason about regardless of whether they sampled.
    if sampling:
        result = df.groupby(level=[n for n in df.index.names if n != "sample_id"],
                            observed=True, sort=False).mean()
    else:
        result = df

    # `params` is captured by the decorator from the call signature, so the old
    # df.attrs["params"] hand-roll is gone -- attrs does not survive most pandas operations
    # and did not survive a write_h5ad at all.
    # n_draws is what ACTUALLY happened, which the call arguments cannot say on their own --
    # the same distinction as epochs_actual vs max_epochs in the training record. Recorded as an
    # effective value so the provenance answers "how many draws is this table built from".
    return with_resolved_params({"table": df, "result": result}, n_draws=int(n_draws))
