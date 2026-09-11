"""``perturb.gene_importance`` -- how far silencing each gene moves the mean phenotype call.

For a gene ``j`` and a set of cells ``C``::

    φ_i(X)     per-cell phenotype probability, as predict() computes it
    φ̄_C(X)    its mean over the cells
    shift_j,p  = φ̄_C(X)_p − φ̄_C(X^(j))_p          X^(j): column j zeroed
    I_j        = Σ_p | shift_j,p |                 in [0, 2]

One pass over the data: the baseline and the ``p_ct`` draws are computed once per batch and
shared by every gene; each gene costs one encoder forward per batch. The per-draw values are
the ``table`` the shared reducers summarise; the signed decomposition is the ``shift`` slot.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from .._compute._tables import build_result, resolve_groupby, validate_splitby
from .._state import keys as K
from .._state import schemas
from .._state.storage import tl_result, with_resolved_params
from ._kernel import (batches, combine, group_sums, head_logits, log_prior_for,
                      prior_draws, resolve_genes)
from ._tables import cell_groups, importance_table, shift_table, stats_per_gene

__all__ = ["gene_importance"]


def _accumulate(model, adata, *, positions, gid, n_groups, draws, use_gate, batch_size):
    """Per-group sums of baseline and per-gene perturbed probabilities, plus cell counts.

    Returns float64 CPU tensors ``base[S', n_groups, P]``, ``pert[G, S', n_groups, P]`` and
    ``counts[n_groups]``. Each gene zeroes one column of a shared buffer and restores it, so
    the per-gene cost is the encoder forward and nothing else.
    """
    module = model.module
    module.eval()
    S, _n_ct, P = draws.shape
    G = len(positions)
    base = torch.zeros(S, n_groups, P, dtype=torch.float64)
    pert = torch.zeros(G, S, n_groups, P, dtype=torch.float64)
    counts = torch.zeros(n_groups, dtype=torch.float64)
    gid_all = torch.as_tensor(gid, dtype=torch.long)
    gate = module.gate_prob

    offset = 0
    with torch.no_grad():
        for x, b, idx, n in batches(model, adata, batch_size=batch_size):
            # labels follow ROW order (the loader is sequential, as predict() relies on);
            # the prior follows the GLOBAL id (ct_array is indexed by training cell id)
            g_rows = gid_all[offset:offset + n]
            offset += n
            keep = g_rows >= 0
            if not bool(keep.any()):
                continue
            keep_dev = keep.to(x.device)
            x, b, idx = x[keep_dev], b[keep_dev], idx[keep_dev]
            g_cpu = g_rows[keep]
            g_dev = g_cpu.to(x.device)
            counts.index_add_(0, g_cpu, torch.ones(int(keep.sum()), dtype=torch.float64))

            log_prior = log_prior_for(draws, module, idx) if use_gate else None
            base += group_sums(combine(head_logits(module, x, b), log_prior, gate, use_gate),
                               g_dev, n_groups)
            buf = x.clone()
            for k, j in enumerate(positions):
                col = buf[:, j].clone()
                buf[:, j] = 0.0
                probs = combine(head_logits(module, buf, b), log_prior, gate, use_gate)
                pert[k] += group_sums(probs, g_dev, n_groups)
                buf[:, j] = col
    return base, pert, counts


@tl_result(key=K.GENE_IMPORTANCE, version=1, schema=schemas.GeneImportance, data_param="adata",
           default_null="phenotype", reference_arg="model", per_gene_stats=True)
def gene_importance(model, adata, *, genes=None, covariate=None, groupby=None, splitby=None,
                    n_samples=0, use_gate=True, batch_size=4096, random_state=None,
                    null_model="auto", key_added=None, inplace=True) -> dict:
    """Importance of each gene to the phenotype call: the L1 distance the mean phenotype
    distribution moves when the gene is silenced. Computed once, cached, returned.

    Parameters
    ----------
    model
        A fitted :class:`~tcri.model.TCRIModel`; parameters are held fixed.
    adata
        The object to score and store into (``uns[key_added or "tcri_gene_importance"]``).
    genes
        Names or positions, one importance each, in this order; ``None`` is every gene.
    covariate
        Restrict to the cells at one covariate level; ``None`` is every cell.
    groupby
        The replicate axis (default: the column registered as ``replicate``); one importance
        per (gene, group), and the unit of the ``splitby`` contrast.
    splitby
        A per-group label to contrast; the contrast is per gene, in ``stats``.
    n_samples
        ``0``: the plug-in at the posterior mean of ``p_ct``. ``N > 0``: ``N`` Dirichlet draws
        of ``p_ct`` from the guide, shared by every gene, entering only through the gate; the
        importance is computed per draw and summarised (mean, sd, HDI).
    use_gate
        ``True``: the model's own rule, as ``predict()``, so the importance is about the calls
        users see. ``False``: the head alone, invariant to the gate and the clone structure.
    batch_size, random_state
        Cells per encoder pass; the seed for the Dirichlet draws.
    null_model
        The permutation reference. ``"auto"`` is the phenotype null, rebuilt from the store and
        this object and run on the same genes, cells and rule; ``None`` reports the bare
        importance; a fit name selects another; a fitted ``TCRIModel`` is used directly and
        skips the rebuild. There is no ``fit=``: this is a query on a MODEL rather than on a
        stored substrate, so a different fit is reached by handing it a different model.

    Returns
    -------
    dict
        ``table`` (one row per gene, group, draw), ``result`` (per gene and group, with
        ``sd``/``hdi_*`` over draws), ``stats`` (per-gene contrast, or ``None``) and
        ``shift`` (per gene, phenotype and group: ``baseline``, ``perturbed``, ``shift``).
    """
    adata = model._validate_anndata(adata)
    module = model.module
    reg = model.adata_manager.registry
    gkey, resolved = resolve_groupby(adata, groupby)
    validate_splitby(adata.obs, gkey, splitby)
    positions, names = resolve_genes(adata, genes)

    if not use_gate and n_samples and int(n_samples) > 0:
        warnings.warn(
            "n_samples has no effect when use_gate=False: the head alone does not depend on "
            "p_ct, so there is no posterior to draw from. Computing the plug-in.",
            UserWarning, stacklevel=2,
        )
        n_samples = 0
    draws = prior_draws(module, n_samples=n_samples, random_state=random_state)
    n_draws = int(n_samples) if (n_samples and int(n_samples) > 0) else 0

    gid, labels, split = cell_groups(adata, cov_col=reg["covariate_col"], covariate=covariate,
                                     groupby=gkey, splitby=splitby)
    n_groups = len(labels)
    base, pert, counts = _accumulate(model, adata, positions=positions, gid=gid,
                                     n_groups=n_groups, draws=draws, use_gate=use_gate,
                                     batch_size=batch_size)

    present = (counts > 0).numpy()
    denom = counts.clamp(min=1.0).numpy()[None, :, None]
    baseline = base.numpy() / denom                       # [S', n_groups, P]
    perturbed = pert.numpy() / denom[None]                # [G, S', n_groups, P]
    importance = np.abs(baseline[None] - perturbed).sum(-1)   # [G, S', n_groups]

    phenotypes = model.adata.obs[reg["phenotype_col"]].astype("category").cat.categories.tolist()
    labels_kw = dict(names=names, labels=labels, present=present, covariate=covariate,
                     groupby=gkey, splitby=splitby, split=split)
    table = importance_table(importance, **labels_kw)
    result = build_result(table)
    # build_result sorts its keys; the caller's gene order is the contract ("one importance
    # each, in this order"), so restore it with a stable sort that keeps the group order
    rank = {g: i for i, g in enumerate(names)}
    order = np.argsort(result["gene"].map(rank).to_numpy(), kind="stable")
    result = result.iloc[order].reset_index(drop=True)
    stats = stats_per_gene(result, groupby=gkey, splitby=splitby)
    shift = shift_table(baseline, perturbed, phenotypes=phenotypes, **labels_kw)

    payload = {"table": table, "result": result, "stats": stats, "shift": shift}
    gp = module.gate_prob
    effective = {"gate_prob": (float(gp) if gp is not None else None), "n_draws": n_draws}
    if resolved:
        effective["groupby"] = gkey
    return with_resolved_params(payload, **effective)
