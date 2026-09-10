"""``perturb.knockout`` -- per-cell phenotype probabilities with genes silenced.

The primitive of the module and the counterpart of :meth:`TCRIModel.predict`: the same
per-cell rule, the same frame, with the named columns of the expression matrix set to zero
before the encoder. ``knockout(genes=[])`` is ``predict()``. Several genes are silenced
together, so a gene program is one call.

It stores nothing by default. Per-cell probabilities are not a group table; they go to
``obsm`` (``key_added=``) beside the baseline ``X_tcri_probabilities`` when the caller wants
them on the object, or straight into a metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ._kernel import (batches, combine, head_logits, log_prior_for, prior_draws,
                      resolve_genes)

__all__ = ["knockout"]


def knockout(model, adata=None, *, genes, use_gate=True, batch_size=4096, key_added=None):
    """Per-cell phenotype probabilities with ``genes`` silenced.

    Parameters
    ----------
    model
        A fitted :class:`~tcri.model.TCRIModel`. Parameters are held fixed.
    adata
        The object to score; ``None`` is the model's own. Must carry the registered fields.
    genes
        Gene names or positions; all of them are zeroed together. ``[]`` silences nothing
        and reproduces ``predict()``.
    use_gate
        ``True``: the model's own rule for combining the head with the clone x covariate
        prior, exactly as ``predict()``. ``False``: the head alone, ``softmax(f_cls(μ_i))``.
    batch_size
        Cells per encoder pass.
    key_added
        When given, also written to ``adata.obsm[key_added]`` as float32.

    Returns
    -------
    pandas.DataFrame
        Index ``adata.obs_names``, columns the phenotype categories, rows summing to one.
    """
    adata = model._validate_anndata(adata)
    module = model.module
    module.eval()
    positions, _names = resolve_genes(adata, genes)
    draws = prior_draws(module, n_samples=0, random_state=None)
    gate = module.gate_prob

    out = []
    with torch.no_grad():
        for x, b, idx, _n in batches(model, adata, batch_size=batch_size):
            log_prior = log_prior_for(draws, module, idx) if use_gate else None
            logits = head_logits(module, x, b, zero_cols=positions)
            out.append(combine(logits, log_prior, gate, use_gate)[0].cpu())
    probs = torch.cat(out, dim=0).numpy()

    phenotype_col = model.adata_manager.registry["phenotype_col"]
    cats = model.adata.obs[phenotype_col].astype("category").cat.categories.tolist()
    frame = pd.DataFrame(probs, index=adata.obs_names, columns=cats)
    if key_added is not None:
        adata.obsm[key_added] = np.asarray(frame.values, dtype="float32")
    return frame
