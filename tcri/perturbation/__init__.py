"""``tcri.perturb`` -- in-silico perturbation of a fitted model.

A query on the fitted networks with their parameters held fixed: intervene on the input and
read the phenotype call back through the same rule ``predict()`` uses. Two doors on one
kernel, mirroring the split the package already has between ``predict()`` (per cell) and
``tl.*`` (per group, with replicates, statistics and a cache):

``knockout``
    Per-cell probabilities with a gene, or a gene program, silenced. The primitive.
``gene_importance``
    One gene at a time, reduced to a per-group importance with a contrast and a cache, read
    back by ``tcri.get.result(adata, "gene_importance")`` and drawn by ``pl.gene_importance``.

The counterfactual question ("what would this cell be called without gene j?") is answered
by a perturbation of the fitted model, not by a causal model of the cell; the functions are
named for the mechanism. Attribution by gradients and sampling from the generative model are
different mechanisms and are not here.
"""
from ._knockout import knockout
from ._importance import gene_importance

__all__ = ["knockout", "gene_importance"]
