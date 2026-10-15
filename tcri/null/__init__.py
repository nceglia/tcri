"""``tcri.null`` — permutation references, fitted as ordinary fits of the same model.

Three axes, one function each, plus :func:`all` for the common path::

    model.train(max_epochs=200, batch_size=512)
    model.to_anndata(adata)
    nulls = tcri.null.all(model, adata)

Each call refits the parent's model, unchanged in every knob and in its seed, on ``n_nulls``
permuted label vectors, and writes each fit into ``adata`` as a named fit beside the main one
(``null.phenotype[0]``, ``null.phenotype[1]``, ...). From there every metric can read them as one
reference: ``value - null_value`` is the part of a number the structure accounts for, paired
draw for draw with every fit of the reference, and no model-based number need be reported bare.
"""
from ._nulls import all, clonotype, covariate, phenotype

#: `rebuild` is deliberately NOT re-exported: it reconstructs a null for inference and is the
#: perturbation's private route to `null_model="auto"`, not something a user calls. Import it as
#: `from tcri.null._rebuild import rebuild`.
__all__ = ["phenotype", "clonotype", "covariate", "all"]
