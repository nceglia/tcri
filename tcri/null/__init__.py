"""``tcri.null`` — permutation references, fitted as ordinary fits of the same model.

Three axes, one function each, plus :func:`all` for the common path::

    model.train(max_epochs=200, batch_size=512)
    model.to_anndata(adata)
    nulls = tcri.null.all(model, adata)

Each call refits the parent's model, unchanged in every knob and in its seed, on one permuted
label vector, and writes the result into ``adata`` as a named fit beside the main one. From
there every metric can read it: ``value - null_value`` is the part of a number the structure
accounts for, and no model-based number need be reported bare.
"""
from ._nulls import all, clonotype, condition, phenotype

#: `rebuild` is deliberately NOT re-exported: it reconstructs a null for inference and is the
#: perturbation's private route to `null_model="auto"`, not something a user calls. Import it as
#: `from tcri.null._rebuild import rebuild`.
__all__ = ["phenotype", "clonotype", "condition", "all"]
