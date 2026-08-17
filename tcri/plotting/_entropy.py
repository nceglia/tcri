"""``pl.clonotypic_entropy`` / ``pl.phenotypic_entropy`` (§8.1) — cache renderers.

The two differ in one thing, and it is a property of the metric rather than a style choice:
H(c|phi) has one value per PHENOTYPE, a handful of named categories that belong on the x
axis; H(phi|c) has one value per CLONE, which is the whole repertoire and belongs in the
distribution.
"""
from __future__ import annotations

from ._base import render_metric

__all__ = ["clonotypic_entropy", "phenotypic_entropy"]


def clonotypic_entropy(adata, *, key=None, order=None, hue_order=None, palette=None,
                       ax=None, figsize=(8, 4), save=None, show=None, return_df=False):
    """Per-phenotype clonotypic entropy (bits) from the cached ``tl.clonotypic_entropy``."""
    return render_metric(adata, "clonotypic_entropy", ylabel="clonotypic entropy (bits)",
                         item_col="phenotype", item_as_x=True, key=key, order=order,
                         hue_order=hue_order, palette=palette, ax=ax, figsize=figsize,
                         save=save, show=show, return_df=return_df)


def phenotypic_entropy(adata, *, key=None, order=None, hue_order=None, palette=None,
                       ax=None, figsize=(8, 4), save=None, show=None, return_df=False):
    """Per-clone phenotypic entropy — plasticity — from the cached ``tl.phenotypic_entropy``."""
    return render_metric(adata, "phenotypic_entropy", ylabel="phenotypic entropy (bits)",
                         item_col="clonotype", key=key, order=order, hue_order=hue_order,
                         palette=palette, ax=ax, figsize=figsize, save=save, show=show,
                         return_df=return_df)
