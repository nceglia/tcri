"""``pl.delta_*`` (§8.4) — cache renderers for the paired entropies.

Two views of one cached result, selected by ``kind``:

``"delta"`` (default)
    The change itself, with a zero rule. Mass either side of zero is the direction.

``"endpoints"``
    ``cov_from`` and ``cov_to`` side by side, connected per replicate. This view exists
    *here* rather than on ``pl.phenotypic_entropy`` for a reason worth stating: the endpoints
    live in the delta's payload, where they were computed over the intersected clone set. The
    same figure drawn from two separate single-covariate results would use a DIFFERENT clone
    set on each side, and the values differ substantially. Rendering it only from the delta
    result makes the unmatched version unreachable rather than merely discouraged.
"""
from __future__ import annotations

from ._base import render_delta

__all__ = ["delta_clonotypic_entropy", "delta_phenotypic_entropy"]


def delta_clonotypic_entropy(adata, *, kind="delta", key=None, order=None, hue_order=None,
                             palette=None, ax=None, figsize=(8, 4), save=None, show=None,
                             return_df=False):
    """ΔH[P(c|φ)] per phenotype, from the cached ``tl.delta_clonotypic_entropy``.

    No connecting lines in the ``"endpoints"`` view, and no matched-count sizing: the item is a
    phenotype, a category measured twice rather than an entity that persisted. A line would
    assert persistence, and the matched CLONE count is not in this result at all — those clones
    were summed over inside H(c|phi), so sizing by item rows would size by phenotype.
    """
    return render_delta(adata, "delta_clonotypic_entropy",
                        ylabel="Δ clonotypic entropy (bits)", item_col="phenotype",
                        item_as_x=True, entity_matched=False, kind=kind, key=key,
                        order=order,
                        hue_order=hue_order, palette=palette, ax=ax, figsize=figsize,
                        save=save, show=show, return_df=return_df)


def delta_phenotypic_entropy(adata, *, kind="delta", key=None, order=None, hue_order=None,
                             palette=None, ax=None, figsize=(8, 4), save=None, show=None,
                             return_df=False):
    """ΔH[P(φ|c)] per clone, from the cached ``tl.delta_phenotypic_entropy``.

    The ``"endpoints"`` view connects each replicate across the two levels — the same clone
    set on both sides, so the line is a matched-identity claim the data supports. Dot area is
    the number of clones matched for that replicate, which varies per replicate and is the n
    the value rests on.
    """
    return render_delta(adata, "delta_phenotypic_entropy",
                        ylabel="Δ phenotypic entropy (bits)", item_col="clonotype",
                        entity_matched=True, kind=kind, key=key, order=order,
                        hue_order=hue_order,
                        palette=palette, ax=ax, figsize=figsize, save=save, show=show,
                        return_df=return_df)
