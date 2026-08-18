"""Canonical categorical colours for ``tcri.pl`` — one home for the palette.

Every categorical plot routes through :func:`resolve_colors`, so a label keeps the same
colour across views and a user-set ``uns["<key>_colors"]`` propagates everywhere. Colours
persist under scanpy's ``uns["<obs_key>_colors"]`` convention (via ``K.colors``), which is
also what ``sc.pl.umap`` reads — so setting a response palette here colours the UMAP too.

Continuous colormaps stay per-plot ``cmap=`` arguments; this module is categorical only.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from .._state import keys as K

__all__ = ["tcri_colors", "NA_COLOR", "resolve_colors"]

#: The categorical cycle. There were two of these -- this list and a 30-entry one in
#: ``utils/_utils.py`` that led with ``#272822`` (the Monokai *background*, so the first
#: category rendered as a near-black block). They disagreed on both contents and order, and
#: nothing imported the utils copy. Merged here: this order, plus the colours only the utils
#: list had.
tcri_colors = [
    "#AE81FF", "#FD971F", "#66D9EF", "#A6E22E", "#F92672", "#E6DB74", "#75715E",
    "#D65F0E", "#004d47", "#D291BC", "#3A506B", "#5D8A5E", "#A6A1E2", "#E97451",
    "#6C8D67", "#832232", "#1E1E1E", "#F92659", "#272822", "#8B4513",
    "#669999", "#C08497", "#587B7F", "#9A8C98", "#F28E7F", "#F3B61F", "#6A6E75",
    "#FFD8B1", "#88AB75", "#C38D94", "#6D6A75",
]

#: Absent / non-significant categories.
NA_COLOR = "lightgray"


def resolve_colors(adata, cat_key, categories=None, *, palette=None, persist=True):
    """Resolve a ``{category: hex}`` map for a categorical, cached under
    ``uns[K.colors(cat_key)]``.

    Priority: an explicit ``palette`` (a ``dict`` ``{cat: colour}``, a ``list`` cycled, or a
    matplotlib colormap name) -> an existing ``uns["<cat_key>_colors"]`` **whose length
    matches** -> :data:`tcri_colors`, cycled. A partial ``dict`` fills its gaps from the
    canonical cycle rather than erroring, so ``palette={"R": "red"}`` is a legal way to pin
    one level and leave the rest alone.

    ``categories`` defaults to the categories of ``adata.obs[cat_key]``. Pass it explicitly
    when colouring something that is not an obs column -- a phenotype axis read off a metric
    result, say.

    This replaces ``resolve_palette``, which took a LIST of columns, always overwrote
    ``uns``, and had no way to read an existing assignment back. That last part is the point:
    a plot that cannot see the colours already stored assigns its own, so the same patient
    changed colour between two figures in the same notebook.
    """
    if categories is None:
        categories = adata.obs[cat_key].astype("category").cat.categories
    requested = list(dict.fromkeys(categories))

    # The colour is a property of the LEVEL, never of its position in this particular figure.
    # Assignment therefore runs over a canonical order -- scanpy's convention, the obs
    # categorical's categories, which is also what `uns[<key>_colors]` is aligned to -- and the
    # requested order only selects from the result.
    #
    # Zipping against the caller's order instead made the colour follow position: the renderer
    # sorts x by median, so a `response` panel where NR sorted first drew NR purple, and the
    # panel beside it where R sorted first drew R purple. Same variable, same figure, swapped.
    # Without an obs column to appeal to, sorting is enough to make it order-independent.
    if cat_key in adata.obs:
        canon = list(adata.obs[cat_key].astype("category").cat.categories)
        canon += [c for c in requested if c not in canon]
    else:
        canon = sorted(requested, key=str)
    n = len(canon)

    if isinstance(palette, dict):
        raw = [palette.get(c, tcri_colors[i % len(tcri_colors)]) for i, c in enumerate(canon)]
    elif isinstance(palette, (list, tuple)):
        raw = [palette[i % len(palette)] for i in range(n)]
    elif isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        raw = [cmap(i % cmap.N) for i in range(n)]
    else:
        existing = adata.uns.get(K.colors(cat_key))
        # a length mismatch means the categories changed under the stored list; reassigning
        # is right, silently zipping a short list against long categories is not
        if existing is not None and len(existing) == n:
            raw = list(existing)
        else:
            raw = [tcri_colors[i % len(tcri_colors)] for i in range(n)]

    hexes = [to_hex(c) for c in raw]
    if persist:
        adata.uns[K.colors(cat_key)] = hexes
    mapping = dict(zip(canon, hexes))
    return {c: mapping[c] for c in requested}
