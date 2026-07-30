"""``pl`` colors — the shared categorical palette and `resolve_palette` (§8.6)."""
from __future__ import annotations

__all__ = ["tcri_colors", "resolve_palette"]

tcri_colors = [
    "#AE81FF", "#FD971F", "#66D9EF", "#A6E22E", "#F92672", "#E6DB74", "#75715E",
    "#D65F0E", "#004d47", "#D291BC", "#3A506B", "#5D8A5E", "#A6A1E2", "#E97451",
    "#6C8D67", "#832232", "#1E1E1E", "#F92659", "#272822", "#8B4513",
]


def resolve_palette(adata, columns, *, palette=None):
    """Assign `tcri_colors` to each `obs` column's categories, store in
    ``uns["<col>_colors"]``, and return ``{col: {category: color}}``.

    Mutates ``adata`` **in place** (fixes the old ``set_color_palette`` bug of writing
    onto a throwaway copy). ``palette`` overrides the default color cycle.
    """
    cols = [columns] if isinstance(columns, str) else list(columns)
    cycle = list(palette) if palette is not None else tcri_colors
    out = {}
    for col in cols:
        cats = adata.obs[col].astype("category").cat.categories.tolist()
        mapping = {c: cycle[i % len(cycle)] for i, c in enumerate(cats)}
        adata.uns[f"{col}_colors"] = [mapping[c] for c in cats]
        out[col] = mapping
    return out
