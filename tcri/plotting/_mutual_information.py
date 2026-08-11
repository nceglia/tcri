"""``pl.mutual_information`` (§8.2) — cache renderer for I(c;phi)."""
from __future__ import annotations

from ._base import render_metric

__all__ = ["mutual_information"]


def mutual_information(adata, *, key=None, order=None, hue_order=None, palette=None,
                       ax=None, figsize=(8, 4), save=None, show=None, return_df=False):
    """Clone<->phenotype MI (bits, normalized) from the cached ``tl.mutual_information``.

    Run ``tl.mutual_information`` first; the axes here are whatever that call used. With
    ``groupby`` it boxes one MI per group, with ``splitby`` it boxes by split and brackets
    the contrast from ``stats``.
    """
    return render_metric(adata, "mutual_information", ylabel="mutual information (bits)",
                         key=key, order=order, hue_order=hue_order, palette=palette, ax=ax,
                         figsize=figsize, save=save, show=show, return_df=return_df)
