"""Preprocessing helpers: rare-clone pooling, clone sizes and the MuData adapter.

Deliberately light on imports: this module is loaded by ``import tcri``, so anything
heavy imported here is paid for on every import of the package.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .._compute._repertoire import TCRIDataWarning, _missing_clonotypes, _pool_rare_clones
from .._state import keys as K
from .._state._resolution import clone_like_candidates

__all__ = ["pool_rare_clones", "clone_size", "from_mudata"]


def pool_rare_clones(adata, *, clonotype_key: str, groupby: str, min_cells: int = 3,
                     samples: str | None = None, key_added: str | None = None) -> None:
    """Pool each individual's rare clones into one label per individual.

    A clone is a set of cells within one individual, so cells are counted within each
    ``groupby`` group, and a clonotype carried by two individuals is two clones. A clone with
    fewer than ``min_cells`` cells, all from one sample, is rare, and each group's rare clones
    are pooled into one ``pooled@{group}`` label. Every other clone is kept, as
    ``{clonotype}@{group}``; an id that already ends in ``@{group}`` is not suffixed again.

    **Why the sample matters.** Doublets and ambient TCR can attach stray cells to the wrong
    clone within one capture, so a clone seen in only one sample needs several cells before it
    is trusted. They cannot carry a clone into another capture, so a clone seen in two samples is
    real however small, and it is kept whatever ``min_cells`` is.

    **Choosing min_cells.** Raising it keeps fewer clones, and each kept clone's estimates rest
    on more cells. A clone seen through very few cells in a condition looks purer there than it
    is, which pushes mutual information up most when phenotype is only weakly tied to clone.
    Raise it when precision matters more than keeping clones.

    **When to pass samples.** Pass it when individuals were captured in more than one sample.
    Without it, each group counts as one sample, and the rule is ``min_cells`` alone.

    Run it on the source clonotype column before ``TCRIModel.setup_anndata``, and register the
    column it writes. It writes ``obs[key_added]``, a categorical with sorted categories, and
    records the step in ``uns['tcri_clonotype_derivations']``; a second call with the same
    ``key_added`` replaces both. One line per group is logged on the ``tcri`` logger: clones
    kept, clones pooled, and cells pooled with their share of the group.

    Parameters
    ----------
    adata
        The object to pool, with the columns in ``obs``.
    clonotype_key
        The source clonotype column. It is never modified.
    groupby
        The column naming each cell's individual, usually the column registered as
        ``replicate`` at setup.
    min_cells
        A clone with fewer cells than this, all from one sample, is pooled. A positive
        integer.
    samples
        The column naming each cell's capture, its 10x sample or library, which need not be the
        condition. ``None`` counts each group as one sample.
    key_added
        The column to write; ``f"{clonotype_key}_pooled"`` when ``None``. It must differ from
        ``clonotype_key``.

    Warns
    -----
    UserWarning
        When a group keeps no clone, naming the group: every cell of that group is pooled.

    Raises
    ------
    KeyError
        When ``clonotype_key``, ``groupby`` or ``samples`` is not a column of ``obs``.
    TypeError
        When ``min_cells`` is not an integer.
    ValueError
        When ``min_cells`` is not positive; when ``key_added`` is ``clonotype_key``; when a cell
        has no clonotype (NaN or None, an empty or whitespace-only string, or the string
        ``"nan"``), no group or no sample; and when two kept clones of one group would get the
        same id. Nothing is written when it raises.
    """
    _pool_rare_clones(adata, clonotype_key=clonotype_key, groupby=groupby, min_cells=min_cells,
                      samples=samples, key_added=key_added)


def clone_size(adata, *, key_added=K.CLONE_SIZE, return_counts=False):
    # Canonical source is uns[METADATA]['clone_col'], written by to_anndata.
    meta = adata.uns.get(K.METADATA)
    if not meta or K.CLONE_COL not in meta:
        raise KeyError(
            f"adata.uns[{K.METADATA!r}][{K.CLONE_COL!r}] is missing — run "
            "model.to_anndata(adata) first (or load a session) so the clonotype "
            "column is registered."
        )
    tcr_key = meta[K.CLONE_COL]
    res = np.unique(adata.obs[tcr_key].tolist(), return_counts=True)
    clone_sizes = dict(zip(res[0],res[1]))
    sizes = []
    for clone in adata.obs[tcr_key]:
        sizes.append(clone_sizes[clone])
    adata.obs[key_added] = sizes
    if return_counts:
        return clone_sizes


def _read_column(mdata, key: str, *, gex_mod: str, airr_mod: str, argument: str,
                 hint: bool = False) -> tuple[pd.Series, str]:
    """The first column named ``key``, and where it was found.

    The places are tried in order: the GEX modality's ``obs``, the AIRR modality's ``obs``, then
    ``mdata.obs`` under ``key``, ``<gex_mod>:key`` and ``<airr_mod>:key``. The label names the
    place, as ``mdata.mod['airr'].obs['clone_id']`` or ``mdata.obs['airr:clone_id']``. With
    ``hint``, the KeyError lists the clone-like columns it saw.
    """
    gex_obs, airr_obs = mdata.mod[gex_mod].obs, mdata.mod[airr_mod].obs
    places = [
        (gex_obs, key, f"mdata.mod[{gex_mod!r}].obs[{key!r}]"),
        (airr_obs, key, f"mdata.mod[{airr_mod!r}].obs[{key!r}]"),
    ] + [
        (mdata.obs, name, f"mdata.obs[{name!r}]")
        for name in (key, f"{gex_mod}:{key}", f"{airr_mod}:{key}")
    ]
    for frame, column, label in places:
        if column in frame.columns:
            return frame[column], label
    message = (
        f"{argument}={key!r} was not found in mdata.mod[{gex_mod!r}].obs, "
        f"mdata.mod[{airr_mod!r}].obs, or mdata.obs as {key!r}, {gex_mod + ':' + key!r} or "
        f"{airr_mod + ':' + key!r}."
    )
    if hint:
        seen = sorted({c for frame in (gex_obs, airr_obs, mdata.obs)
                       for c in clone_like_candidates(frame, source_prefix=airr_mod)})
        message += (f" Clone-like columns include: {', '.join(seen[:8])}." if seen
                    else " No clone-like columns were found.")
    raise KeyError(message)


def _holds_counts(matrix) -> bool:
    """Whether every stored value of ``matrix``, dense or sparse, is a non-negative integer."""
    from scipy import sparse

    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if np.issubdtype(values.dtype, np.integer):
        return bool((values >= 0).all())
    return bool(np.isfinite(values).all() and (values >= 0).all()
                and (values == np.round(values)).all())


def from_mudata(
    mdata,
    *,
    clonotype_key: str,
    covariate_key: str | None = None,
    gex_mod: str = "gex",
    airr_mod: str = "airr",
    counts_layer: str = "counts",
    drop_missing_clonotype: bool = True,
):
    """Return a tcri-ready AnnData from a MuData object with a GEX and an AIRR modality.

    The output holds the cells of the GEX modality that are also in the AIRR modality, with the
    GEX modality's ``X``, layers and ``obs``; GEX cells without AIRR data are dropped with a
    warning. The clonotype column and, when given, the covariate column are each read from the
    first place that has them: the GEX modality's ``obs``, the AIRR modality's ``obs``, then
    ``mdata.obs`` under the bare name or with a ``<gex_mod>:`` or ``<airr_mod>:`` prefix. Each is
    written to ``obs`` under the bare name, as a categorical with string categories. ``mdata`` is
    not modified.

    A cell has no clonotype when its value is NaN or None, an empty or whitespace-only string, or
    the string ``"nan"``; these are the cells ``TCRIModel.setup_anndata`` refuses.

    Parameters
    ----------
    mdata
        A MuData with a GEX and an AIRR modality, as Scirpy writes it.
    clonotype_key
        The clonotype column, by its bare name, such as ``"clone_id"`` or ``"cc_aa_tcrdist"``.
    covariate_key
        The covariate column, by its bare name; ``None`` reads no covariate. Build a composite
        covariate as a single column first.
    gex_mod
        The gene-expression modality.
    airr_mod
        The AIRR modality.
    counts_layer
        The layer of the GEX modality that holds the raw counts, as non-negative integers.
    drop_missing_clonotype
        Drop the cells without a clonotype. With ``False`` they raise.

    Returns
    -------
    anndata.AnnData
        A new object. ``uns['tcri_adapter']['resolved']`` records the modalities, each column's
        key and where it was read from (``clonotype_source``, ``covariate_source``), the counts
        layer, and the cell counts: ``n_obs_gex`` and ``n_obs_airr`` in the two modalities,
        ``n_obs_shared`` in both, and ``n_obs_output`` returned.

    Warns
    -----
    UserWarning
        When GEX cells have no AIRR data, with the number dropped.

    Raises
    ------
    KeyError
        When a modality is missing, ``clonotype_key`` or ``covariate_key`` is in none of the
        places above, or ``counts_layer`` is not a layer of the GEX modality.
    ValueError
        When the modalities share no cells, cells have no clonotype and
        ``drop_missing_clonotype=False``, cells have no covariate value, or ``counts_layer``
        holds negative or non-integer values. The message gives the number of cells.
    TypeError
        When ``clonotype_key``, or ``covariate_key`` when given, is not a non-empty string.
    """
    if not isinstance(clonotype_key, str) or not clonotype_key:
        raise TypeError(
            f"clonotype_key must be one column name, a non-empty string; got {clonotype_key!r}."
        )
    if covariate_key is not None and (not isinstance(covariate_key, str) or not covariate_key):
        raise TypeError(
            f"covariate_key must be one column name, a non-empty string, or None; got "
            f"{covariate_key!r}. Build a combined covariate as a single column first."
        )
    for mod in (gex_mod, airr_mod):
        if mod not in mdata.mod:
            raise KeyError(f"mdata.mod has no {mod!r} modality")
    gex, airr = mdata.mod[gex_mod], mdata.mod[airr_mod]
    shared = gex.obs_names.intersection(airr.obs_names)
    if len(shared) == 0:
        raise ValueError(
            f"no shared obs_names between mdata.mod[{gex_mod!r}] and mdata.mod[{airr_mod!r}]"
        )
    mods = dict(gex_mod=gex_mod, airr_mod=airr_mod)
    values, clonotype_source = _read_column(mdata, clonotype_key, argument="clonotype_key",
                                            hint=True, **mods)
    clone = values.reindex(shared).astype(object)
    missing = _missing_clonotypes(clone).to_numpy()
    if missing.any() and not drop_missing_clonotype:
        raise ValueError(
            f"{int(missing.sum())} of {len(shared)} cells have no clonotype in "
            f"{clonotype_source} (NaN or None, an empty or whitespace-only string, or the string "
            "'nan'). Pass drop_missing_clonotype=True to drop them."
        )
    cells = shared[~missing]

    covariate_source = None
    if covariate_key is not None:
        values, covariate_source = _read_column(mdata, covariate_key, argument="covariate_key",
                                                **mods)
        covariate = values.reindex(cells)
        n_missing = int(covariate.isna().sum())
        if n_missing:
            raise ValueError(
                f"{n_missing} of {len(cells)} cells have no covariate value in "
                f"{covariate_source}. Drop those cells or fill the values in the MuData first."
            )

    copy_x = (f"mdata.mod[{gex_mod!r}].layers[{counts_layer!r}] = "
              f"mdata.mod[{gex_mod!r}].X.copy()")
    if counts_layer not in gex.layers:
        raise KeyError(
            f"mdata.mod[{gex_mod!r}] has no layer {counts_layer!r}. If X holds the raw counts, "
            f"copy it in first: {copy_x}"
        )

    adata = gex[cells].copy()
    if not _holds_counts(adata.layers[counts_layer]):
        raise ValueError(
            f"layer {counts_layer!r} of mdata.mod[{gex_mod!r}] does not hold raw counts: it has "
            f"negative or non-integer values. Pass the layer that holds them as counts_layer, or, "
            f"if X holds them, copy it in first: {copy_x}"
        )
    # Plain-string categories, not pandas' nullable "string": anndata 0.12 does not write those,
    # so the output would not save to h5ad.
    adata.obs[clonotype_key] = clone[cells].astype("string").astype(object).astype("category")
    if covariate_key is not None:
        adata.obs[covariate_key] = covariate.astype(str).astype("category")

    n_gex_only = gex.n_obs - len(shared)
    if n_gex_only:
        warnings.warn(
            f"{n_gex_only} of {gex.n_obs} cells of mdata.mod[{gex_mod!r}] have no AIRR data in "
            f"mdata.mod[{airr_mod!r}] and are dropped.",
            TCRIDataWarning, stacklevel=2,
        )

    adata.uns["tcri_adapter"] = {
        "source": "tcri.pp.from_mudata",
        "resolved": {
            "gex_mod": gex_mod,
            "airr_mod": airr_mod,
            "clonotype_key": clonotype_key,
            "clonotype_source": clonotype_source,
            "covariate_key": covariate_key,
            "covariate_source": covariate_source,
            "counts_layer": counts_layer,
            "n_obs_gex": int(gex.n_obs),
            "n_obs_airr": int(airr.n_obs),
            "n_obs_shared": int(len(shared)),
            "n_obs_output": int(adata.n_obs),
        },
    }
    return adata
