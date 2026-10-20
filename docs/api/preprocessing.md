# Preprocessing (`tcri.pp`)

Lightweight clone bookkeeping applied to an `AnnData` before modeling: pool
rare clones, leave the TCR V(D)J segment genes out of the fitted genes
(`exclude_tcr_genes`), derive clone sizes, summarize cells and clones per group, which
clones persist across covariate levels and how clonotypes are shared across
groups, and adapt Scirpy/MuData objects into a
tcri-ready `AnnData`. Exposed as ``tcri.pp``. `from_mudata` reads one clonotype
column and at most one covariate column; build composite covariates upstream. It needs the
`mudata` extra: `pip install "tcri[mudata]"`.

```{note}
Model registration and the joint distribution are **not** preprocessing steps.
Registration is `tcri.ml.TCRIModel.setup_anndata`, and the joint distribution is
computed by `tcri.tl.joint_distribution` after `model.to_anndata`.
```

```{eval-rst}
.. automodule:: tcri.preprocessing._preprocessing
   :members:
```

## Repertoire summaries

Each summary stores its result in `adata.uns`, read back with `tcri.get.result`, and the
`tcri.pl` function of the same name draws it; `inplace=False` returns the result instead.

```{eval-rst}
.. automodule:: tcri.preprocessing._repertoire
   :members:
```
