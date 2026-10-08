# Preprocessing (`tcri.pp`)

Lightweight clone bookkeeping applied to an `AnnData` before modeling: pool
rare clones, derive clone sizes, summarize cells and clones per group and which
clones persist across covariate levels, and adapt Scirpy/MuData objects into a
tcri-ready `AnnData`. Exposed as ``tcri.pp``. For MuData adaptation, covariates
are passed as a single source column; build composite covariates upstream.

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

```{eval-rst}
.. automodule:: tcri.preprocessing._repertoire
   :members:
```
