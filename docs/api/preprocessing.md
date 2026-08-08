# Preprocessing (`tcri.pp`)

Lightweight clone bookkeeping applied to an `AnnData` before modelling: derive
clone sizes and collapse singleton clonotypes. Exposed as ``tcri.pp``.

```{note}
Model registration and the joint distribution are **not** preprocessing steps.
Registration is `tcri.ml.TCRIModel.setup_anndata`, and the joint distribution is
computed by `tcri.tl.joint_distribution` after `model.to_anndata`.
```

```{eval-rst}
.. automodule:: tcri.preprocessing._preprocessing
   :members:
```
