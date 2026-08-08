# Model (`tcri.ml`)

The deep-learning model that jointly embeds gene expression and clonotype
information and learns the clone–phenotype distribution. Exposed as
``tcri.ml.TCRIModel``.

The generative mathematics are frozen by the [model contract](../contracts/index.md) and
described conceptually in [The model](../concepts/model.md).

```{eval-rst}
.. autoclass:: tcri.model._model.TCRIModel
   :members: setup_anndata, train, get_latent_representation, predict, get_p_ct, to_anndata
   :member-order: bysource
```

```{note}
Model persistence is handled by the session helpers in the
[Utilities](utils.md) API — ``save_tcri_session`` and ``load_tcri_session`` —
not by methods on the model object.
```
