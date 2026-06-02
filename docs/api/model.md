# Model

The deep-learning model that jointly embeds gene expression and clonotype
information and learns the clone–phenotype distribution.

```{eval-rst}
.. autoclass:: tcri.model._model.TCRIModel
```

```{note}
Model persistence is handled by the session helpers in the
[Utilities](utils.md) API — ``save_tcri_session`` and ``load_tcri_session`` —
not by methods on the model object.
```
