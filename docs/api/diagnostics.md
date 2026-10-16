# Diagnostics (`tcri.diag`)

Model-checking and metric-validation utilities: a posterior-predictive check of the clone x
phenotype fit, phenotype-probability calibration, reconstruction PPCs, permutation nulls for
the metrics, and training-curve inspection. Exposed as ``tcri.diag``. Each check returns what it
computed and, with ``plot=True`` (the default), draws its figure from those values; ``ax=`` and
``save=`` place and write it.

## Posterior-predictive & null checks

```{eval-rst}
.. automodule:: tcri.diagnostics._ppc
   :members:
```

## Training diagnostics

```{eval-rst}
.. automodule:: tcri.diagnostics._training
   :members:
```
