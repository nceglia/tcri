# Diagnostics (`tcri.diag`)

Model-checking and metric-validation utilities: a posterior-predictive check of the clone x
phenotype fit, phenotype-probability calibration, reconstruction PPCs, permutation nulls for
the metrics, and training-curve inspection. Exposed as ``tcri.diag``. Each check draws its
figure (``plot=True``; ``ax=`` and ``save=`` place and write it), prints a table of what it
computed (``print=True``), and returns it (``return_dict=True``), all by default; turning all
three off raises. ``loss`` and ``archetypes`` only draw.

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
