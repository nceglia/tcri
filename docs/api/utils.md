# Utilities (`tcri.ut`)

Session persistence and AUROC statistics, exposed as ``tcri.ut``.

## Session persistence

Save and restore a fitted model together with its `AnnData`.

```{eval-rst}
.. autofunction:: tcri.utils._utils.save_tcri_session
.. autofunction:: tcri.utils._utils.load_tcri_session
```

## AUROC statistics

Compare a selected positive label against all other labels. Empty labels, a single
observed class, or an absent `pos_label` return undefined results without warnings:
`(nan, nan, array([]), "degenerate")` for permutation testing and `array([nan, nan])`
for the bootstrap confidence interval.

```{eval-rst}
.. autofunction:: tcri._stats._core.auc_and_label_permutation
.. autofunction:: tcri._stats._core.bootstrap_auc
```

```{note}
The four functions above are part of the public API contract. Other names reachable
under `tcri.ut` are internal helpers and may change without notice.
```
