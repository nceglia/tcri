# Metrics (`tcri.tl`)

Information-theoretic metrics over the clone × phenotype joint distribution: the joint
engine, the two entropies, mutual information, phenotypic flux, and group comparisons.
All entropies and MI are in **bits** (log base 2). Exposed as ``tcri.tl``.

The definitions are frozen by the [metrics contract](../contracts/index.md); the
conceptual reference is [Information-theoretic metrics](../concepts/index.md).

## Joint distribution

```{eval-rst}
.. automodule:: tcri.tools._joint
   :members:
```

## Entropies

```{eval-rst}
.. automodule:: tcri.tools._entropy
   :members:
```

## Mutual information

```{eval-rst}
.. automodule:: tcri.tools._mutual_information
   :members:
```

## Phenotypic flux

```{eval-rst}
.. automodule:: tcri.tools._flux
   :members:
```

## Group comparison

`compare_groups` is **not public API** — it is not on `tcri.tl`, and
`tests/test_removal_ledger.py` pins that it stays off. It is reached only through a metric's
`splitby` argument, never as a step you perform. It is documented here because the contrast it
computes is part of what a metric with `splitby` *means*, not because you should call it.

```{eval-rst}
.. automodule:: tcri._stats._compare
   :members:
```
