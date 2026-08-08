# Metrics (`tcri.tl`)

Information-theoretic metrics over the clone × phenotype joint distribution: the joint
engine, the two entropies, mutual information, phenotypic flux, and group comparisons.
All entropies and MI are in **bits** (log base 2). Exposed as ``tcri.tl``.

The definitions are frozen by the [metrics contract](../contracts/index.md); the
conceptual reference is [Information-theoretic metrics](../concepts/metrics.md).

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

```{eval-rst}
.. automodule:: tcri.tools._compare
   :members:
```
