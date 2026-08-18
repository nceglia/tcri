"""Result shapes for the ``tl`` metrics, as ``TypedDict``s.

``@tl_result(schema=...)`` checks required-key PRESENCE on dict results. The shapes in the
trailing comments are documentation — nothing parses them — but they are the only place the
payload layout is written down, so keep them accurate.

Every metric returns the same two payload keys:

``table``
    The substrate. One row per (covariate, group, item[, draw]). Never reduced. All label
    columns present, so a caller can group it any way they like.

``result``
    The reduced frame, built FROM ``table`` — one row per group (or per item when there is no
    groupby). This is what the box/swarm plots and the comparison consume. Statistics live here
    as **columns**, following grafiti: there is no separate ``stats`` slot, and ``pl`` dispatches
    on column presence (``"p" in result.columns``) rather than re-reading params.
"""
from __future__ import annotations

from typing import TypedDict

import pandas as pd

__all__ = [
    "JointDistribution",
    "MutualInformation",
    "ClonotypicEntropy",
    "PhenotypicEntropy",
    "PhenotypicFlux",
    "DeltaClonotypicEntropy",
    "DeltaPhenotypicEntropy",
    "validate",
]


class JointDistribution(TypedDict):
    """The clone x phenotype table every other metric reduces."""

    table: pd.DataFrame     # index (covariate, clonotype[, draw]); columns = phenotypes
    result: pd.DataFrame    # same, reduced over draws when n_samples > 0


class MutualInformation(TypedDict):
    """I(c; phi) — the one metric with no item axis, so ``table`` has ``item=None``."""

    table: pd.DataFrame     # cols: covariate, [groupby], [splitby], draw, value
    result: pd.DataFrame    # one row per group; value + sd/hdi_* (draws) + ci_*/n_groups
                            # (across groups) + p/stat/stars when splitby is set


class ClonotypicEntropy(TypedDict):
    """H(c | phi) — one value per PHENOTYPE, not per clone."""

    table: pd.DataFrame     # cols: covariate, [groupby], [splitby], phenotype, draw, value
    result: pd.DataFrame


class PhenotypicEntropy(TypedDict):
    """H(phi | c) — one value per clone."""

    table: pd.DataFrame     # cols: covariate, [groupby], [splitby], clonotype, draw, value
    result: pd.DataFrame


class PhenotypicFlux(TypedDict):
    """Distance between a clone's phenotype distribution at two covariate levels.

    Clones absent from either side are dropped, not NaN-filled — a flux needs both endpoints.
    """

    table: pd.DataFrame     # cols: cov_from, cov_to, [groupby], [splitby], clonotype, draw, value
    result: pd.DataFrame


class DeltaClonotypicEntropy(TypedDict):
    """H(c|phi) at ``cov_to`` minus at ``cov_from`` — one value per PHENOTYPE.

    The clone set is intersected across the two levels within each replicate, so ``log2(C)``
    is the same on both sides and cancels out of the difference. That is why no special
    ``n_clones_ref`` default is needed here.
    """

    table: pd.DataFrame     # cols: cov_from, cov_to, [groupby], [splitby], phenotype, draw,
                            #       value, value_from, value_to
    result: pd.DataFrame
    stats: object


class DeltaPhenotypicEntropy(TypedDict):
    """H(phi|c) at ``cov_to`` minus at ``cov_from`` — one value per CLONE.

    The only metric whose item axis is entity-matched: the same clonotype observed at both
    levels, a biological barcode rather than a category measured twice.
    """

    table: pd.DataFrame     # cols: cov_from, cov_to, [groupby], [splitby], clonotype, draw,
                            #       value, value_from, value_to
    result: pd.DataFrame
    stats: object


def validate(schema, result, *, name: str = "result") -> None:
    """Public twin of the decorator's internal check, for direct use in tests."""
    required = set(getattr(schema, "__required_keys__", None) or schema.__annotations__)
    missing = required - set(result)
    if missing:
        raise ValueError(f"{name}: missing required keys {sorted(missing)}")
