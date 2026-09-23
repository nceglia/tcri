"""Result shapes for the ``tl`` metrics, as ``TypedDict``s.

``@tl_result(schema=...)`` checks required-key PRESENCE on dict results. The shapes in the
trailing comments are documentation of the payload layout; nothing parses them.

Every metric returns the same two payload keys:

``table``
    The substrate. One row per (covariate, group, item[, draw]). Never reduced. All label
    columns present, so a caller can group it any way they like.

``result``
    The reduced frame, built FROM ``table`` — one row per group (or per item when there is no
    groupby). This is what the box/swarm plots and the comparison consume. Per-row statistics
    live here as **columns**, following grafiti, and ``pl`` dispatches on column presence
    (``"p" in result.columns``) rather than re-reading params.

**Reference columns.** When the metric ran with a reference — ``null_model`` not ``None``,
which is the default — ``result`` and ``table`` each gain, beside every native value column
``v``, a ``null_v`` (the same functional on the permutation null, at the same arguments) and an
``excess`` (``value - null_value``; ``excess_from``/``excess_to`` for the delta endpoints). A
DENOMINATOR gets ``null_denom`` and no excess. At ``null_model=None`` none of them exist, on
either frame. The ``excess`` carries no ``sd`` and no interval: it is a difference of two
summaries, and draws are never paired across two fits.

**The ``stats`` slot.** Every metric except :class:`JointDistribution` returns it. Only three
of the ``TypedDict``s below declare it, and a declared key is a required one, so the rest accept
a payload with or without it. It carries one row per (contrast, ``quantity``), the quantity being
``"value"`` or ``"excess"``; without a reference it carries only ``"value"``. Nothing switches
automatically on the presence of a reference: the plot selects, so the star and the marks are
always the same quantity.
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
    "GeneImportance",
    "validate",
]


class JointDistribution(TypedDict):
    """The clone x phenotype table every other metric reduces."""

    table: pd.DataFrame     # index (covariate, clonotype[, draw]); columns = phenotypes
    result: pd.DataFrame    # same, reduced over draws when n_samples > 0


class MutualInformation(TypedDict):
    """I(c; phi) — the one metric with no item axis, so ``table`` has ``item=None``."""

    table: pd.DataFrame     # cols: covariate, [groupby], [splitby], draw, value, denom
    result: pd.DataFrame    # one row per group; value + sd/hdi_* (draws) + ci_*/n_groups
                            # (across groups) + p/stat/stars when splitby is set;
                            # + denom, and null_value/null_denom/excess with a reference.
                            # `denom` is the normaliser this MI was divided by: the null does
                            # not share it, so both are stored and value*denom recovers bits


class ClonotypicEntropy(TypedDict):
    """H(c | phi) — one value per PHENOTYPE, not per clone."""

    table: pd.DataFrame     # cols: covariate, [groupby], [splitby], phenotype, draw, value
    result: pd.DataFrame    # + null_value/excess with a reference


class PhenotypicEntropy(TypedDict):
    """H(phi | c) — one value per clone."""

    table: pd.DataFrame     # cols: covariate, [groupby], [splitby], clonotype, draw, value
    result: pd.DataFrame    # + null_value/excess with a reference


class PhenotypicFlux(TypedDict):
    """Distance between a clone's phenotype distribution at two covariate levels.

    Clones absent from either side are dropped, not NaN-filled — a flux needs both endpoints.
    """

    table: pd.DataFrame     # cols: cov_from, cov_to, [groupby], [splitby], clonotype, draw, value
    result: pd.DataFrame    # + null_value/excess with a reference


class DeltaClonotypicEntropy(TypedDict):
    """H(c|phi) at ``cov_to`` minus at ``cov_from`` — one value per PHENOTYPE.

    The clone set is intersected across the two levels within each replicate, so ``log2(C)``
    is the same on both sides and cancels out of the difference. That is why no special
    ``n_clones_ref`` default is needed here.
    """

    table: pd.DataFrame     # cols: cov_from, cov_to, [groupby], [splitby], phenotype, draw,
                            #       value, value_from, value_to
    result: pd.DataFrame    # + null_value/null_value_from/null_value_to and
                            #   excess/excess_from/excess_to with a reference
    stats: object           # one row per (contrast, quantity)


class DeltaPhenotypicEntropy(TypedDict):
    """H(phi|c) at ``cov_to`` minus at ``cov_from`` — one value per CLONE.

    The only metric whose item axis is entity-matched: the same clonotype observed at both
    levels, a biological barcode rather than a category measured twice.
    """

    table: pd.DataFrame     # cols: cov_from, cov_to, [groupby], [splitby], clonotype, draw,
                            #       value, value_from, value_to
    result: pd.DataFrame    # + null_value/null_value_from/null_value_to and
                            #   excess/excess_from/excess_to with a reference
    stats: object           # one row per (contrast, quantity)


class GeneImportance(TypedDict):
    """``perturb.gene_importance`` — how far silencing a gene moves the mean phenotype call.

    The item is a gene. ``table``/``result``/``stats`` have the metric shape so the shared
    reducers apply unchanged; the signed per-phenotype decomposition of each importance lives
    in ``shift``, averaged over draws, because putting the phenotype axis inside ``table``
    would make ``result`` per phenotype rather than per gene.
    """

    table: pd.DataFrame     # cols: gene, [covariate], [groupby], [splitby], draw, value
    result: pd.DataFrame    # one row per (gene[, covariate][, group]); value + sd/hdi_*,
                            #   + null_value/excess with a reference
    stats: object           # per-gene contrast over groups when splitby is set, else None;
                            #   one row per (gene, contrast, quantity)
    shift: pd.DataFrame     # cols: gene, phenotype, [covariate], [groupby], [splitby],
                            #       baseline, perturbed, shift


def validate(schema, result, *, name: str = "result") -> None:
    """Public twin of the decorator's internal check, for direct use in tests."""
    required = set(getattr(schema, "__required_keys__", None) or schema.__annotations__)
    missing = required - set(result)
    if missing:
        raise ValueError(f"{name}: missing required keys {sorted(missing)}")
