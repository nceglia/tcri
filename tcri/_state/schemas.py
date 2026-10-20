"""Result shapes for the ``tl`` metrics, as ``TypedDict``s.

``@tl_result(schema=...)`` checks required-key PRESENCE on dict results. The shapes in the
trailing comments are documentation of the payload layout; nothing parses them.

**By level.** Mutual information, the two entropies and flux store ``{replicate, item, stats}``:

``replicate``
    ``{draws, summary}``. ``draws`` has one row per (covariate, group, draw), ``summary`` the same
    rows summarized over the draws, one per (covariate, group), read with ``tcri.get.replicates``.
    A change labels its rows ``cov_from`` and ``cov_to`` in place of ``covariate``, and carries
    its endpoints ``value_from`` and ``value_to`` beside ``value``; flux has none.

``item``
    ``{draws, summary}`` for the metric's items, one row per (covariate, group, item[, draw]).
    Mutual information has one value per joint and no ``item``. On a change the items are the
    persistent clones (or, for clonotypic entropy, the phenotypes over them).

``stats``
    The tests on the replicate summary, or ``None`` when nothing is compared.

The other tools store the layout below, with two payload keys:

``table``
    The substrate. One row per (covariate, group, item[, draw]). Never reduced. All label
    columns present, so a caller can group it any way they like.

``result``
    The reduced frame, built FROM ``table`` — one row per group (or per item when there is no
    groupby). This is what the box/swarm plots consume. It carries the value columns and their
    spread — ``sd``, ``hdi_*`` over draws, ``ci_*`` across groups — as columns. The contrast
    between groups is not here: it is the ``stats`` frame below, and ``pl`` reads it from
    there.

**Reference columns.** When the metric ran with a reference — ``null_model`` not ``None``,
which is the default — ``result`` and ``table`` each gain, beside every native value column
``v``, a ``null_v`` (the same functional on the permutation null, at the same arguments) and an
``adjusted`` (``value - null_value``; ``adjusted_from``/``adjusted_to`` for a change's
endpoints). The adjusted value is the one quantity a scored metric reports; ``value`` and
``null_value`` are its inputs and stay stored beside it. A DENOMINATOR gets ``null_denom`` and
no adjusted column. ``table`` carries them per draw, the fit's draw ``d`` paired with the
null's draw ``d``, and ``result`` summarizes the paired draws: ``adjusted`` is their mean, with
``adjusted_sd``, ``adjusted_hdi_low`` and ``adjusted_hdi_high`` beside it, and ``null_v`` is the
null's mean. At ``null_model=None`` none of them exist, on either frame.

**The ``stats`` slot.** Every metric except :class:`JointDistribution` stores it and declares it.
It carries one row per contrast, ``comparison`` saying what it compares (``"split"``, two levels
of ``splitby``; ``"covariate"``, the paired test of a change), and a ``quantity`` column naming
the column contrasted: ``"adjusted"`` when the result carries a usable adjustment, ``"value"``
otherwise (no reference, or one that is non-finite everywhere). ``pl`` stars the quantity on its
axis, so a panel of the value carries no star when the contrast is on the adjusted value.
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
    "GeneImportance",
    "validate",
]


class JointDistribution(TypedDict):
    """The clone x phenotype table every other metric reduces."""

    table: pd.DataFrame     # index (covariate, clonotype[, draw]); columns = phenotypes
    result: pd.DataFrame    # same, reduced over draws when n_samples > 0


class MutualInformation(TypedDict):
    """I(c; phi) — one value per joint, so a replicate level and no item level."""

    replicate: dict         # {draws, summary}. draws: covariate, [groupby], [splitby], draw,
                            # value, denom (+ null, null_value, null_denom, adjusted with a
                            # reference). summary: one row per group; value + sd/hdi_* + denom
                            # (+ null_value/null_denom/adjusted/adjusted_sd/adjusted_hdi_*).
                            # `denom` is the normalizer this MI was divided by: the null does
                            # not share it, so both are stored and value*denom recovers bits.
                            # A change: cov_from, cov_to in place of covariate, and
                            # value_from, value_to, denom_from, denom_to in place of denom
    stats: object           # one row per contrast; `quantity` names the column contrasted


class ClonotypicEntropy(TypedDict):
    """H(c | phi) — per replicate H(C|P), and one value per PHENOTYPE."""

    replicate: dict         # {draws, summary}: covariate, [groupby], [splitby], [draw], value
                            # (+ the reference columns); a change: cov_from, cov_to,
                            # value_from, value_to
    item: dict              # {draws, summary}: the same, per phenotype
    stats: object           # one row per contrast; `quantity` names the column contrasted


class PhenotypicEntropy(TypedDict):
    """H(phi | c) — per replicate H(P|C), and one value per clone."""

    replicate: dict         # {draws, summary}: covariate, [groupby], [splitby], [draw], value
                            # (+ the reference columns); a change: cov_from, cov_to,
                            # value_from, value_to
    item: dict              # {draws, summary}: the same, per clone (`clonotype`)
    stats: object           # one row per contrast; `quantity` names the column contrasted


class PhenotypicFlux(TypedDict):
    """Distance between a clone's phenotype distribution at two covariate levels — per replicate,
    the persistent clones' distances weighted by their ``cov_from`` shares, and per persistent
    clone. A clone at one level only has no item row: a flux needs both endpoints.
    """

    replicate: dict         # {draws, summary}: cov_from, cov_to, [groupby], [splitby], [draw],
                            # value (+ the reference columns)
    item: dict              # {draws, summary}: the same, per persistent clone (`clonotype`)
    stats: object           # one row per contrast; `quantity` names the column contrasted


class GeneImportance(TypedDict):
    """``perturb.gene_importance`` — how far silencing a gene moves the mean phenotype call.

    The item is a gene. ``table``/``result``/``stats`` use the ``{table, result, stats}`` layout so
    the shared
    reducers apply unchanged; the signed per-phenotype decomposition of each importance lives
    in ``shift``, averaged over draws, because putting the phenotype axis inside ``table``
    would make ``result`` per phenotype rather than per gene.
    """

    table: pd.DataFrame     # cols: gene, [covariate], [groupby], [splitby], draw, value
    result: pd.DataFrame    # one row per (gene[, covariate][, group]); value + sd/hdi_*,
                            #   + null_value/adjusted with a reference
    stats: object           # per-gene contrast over groups when splitby is set, else None;
                            #   one row per (gene, contrast); `quantity` names the column
    shift: pd.DataFrame     # cols: gene, phenotype, [covariate], [groupby], [splitby],
                            #       baseline, perturbed, shift


def validate(schema, result, *, name: str = "result") -> None:
    """Public twin of the decorator's internal check, for direct use in tests."""
    required = set(getattr(schema, "__required_keys__", None) or schema.__annotations__)
    missing = required - set(result)
    if missing:
        raise ValueError(f"{name}: missing required keys {sorted(missing)}")
