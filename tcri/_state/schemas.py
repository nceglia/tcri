"""Result shapes for the ``tl`` metrics and the ``pp`` repertoire summaries, as ``TypedDict``s.

``@tl_result(schema=...)`` checks required-key PRESENCE on dict results. The shapes in the
trailing comments are documentation of the payload layout; nothing parses them.

**By level.** Mutual information, the two entropies and ``perturb.gene_importance`` store
``{replicate, item, stats}``:

``replicate``
    ``{draws, summary}``. ``draws`` has one row per (covariate, group, draw), ``summary`` the same
    rows summarized over the draws, one per (covariate, group), read with ``tcri.get.replicates``.

``item``
    ``{draws, summary}`` for the metric's items, one row per (covariate, group, item[, draw]).
    Mutual information has one value per joint and no ``item``.

``stats``
    The tests on the replicate summary, or ``None`` when nothing is compared.

``joint_distribution`` stores ``{summary, draws}``: the clone × phenotype matrix it is, not a
scored quantity. The other tools store the layout below, with two payload keys:

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
``adjusted`` (``value - null_value``; ``adjusted_from``/``adjusted_to`` for the delta
endpoints). The adjusted value is the one quantity a scored metric reports; ``value`` and
``null_value`` are its inputs and stay stored beside it. A DENOMINATOR gets ``null_denom`` and
no adjusted column. ``table`` carries them per draw, the fit's draw ``d`` paired with the
null's draw ``d``, and ``result`` summarizes the paired draws: ``adjusted`` is their mean, with
``adjusted_sd``, ``adjusted_hdi_low`` and ``adjusted_hdi_high`` beside it, and ``null_v`` is the
null's mean. At ``null_model=None`` none of them exist, on either frame.

**The ``stats`` slot.** Every metric except :class:`JointDistribution` stores it. Six of the
``TypedDict``s below declare it; :class:`PhenotypicFlux` does not, and a declared key is a
required one, so it accepts a payload with or without it. It carries one row per contrast and a ``quantity`` column naming
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
    "DeltaClonotypicEntropy",
    "DeltaPhenotypicEntropy",
    "GeneImportance",
    "validate",
]


class JointDistribution(TypedDict):
    """The clone x phenotype table every other metric reduces."""

    summary: pd.DataFrame   # index ([covariate,] clonotype); columns = phenotypes; the
                            #   posterior mean over draws
    draws: pd.DataFrame     # index ([covariate,] clonotype[, draw]); the draws at
                            #   n_samples > 0, the summary at n_samples = 0


class MutualInformation(TypedDict):
    """I(c; phi) — one value per joint, so a replicate level and no item level."""

    replicate: dict         # {draws, summary}. draws: covariate, [groupby], [splitby], draw,
                            # value, denom (+ null, null_value, null_denom, adjusted with a
                            # reference). summary: one row per group; value + sd/hdi_* + denom
                            # (+ null_value/null_denom/adjusted/adjusted_sd/adjusted_hdi_*).
                            # `denom` is the normalizer this MI was divided by: the null does
                            # not share it, so both are stored and value*denom recovers bits
    stats: object           # one row per contrast; `quantity` names the column contrasted


class ClonotypicEntropy(TypedDict):
    """H(c | phi) — per replicate H(C|P), and one value per PHENOTYPE."""

    replicate: dict         # {draws, summary}: covariate, [groupby], [splitby], [draw], value
                            # (+ the reference columns)
    item: dict              # {draws, summary}: the same, per phenotype
    stats: object           # one row per contrast; `quantity` names the column contrasted


class PhenotypicEntropy(TypedDict):
    """H(phi | c) — per replicate H(P|C), and one value per clone."""

    replicate: dict         # {draws, summary}: covariate, [groupby], [splitby], [draw], value
                            # (+ the reference columns)
    item: dict              # {draws, summary}: the same, per clone (`clonotype`)
    stats: object           # one row per contrast; `quantity` names the column contrasted


class PhenotypicFlux(TypedDict):
    """Distance between a clone's phenotype distribution at two covariate levels.

    Clones absent from either side are dropped, not NaN-filled — a flux needs both endpoints.
    """

    table: pd.DataFrame     # cols: cov_from, cov_to, [groupby], [splitby], clonotype, draw, value
    result: pd.DataFrame    # + null_value/adjusted with a reference


class DeltaClonotypicEntropy(TypedDict):
    """H(c|phi) at ``cov_to`` minus at ``cov_from`` — one value per PHENOTYPE.

    The clone set is intersected across the two levels within each replicate, so ``log2(C)``
    is the same on both sides and cancels out of the difference. That is why no special
    ``n_clones_ref`` default is needed here.
    """

    table: pd.DataFrame     # cols: cov_from, cov_to, [groupby], [splitby], phenotype, draw,
                            #       value, value_from, value_to
    result: pd.DataFrame    # + null_value/null_value_from/null_value_to and
                            #   adjusted/adjusted_from/adjusted_to with a reference
    stats: object           # one row per contrast; `quantity` names the column contrasted


class DeltaPhenotypicEntropy(TypedDict):
    """H(phi|c) at ``cov_to`` minus at ``cov_from`` — one value per CLONE.

    The only metric whose item axis is entity-matched: the same clonotype observed at both
    levels, a biological barcode rather than a category measured twice.
    """

    table: pd.DataFrame     # cols: cov_from, cov_to, [groupby], [splitby], clonotype, draw,
                            #       value, value_from, value_to
    result: pd.DataFrame    # + null_value/null_value_from/null_value_to and
                            #   adjusted/adjusted_from/adjusted_to with a reference
    stats: object           # one row per contrast; `quantity` names the column contrasted


class GeneImportance(TypedDict):
    """``perturb.gene_importance`` — how far silencing a gene moves the mean phenotype call.

    The gene is a label on every row. The replicate level holds the importance, and the item
    level the per-phenotype shift whose L1 norm the importance is.
    """

    replicate: dict         # {draws, summary}: gene, [covariate], [groupby], [splitby], [draw],
                            #   value (+ null, null_value, adjusted with a reference); summary
                            #   value + sd/hdi_* (+ null_value/adjusted/adjusted_sd/adjusted_hdi_*)
    item: dict              # {draws, summary}: the same, per phenotype; value is the shift
    stats: object           # per-gene contrast over groups when splitby is set, else None;
                            #   one row per (gene, contrast); `quantity` names the column


def validate(schema, result, *, name: str = "result") -> None:
    """Public twin of the decorator's internal check, for direct use in tests."""
    required = set(getattr(schema, "__required_keys__", None) or schema.__annotations__)
    missing = required - set(result)
    if missing:
        raise ValueError(f"{name}: missing required keys {sorted(missing)}")


class ClonotypeSharing(TypedDict):
    """``pp.clonotype_sharing``: the pairs of groups, the groups, and the shared clonotypes.
    ``pp.clone_persistence`` stores its one frame and has no ``TypedDict``."""

    pairs: pd.DataFrame       # one row per unordered pair of groups: <groupby>_a, <groupby>_b,
                              #   [<splitby>_a, <splitby>_b,] n_clonotypes_a, n_clonotypes_b,
                              #   shared_clonotypes, shared_fraction_a, shared_fraction_b
    groups: pd.DataFrame      # one row per group: <groupby>, [<splitby>,] n_clonotypes,
                              #   shared_clonotypes, shared_fraction
    clonotypes: pd.DataFrame  # one row per shared clonotype: clonotype, n_groups, groups,
                              #   n_cells, cells_per_group
