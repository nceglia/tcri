"""``compare_groups`` — INTERNAL. The one contrast implementation in the package.

Turns a per-replicate frame (one row per replicate and level) into contrasts between the levels
of ``splitby``. Unpaired: a two-sided Mann–Whitney U between the replicates of two levels.
Paired (``paired=True``): a two-sided Wilcoxon signed-rank test on each replicate's difference
between two levels, the replicates matched by the ``replicate`` column.

**Not public.** It is reached only through a metric's ``splitby``, never as a step the user
performs: ``build_stats`` calls it with the per-replicate frame it has already collapsed. It
refuses any frame in which a replicate has more than one row at a level, so items or draws can
never be counted as replicates, whoever calls it.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from ._core import mann_whitney, signed_rank, stars

__all__ = ["compare_groups"]

#: The fewest non-zero paired differences the signed-rank test is run on. Below it the row is
#: kept, with NaN statistics and blank stars: scipy returns p = 1.0 for one difference and raises
#: for one zero, either of which would read as a test that was run.
MIN_PAIRED = 2


def compare_groups(df, *, value, splitby, replicate, paired=False):
    """Contrast ``df[value]`` between every pair of ``df[splitby]`` levels.

    Parameters
    ----------
    df
        One row per replicate and level: the replicate in ``replicate``, the level in
        ``splitby``, the number in ``value``.
    value
        The numeric column contrasted.
    splitby
        The column whose levels are contrasted, every pair of them, in sorted order.
    replicate
        The column naming the replicate. Each replicate may have at most one row per level.
    paired
        ``False`` compares the replicates of one level with those of the other (Mann–Whitney U).
        ``True`` matches the replicates present at both levels and tests their differences,
        level b minus level a (Wilcoxon signed-rank). A replicate whose difference is exactly 0
        leaves the test, and ``n_a``/``n_b`` count the replicates the test used.

    Returns
    -------
    pandas.DataFrame
        One row per pair: ``group_a``, ``group_b``, ``mean_a``, ``mean_b``,
        ``delta`` (``mean_b - mean_a``), ``stat`` (U, or the signed-rank W), ``p``, ``stars``,
        ``n_a``, ``n_b``. Unpaired, a pair with no replicate on a side has no row. Paired, a pair
        with fewer than two non-zero differences has its row, with NaN ``stat`` and ``p`` and
        blank ``stars``. An undefined p always has blank ``stars``.

    Raises
    ------
    ValueError
        When ``value`` is not numeric, or a replicate has more than one row at a level.
    """
    for column in (value, splitby, replicate):
        if column not in df.columns:
            raise ValueError(f"compare_groups: no column {column!r} in the frame")
    if not pd.api.types.is_numeric_dtype(df[value]):
        raise ValueError(
            f"compare_groups: column {value!r} is {df[value].dtype}, not numeric; a rank test "
            f"needs numbers."
        )
    rows_per = df.groupby([splitby, replicate], observed=True, dropna=False).size()
    repeated = rows_per[rows_per > 1]
    if len(repeated):
        (level, rep), n = next(iter(repeated.items()))
        raise ValueError(
            f"compare_groups: replicate {rep!r} has {n} rows at {splitby}={level!r}. A test "
            f"counts replicates, so the frame must hold one row per replicate and level; "
            f"collapse items and draws to their replicate first."
        )

    levels = sorted(df[splitby].dropna().unique().tolist(), key=str)
    rows = []
    for a, b in itertools.combinations(levels, 2):
        side_a = df[df[splitby] == a].set_index(replicate)[value].astype(float)
        side_b = df[df[splitby] == b].set_index(replicate)[value].astype(float)
        row = _paired(a, b, side_a, side_b) if paired else _unpaired(a, b, side_a, side_b)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows, columns=_COLUMNS)


_COLUMNS = ["group_a", "group_b", "mean_a", "mean_b", "delta", "stat", "p", "stars", "n_a", "n_b"]


def _unpaired(a, b, side_a, side_b):
    va = side_a.dropna().to_numpy()
    vb = side_b.dropna().to_numpy()
    if va.size == 0 or vb.size == 0:
        return None
    try:
        U, p = mann_whitney(va, vb)
    except ValueError:  # e.g. all-identical values
        U, p = np.nan, np.nan
    row = _row(a, b, va, vb)
    row.update(stat=float(U), p=float(p), stars=stars(p))
    return row


def _paired(a, b, side_a, side_b):
    both = pd.concat({"a": side_a, "b": side_b}, axis=1, join="inner").dropna()
    diff = (both["b"] - both["a"]).to_numpy()
    used = both[diff != 0]
    row = _row(a, b, used["a"].to_numpy(), used["b"].to_numpy())
    if len(used) >= MIN_PAIRED:
        W, p = signed_rank(used["b"].to_numpy() - used["a"].to_numpy())
        row.update(stat=float(W), p=float(p), stars=stars(p))
    return row


def _row(a, b, va, vb):
    """A contrast row over the replicates used; the statistics stay undefined until a test runs."""
    mean_a = float(np.mean(va)) if len(va) else np.nan
    mean_b = float(np.mean(vb)) if len(vb) else np.nan
    return {"group_a": a, "group_b": b, "mean_a": mean_a, "mean_b": mean_b,
            "delta": mean_b - mean_a, "stat": np.nan, "p": np.nan, "stars": "",
            "n_a": int(len(va)), "n_b": int(len(vb))}
