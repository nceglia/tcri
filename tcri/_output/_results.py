"""The printed table of a stored or returned ``tl``, ``perturb.gene_importance`` or ``pp``
summary result.

A table shows a summary, never draws or the per-item frame, so it stays a few rows long:

- results stored by level (mutual information, the two entropies, flux and ``gene_importance``):
  the replicate summary, one row per replicate or group (per gene and group for
  ``gene_importance``, ranked by the adjusted importance, with the contrasts of the
  :data:`TOP_GENES` highest-ranked genes). A change (``cov_from``/``cov_to``) also shows the two
  endpoints of the quantity it tests;
- ``joint_distribution``: the number of clones and phenotypes per covariate level, not the
  matrix;
- the ``pp`` summaries: ``repertoire_summary`` and ``clone_persistence`` their stored frame,
  ``clonotype_sharing`` its ``groups`` frame.

A ``tl`` table names the metric, the covariate (or ``cov_from → cov_to``) and the reference it
was adjusted against, and ends with one line per row of ``stats``: a change's paired test of
``cov_from`` against ``cov_to`` (for flux, of the adjusted value against 0), and the contrast
between ``splitby`` levels. A ``pp`` table names the summary and its ``groupby``. A table cut at
:data:`._render.MAX_ROWS` rows says where the rest is.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ._render import cell, frame_table

#: The interval every summary frame carries (``tcri._compute._tables.summarize``'s default
#: ``hdi_prob``; ``tests/test_console_output.py`` checks that they agree).
INTERVAL = "94% HDI"

#: The ``pp`` summaries, which have no covariate subject and no reference.
_SUMMARIES = ("repertoire_summary", "clone_persistence", "clonotype_sharing")

#: The genes whose contrasts a ``gene_importance`` table prints, by rank.
TOP_GENES = 5


def accessor(result) -> str:
    """The ``tcri.get`` accessor that reads a result's printed frame back whole: ``replicates``
    for a result stored by level, ``result`` for any other."""
    if isinstance(result, dict) and isinstance(result.get("replicate"), dict):
        return "replicates"
    return "result"


def _part(rest, part) -> str:
    """Where one entry of a stored result is: ``rest`` (the ``tcri.get`` call that reads the
    whole result) indexed by ``part``, or that entry of a returned result."""
    if rest and rest.startswith("tcri.get."):
        return f"{rest}[{part!r}]"
    return f"{part!r} of the returned result"


def _stats_call(rest) -> str:
    """Where a result's ``stats`` is: ``rest`` with its accessor swapped for ``tcri.get.stats``."""
    if rest and rest.startswith("tcri.get."):
        return re.sub(r"^tcri\.get\.\w+\(", "tcri.get.stats(", rest)
    return _part(rest, "stats")


def _subject(params) -> str:
    if params.get("cov_from") is not None or params.get("cov_to") is not None:
        return f"{params.get('cov_from')} → {params.get('cov_to')}"
    covariate = params.get("covariate")
    return "all levels" if covariate is None else str(covariate)


def _reference(null_model, n_references) -> str:
    if null_model is None:
        return "no reference"
    if not isinstance(null_model, str) and np.ndim(null_model) == 1:
        # several fits: a list as called, an array once read back from .h5ad
        null_model = ", ".join(map(str, null_model))
    fits = f" ({n_references} fit{'s' if n_references != 1 else ''})" if n_references else ""
    return f"against {null_model}{fits}"


def _interval(frame, prefix) -> pd.Series | None:
    """``low – high`` per row, or ``None`` when there is no interval (a plug-in result)."""
    lo, hi = f"{prefix}hdi_low", f"{prefix}hdi_high"
    if lo not in frame or hi not in frame or frame[[lo, hi]].isna().all(axis=None):
        return None
    return pd.Series([f"{cell(a)} – {cell(b)}" for a, b in zip(frame[lo], frame[hi])],
                     index=frame.index)


def _value_view(frame, ids) -> pd.DataFrame:
    """The id columns, then the value and the reference, the endpoints of a change, and the
    tested quantity with its interval: the adjusted value when there is a reference, else the
    value.

    A change's endpoints are those of the tested quantity, by their stored names
    (``adjusted_from``/``adjusted_to`` or ``value_from``/``value_to``): the two numbers its
    paired test compares. The interval is on the change only.
    """
    adjusted = "adjusted" in frame
    value = "adjusted" if adjusted else "value"
    view = frame[ids].copy()
    if adjusted and "value" in frame:
        view["value"] = frame["value"]
        if "null_value" in frame:
            view["null"] = frame["null_value"]
    for end in (f"{value}_from", f"{value}_to"):
        if end in frame:
            view[end] = frame[end]
    view[value] = frame[value]
    interval = _interval(frame, "adjusted_" if adjusted else "")
    if interval is not None:
        view[f"{value} {INTERVAL}"] = interval
    return view


def _unit(row, n) -> str:
    unit = row.get("replicate_unit")
    return f"n {cell(n)}" + (f" {unit}" if isinstance(unit, str) and unit else "")


def _covariate_line(row, splitby, distance) -> str:
    """One covariate row of ``stats``: the paired test of a change, or, for a distance (flux),
    the test of the adjusted value against 0."""
    where = (f"{row[splitby]}: " if splitby and splitby in row and pd.notna(row.get(splitby))
             else "")
    stars = f" {row['stars']}" if str(row.get("stars", "")).strip() else ""
    quantity = row["quantity"]
    if distance:
        if pd.isna(row["p"]) and quantity != "adjusted":
            return f"{where}{quantity} vs 0: not tested without a reference"
        return (f"{where}{quantity} vs 0: mean {cell(row['delta'])}, signed-rank p "
                f"{cell(row['p'])}{stars} ({_unit(row, row['n_b'])})")
    return (f"{where}{row['level_a']} → {row['level_b']}: Δ {quantity} {cell(row['delta'])}, "
            f"signed-rank p {cell(row['p'])}{stars} ({_unit(row, row['n_b'])}, paired)")


def _split_line(row) -> str:
    gene = f"{row['gene']}: " if "gene" in row and pd.notna(row.get("gene")) else ""
    stars = f" {row['stars']}" if str(row.get("stars", "")).strip() else ""
    unit = row.get("replicate_unit")
    unit = f" {unit}" if isinstance(unit, str) and unit else ""
    return (f"{gene}{row['level_b']} vs {row['level_a']}: Δ {row['quantity']} "
            f"{cell(row['delta'])}, p {cell(row['p'])}{stars} "
            f"(n {cell(row['n_b'])} vs {cell(row['n_a'])}{unit})")


def _contrast_lines(stats, *, splitby=None, distance=False) -> list[str]:
    """One line per row of ``stats``, the covariate rows of a change first, as stored."""
    if stats is None or not isinstance(stats, pd.DataFrame) or stats.empty:
        return []
    lines = []
    for _, row in stats.iterrows():
        if row.get("comparison") == "covariate":
            lines.append(_covariate_line(row, splitby, distance))
        else:
            lines.append(_split_line(row))
    return lines


def _gene_contrasts(ranked, stats, rest, *, splitby, distance) -> list[str]:
    """The contrast lines of the :data:`TOP_GENES` genes that rank highest in ``ranked``, in that
    order, and where the others are."""
    if not isinstance(stats, pd.DataFrame) or stats.empty or "gene" not in stats:
        return _contrast_lines(stats, splitby=splitby, distance=distance)
    order = list(dict.fromkeys(ranked["gene"]))
    top = order[:TOP_GENES]
    shown = stats[stats["gene"].isin(top)].copy()
    shown["_rank"] = shown["gene"].map({g: i for i, g in enumerate(top)})
    shown = shown.sort_values("_rank", kind="stable").drop(columns="_rank")
    lines = _contrast_lines(shown, splitby=splitby, distance=distance)
    others = stats["gene"].nunique() - shown["gene"].nunique()
    if others > 0:
        lines.append(f"contrasts of {others:,} more genes: {_stats_call(rest)}")
    return lines


def _summary_table(name, result, params, rest):
    """The table of a ``pp`` summary: its stored frame (``clonotype_sharing``: its ``groups``
    frame, with the other two one accessor away)."""
    from rich.text import Text

    groupby, splitby = params.get("groupby"), params.get("splitby")
    title = Text.assemble((name, "bold"),
                          "".join(f" · {c}" for c in (groupby, splitby) if c))
    notes = []
    if name == "clonotype_sharing" and isinstance(result, dict):
        frame = result.get("groups")
        if not isinstance(frame, pd.DataFrame):
            return None
        notes = [f"pairs of groups: {_part(rest, 'pairs')}",
                 f"{len(result.get('clonotypes', ())):,} shared clonotypes: "
                 f"{_part(rest, 'clonotypes')}"]
        rest = _part(rest, "groups")
    elif isinstance(result, pd.DataFrame):
        frame = result
    else:
        return None
    if not len(frame):
        what = (f"no group has clones at two levels of {params.get('covariate_key')}"
                if name == "clone_persistence" else "no rows")
        title.append(f": {what}")
        return title
    return frame_table(frame, title=title, rest=rest, notes=notes)


def result_table(name, result, params, *, null_model=None, n_references=0, rest=None,
                 distance=False):
    """The printed table of one result of tool ``name``.

    Parameters
    ----------
    name
        The tool's name (``"mutual_information"``, ``"phenotypic_flux"``, ...).
    result
        The result the tool computed, as it returns it with ``inplace=False``.
    params
        The arguments it ran with (``covariate`` or ``cov_from``/``cov_to``, ``groupby``,
        ``splitby``).
    null_model, n_references
        The reference it was adjusted against and how many fits it has.
    rest
        Where the full result is, for a table cut short: a ``tcri.get`` call or a description.
    distance
        The tool's change is a distance between the two levels (flux), so a covariate row of
        ``stats`` tests the adjusted value against 0.

    Returns
    -------
    rich.console.RenderableType or None
        ``None`` for a result of a shape no table is defined for (a tool built on
        ``tl_result`` outside tcri, say), which then prints nothing.
    """
    from rich.text import Text

    if name in _SUMMARIES:
        return _summary_table(name, result, params, rest)
    if not isinstance(result, dict):
        return None

    title = Text.assemble((name, "bold"), f" · {_subject(params)} · "
                          f"{_reference(null_model, n_references)}")
    groupby, splitby = params.get("groupby"), params.get("splitby")
    stats = result.get("stats")
    notes = _contrast_lines(stats, splitby=splitby, distance=distance)

    if isinstance(result.get("replicate"), dict):
        frame = result["replicate"]["summary"]
        # a covariate with one level, and a change's two levels, are in the title already
        covariate = ("covariate" if "covariate" in frame and frame["covariate"].nunique() > 1
                     else None)
        ids = [c for c in ("gene", covariate, groupby, splitby) if c and c in frame]
        if "gene" in frame and "adjusted" in frame:
            frame = frame.sort_values("adjusted", ascending=False, kind="stable")
            notes = _gene_contrasts(frame, stats, rest, splitby=splitby, distance=distance)
        view = _value_view(frame, list(dict.fromkeys(ids)))
        return frame_table(view, title=title, rest=rest, notes=notes)

    if name == "joint_distribution" and isinstance(result.get("summary"), pd.DataFrame):
        summary = result["summary"]
        if isinstance(summary.index, pd.MultiIndex) and "covariate" in summary.index.names:
            counts = (summary.groupby(level="covariate", observed=True).size()
                      .rename("clones").reset_index())
        else:
            counts = pd.DataFrame({"covariate": [_subject(params)], "clones": [len(summary)]})
        counts["phenotypes"] = summary.shape[1]
        return frame_table(counts, title=title, rest=rest, notes=notes)
    return None
