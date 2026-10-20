"""The reference run, and the columns it adds.

A model-based number on its own says nothing about whether the structure it measures is there.
Every scored metric therefore computes a **reference**: the same functional, at the caller's own
arguments, on a permutation null of the same fit. The null is the value expected under no
association, and ``adjusted = value - null_value`` is the part of the observed number the
structure accounts for: the one quantity a scored metric reports, with ``value`` and
``null_value`` stored beside it as its inputs.

Two rules do the work here.

**The reference run is the caller's own call on the null.** Not a call at defaults. ``groupby``,
``splitby``, ``clones``, ``exclude_pools``, ``weighted``, ``normalized``, ``normalize_mode``,
``n_clones_ref``, ``distance_metric``, ``temperature`` and ``n_samples`` each change the
estimand, so a reference computed at defaults is a different quantity subtracted from a
different quantity. The arguments are taken from the decorator's own ``bind``, so the forwarded
set cannot drift from the signature. Only the fit, ``null_model`` and the seed of the draws
differ: the null's draws are seeded by :func:`null_seed`, derived from the caller's
``random_state``.

**``adjusted`` is computed per draw, against every fit of the reference.** A reference is the
fits ``tcri.null.*`` wrote for it, ``null.phenotype[0]`` to ``null.phenotype[K-1]``; the caller's
call runs on each. The fit and each null are independent fits, so draw ``d`` of the one is
paired with draw ``d`` of every null: ``table`` carries ``null`` (which of the K), ``null_value``
and ``adjusted`` on every pair, and ``result`` summarizes ``adjusted`` over all of them like any
value, with its mean, ``adjusted_sd`` and an HDI. Pooling the pairs is what puts the null's
fit-to-fit spread into the interval; one null's draws do not carry it, and the mean of the
nulls' draws would hide it. Each null draws from its own seed because two samples drawn from one
seed share their underlying noise, and the difference of two correlated samples has a narrower
spread than two independent fits support. At ``n_samples=0`` each side has one draw, so a row
has K pairs: ``adjusted`` is their mean, ``adjusted_sd`` the sd between the nulls, and the HDI is
NaN, because an interval read off K points is not one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import keys as K

#: Columns :func:`.._compute._tables.summarize` adds beside the value. Not labels, not values:
#: the reference merge has to tell all three apart to know what to join on.
SUMMARY_COLUMNS = ("sd", "hdi_low", "hdi_high", "ci_low", "ci_high", "n_groups")

#: The spread of ``adjusted`` over the paired draws, under the adjusted value's own names: the
#: ``sd`` and ``hdi_*`` that :func:`.._compute._tables.build_result` gives the value.
ADJUSTED_SUMMARY = {"sd": "adjusted_sd", "hdi_low": "adjusted_hdi_low",
                    "hdi_high": "adjusted_hdi_high"}

#: The spawn key of the reference's draws, beside the stream the caller's seed starts.
_NULL_STREAM = 1

#: The ``table`` column numbering the fits of a reference, in the order they were read.
NULL_COLUMN = "null"

#: ``fit=`` is set, so the number is ABOUT that fit; "auto" then resolves to no reference. A fit
#: being measured is never simultaneously measured against something else, and this avoids
#: needing a predicate for "is this fit a null" -- which is deliberately undefined, since any
#: fit may serve as a reference and nothing requires the ``null.`` prefix.
AUTO = "auto"


def null_seed(random_state, index=0):
    """The seed of null ``index``'s draws, derived from the caller's ``random_state``.

    ``None`` stays ``None``: both runs then draw unseeded and independently. A NumPy
    ``Generator`` gives one integer off its own stream, so a seeded generator still reproduces.
    An integer, or a ``torch.Generator``'s initial seed, starts a second stream of the same
    seed sequence, one per ``index``, so the reference never draws from the seed the fit drew
    from and no two nulls of a reference draw from one seed. ``index`` is the fit's own index in
    its reference, so a null's draws are the same whether it is read alone or with the others.
    """
    if random_state is None:
        return None
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, 2**31 - 1))
    if hasattr(random_state, "initial_seed"):     # a torch.Generator
        base = int(random_state.initial_seed())
    elif isinstance(random_state, (int, np.integer)):
        base = int(random_state)
    else:
        base = int(np.random.default_rng(random_state).integers(0, 2**63 - 1))
    spawn = (_NULL_STREAM,) + ((int(index),) if index else ())
    stream = np.random.SeedSequence(base % 2**64, spawn_key=spawn)
    return int(stream.generate_state(1)[0] % (2**31 - 1))


def summary_column(quantity: str, stat: str) -> str:
    """The ``result`` column holding ``stat`` (``"sd"``, ``"hdi_low"``, ``"hdi_high"``) of
    ``quantity``: the bare name for the value, ``adjusted_<stat>`` for the adjusted value."""
    return stat if quantity == "value" else f"{quantity}_{stat}"


def null_column(value: str) -> str:
    return f"null_{value}"


def adjusted_column(value: str) -> str:
    """``value`` -> ``adjusted``, ``value_from`` -> ``adjusted_from``, ``value_to`` ->
    ``adjusted_to``."""
    if not value.startswith("value"):
        raise ValueError(f"a native value column must start with 'value', got {value!r}")
    return "adjusted" + value[len("value"):]


def column_pairs(values, denominators=()):
    """``(native, null, adjusted)`` per native value column; ``adjusted`` is ``None`` for a
    denominator, which has a reference but no adjusted column -- a difference of two normalizers
    is not a quantity anyone reports."""
    return [(v, null_column(v), None if v in denominators else adjusted_column(v))
            for v in values]


def usable_reference(frame, column="null_value") -> bool:
    """Whether ``frame[column]`` is a reference: present, with something finite in it.

    A column with nothing finite is not a reference. ``stats`` is then contrasted on the value
    rather than on an all-NaN adjustment, and a panel draws the value with "(no reference)" on
    the label rather than handing seaborn an all-NaN y, which it refuses.
    """
    if frame is None or not len(frame) or column not in frame.columns:
        return False
    return bool(np.isfinite(
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)).any())


def resolve_reference(adata, *, null_model, fit, default_null, metric):
    """The fits (or models) the reference is computed on, as a list, or ``None`` for none.

    ``None`` -> no reference. ``"auto"`` -> every fit of the metric's default reference from the
    contract, unless ``fit`` is set. A name selects one fit when it names one (``"myalt"``,
    ``"null.phenotype[2]"``, or ``"phenotype[2]"``) and every fit of a reference when it names a
    reference (``"phenotype"``, ``"null.phenotype"``, ``"null.phenotype.b"``), so a bare kind and
    a full name are one rule, as in :func:`..keys.resolve_fit`. A model is that one fit, and a
    list of models is a reference of those fits: the list ``tcri.null.*`` returns can be passed
    straight back.
    """
    if null_model is None:
        return None
    if isinstance(null_model, (list, tuple)):
        references = list(null_model)
        if not references or any(r is None or isinstance(r, str) for r in references):
            raise TypeError(
                f"null_model={null_model!r}: a list names a reference by its fitted models, "
                f"such as the list tcri.null.* returns. Name a reference or one fit by its name "
                f"instead, e.g. null_model='phenotype'."
            )
        return references
    if not isinstance(null_model, str):
        return [null_model]
    if null_model == AUTO:
        if fit is not None:
            return None
        if default_null is None:
            return None
        group = K.resolve_group(adata, default_null)
        if group is None:
            raise KeyError(
                f"{metric} defaults to the {default_null!r} null and this AnnData carries no "
                f"such reference. Build the references once with `tcri.null.all(model, adata)` "
                f"(or `tcri.null.{default_null}(model, adata)` for this one), or pass "
                f"`null_model=None` to report the bare value. A metric never fits a null "
                f"itself: that is a decision with a cost, and it is yours."
            )
        return list(group[1])
    known = K.fits(adata)
    if null_model in known or f"null.{null_model}" in known:
        return [K.resolve_fit(adata, null_model)]
    group = K.resolve_group(adata, null_model)
    if group is not None:
        return list(group[1])
    return [K.resolve_fit(adata, null_model)]   # raises, naming what the object carries


def fit_of(reference_of, adata):
    """The fit a substrate metric's reference is computed on.

    A name passes through. A model is read as the fit ``adata`` records for it; a model with no
    fit on this object raises, because a metric that reads a stored substrate cannot compute on
    a model's networks.
    """
    if isinstance(reference_of, str):
        return reference_of
    name = name_of(reference_of, adata)
    if name in K.fits(adata):
        return name
    raise TypeError(
        f"this metric reads a stored substrate, and the model {name!r} has no fit on this "
        f"AnnData. Write its substrate first with model.to_anndata(adata, fit='myalt'), then "
        f"pass null_model='myalt', or pass the models tcri.null.* returned for this object."
    )


def index_of(reference_of, adata):
    """A reference fit's own index in its reference, from its record; 0 when it has none."""
    name = reference_of if isinstance(reference_of, str) else name_of(reference_of, adata)
    record = adata.uns.get(K.fit_key(K.FIT_SETTINGS, name)) or {}
    index = record.get("index")
    return 0 if index is None else int(index)


#: The five things that have to agree before one fit's numbers can be subtracted from another's.
#: Written per fit by ``to_anndata``, because the three category lists are SHARED keys: comparing
#: them between two fits of one object compares an object with itself and always passes.
JOINABLE = ("n_obs", "n_ct", K.PHENOTYPE_CATEGORIES, K.CLONOTYPE_CATEGORIES,
            K.COVARIATE_CATEGORIES)


def check_joinable(adata, fit, *, against=None):
    """Refuse a reference whose rows do not correspond to the main fit's.

    Holds by construction for a permutation null (the ct index is the parent's) and for nothing
    else automatically. Without the check an incomparable fit writes its substrate happily and
    the join comes back misaligned or NaN with no error anywhere -- the same class of silent
    wrongness the module exists to remove.
    """
    mine = adata.uns.get(K.fit_key(K.FIT_SETTINGS, against))
    theirs = adata.uns.get(K.fit_key(K.FIT_SETTINGS, fit))
    if mine is None or theirs is None:
        return  # a fit written before the record existed; the substrate guards still apply
    for field in JOINABLE:
        a, b = mine.get(field), theirs.get(field)
        if a is None or b is None:
            continue
        same = (list(a) == list(b)) if isinstance(a, (list, np.ndarray)) else (int(a) == int(b))
        if not same:
            raise ValueError(
                f"fit {fit!r} is not comparable to fit {against!r}: they differ in {field!r} "
                f"({a!r} against {b!r}). A reference has to describe the same cells and the "
                f"same label spaces, or its rows do not correspond to this fit's and the "
                f"difference is between two different quantities. Build the reference from "
                f"this fit with tcri.null.*, or register the other model on the same "
                f"categories."
            )


def _labels(result, values, reference_columns=()):
    """The columns that identify a row: not a value, not a summary of one.

    Excluded by EXACT NAME, never by prefix. `groupby` and `splitby` are arbitrary `obs` column
    names, so a user column called `adjusted_patient` or `null_arm` is a perfectly legitimate row
    label; excluding it by prefix drops a join key and the merge fans out.
    """
    drop = set(values) | set(SUMMARY_COLUMNS) | set(reference_columns)
    return [c for c in result.columns if c not in drop]


def attach(payload, references, *, values, denominators=()):
    """Add the reference and adjusted columns to a ``{table, result}`` payload, in place.

    ``references`` is the caller's call run on each fit of the reference, in order; each run's
    ``table`` is paired with this one's by :func:`_attach_frames`.
    """
    table, result = _attach_frames(payload.get("table"), payload.get("result"),
                                   [r.get("table") for r in references],
                                   values=values, denominators=denominators)
    payload["table"], payload["result"] = table, result
    return payload


def attach_levels(payload, references, *, values, denominators=()):
    """Add the reference and adjusted columns to every level of a ``{replicate, item, stats}``
    payload, in place: each level's ``draws`` is paired with the same level's ``draws`` of every
    run of the reference, and its ``summary`` gains the summaries (:func:`_attach_frames`)."""
    for level in ("replicate", "item"):
        own = payload.get(level)
        if own is None:
            continue
        draws, summary = _attach_frames(own.get("draws"), own.get("summary"),
                                        [(r.get(level) or {}).get("draws") for r in references],
                                        values=values, denominators=denominators)
        own["draws"], own["summary"] = draws, summary
    return payload


def _attach_frames(table, result, ref_tables, *, values, denominators=()):
    """Return ``(table, result)`` with the reference and adjusted columns added.

    ``ref_tables`` are the per-draw frames of the caller's call run on each fit of the reference,
    in order. They are stacked with ``null`` numbering them, and joined onto ``table`` on the row
    LABELS and the draw, never positionally: a null's rows are the parent's by construction, but
    a join says so and an alignment by position only assumes it. Draw ``d`` of the fit is paired
    with draw ``d`` of every null, and every adjusted column is their difference on that pair.
    ``result`` then gains the nulls' mean over all their draws and the adjusted values
    summarized over all the pairs: ``adjusted`` with ``adjusted_sd``, ``adjusted_hdi_low`` and
    ``adjusted_hdi_high``, and a change's endpoints, ``adjusted_from`` and ``adjusted_to``, as
    means. With one draw per side (``n_samples=0``) a row has one pair per null: ``adjusted_sd``
    is then the sd between the nulls and the HDI is NaN, since an interval read off that few
    points is not one. A row with no reference row gets NaN rather than being dropped, so the
    caller's own rows stay authoritative.

    ``adjusted`` equals ``value - null_value`` whenever every draw is finite on both sides. A
    draw that is non-finite on one side leaves the mean of that side and drops out of the
    paired mean.
    """
    from .._compute._tables import build_result   # the lower layer, reached lazily

    frames = [f.assign(**{NULL_COLUMN: i}) for i, f in enumerate(ref_tables)
              if f is not None and len(f)]
    if not frames or any(frame is None or not len(frame) for frame in (result, table)):
        return table, result
    ref_table = pd.concat(frames, ignore_index=True)

    present = [v for v in values if v in table.columns and v in ref_table.columns]
    if "value" not in present:
        return table, result
    pairs = column_pairs(present, denominators)
    per_draw = [c for _, n, a in pairs for c in (n, a) if c]
    labels = [c for c in _labels(table, (*values, "draw"),
                                 (*per_draw, *ADJUSTED_SUMMARY.values(), NULL_COLUMN))
              if c in ref_table.columns]
    keys = labels + (["draw"] if "draw" in table.columns and "draw" in ref_table.columns
                     else [])
    if not keys:
        raise ValueError("cannot pair a reference with no label or draw columns")

    ref = ref_table[keys + [NULL_COLUMN] + present].rename(columns={v: n for v, n, _ in pairs})
    merged = table.merge(ref, on=keys, how="left", validate="one_to_many")
    for native, null, adjusted in pairs:
        if adjusted is not None:
            merged[adjusted] = (merged[native].to_numpy(dtype=float)
                                - merged[null].to_numpy(dtype=float))

    group = [c for c in labels if c in result.columns]
    # a change and its endpoints share one mask over the (null, draw) pairs, so
    # adjusted == adjusted_to - adjusted_from holds in the summary as it does on every pair
    summary = build_result(merged[group + ["draw"] + per_draw], value="adjusted",
                           extra_values=[c for c in per_draw if c != "adjusted"],
                           one_mask="value_from" in present)
    summary = summary.rename(columns=ADJUSTED_SUMMARY)
    if "draw" not in merged.columns or merged["draw"].nunique() <= 1:
        summary[ADJUSTED_SUMMARY["hdi_low"]] = np.nan
        summary[ADJUSTED_SUMMARY["hdi_high"]] = np.nan
    nulls = [n for _, n, _ in pairs]
    differences = [a for _, _, a in pairs if a is not None]
    summary = summary[group + nulls + differences[:1] + list(ADJUSTED_SUMMARY.values())
                      + differences[1:]]
    if group:
        result = result.merge(summary, on=group, how="left", validate="many_to_one")
    else:
        result = result.assign(**summary.iloc[0].to_dict())
    return merged, result


def mask_unmovable(payload, adata, references, *, arguments, values, denominators=(), metric):
    """Blank the reference of every group a covariate null cannot move, in place.

    A covariate null shuffles levels within (clonotype, batch), so a group whose counted clones
    sit at one level in every batch is left exactly as the fit has it, and its adjusted value is 0
    by construction rather than by the data. Such a group has no reference: its value is kept, and
    every reference column of its replicate and item rows is NaN, so it leaves the tests. It is
    named in a warning. Applies only when the reference is a covariate null; which groups qualify
    is decided from ``obs`` (:func:`~tcri._compute._tables.movable_clones`), never from a drawn
    shuffle. Returns the label rows of the groups blanked, for :func:`restat_levels`.
    """
    import os
    import warnings

    from .._compute._tables import unmovable_groups   # the lower layer, reached lazily

    kinds = {(adata.uns.get(K.fit_key(K.FIT_SETTINGS, r)) or {}).get("kind")
             for r in references if isinstance(r, str)}
    if "covariate" not in kinds:
        return []
    if arguments.get("cov_from") is not None:
        levels = (arguments["cov_from"], arguments.get("cov_to"))
    else:
        levels = (arguments.get("covariate"),)
    groupby = arguments.get("groupby")
    unmoved = unmovable_groups(adata, levels=levels, groupby=groupby,
                               splitby=arguments.get("splitby"), clones=arguments.get("clones"),
                               exclude_pools=arguments.get("exclude_pools", True),
                               fit=arguments.get("fit"))
    if not unmoved:
        return []
    pairs = column_pairs(values, denominators)
    columns = ([n for _, n, _ in pairs] + [a for _, _, a in pairs if a is not None]
               + list(ADJUSTED_SUMMARY.values()))
    labels = [row.get(groupby) for row in unmoved] if groupby is not None else None
    for level in ("replicate", "item"):
        own = payload.get(level)
        for frame_name in ("draws", "summary"):
            frame = (own or {}).get(frame_name)
            if frame is None or not len(frame):
                continue
            rows = (np.ones(len(frame), dtype=bool) if labels is None
                    else frame[groupby].isin(labels).to_numpy())
            present = [c for c in columns if c in frame.columns]
            frame.loc[rows, present] = np.nan
    where = (f"{groupby} {', '.join(repr(x) for x in labels)}" if labels is not None
             else "the whole object")
    warnings.warn(
        f"{metric}: the covariate null cannot move any cell of {where} at "
        f"{', '.join(repr(x) for x in levels)}: within each of its clones, every batch holds one "
        f"covariate level, so a shuffle within (clonotype, batch) changes nothing and there is no "
        f"reference. Its adjusted values are NaN and it leaves the tests.",
        UserWarning,
        skip_file_prefixes=(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), ""),),
    )
    return unmoved


def restat(payload, *, groupby, splitby, per_gene=False):
    """Recompute ``stats`` on the quantity this result reports: the adjusted value, each row's
    mean over its paired draws, or the value when there is no usable adjustment.

    The frame is replaced, not appended to, so a result never carries two contrasts. The adjusted
    column is contrasted only when something in it is finite: a reference that is non-finite
    everywhere is no reference, and the frame then says ``"value"``. Rebuilding on the value
    rather than leaving the payload's frame alone is what makes the stored frame self-describing,
    and what makes this function idempotent: the payload it receives may already carry an
    adjusted frame from an earlier pass.
    """
    if "stats" not in payload:
        return payload
    result = payload.get("result")
    if result is None or not len(result) or splitby is None:
        return payload
    target = "adjusted" if usable_reference(result, "adjusted") else "value"
    if per_gene:
        from ..perturbation._tables import stats_per_gene
        rebuilt = stats_per_gene(result, groupby=groupby, splitby=splitby, value=target)
    else:
        from .._compute._tables import build_stats
        rebuilt = build_stats(result, groupby=groupby, splitby=splitby, value=target)
    # Never replace a frame with nothing: the frame already there is the one the result
    # legitimately has.
    if rebuilt is not None and len(rebuilt):
        payload["stats"] = rebuilt
    return payload


def restat_levels(payload, *, groupby, splitby, change=None, without=()):
    """Recompute a ``{replicate, item, stats}`` payload's ``stats`` on the replicate summary, as
    :func:`restat` does for a ``{table, result, stats}`` one: on the adjusted value when it has
    something finite, else on the value. ``change`` is the covariate-row settings of a change
    call (:func:`~tcri._compute._tables.build_stats`), whose stats exist with or without
    ``splitby``. ``without`` names groups (label rows, :func:`mask_unmovable`) that have no
    reference and leave the tests, on the value as on the adjusted value."""
    summary = (payload.get("replicate") or {}).get("summary")
    if summary is None or not len(summary) or (splitby is None and change is None):
        return payload
    from .._compute._tables import build_stats
    target = "adjusted" if usable_reference(summary, "adjusted") else "value"
    if without:
        summary = summary.copy()
        if groupby is not None and groupby in summary.columns:
            gone = summary[groupby].isin([row.get(groupby) for row in without]).to_numpy()
        else:
            gone = np.ones(len(summary), dtype=bool)
        own = [c for c in ("value", "value_from", "value_to") if c in summary.columns]
        summary.loc[gone, own] = np.nan
    rebuilt = build_stats(summary, groupby=groupby, splitby=splitby, value=target, change=change)
    if rebuilt is not None and len(rebuilt):
        payload["stats"] = rebuilt
    return payload


def label_for(result, quantity, ylabel):
    """The y label, saying when there is no reference to compare against.

    Only when the frame HAS rows: an empty result is a real outcome (no clone at both covariate
    levels within any replicate) and saying "no reference" about it would name the wrong cause.
    """
    if result is None or not len(result):
        return ylabel
    if quantity == "adjusted":
        return f"adjusted {ylabel}"
    return ylabel if "null_value" in result.columns else f"{ylabel} (no reference)"


def name_of(reference_of, adata=None):
    """The h5ad-writable name of whatever the reference was computed on.

    A fit name passes through. A null built by ``tcri.null.*`` or rebuilt becomes its fit name
    (``null.phenotype[0]``), which ``fit=`` resolves, when ``adata``'s record for that fit has the
    null's ``name``, its parameter-store namespace, as its namespace. Every fit has a namespace
    of its own, so a null whose fit has since been refit, by its parent or another, a null with
    no record on ``adata``, and any other model object become their own name. A list of
    references is named by its one entry, by the reference whose fits it is exactly
    (``null.phenotype``), or else as the list of its entries' names. A model in the params block
    would break the `.h5ad` write and tell a later reader nothing it can act on.
    """
    if isinstance(reference_of, (list, tuple)):
        names = [name_of(r, adata) for r in reference_of]
        if len(names) == 1:
            return names[0]
        if adata is not None:
            groups = {str((adata.uns.get(K.fit_key(K.FIT_SETTINGS, n)) or {}).get("reference"))
                      for n in names}
            if len(groups) == 1:
                (group,) = groups
                if K.members(adata, group) == names:
                    return group
        return names
    if reference_of is None or isinstance(reference_of, str):
        return reference_of
    name = str(getattr(reference_of, "name", "") or "model")
    fit = getattr(reference_of, "_fit_name", None)
    if fit and adata is not None:
        record = adata.uns.get(K.fit_key(K.FIT_SETTINGS, fit)) or {}
        if str(record.get("namespace", "")) == name:
            return str(fit)
    return name
