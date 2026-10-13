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

**``adjusted`` is computed per draw.** The fit and the null are independent fits, so draw ``d``
of the one is paired with draw ``d`` of the other: ``table`` carries ``null_value`` and
``adjusted`` on every draw, and ``result`` summarizes ``adjusted`` over those draws like any
value, with its mean, ``adjusted_sd`` and an HDI. The null draws from its own seed because two
samples drawn from one seed share their underlying noise, and the difference of two correlated
samples has a narrower spread than two independent fits support. At ``n_samples=0`` each side
has one draw, and ``adjusted`` is the difference of the two point values.
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

#: ``fit=`` is set, so the number is ABOUT that fit; "auto" then resolves to no reference. A fit
#: being measured is never simultaneously measured against something else, and this avoids
#: needing a predicate for "is this fit a null" -- which is deliberately undefined, since any
#: fit may serve as a reference and nothing requires the ``null.`` prefix.
AUTO = "auto"


def null_seed(random_state):
    """The seed of the reference's draws, derived from the caller's ``random_state``.

    ``None`` stays ``None``: both runs then draw unseeded and independently. A NumPy
    ``Generator`` gives one integer off its own stream, so a seeded generator still reproduces.
    An integer, or a ``torch.Generator``'s initial seed, starts a second stream of the same
    seed sequence, so the reference never draws from the seed the fit drew from.
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
    stream = np.random.SeedSequence(base % 2**64, spawn_key=(_NULL_STREAM,))
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
    """The fit name the reference is computed on, or ``None`` for no reference.

    ``None`` -> no reference. ``"auto"`` -> the metric's default kind from the contract, unless
    ``fit`` is set. Anything else goes through :func:`..keys.resolve_fit`, so a bare kind
    (``"clonotype"``), a full fit name (``"null.clonotype"``) and a hand-written fit
    (``"myalt"``) are one rule.
    """
    if null_model is None:
        return None
    if not isinstance(null_model, str):
        # A fitted model, accepted only where the reference needs networks rather than arrays
        # (`perturb.gene_importance`); the decorator refuses it everywhere else.
        return null_model
    if null_model == AUTO:
        if fit is not None:
            return None
        if default_null is None:
            return None
        try:
            return K.resolve_fit(adata, default_null)
        except KeyError:
            raise KeyError(
                f"{metric} defaults to the {default_null!r} null and this AnnData carries no "
                f"such fit. Build the references once with `tcri.null.all(model, adata)` (or "
                f"`tcri.null.{default_null}(model, adata)` for this one), or pass "
                f"`null_model=None` to report the bare value. A metric never fits a null "
                f"itself: that is a decision with a cost, and it is yours."
            ) from None
    return K.resolve_fit(adata, null_model)


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


def attach(payload, reference, *, values, denominators=()):
    """Add the reference and adjusted columns to ``table`` and ``result``, in place.

    The reference's ``table`` is joined onto this one on the row LABELS and the draw, never
    positionally: a null's rows are the parent's by construction, but a join says so and an
    alignment by position only assumes it. Draw ``d`` of the fit is paired with draw ``d`` of
    the null, and every adjusted column is their difference on that draw. ``result`` then gains
    the null's mean over its draws and the adjusted values summarized over the paired draws:
    ``adjusted`` with ``adjusted_sd``, ``adjusted_hdi_low`` and ``adjusted_hdi_high``, and the
    delta endpoints' ``adjusted_from`` and ``adjusted_to`` as means. A row with no reference
    row gets NaN rather than being dropped, so the caller's own rows stay authoritative.

    ``adjusted`` equals ``value - null_value`` whenever every draw is finite on both sides. A
    draw that is non-finite on one side leaves the mean of that side and drops out of the
    paired mean.
    """
    from .._compute._tables import build_result   # the lower layer, reached lazily

    result, table = payload.get("result"), payload.get("table")
    ref_table = reference.get("table")
    if any(frame is None or not len(frame) for frame in (result, table, ref_table)):
        return payload

    present = [v for v in values if v in table.columns and v in ref_table.columns]
    if "value" not in present:
        return payload
    pairs = column_pairs(present, denominators)
    per_draw = [c for _, n, a in pairs for c in (n, a) if c]
    labels = [c for c in _labels(table, (*values, "draw"),
                                 (*per_draw, *ADJUSTED_SUMMARY.values()))
              if c in ref_table.columns]
    keys = labels + (["draw"] if "draw" in table.columns and "draw" in ref_table.columns
                     else [])
    if not keys:
        raise ValueError("cannot pair a reference with no label or draw columns")

    ref = ref_table[keys + present].rename(columns={v: n for v, n, _ in pairs})
    merged = table.merge(ref, on=keys, how="left", validate="many_to_one")
    for native, null, adjusted in pairs:
        if adjusted is not None:
            merged[adjusted] = (merged[native].to_numpy(dtype=float)
                                - merged[null].to_numpy(dtype=float))
    payload["table"] = merged

    group = [c for c in labels if c in result.columns]
    summary = build_result(merged[group + ["draw"] + per_draw], value="adjusted",
                           extra_values=[c for c in per_draw if c != "adjusted"])
    summary = summary.rename(columns=ADJUSTED_SUMMARY)
    nulls = [n for _, n, _ in pairs]
    differences = [a for _, _, a in pairs if a is not None]
    summary = summary[group + nulls + differences[:1] + list(ADJUSTED_SUMMARY.values())
                      + differences[1:]]
    if group:
        payload["result"] = result.merge(summary, on=group, how="left",
                                         validate="many_to_one")
    else:
        payload["result"] = result.assign(**summary.iloc[0].to_dict())
    return payload


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
    (``null.phenotype``), which ``fit=`` resolves, when ``adata``'s record for that fit has the
    null's ``name``, its parameter-store namespace, as its namespace. A null whose fit another
    parent has since rewritten, a null with no record on ``adata``, and any other model object
    become their own name. An earlier null of the same parent shares the namespace and so takes
    the fit name. A model in the params block would break the `.h5ad` write and tell a later
    reader nothing it can act on.
    """
    if reference_of is None or isinstance(reference_of, str):
        return reference_of
    name = str(getattr(reference_of, "name", "") or "model")
    fit = getattr(reference_of, "_fit_name", None)
    if fit and adata is not None:
        record = adata.uns.get(K.fit_key(K.FIT_SETTINGS, fit)) or {}
        if str(record.get("namespace", "")) == name:
            return str(fit)
    return name
