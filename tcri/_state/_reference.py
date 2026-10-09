"""The reference run, and the columns it adds.

A model-based number on its own says nothing about whether the structure it measures is there.
Every scored metric therefore computes a **reference**: the same functional, at the caller's own
arguments, on a permutation null of the same fit. ``adjusted = value - null_value`` is the part
of the observed number the structure accounts for, and the one quantity a scored metric reports;
``value`` and ``null_value`` are stored beside it as its inputs.

Two rules do the work here, and both exist because the obvious shortcut is wrong.

**The reference run is the caller's own call with two arguments changed.** Not a call at
defaults. ``groupby``, ``splitby``, ``clones``, ``weighted``, ``normalized``, ``normalize_mode``,
``n_clones_ref``, ``distance_metric``, ``temperature`` and ``n_samples`` each change the
estimand, so a reference computed at defaults is a different quantity subtracted from a
different quantity. The arguments are taken from the decorator's own ``bind``, so the forwarded
set cannot drift from the signature.

**``adjusted`` is a difference of two summaries, never a summary of a difference.** Draws are
never paired across two fits -- there is no correspondence between the parent's draw 7 and the
null's -- so the adjusted value carries no ``sd`` and no interval, and nothing in this module
produces one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import keys as K

#: Columns :func:`.._compute._tables.summarize` adds beside the value. Not labels, not values:
#: the reference merge has to tell all three apart to know what to join on.
SUMMARY_COLUMNS = ("sd", "hdi_low", "hdi_high", "ci_low", "ci_high", "n_groups")

#: ``fit=`` is set, so the number is ABOUT that fit; "auto" then resolves to no reference. A fit
#: being measured is never simultaneously measured against something else, and this avoids
#: needing a predicate for "is this fit a null" -- which is deliberately undefined, since any
#: fit may serve as a reference and nothing requires the ``null.`` prefix.
AUTO = "auto"


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

    The reference frame is joined on the row LABELS, never positionally: a null's rows are the
    parent's by construction, but a join says so and an alignment by position only assumes
    it. A label with no reference row gets NaN rather than being dropped, so the caller's own
    rows stay authoritative.
    """
    result, ref_result = payload.get("result"), reference.get("result")
    if result is None or ref_result is None or not len(result) or not len(ref_result):
        return payload

    present = [v for v in values if v in result.columns and v in ref_result.columns]
    if not present:
        return payload
    pairs = column_pairs(present, denominators)
    created = [c for _, n, a in pairs for c in (n, a) if c]
    labels = [c for c in _labels(result, values, created) if c in ref_result.columns]
    renames = {v: n for v, n, _ in pairs}

    ref = ref_result[labels + present].rename(columns=renames)
    if labels:
        merged = result.merge(ref, on=labels, how="left")
    else:
        # No labels at all: one row against one row (a scalar metric with no covariate and no
        # groupby). A merge on nothing is a cross join, which is right for 1x1 and wrong for
        # anything else, so say so rather than producing a silent product.
        if len(result) != 1 or len(ref) != 1:
            raise ValueError(
                f"cannot join a reference with no label columns onto {len(result)} rows"
            )
        merged = result.assign(**{c: ref[c].iloc[0] for c in renames.values()})
    for native, null, adjusted in pairs:
        if adjusted is not None:
            merged[adjusted] = (merged[native].to_numpy(dtype=float)
                                - merged[null].to_numpy(dtype=float))
    payload["result"] = merged

    # `table` carries them too, broadcast to every draw: the violin path reads `table` rather
    # than `result`, so without this the adjusted panel is a seaborn KeyError on any metric
    # whose coarsest varying unit is the draw.
    table = payload.get("table")
    if table is not None and len(table):
        cols = [c for _, c, _ in pairs] + [a for _, _, a in pairs if a is not None]
        keep = [c for c in labels if c in table.columns]
        if keep:
            payload["table"] = table.merge(merged[keep + cols].drop_duplicates(keep),
                                           on=keep, how="left")
        elif len(merged) == 1:
            payload["table"] = table.assign(**{c: merged[c].iloc[0] for c in cols})
    return payload


def restat(payload, *, groupby, splitby, per_gene=False):
    """Recompute ``stats`` on the quantity this result reports: the adjusted value, or the
    value when there is no usable adjustment.

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
