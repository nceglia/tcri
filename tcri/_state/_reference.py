"""The reference run, and the columns it adds.

A model-based number on its own says nothing about whether the structure it measures is there.
Every scored metric therefore computes a **reference**: the same functional, at the caller's own
arguments, on a permutation null of the same fit. ``excess = value - null_value`` is the part of
the observed number the structure accounts for.

Two rules do the work here, and both exist because the obvious shortcut is wrong.

**The reference run is the caller's own call with two arguments changed.** Not a call at
defaults. ``groupby``, ``splitby``, ``clones``, ``weighted``, ``normalized``, ``normalize_mode``,
``n_clones_ref``, ``distance_metric``, ``temperature`` and ``n_samples`` each change the
estimand, so a reference computed at defaults is a different quantity subtracted from a
different quantity. The arguments are taken from the decorator's own ``bind``, so the forwarded
set cannot drift from the signature.

**``excess`` is a difference of two summaries, never a summary of a difference.** Draws are
never paired across two fits -- there is no correspondence between the parent's draw 7 and the
null's -- so the excess carries no ``sd`` and no interval, and nothing in this module produces
one.
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


def excess_column(value: str) -> str:
    """``value`` -> ``excess``, ``value_from`` -> ``excess_from``, ``value_to`` -> ``excess_to``."""
    if not value.startswith("value"):
        raise ValueError(f"a native value column must start with 'value', got {value!r}")
    return "excess" + value[len("value"):]


def column_pairs(values, denominators=()):
    """``(native, null, excess)`` per native value column; ``excess`` is ``None`` for a
    denominator, which has a reference but no excess -- a difference of two normalisers is not
    a quantity anyone reports."""
    return [(v, null_column(v), None if v in denominators else excess_column(v)) for v in values]


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
    names, so a user column called `excess_patient` or `null_arm` is a perfectly legitimate row
    label; excluding it by prefix drops a join key and the merge fans out. Measured: a groupby
    named `excess_patient` took a 24-row result to 144.
    """
    drop = set(values) | set(SUMMARY_COLUMNS) | set(reference_columns)
    return [c for c in result.columns if c not in drop]


def attach(payload, reference, *, values, denominators=()):
    """Add the reference and difference columns to ``table`` and ``result``, in place.

    The reference frame is joined on the row LABELS, never positionally: a null's rows are the
    parent's by construction (§3.1), but a join says so and an alignment by position only
    assumes it. A label with no reference row gets NaN rather than being dropped, so the
    caller's own rows stay authoritative.
    """
    result, ref_result = payload.get("result"), reference.get("result")
    if result is None or ref_result is None or not len(result) or not len(ref_result):
        return payload

    present = [v for v in values if v in result.columns and v in ref_result.columns]
    if not present:
        return payload
    pairs = column_pairs(present, denominators)
    created = [c for _, n, e in pairs for c in (n, e) if c]
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
    for native, null, excess in pairs:
        if excess is not None:
            merged[excess] = merged[native].to_numpy(dtype=float) - merged[null].to_numpy(dtype=float)
    payload["result"] = merged

    # `table` carries them too, broadcast to every draw: the violin path reads `table` rather
    # than `result`, so without this `quantity="excess"` is a seaborn KeyError on any metric
    # whose coarsest varying unit is the draw.
    table = payload.get("table")
    if table is not None and len(table):
        cols = [c for _, c, _ in pairs] + [e for _, _, e in pairs if e is not None]
        keep = [c for c in labels if c in table.columns]
        if keep:
            payload["table"] = table.merge(merged[keep + cols].drop_duplicates(keep),
                                           on=keep, how="left")
        elif len(merged) == 1:
            payload["table"] = table.assign(**{c: merged[c].iloc[0] for c in cols})
    return payload


def restat(payload, *, groupby, splitby, values, denominators=(), per_gene=False):
    """Recompute ``stats`` over the value AND its excess, under one collapse.

    Replaces the body's own single-quantity frame rather than appending to it, because the two
    have to come from the same ``collapse_to_replicates`` call: a second collapse would drop a
    different non-finite set and the two rows of one frame would describe different replicate
    sets while looking directly comparable.
    """
    if "stats" not in payload:
        return payload
    result = payload.get("result")
    if result is None or not len(result) or splitby is None:
        return payload
    quantities = ["value"] + [e for _, _, e in column_pairs(values, denominators)
                              if e is not None and e in result.columns]
    # A quantity that is non-finite on every row would empty the SHARED collapse mask and take
    # the value rows down with it, so the payload would lose a stats frame it legitimately had.
    # Dropping it here keeps the frame the body already built.
    quantities = [q for q in dict.fromkeys(quantities)
                  if q in result.columns and np.isfinite(
                      pd.to_numeric(result[q], errors="coerce").to_numpy(dtype=float)).any()]
    if len(quantities) < 2:
        return payload
    if per_gene:
        from ..perturbation._tables import stats_per_gene
        rebuilt = stats_per_gene(result, groupby=groupby, splitby=splitby, value=quantities)
    else:
        from .._compute._tables import build_stats
        rebuilt = build_stats(result, groupby=groupby, splitby=splitby, value=quantities)
    # Never replace a frame with nothing: the body's own single-quantity frame is still correct
    # and still carries its `quantity` column.
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
    if quantity != "value":
        return f"{ylabel} - null"
    return ylabel if "null_value" in result.columns else f"{ylabel} (no reference)"


def name_of(reference_of):
    """The h5ad-writable name of whatever the reference was computed on.

    A fit name passes through; a model object becomes its own name, because a model in the
    params block would break the `.h5ad` write and tell a later reader nothing it can act on.
    """
    if reference_of is None or isinstance(reference_of, str):
        return reference_of
    return str(getattr(reference_of, "name", "") or "model")
