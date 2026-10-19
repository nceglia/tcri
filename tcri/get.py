"""Accessors for cached ``tl`` results — ``tcri.get.*``.

Every ``tl`` metric stores its result under a namespaced ``uns`` key. This module is how
everything else reads it back without knowing the blob format: ``pl`` renders from here, and so
should any user code that wants the frame a plot was drawn from.

A ``tl`` call stores its result and returns ``None``; this module is how it is read. Mutual
information, the two entropies and ``perturb.gene_importance`` store their result by level,
``{replicate, item, stats}``, each level holding ``draws`` and ``summary``:

    tcri.tl.phenotypic_entropy(adata, covariate="pre", splitby="response")
    tcri.get.replicates(adata, "phenotypic_entropy")  # one row per replicate
    tcri.get.items(adata, "phenotypic_entropy")       # one row per replicate and clone
    tcri.get.stats(adata, "phenotypic_entropy")       # the tests
    tcri.get.draws(adata, "phenotypic_entropy")       # replicate rows, one per draw

:func:`result` returns the whole stored result, provenance stripped, and :func:`params` the
provenance. ``joint_distribution`` stores ``{summary, draws}``: its summary is read with
:func:`result` and its draws with :func:`draws`.
"""
from __future__ import annotations

# Aliased private: this module IS the public accessor namespace, so anything bound at module
# scope becomes tcri.get.<name>. Without the underscores, `dir(tcri.get)` would advertise K,
# load_result and load_result_params -- the storage internals the accessors exist to hide.
from ._state import keys as _K
from ._state.storage import load_result as _load_result, load_result_params as _load_result_params

__all__ = ["result", "params", "replicates", "items", "stats", "draws", "fits"]

#: tool name -> canonical uns key
_RESULTS = {
    "joint_distribution": _K.JOINT_DISTRIBUTION,
    "mutual_information": _K.MUTUAL_INFORMATION,
    "clonotypic_entropy": _K.CLONOTYPIC_ENTROPY,
    "phenotypic_entropy": _K.PHENOTYPIC_ENTROPY,
    "phenotypic_flux": _K.PHENOTYPIC_FLUX,
    "delta_clonotypic_entropy": _K.DELTA_CLONOTYPIC_ENTROPY,
    "delta_phenotypic_entropy": _K.DELTA_PHENOTYPIC_ENTROPY,
    "gene_importance": _K.GENE_IMPORTANCE,
}

#: The call that fills a key, for results that are not ``tcri.tl.<name>(adata, ...)``.
#: ``perturb`` queries take the fitted model first.
_CALLS = {
    "gene_importance": "tcri.perturb.gene_importance(model, adata, ...)",
}

#: Written beside every result; stripped by :func:`result`, which returns the stored result, the
#: same one ``inplace=False`` returns.
_PROVENANCE = ("params", "version", "tool", "tcri_version")


def fits(adata) -> list:
    """The named fits this object carries, beside the main one."""
    return _K.fits(adata)


def _resolve_key(name: str, key, fit=None):
    """Explicit ``key`` wins; then the registry; then a raw ``tcri_``-prefixed uns key.

    ``fit`` suffixes whichever base is chosen, never only the default: a metric computed on a
    named fit with a ``key_added`` of its own lands at ``<key_added>_<fit>``, so a result on a
    reference can never overwrite the main fit's cached result.
    """
    if key is not None:
        return _K.fit_key(key, fit)
    if name in _RESULTS:
        return _K.fit_key(_RESULTS[name], fit)
    if isinstance(name, str) and name.startswith("tcri_"):
        return _K.fit_key(name, fit)
    raise KeyError(
        f"unknown tcri result {name!r}; expected one of {sorted(_RESULTS)} or a 'tcri_*' "
        f"uns key (pass key= for a custom key_added)"
    )


def _require(adata, name, key, fit=None):
    """Resolve the uns key, or raise naming the exact call that would fill it.

    With ``pl`` reading the cache instead of recomputing, "I plotted before I computed" is
    the most common way to get this wrong, so the message names the exact call that fills the
    key.
    """
    fit = _K.resolve_fit(adata, fit)
    resolved = _resolve_key(name, key, fit)
    if resolved not in adata.uns:
        if name in _CALLS:
            call = _CALLS[name]
        elif name in _RESULTS:
            call = f"tcri.tl.{name}(adata, ...)"
        else:
            call = f"the tool writing {resolved!r}"
        suffix = f", key_added={key!r}" if key is not None else ""
        suffix += f", fit={fit!r}" if fit is not None else ""
        raise KeyError(
            f"adata.uns[{resolved!r}] not found. Run {call}{suffix} first — "
            f"tcri.pl.* renders the stored result and never recomputes it."
        )
    return resolved


def result(adata, name: str, *, key=None, fit=None):
    """The stored result, provenance stripped: ``{replicate, item, stats}`` for a metric stored
    by level, the payload as stored otherwise. With ``inplace=False`` the tool returns this same
    result instead of storing it.

    ``fit`` selects which fit's RESULT BLOB to read, which is a different question from the
    ``fit`` a metric takes: that one selects which fit the number is computed on. They resolve
    through the same :func:`~tcri._state.keys.fit_key`, which is why one spelling serves both.
    """
    payload = _load_result(adata, _require(adata, name, key, fit), tool=_RESULTS.get(name))
    if isinstance(payload, dict):
        return {k: v for k, v in payload.items() if k not in _PROVENANCE}
    return payload


def params(adata, name: str, *, key=None, fit=None) -> dict:
    """The provenance block: every argument the tool ran with, including untouched defaults."""
    return _load_result_params(adata, _require(adata, name, key, fit))


def _level(adata, name, level, *, key=None, fit=None):
    return _pick(result(adata, name, key=key, fit=fit), name, level)


def _pick(payload, name, level):
    if not isinstance(payload, dict) or "replicate" not in payload:
        raise KeyError(
            f"the stored {name!r} result is not stored by level; read it with "
            f"tcri.get.result(adata, {name!r})."
        )
    if payload.get(level) is None:
        raise KeyError(
            f"{name!r} has no {level} level: it has one value per replicate and no items."
        )
    return payload[level]


def replicates(adata, name: str, *, key=None, fit=None):
    """The replicate summary of a stored result: one row per (covariate, replicate), with the
    reference columns when a reference was computed.

    Raises ``KeyError`` for a result not stored by level.
    """
    return _level(adata, name, "replicate", key=key, fit=fit)["summary"]


def items(adata, name: str, *, key=None, fit=None):
    """The item summary of a stored result: one row per (covariate, group, item), with the
    reference columns when a reference was computed.

    Raises ``KeyError`` for a metric with no items (mutual information).
    """
    return _level(adata, name, "item", key=key, fit=fit)["summary"]


def stats(adata, name: str, *, key=None, fit=None):
    """The tests of a stored result, one row per contrast, or ``None`` when nothing was
    compared."""
    return result(adata, name, key=key, fit=fit).get("stats")


def draws(adata, name: str, *, items: bool = False, key=None, fit=None):
    """The per-draw rows of a stored result: the replicate level, or the item level with
    ``items=True``. With a reference, one row per (row, null, draw).

    For ``joint_distribution``, its ``draws`` frame, which has no levels; ``items=True`` raises
    ``KeyError`` there.
    """
    payload = result(adata, name, key=key, fit=fit)
    if isinstance(payload, dict) and "replicate" not in payload and "draws" in payload:
        if items:
            raise KeyError(f"{name!r} has no item level: its draws are one frame.")
        return payload["draws"]
    return _pick(payload, name, "item" if items else "replicate")["draws"]
