"""Accessors for cached ``tl`` results — ``tcri.get.*``.

Every ``tl`` metric stores its result under a namespaced ``uns`` key and returns the same
object. This module is how everything else reads it back without knowing the blob format:
``pl`` renders from here, and so should any user code that wants the table a plot was drawn
from.

The invariant: :func:`result` reconstructs exactly what the ``tl`` function **returned** —
provenance stripped — so

    res = tcri.tl.mutual_information(adata, covariate="pre", groupby="patient")
    assert res == tcri.get.result(adata, "mutual_information")

holds for every metric. :func:`params` reads the provenance separately.
"""
from __future__ import annotations

# Aliased private: this module IS the public accessor namespace, so anything bound at module
# scope becomes tcri.get.<name>. Without the underscores, `dir(tcri.get)` advertised K,
# load_result and load_result_params -- the storage internals the accessors exist to hide.
from ._state import keys as _K
from ._state.storage import load_result as _load_result, load_result_params as _load_result_params

__all__ = [
    "result",
    "params",
    "joint_distribution",
    "mutual_information",
    "clonotypic_entropy",
    "phenotypic_entropy",
    "phenotypic_flux",
    "delta_clonotypic_entropy",
    "delta_phenotypic_entropy",
    "gene_importance",
    "table",
    "fits",
]

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

_PROVENANCE = ("params", "version")


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
    now the most common way to get this wrong — so the message has to be the fix, not a
    description of the problem. ``run the matching tcri.tl tool first`` was neither.
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
    """The cached result, exactly as the ``tl`` function returned it.

    Strips ``params``/``version``, which ``load_result`` carries through for dict payloads but
    not for DataFrame ones — normalising that asymmetry is most of this function's job.

    ``fit`` selects which fit's RESULT BLOB to read, which is a different question from the
    ``fit`` a metric takes: that one selects which fit the number is computed on. They resolve
    through the same :func:`~tcri._state.keys.fit_key`, which is why one spelling serves both.
    """
    payload = _load_result(adata, _require(adata, name, key, fit))
    if isinstance(payload, dict):
        return {k: v for k, v in payload.items() if k not in _PROVENANCE}
    return payload


def params(adata, name: str, *, key=None, fit=None) -> dict:
    """The provenance block: every argument the tool ran with, including untouched defaults."""
    return _load_result_params(adata, _require(adata, name, key, fit))


def table(adata, name: str, *, key=None, which: str = "result", fit=None):
    """A named payload frame from a cached result.

    ``which="result"`` (default) is the reduced, per-group frame the plots consume;
    ``which="table"`` is the unreduced substrate, one row per (covariate, group, item[, draw]).
    """
    payload = _load_result(adata, _require(adata, name, key, fit))
    if not isinstance(payload, dict) or which not in payload:
        raise KeyError(
            f"cached result for {name!r} has no {which!r} frame "
            f"(present: {sorted(k for k in payload if k not in _PROVENANCE)})"
            if isinstance(payload, dict) else
            f"cached result for {name!r} is not a dict payload"
        )
    return payload[which]


def joint_distribution(adata, *, key=None, which: str = "result", fit=None):
    return table(adata, "joint_distribution", key=key, which=which, fit=fit)


def mutual_information(adata, *, key=None, which: str = "result", fit=None):
    return table(adata, "mutual_information", key=key, which=which, fit=fit)


def clonotypic_entropy(adata, *, key=None, which: str = "result", fit=None):
    return table(adata, "clonotypic_entropy", key=key, which=which, fit=fit)


def phenotypic_entropy(adata, *, key=None, which: str = "result", fit=None):
    return table(adata, "phenotypic_entropy", key=key, which=which, fit=fit)


def phenotypic_flux(adata, *, key=None, which: str = "result", fit=None):
    return table(adata, "phenotypic_flux", key=key, which=which, fit=fit)


def delta_clonotypic_entropy(adata, *, key=None, which: str = "result", fit=None):
    return table(adata, "delta_clonotypic_entropy", key=key, which=which, fit=fit)


def delta_phenotypic_entropy(adata, *, key=None, which: str = "result", fit=None):
    return table(adata, "delta_phenotypic_entropy", key=key, which=which, fit=fit)


def gene_importance(adata, *, key=None, which: str = "result", fit=None):
    """``which`` may also be ``"shift"``: the per-phenotype decomposition of each importance."""
    return table(adata, "gene_importance", key=key, which=which, fit=fit)
