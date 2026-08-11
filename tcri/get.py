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

from ._state import keys as K
from ._state.storage import load_result, load_result_params

__all__ = [
    "result",
    "params",
    "joint_distribution",
    "mutual_information",
    "clonotypic_entropy",
    "phenotypic_entropy",
    "phenotypic_flux",
    "table",
]

#: The decoded payload IS the table (no sub-key to pull).
_SELF = object()

#: tool name -> (canonical uns key, the sub-key holding the primary tidy table)
_RESULTS = {
    "joint_distribution": (K.JOINT_DISTRIBUTION, "result"),
    "mutual_information": (K.MUTUAL_INFORMATION, "result"),
    "clonotypic_entropy": (K.CLONOTYPIC_ENTROPY, "result"),
    "phenotypic_entropy": (K.PHENOTYPIC_ENTROPY, "result"),
    "phenotypic_flux": (K.PHENOTYPIC_FLUX, "result"),
}

_PROVENANCE = ("params", "version")


def _resolve_key(name: str, key):
    """Explicit ``key`` wins; then the registry; then a raw ``tcri_``-prefixed uns key."""
    if key is not None:
        return key
    if name in _RESULTS:
        return _RESULTS[name][0]
    if isinstance(name, str) and name.startswith("tcri_"):
        return name
    raise KeyError(
        f"unknown tcri result {name!r}; expected one of {sorted(_RESULTS)} or a 'tcri_*' "
        f"uns key (pass key= for a custom key_added)"
    )


def _require(adata, name, key):
    """Resolve the uns key, or raise naming the exact call that would fill it.

    With ``pl`` reading the cache instead of recomputing, "I plotted before I computed" is
    now the most common way to get this wrong — so the message has to be the fix, not a
    description of the problem. ``run the matching tcri.tl tool first`` was neither.
    """
    resolved = _resolve_key(name, key)
    if resolved not in adata.uns:
        call = f"tcri.tl.{name}(adata, ...)" if name in _RESULTS else f"the tool writing {resolved!r}"
        suffix = f", key_added={key!r}" if key is not None else ""
        raise KeyError(
            f"adata.uns[{resolved!r}] not found. Run {call}{suffix} first — "
            f"tcri.pl.* renders the stored result and never recomputes it."
        )
    return resolved


def result(adata, name: str, *, key=None):
    """The cached result, exactly as the ``tl`` function returned it.

    Strips ``params``/``version``, which ``load_result`` carries through for dict payloads but
    not for DataFrame ones — normalising that asymmetry is most of this function's job.
    """
    payload = load_result(adata, _require(adata, name, key))
    if isinstance(payload, dict):
        return {k: v for k, v in payload.items() if k not in _PROVENANCE}
    return payload


def params(adata, name: str, *, key=None) -> dict:
    """The provenance block: every argument the tool ran with, including untouched defaults."""
    return load_result_params(adata, _require(adata, name, key))


def table(adata, name: str, *, key=None, which: str = "result"):
    """A named payload frame from a cached result.

    ``which="result"`` (default) is the reduced, per-group frame the plots consume;
    ``which="table"`` is the unreduced substrate, one row per (covariate, group, item[, draw]).
    """
    payload = load_result(adata, _require(adata, name, key))
    subkey = _RESULTS[name][1] if name in _RESULTS else which
    if subkey is _SELF:
        return payload
    if not isinstance(payload, dict) or which not in payload:
        raise KeyError(
            f"cached result for {name!r} has no {which!r} frame "
            f"(present: {sorted(k for k in payload if k not in _PROVENANCE)})"
            if isinstance(payload, dict) else
            f"cached result for {name!r} is not a dict payload"
        )
    return payload[which]


def joint_distribution(adata, *, key=None, which: str = "result"):
    return table(adata, "joint_distribution", key=key, which=which)


def mutual_information(adata, *, key=None, which: str = "result"):
    return table(adata, "mutual_information", key=key, which=which)


def clonotypic_entropy(adata, *, key=None, which: str = "result"):
    return table(adata, "clonotypic_entropy", key=key, which=which)


def phenotypic_entropy(adata, *, key=None, which: str = "result"):
    return table(adata, "phenotypic_entropy", key=key, which=which)


def phenotypic_flux(adata, *, key=None, which: str = "result"):
    return table(adata, "phenotypic_flux", key=key, which=which)
