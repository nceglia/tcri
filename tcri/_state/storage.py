"""Result-storage convention for ``tl`` functions — realized once as a decorator.

The contract: every ``tl`` writes ``uns[key]`` as a **dict-of-arrays + a ``params`` block + a
``version`` int** (the scanpy ``rank_genes_groups`` pattern) and **returns** the natural result.
``@tl_result`` is that convention as code, so the metrics cannot drift from it.

Why this exists rather than each tool writing its own ``uns`` entry: every ``pl.*`` used to take
``adata`` and recompute the metric internally, so a plot could silently disagree with the table
the user had in hand. Storing once and reading from the cache makes that impossible. The
recompute is also why ``pl.phenotypic_flux`` manufactured a ``groupby`` from ``batch_col`` — it
could only consume the tidy frame, which only existed when ``groupby`` was set.

Usage — the wrapped function declares the full contract signature (including ``key_added`` /
``inplace``, for signature conformance) and returns its *natural* result. The decorator owns
storage; the body does not::

    @tl_result(key=K.MUTUAL_INFORMATION, version=1, schema=schemas.MutualInformation)
    def mutual_information(adata, *, covariate, groupby=None, ...,
                           key_added=None, inplace=True):
        ...
        return {"table": table, "result": result}

The decorator reads ``inplace`` / ``key_added`` from the call, optionally checks the result
against a ``schema`` TypedDict, auto-captures the remaining arguments as ``params``, serializes
to an ``.h5ad``-safe dict-of-arrays blob, writes it to ``uns[key_added or key]`` when
``inplace``, and returns the natural result. :func:`decode_blob` / :func:`load_result` invert it
for the ``tcri.get.*`` accessors.

A body that *resolves* a param the caller left implicit (``groupby=None`` -> the registry's
``replicate``) records the effective value with :func:`with_resolved_params`, so the cached
provenance — and every ``pl`` / ``tcri.get`` reader of it — sees the real column rather than the
``None`` placeholder::

    return with_resolved_params({"table": table, "result": result}, groupby=gkey)

**h5ad safety.** h5py treats ``/`` as a path separator, so a clonotype id like ``"TRB/1"`` used
as a dict key would silently split into nested groups on write. The encoder therefore never
writes a user label as a dict key: DataFrame columns are stored positionally with the labels as
array *values*, and a label-keyed dict is tagged and stored as parallel key/value arrays.
"""
from __future__ import annotations

import functools
import inspect

import numpy as np
import pandas as pd

__all__ = [
    "tl_result",
    "decode_blob",
    "load_result",
    "load_result_params",
    "with_resolved_params",
]

#: Never recorded as provenance. The data argument is excluded by POSITION (see ``tl_result``)
#: rather than by name, so a tool is free to call its first parameter whatever fits.
_RESERVED = {"key_added", "inplace"}

_DF = "__tcri_df__"
_SERIES = "__tcri_series__"
_MAP = "__tcri_map__"

#: Tag a body attaches (via :func:`with_resolved_params`) to carry effective values.
_RESOLVED_PARAMS = "__tcri_resolved_params__"


def _key_safe(k) -> bool:
    """Is ``k`` usable as an h5py group name?

    ``/`` is a path separator, so a user label containing one would silently become nested
    groups. Empty, ``.`` and ``..`` are also reserved.
    """
    return isinstance(k, str) and "/" not in k and k not in ("", ".", "..")


def _encode(obj):
    """Convert a result into an ``.h5ad``-safe structure.

    User labels are never used as dict KEYS — only as array values. Everything not explicitly
    handled (ndarray, str, int, float, None, list) passes through untouched, because anndata
    already round-trips those.
    """
    if isinstance(obj, pd.DataFrame):
        return {
            _DF: 1,
            "index": obj.index.to_numpy(),
            "index_name": obj.index.name,
            # positional column storage: labels live as VALUES, so "/" is harmless and
            # duplicate labels survive
            "columns": np.asarray([str(c) for c in obj.columns], dtype=object),
            "columns_name": obj.columns.name,
            "data": {str(i): obj.iloc[:, i].to_numpy() for i in range(obj.shape[1])},
        }
    if isinstance(obj, pd.Series):
        return {
            _SERIES: 1,
            "index": obj.index.to_numpy(),
            "index_name": obj.index.name,
            "values": obj.to_numpy(),
            "name": obj.name,
        }
    if isinstance(obj, dict):
        if all(_key_safe(k) for k in obj):
            return {k: _encode(v) for k, v in obj.items()}
        keys = list(obj)
        return {
            _MAP: 1,
            "keys": np.asarray([str(k) for k in keys], dtype=object),
            "values": {str(i): _encode(obj[k]) for i, k in enumerate(keys)},
        }
    return obj


def _decode(obj):
    """Exact inverse of :func:`_encode`."""
    if isinstance(obj, dict):
        if obj.get(_DF):
            cols = [str(c) for c in obj["columns"]]
            data = obj["data"]
            frame = pd.DataFrame(
                {i: np.asarray(data[str(i)]) for i in range(len(cols))},
                index=pd.Index(np.asarray(obj["index"]), name=obj.get("index_name")),
            )
            # assign AFTER construction so duplicate labels survive
            frame.columns = pd.Index(cols, name=obj.get("columns_name"))
            return frame
        if obj.get(_SERIES):
            return pd.Series(
                np.asarray(obj["values"]),
                index=pd.Index(np.asarray(obj["index"]), name=obj.get("index_name")),
                name=obj.get("name"),
            )
        if obj.get(_MAP):
            keys = [str(k) for k in obj["keys"]]
            values = obj["values"]
            return {k: _decode(values[str(i)]) for i, k in enumerate(keys)}
        return {k: _decode(v) for k, v in obj.items()}
    return obj


def decode_blob(blob: dict):
    """Decode a stored blob back to the tool's natural result (``params``/``version`` kept)."""
    return _decode(blob)


def load_result(adata, key: str):
    """Read and decode ``adata.uns[key]``; raise if absent.

    ``tcri.get`` and every ``pl`` cache renderer go through here, so the "run the tool first"
    message is written once.
    """
    if key not in adata.uns:
        raise KeyError(
            f"adata.uns[{key!r}] not found — run the matching tcri.tl tool first."
        )
    return decode_blob(adata.uns[key])


def load_result_params(adata, key: str, default=None) -> dict:
    """Read the provenance ``params`` block for a cached ``tl`` result.

    Raises on a missing KEY but returns ``default`` on a missing params block — the asymmetry is
    deliberate: a blob written by an older version still renders, an absent tool does not.
    """
    if key not in adata.uns:
        raise KeyError(
            f"adata.uns[{key!r}] not found — run the matching tcri.tl tool first."
        )
    blob = adata.uns[key]
    fallback = {} if default is None else default
    if not isinstance(blob, dict) or "params" not in blob:
        return fallback
    params = _decode(blob["params"])
    return params if isinstance(params, dict) else fallback


def with_resolved_params(result: dict, **resolved) -> dict:
    """Tag a ``tl`` result with the EFFECTIVE values of params the caller left implicit.

    Without this, ``params`` records ``groupby=None`` even when the tool resolved it to the
    registry's ``replicate`` column — and every reader of that provenance (``pl``, ``tcri.get``,
    a user six months later) sees a placeholder instead of the column actually used.
    """
    return {**result, _RESOLVED_PARAMS: resolved}


def _check_schema(schema, result, fn_name: str) -> None:
    """Presence-only check of a TypedDict's required keys.

    Reads ``__required_keys__`` off the class so this module has no import dependency on
    ``schemas``. DataFrame results are not checked — their columns are asserted per-tool.
    """
    required = set(getattr(schema, "__required_keys__", None) or schema.__annotations__)
    missing = required - set(result)
    if missing:
        raise ValueError(f"{fn_name}: result missing required keys {sorted(missing)}")


def tl_result(*, key: str, version: int = 1, schema=None):
    """Store the wrapped ``tl``'s result under ``uns[key_added or key]`` and return it.

    ``functools.wraps`` keeps the wrapped signature, so the ``.pyi`` conformance check sees the
    contract signature exactly — which is why every wrapped body must DECLARE ``key_added`` and
    ``inplace`` even though it never reads them.

    ``params`` captures every declared argument except :data:`_RESERVED`, **including defaults
    the caller never passed**, via ``bind`` + ``apply_defaults``. That is the point: provenance
    that only records explicit arguments cannot answer "what was this run with".
    """
    def deco(fn):
        sig = inspect.signature(fn)
        # the data argument, whatever it is called
        data_param = next(iter(sig.parameters))

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            adata = arguments[data_param]

            result = fn(*args, **kwargs)

            # pop BEFORE the schema check so the tag never trips required-key validation
            # and never leaks into the returned object
            resolved = (result.pop(_RESOLVED_PARAMS, None)
                        if isinstance(result, dict) else None)
            if schema is not None and isinstance(result, dict):
                _check_schema(schema, result, fn.__name__)

            if arguments.get("inplace", True):
                blob = _encode(result)
                if not isinstance(blob, dict):
                    blob = {"value": blob}
                params = {k: v for k, v in arguments.items()
                          if k not in _RESERVED and k != data_param}
                if resolved:
                    params.update(resolved)
                blob = {**blob, "params": _encode(params), "version": int(version)}
                adata.uns[arguments.get("key_added") or key] = blob
            return result

        return wrapper

    return deco
