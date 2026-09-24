"""Result-storage convention for ``tl`` functions — realized once as a decorator.

The contract: every ``tl`` writes ``uns[key]`` as a **dict-of-arrays + a provenance block**
(``params``, the schema ``version`` int, the writing ``tool`` and the ``tcri_version`` that wrote
it; the scanpy ``rank_genes_groups`` pattern) and **returns** the natural result.
``@tl_result`` is that convention as code, so the metrics cannot drift from it.

Why this exists rather than each tool writing its own ``uns`` entry: the metric is computed and
stored once, and every ``pl.*`` reads that cache instead of recomputing, so a plot cannot
disagree with the table the user holds.

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
import importlib.metadata
import inspect
from typing import Callable

import numpy as np
import pandas as pd

from . import _reference
from . import keys as _keys

__all__ = [
    "tl_result",
    "decode_blob",
    "load_result",
    "load_result_params",
    "with_resolved_params",
]

#: Canonical ``uns`` key -> the wrapped ``tl`` that writes it. Only tools defined in ``tcri`` are
#: registered, so a tool defined in a test cannot displace a real one.
_REGISTRY: dict[str, Callable] = {}


def _tcri_version() -> str:
    """The installed package version, recorded in every result it writes."""
    try:
        return importlib.metadata.version("tcri")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+unknown"



#: Never recorded as provenance. The data argument is excluded by POSITION (see ``tl_result``)
#: rather than by name, so a tool is free to call its first parameter whatever fits.
_RESERVED = {"key_added", "inplace"}

_DF = "__tcri_df__"
_SERIES = "__tcri_series__"
_MAP = "__tcri_map__"
_MULTI = "__tcri_multiindex__"

#: Tag a body attaches (via :func:`with_resolved_params`) to carry effective values.
_RESOLVED_PARAMS = "__tcri_resolved_params__"


def _key_safe(k) -> bool:
    """Is ``k`` usable as an h5py group name?

    ``/`` is a path separator, so a user label containing one would silently become nested
    groups. Empty, ``.`` and ``..`` are also reserved.
    """
    return isinstance(k, str) and "/" not in k and k not in ("", ".", "..")


def _encode_index(index):
    """Index -> h5ad-safe arrays, flattening a MultiIndex into one array per level.

    ``index.to_numpy()`` on a MultiIndex yields an object array of TUPLES, and h5py has no
    writer for a tuple -- it fails with "Can't implicitly convert non-string objects to
    strings", named against the enclosing group rather than the index, which is a long way
    from the cause. ``joint_distribution(n_samples>0)`` produces exactly that: a
    (clonotype, sample_id) MultiIndex on its `table`. Flattening it level by level keeps the
    result writable to .h5ad, which is what this module exists to guarantee.
    """
    if isinstance(index, pd.MultiIndex):
        return {
            _MULTI: 1,
            "levels": {str(i): np.asarray(index.get_level_values(i))
                       for i in range(index.nlevels)},
            "level_names": np.asarray([("" if n is None else str(n)) for n in index.names],
                                      dtype=object),
        }
    return {"index": index.to_numpy(), "index_name": index.name}


def _decode_index(obj):
    """Inverse of :func:`_encode_index`."""
    if obj.get(_MULTI):
        levels = obj["levels"]
        names = [n or None for n in obj["level_names"]]
        arrays = [np.asarray(levels[str(i)]) for i in range(len(names))]
        return pd.MultiIndex.from_arrays(arrays, names=names)
    return pd.Index(np.asarray(obj["index"]), name=obj.get("index_name"))


def _encode(obj):
    """Convert a result into an ``.h5ad``-safe structure.

    User labels are never used as dict KEYS — only as array values. Everything not explicitly
    handled (ndarray, str, int, float, None, list) passes through untouched, because anndata
    already round-trips those.
    """
    if isinstance(obj, pd.DataFrame):
        return {
            _DF: 1,
            **_encode_index(obj.index),
            # positional column storage: labels live as VALUES, so "/" is harmless and
            # duplicate labels survive
            "columns": np.asarray([str(c) for c in obj.columns], dtype=object),
            "columns_name": obj.columns.name,
            "data": {str(i): obj.iloc[:, i].to_numpy() for i in range(obj.shape[1])},
        }
    if isinstance(obj, pd.Series):
        return {
            _SERIES: 1,
            **_encode_index(obj.index),
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
                index=_decode_index(obj),
            )
            # assign AFTER construction so duplicate labels survive
            frame.columns = pd.Index(cols, name=obj.get("columns_name"))
            return frame
        if obj.get(_SERIES):
            return pd.Series(
                np.asarray(obj["values"]),
                index=_decode_index(obj),
                name=obj.get("name"),
            )
        if obj.get(_MAP):
            keys = [str(k) for k in obj["keys"]]
            values = obj["values"]
            return {k: _decode(values[str(i)]) for i, k in enumerate(keys)}
        return {k: _decode(v) for k, v in obj.items()}
    return obj


def decode_blob(blob: dict):
    """Decode a stored blob back to the tool's natural result (the provenance keys kept)."""
    return _decode(blob)


def load_result(adata, key: str, *, tool: str | None = None):
    """Read and decode ``adata.uns[key]``; raise if absent.

    ``tcri.get`` and every ``pl`` cache renderer go through here, so the "run the tool first"
    message is written once.

    The stored schema ``version`` decides what happens next. Results written before 0.13 carry no
    ``tool``, so ``tcri.get`` passes the one it resolved the key from; without it the blob is
    decoded as it stands. A result from a NEWER schema than this tcri knows is refused rather than
    half-read, and an older one is refused with a request to recompute it.
    """
    if key not in adata.uns:
        raise KeyError(
            f"adata.uns[{key!r}] not found — run the matching tcri.tl tool first."
        )
    blob = adata.uns[key]
    if not isinstance(blob, dict):
        return decode_blob(blob)
    tool = blob.get("tool") or tool
    if tool not in _REGISTRY:
        return decode_blob(blob)
    stored = int(blob.get("version", 1))
    current = _REGISTRY[tool].tcri_schema_version
    if stored > current:
        raise ValueError(
            f"adata.uns[{key!r}] was written by tcri {blob.get('tcri_version', '?')} with "
            f"{tool} schema v{stored}; this tcri ({_tcri_version()}) reads up to v{current}. "
            f"Upgrade tcri to read it."
        )
    if stored < current:
        raise ValueError(
            f"adata.uns[{key!r}] uses {tool} schema v{stored}, which this tcri can no longer "
            f"read (current v{current}). Recompute it with tcri."
        )
    return decode_blob(blob)


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


def tl_result(*, key: str, version: int = 1, schema=None, data_param: str | None = None,
              values=("value",), denominators=(), default_null=None, reference_arg="fit",
              per_gene_stats=False):
    """Store the wrapped ``tl``'s result under ``uns[key_added or key]`` and return it.

    ``functools.wraps`` keeps the wrapped signature, so the ``.pyi`` conformance check sees the
    contract signature exactly — which is why every wrapped body must DECLARE ``key_added`` and
    ``inplace`` even though it never reads them.

    ``params`` captures every declared argument except :data:`_RESERVED`, **including defaults
    the caller never passed**, via ``bind`` + ``apply_defaults``. That is the point: provenance
    that only records explicit arguments cannot answer "what was this run with".

    ``data_param`` names the AnnData argument when it is not the first parameter. A tool that
    takes the fitted model first (``perturb.gene_importance(model, adata, ...)``) stores into
    ``adata``; every parameter *before* the data argument is an object the tool operates on,
    not a setting, and is excluded from ``params`` with it. Otherwise the model itself would be
    recorded as provenance and the ``.h5ad`` write would fail on it.

    **The reference.** When the wrapped function declares ``null_model``, the decorator runs it
    a second time against a permutation null and adds the ``null_*``/``excess*`` columns
    (:mod:`tcri._state._reference`). It lives here rather than in each body for one reason that
    is not tidiness: the reference has to be *the caller's own call with two arguments changed*,
    and only the decorator holds the caller's bound arguments. A body re-invoking itself would
    have to re-list what to forward, and that list would drift from the signature the first time
    a knob was added.

    ``values`` names the metric's NATIVE value columns, before any reference column exists --
    ``("value",)`` for the scalar metrics, ``("value", "value_from", "value_to")`` for the two
    deltas, ``("value", "denom")`` for mutual information. ``denominators`` names those that get
    a reference but no excess. ``default_null`` is the metric's entry in the contract's
    ``DEFAULT_NULL`` table, read back off the live function by the conformance test so the two
    cannot drift. ``reference_arg`` is the parameter the reference run substitutes: ``"fit"``
    for a metric that reads a substrate, or the name of the model parameter for a query on a
    fitted model, which needs the null's networks rather than its arrays.
    """
    def deco(fn):
        sig = inspect.signature(fn)
        names = list(sig.parameters)
        takes_null = "null_model" in names
        takes_fit = "fit" in names
        if data_param is None:
            # the data argument, whatever it is called
            _data_param, leading = names[0], set()
        else:
            if data_param not in names:
                raise TypeError(f"{fn.__name__} has no parameter {data_param!r}")
            _data_param = data_param
            leading = set(names[: names.index(data_param)])

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            adata = arguments[_data_param]

            # The fit is CANONICALISED before anything reads it, so `fit="phenotype"` and
            # `fit="null.phenotype"` are one call rather than two `uns` keys and two params
            # blocks. `resolve_fit` is idempotent on the canonical name, which is what makes
            # doing it here safe.
            if takes_fit and arguments.get("fit") is not None:
                arguments["fit"] = _keys.resolve_fit(adata, arguments["fit"])
                result = fn(**dict(arguments))
            else:
                result = fn(*args, **kwargs)

            # pop BEFORE the schema check so the tag never trips required-key validation
            # and never leaks into the returned object
            resolved = (result.pop(_RESOLVED_PARAMS, None)
                        if isinstance(result, dict) else None)
            if resolved:
                # the effective groupby/splitby, needed by the reference restat below
                arguments.update({k: v for k, v in resolved.items() if k in arguments})
            if schema is not None and isinstance(result, dict):
                _check_schema(schema, result, fn.__name__)

            # ── the reference ────────────────────────────────────────────────
            reference_of = None
            if takes_null and isinstance(result, dict):
                reference_of = _reference.resolve_reference(
                    adata, null_model=arguments.get("null_model"),
                    fit=arguments.get("fit") if takes_fit else None,
                    default_null=default_null, metric=fn.__name__,
                )
                if reference_of is not None:
                    if isinstance(reference_of, str):
                        _reference.check_joinable(
                            adata, reference_of,
                            against=arguments.get("fit") if takes_fit else None)
                    reference = wrapper(**_reference_call(arguments, reference_of))
                    _reference.attach(result, reference, values=values,
                                      denominators=denominators)
                    _reference.restat(result, groupby=arguments.get("groupby"),
                                      splitby=arguments.get("splitby"), values=values,
                                      denominators=denominators, per_gene=per_gene_stats)

            if arguments.get("inplace", True):
                blob = _encode(result)
                if not isinstance(blob, dict):
                    blob = {"value": blob}
                params = {k: v for k, v in arguments.items()
                          if k not in _RESERVED and k != _data_param and k not in leading}
                if takes_null:
                    # the RESOLVED reference, never the caller's "auto" and never a model
                    # object: the params block goes into `uns` and has to be readable and
                    # h5ad-writable. An unresolved model here breaks the .h5ad write.
                    params["null_model"] = _reference.name_of(reference_of)
                if resolved:
                    params.update(resolved)
                blob = {**blob, "params": _encode(params), "version": int(version),
                        "tool": key, "tcri_version": _tcri_version()}
                dest = arguments.get("key_added") or key
                adata.uns[_keys.fit_key(dest, arguments.get("fit") if takes_fit else None)] = blob
            return result

        def _reference_call(arguments, reference_of):
            """The caller's own arguments, with the two the reference changes.

            Every other argument is forwarded VERBATIM -- `groupby`, `splitby`, `clones`,
            `weighted`, `normalized`, `normalize_mode`, `n_clones_ref`, `distance_metric`,
            `temperature`, `n_samples`, `random_state` and `device` each change the estimand,
            and a reference computed at defaults is a different quantity subtracted from a
            different quantity. `inplace` is honoured as the caller gave it, so the reference's
            own frame is cached under its fit key: plottable by `key=`, available to the
            identity test, and not computed twice.
            """
            call = {k: v for k, v in arguments.items() if k != "key_added"}
            call["null_model"] = None          # terminates the recursion
            if reference_arg == "fit":
                if not isinstance(reference_of, str):
                    raise TypeError(
                        f"{fn.__name__} reads a stored substrate, so its null_model names a "
                        f"FIT on this AnnData, not a model object. Write the other model's "
                        f"substrate first with other.to_anndata(adata, fit='myalt'), then pass "
                        f"null_model='myalt'."
                    )
                call["fit"] = reference_of
            else:
                # A query on a fitted model needs the null's NETWORKS, not its arrays, so the
                # reference run substitutes the model itself. `key_added` then has to be set
                # explicitly: with no `fit` parameter there is nothing for the destination to
                # be suffixed by, and the reference would land on the caller's own key.
                # The FIT name, captured before `reference_of` is rebound to a model: a rebuilt
                # null's `name` is its parameter-store namespace (`<parent>.null.phenotype`),
                # which is not what `params["null_model"]` records and not what `fit=` resolves.
                # Keying the blob by it would put the reference somewhere no reader looks.
                fit_name = _reference.name_of(reference_of)
                if isinstance(reference_of, str):
                    from ..null._rebuild import rebuild   # a view package: lazy (test_layout.py)
                    reference_of = rebuild(arguments[reference_arg],
                                           arguments[_data_param], reference_of)
                call[reference_arg] = reference_of
                call["key_added"] = _keys.fit_key(key, fit_name)
            return call

        wrapper.tcri_default_null = default_null
        wrapper.tcri_values = tuple(values)
        wrapper.tcri_denominators = tuple(denominators)
        wrapper.tcri_key = key
        wrapper.tcri_schema_version = int(version)
        wrapper.tcri_schema = schema
        if fn.__module__.startswith("tcri."):
            previous = _REGISTRY.get(key)
            if previous is not None and (
                (previous.__module__, previous.__qualname__, previous.tcri_schema_version)
                != (wrapper.__module__, wrapper.__qualname__, int(version))
            ):
                raise RuntimeError(f"two tl tools declare uns key {key!r}")
            _REGISTRY[key] = wrapper
        return wrapper

    return deco
