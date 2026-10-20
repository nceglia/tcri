"""Result-storage convention for ``tl`` functions — realized once as a decorator.

The contract: every ``tl`` writes ``uns[key]`` as a **dict-of-arrays + a provenance block**
(``params``, the schema ``version`` int, the writing ``tool`` and the ``tcri_version`` that wrote
it; the scanpy ``rank_genes_groups`` pattern) and **returns** ``None``; with ``inplace=False`` it
stores nothing and returns the result.
``@tl_result`` is that convention as code, so the metrics cannot drift from it.

Why this exists rather than each tool writing its own ``uns`` entry: the metric is computed and
stored once, and every ``pl.*`` reads that cache instead of recomputing, so a plot cannot
disagree with the table the user holds.

Usage — the wrapped function declares the parameters it computes with and returns its
*natural* result. The decorator owns storage and the reference, and the parameters that steer
them: it adds ``key_added``, ``inplace`` and, for a tool with a reference, ``null_model`` to the
published signature::

    @tl_result(key=K.MUTUAL_INFORMATION, version=4, schema=schemas.MutualInformation,
               default_null="phenotype")
    def mutual_information(adata, *, covariate, groupby=None, ..., fit=None):
        ...
        return {"table": table, "result": result}

    # published: mutual_information(adata, *, covariate, groupby=None, ..., fit=None,
    #                               null_model="auto", key_added=None, inplace=True)

The decorator reads ``inplace`` / ``key_added`` from the call, optionally checks the result
against a ``schema`` TypedDict, auto-captures the remaining arguments as ``params``, serializes
to an ``.h5ad``-safe dict-of-arrays blob, and writes it to ``uns[key_added or key]``, returning
``None``; with ``inplace=False`` it writes nothing and returns the result. :func:`decode_blob` / :func:`load_result` invert it
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
import re
import warnings
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



#: The parameters ``tl_result`` adds to a wrapped tool's signature and reads itself. A wrapped
#: function never declares them; ``null_model`` is added only for a tool with a reference.
_OWNED = ("null_model", "key_added", "inplace")

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


def _null_for(default_null, change):
    """The default null of a call: the one given, or its mode's entry of a per-mode table."""
    if isinstance(default_null, dict):
        return default_null["change" if change is not None else "covariate"]
    return default_null


def _has_levels(result) -> bool:
    """Whether a result is laid out by level, ``{replicate, item, stats}``, rather than as
    ``{table, result, stats}``."""
    return isinstance(result, dict) and "replicate" in result


def _check_schema(schema, result, fn_name: str) -> None:
    """Presence-only check of a TypedDict's required keys.

    Reads ``__required_keys__`` off the class so this module has no import dependency on
    ``schemas``. DataFrame results are not checked — their columns are asserted per-tool.
    """
    required = set(getattr(schema, "__required_keys__", None) or schema.__annotations__)
    missing = required - set(result)
    if missing:
        raise ValueError(f"{fn_name}: result missing required keys {sorted(missing)}")


def _with_owned(sig: inspect.Signature, owned) -> inspect.Signature:
    """``sig`` with the decorator's keyword-only parameters added after its own, and its return
    annotation widened by ``None``, which a storing call returns.

    They go before a ``**kwargs``, which must stay last.
    """
    params = list(sig.parameters.values())
    tail = [p for p in params if p.kind is inspect.Parameter.VAR_KEYWORD]
    head = [p for p in params if p.kind is not inspect.Parameter.VAR_KEYWORD]
    returns = sig.return_annotation
    if returns is not inspect.Signature.empty:
        name = returns if isinstance(returns, str) else getattr(returns, "__name__", str(returns))
        returns = f"{name} | None"
    return sig.replace(parameters=[*head, *owned, *tail], return_annotation=returns)


def tl_result(*, key: str, version: int = 1, schema=None, data_param: str | None = None,
              values=("value",), denominators=(), default_null=None, reference_arg="fit",
              per_gene_stats=False, distance=False):
    """Store the wrapped ``tl``'s result under ``uns[key_added or key]`` and return ``None``.

    With ``inplace=False`` nothing is stored and the whole result is returned. A stored result is
    read back through :mod:`tcri.get` only, so there is one copy of it, which cannot drift from a
    returned one.

    The decorator owns three parameters, :data:`_OWNED`: ``key_added`` and ``inplace`` on every
    tool, and ``null_model`` on a tool with a ``default_null``. It declares them on its own
    wrapper, reads them there, and publishes the wrapped function's signature followed by them,
    so the ``.pyi`` conformance check sees the whole contract signature. The wrapped function
    declares only the parameters it computes with; declaring an owned one raises ``TypeError``,
    because the body would accept a value it never reads.

    ``params`` captures every argument the wrapped function declares, **including defaults the
    caller never passed**, via ``bind`` + ``apply_defaults``. That is the point: provenance that
    only records explicit arguments cannot answer "what was this run with". ``key_added`` and
    ``inplace`` say where the result goes rather than what it is, and are not recorded;
    ``null_model`` is recorded as the reference it resolved to.

    ``data_param`` names the AnnData argument when it is not the first parameter. A tool that
    takes the fitted model first (``perturb.gene_importance(model, adata, ...)``) stores into
    ``adata``; every parameter *before* the data argument is an object the tool operates on,
    not a setting, and is excluded from ``params`` with it. Otherwise the model itself would be
    recorded as provenance and the ``.h5ad`` write would fail on it.

    **The reference.** When the tool has a ``default_null``, the decorator runs it
    again on each fit of a permutation reference and adds the ``null``/``null_*``/``adjusted*``
    columns, per pair of draws on the per-draw frames (each level's ``draws``, or ``table``) and
    summarized on the summaries (each level's ``summary``, or ``result``)
    (:mod:`tcri._state._reference`). It lives here rather than in each body for one reason that
    is not tidiness: the reference has to be *the caller's own call on the null*, and only the
    decorator holds the caller's bound arguments. A body re-invoking itself would have to re-list what to forward, and that list
    would drift from the signature the first time a knob was added.

    ``values`` names every NATIVE value column the metric can return, before any reference
    column exists -- ``("value",)`` for a scalar metric, with ``"value_from"`` and ``"value_to"``
    for one that takes a change, and ``"denom"`` (``"denom_from"``, ``"denom_to"``) for mutual
    information; the columns a call does not return are skipped. An unlisted value column would
    be read as a row label. ``denominators`` names those that get a reference but no adjusted
    column. ``default_null`` is the metric's entry in the contract's ``DEFAULT_NULL`` table, read
    back off the live function by the conformance test so the two cannot drift: one null, or
    ``{"covariate": ..., "change": ...}`` for a metric whose reference depends on whether the call
    is at one covariate level or a change (``cov_from``/``cov_to``). ``reference_arg`` is the
    parameter the reference run substitutes: ``"fit"`` for a metric that reads a substrate, or
    the name of the model parameter for a query on a fitted model, which needs the null's
    networks rather than its arrays.

    A stateful ``random_state`` (a numpy ``Generator``, ``RandomState``, ``BitGenerator`` or
    ``SeedSequence``, or a ``torch.Generator``) is turned into one integer seed before the call
    (:func:`~tcri._compute._tables.resolve_seed`). That integer is what the call and its reference
    runs draw from and what ``params`` records, so the stored result writes to ``.h5ad`` and the
    call reruns from its ``params``.

    A change call's ``stats`` holds its covariate rows (:func:`~tcri._compute._tables.build_stats`)
    beside the split rows, rebuilt on the adjusted value once the reference is attached.
    ``distance=True`` marks a metric whose change is a distance between the two levels (flux):
    its covariate row tests the adjusted value against 0.
    """
    def deco(fn):
        sig = inspect.signature(fn)
        names = list(sig.parameters)
        if data_param is None:
            # the data argument, whatever it is called
            _data_param, leading = names[0], set()
        else:
            if data_param not in names:
                raise TypeError(f"{fn.__name__} has no parameter {data_param!r}")
            _data_param = data_param
            leading = set(names[: names.index(data_param)])
        clash = [n for n in names if n in _OWNED]
        if clash:
            raise TypeError(
                f"{fn.__name__} declares {clash}: tl_result adds these to the signature and "
                f"reads them itself, so the wrapped function must not declare them")
        takes_null = default_null is not None
        takes_fit = "fit" in names

        def run(args, kwargs, *, null_model, key_added, inplace):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            seed = arguments.get("random_state")
            if seed is not None and not isinstance(seed, (int, np.integer)):
                # a stateful source becomes the one integer the whole call runs on, the fit and
                # every reference run, and that integer is what params records
                from .._compute._tables import resolve_seed   # the lower layer, reached lazily
                arguments["random_state"] = resolve_seed(seed)
            result = _run(arguments, null_model=null_model, key_added=key_added,
                          inplace=inplace)
            return None if inplace else result

        def _run(arguments, *, null_model, key_added, inplace):
            """Compute, attach the reference, store; return the full result.

            The reference runs come back through here too, so they return their draws, which
            the pairing needs, whatever the outer call returns.
            """
            adata = arguments[_data_param]

            # The fit is CANONICALIZED before anything reads it, so `fit="phenotype[0]"` and
            # `fit="null.phenotype[0]"` are one call rather than two `uns` keys and two params
            # blocks. `resolve_fit` is idempotent on the canonical name, which is what makes
            # doing it here safe.
            if takes_fit and arguments.get("fit") is not None:
                arguments["fit"] = _keys.resolve_fit(adata, arguments["fit"])
            result = fn(**arguments)

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
            references = None
            change = _change_of(arguments)
            if takes_null and isinstance(result, dict):
                references = _reference.resolve_reference(
                    adata, null_model=null_model,
                    fit=arguments.get("fit") if takes_fit else None,
                    default_null=_null_for(default_null, change), metric=fn.__name__,
                )
                if references is not None:
                    if reference_arg == "fit":
                        references = [_reference.fit_of(r, adata) for r in references]
                        for r in references:
                            _reference.check_joinable(
                                adata, r, against=arguments.get("fit") if takes_fit else None)
                    with warnings.catch_warnings():
                        # the call has warned about its own groups; its runs on the reference
                        # read the same groups and would repeat it once per fit
                        warnings.filterwarnings("ignore", message=rf"{re.escape(fn.__name__)}: ",
                                                category=UserWarning)
                        runs = [_run_call(_reference_call(arguments, r,
                                                          _reference.index_of(r, adata),
                                                          inplace=inplace,
                                                          key_added=key_added))
                                for r in references]
                    if _has_levels(result):
                        _reference.attach_levels(result, runs, values=values,
                                                 denominators=denominators)
                        unmoved = _reference.mask_unmovable(
                            result, adata, references, arguments=arguments, values=values,
                            denominators=denominators, metric=fn.__name__)
                        _reference.restat_levels(result, groupby=arguments.get("groupby"),
                                                 splitby=arguments.get("splitby"),
                                                 change=change, without=unmoved)
                    else:
                        _reference.attach(result, runs, values=values, denominators=denominators)
                        _reference.restat(result, groupby=arguments.get("groupby"),
                                          splitby=arguments.get("splitby"),
                                          per_gene=per_gene_stats)

            if inplace:
                blob = _encode(result)
                if not isinstance(blob, dict):
                    blob = {"value": blob}
                params = {k: v for k, v in arguments.items()
                          if k != _data_param and k not in leading}
                if takes_null:
                    # the RESOLVED reference, never the caller's "auto" and never a model
                    # object: the params block goes into `uns` and has to be readable and
                    # h5ad-writable. An unresolved model here breaks the .h5ad write.
                    params["null_model"] = _reference.name_of(references, adata)
                if resolved:
                    params.update(resolved)
                blob = {**blob, "params": _encode(params), "version": int(version),
                        "tool": key, "tcri_version": _tcri_version()}
                dest = key_added or key
                adata.uns[_keys.fit_key(dest, arguments.get("fit") if takes_fit else None)] = blob
            return result

        def _change_of(arguments):
            """The covariate-row settings of a change call, or ``None`` for one at a level."""
            if arguments.get("cov_from") is None:
                return None
            return {"cov_from": arguments["cov_from"], "cov_to": arguments.get("cov_to"),
                    "distance": distance}

        def _run_call(call):
            """Run a call mapping, the decorator's own parameters included, through `_run`."""
            call = dict(call)
            owned = {"null_model": call.pop("null_model", None),
                     "key_added": call.pop("key_added", None),
                     "inplace": call.pop("inplace", True)}
            bound = sig.bind(**call)
            bound.apply_defaults()
            return _run(dict(bound.arguments), **owned)

        def _reference_call(arguments, reference_of, index, *, inplace, key_added):
            """The caller's own arguments, on one fit of the reference.

            The fit (or model) and `null_model` change, and the draws take their own seed,
            derived from the caller's `random_state` and the fit's `index` in its reference by
            `_reference.null_seed`: the fit and each null are independent fits whose draws are
            paired, and two samples drawn from one seed share their noise. Every other argument is forwarded VERBATIM -- `groupby`,
            `splitby`, `clones`, `exclude_pools`, `weighted`, `normalized`, `normalize_mode`,
            `n_clones_ref`, `distance_metric`, `temperature`, `n_samples` and `device` each
            change the estimand, and a reference computed at defaults is a different quantity
            subtracted from a different quantity. `inplace` is honored as the caller gave it, so
            the reference's own frame is cached under its fit key: plottable by `key=`,
            available to the identity test, and not computed twice. The fit key is suffixed onto
            the caller's `key_added` when it gives one, so two results under different keys keep
            separate references.
            """
            call = dict(arguments)
            call["null_model"] = None          # terminates the recursion
            call["inplace"] = inplace
            call["key_added"] = key_added
            if "random_state" in call:
                call["random_state"] = _reference.null_seed(call["random_state"], index)
            if reference_arg == "fit":
                call["fit"] = reference_of
            else:
                # A query on a fitted model needs the null's NETWORKS, not its arrays, so the
                # reference run substitutes the model itself. `key_added` then has to be set
                # explicitly: with no `fit` parameter there is nothing for the destination to
                # be suffixed by, and the reference would land on the caller's own key.
                # The FIT name, captured before `reference_of` is rebound to a model: a rebuilt
                # null's `name` is its parameter-store namespace
                # (`<parent>.null.phenotype[0].<token>`), which is not what `params["null_model"]`
                # records and not what `fit=` resolves.
                # Keying the blob by it would put the reference somewhere no reader looks.
                fit_name = _reference.name_of(reference_of, arguments[_data_param])
                if isinstance(reference_of, str):
                    from ..null._rebuild import rebuild   # a view package: lazy (test_layout.py)
                    reference_of = rebuild(arguments[reference_arg],
                                           arguments[_data_param], reference_of)
                call[reference_arg] = reference_of
                call["key_added"] = _keys.fit_key(key_added or key, fit_name)
            return call

        # The parameters the decorator owns are declared HERE, on the wrapper, and read here;
        # the published signature is the wrapped function's own followed by these. Their
        # defaults therefore have one definition, and a tool without a reference has no
        # `null_model` to accept.
        if takes_null:
            def wrapper(*args, null_model="auto", key_added=None, inplace=True, **kwargs):
                return run(args, kwargs, null_model=null_model, key_added=key_added,
                           inplace=inplace)
        else:
            def wrapper(*args, key_added=None, inplace=True, **kwargs):
                return run(args, kwargs, null_model=None, key_added=key_added, inplace=inplace)

        owned = [p for p in inspect.signature(wrapper).parameters.values()
                 if p.kind is inspect.Parameter.KEYWORD_ONLY]
        functools.update_wrapper(wrapper, fn)
        wrapper.__signature__ = _with_owned(sig, owned)

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
