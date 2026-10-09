"""``tcri.null``: one function per permutable axis.

A null is not a different estimator and not a bootstrap. It is **the same model, fitted the same
way, on one permuted label vector** -- so the number it produces is the floor the pipeline
reports when the structure being measured is absent, and ``value - null_value`` is the part of
the observed number that the structure accounts for.

Nulls are explicit. No metric trains one: a metric reads the reference from the substrate and
raises, naming the call to run, when it is not there. Fitting is a decision with a cost, and a
function that quietly triples its own runtime is worse than one that says what it needs.
"""
from __future__ import annotations

import builtins
import warnings

import numpy as np

from .._state import keys as K
from . import _permute
from ._permute import KINDS, OFFSET

__all__ = ["phenotype", "clonotype", "condition", "all"]


def _init_params(model):
    """The parent's constructor arguments, flattened and stripped of what a null replaces.

    scvi's ``_get_init_params`` returns ``{"kwargs": {...}, "non_kwargs": {...}}``, so the two
    have to be merged before they can be splatted. ``kwargs`` itself is a nested dict of the
    ``**kwargs`` the parent swallowed; it is flattened one more level for the same reason.
    """
    init = dict(model.init_params_.get("non_kwargs") or {})
    extra = dict(model.init_params_.get("kwargs") or {})
    init.update(extra.pop("kwargs", {}) or {})
    init.update(extra)
    for drop in ("adata", "name", "permutation"):
        init.pop(drop, None)
    return init


def _fit_name(kind, key_added):
    return f"null.{kind}" if key_added is None else f"null.{kind}.{key_added}"


def _replicate_index(adata, kind, key_added):
    """Position of ``key_added`` among this kind's existing named nulls, 0 when unnamed.

    Two nulls of one kind must not share a permutation stream, or the second would be the first
    with a different name. The index is positional in the SORTED list of existing keys, so it is
    reproducible from the object rather than from the order the calls happened to be made in.
    """
    if key_added is None:
        return 0
    prefix = f"null.{kind}."
    keys = sorted({f[len(prefix):] for f in K.fits(adata) if f.startswith(prefix)} | {str(key_added)})
    return keys.index(str(key_added)) + 1


def _check_the_name_is_free(adata, model, fit):
    """A fit name is claimed by one parent.

    ``tcri.null.all(a, adata)`` then ``tcri.null.all(b, adata)`` would have both claim
    ``null.phenotype`` and the second silently overwrite the first -- two ordinary calls,
    reachable because two models can coexist in one process -- so a different parent raises.
    Re-running the SAME parent rewrites in place, which is the ``tl`` convention and not a
    collision. When neither parent was given ``name=`` (a generated name, or the empty name a
    record from an earlier release carries) and the new parent has written the object's main
    fit, the ordinary way to arrive here is refitting a model and re-running its nulls on the
    same object, so the fit is rewritten with a warning. Otherwise it raises: a parent given
    ``name=`` on either side, or a new parent whose main fit is not on this object, whose null
    would sit beside another model's values.
    """
    from ..model._model import _is_generated_name

    record = adata.uns.get(K.fit_key(K.FIT_SETTINGS, fit))
    if not record or "parent" not in record:
        return
    held = str(record["parent"])
    if held == str(model.name):
        return
    main = str((adata.uns.get(K.FIT_SETTINGS) or {}).get("name", ""))
    unnamed = (not held or _is_generated_name(held)) and _is_generated_name(model.name)
    if unnamed and main == str(model.name):
        warnings.warn(
            f"fit {fit!r} on this object was written by parent {held!r}. Neither that model nor "
            f"this one ({model.name!r}) was given `name=`, and this one wrote the object's main "
            f"fit, so this is taken as a refit and the fit is rewritten. Give the models `name=` "
            f"at construction if they are different models.",
            UserWarning, stacklevel=3,
        )
        return
    # A copy of the AnnData is not a way out: it carries the same record and raises the same.
    refit = (" If this model is a refit of that one, write its main fit first with "
             "model.to_anndata(adata); the null is then rewritten as a refit."
             if unnamed else "")
    raise ValueError(
        f"fit {fit!r} on this object belongs to parent {held!r}, not to {model.name!r}. "
        f"Writing it would overwrite another model's reference, and the two are not "
        f"interchangeable. Pass key_added= to name this one (it would land as "
        f"{_fit_name(fit.split('.')[1], 'b')!r}).{refit}"
    )


def _training_for_uns(null):
    """A null's ``history_`` and ``training_record_``, in a form h5ad writes.

    A null has no saved model of its own, so this is what lets ``rebuild`` hand
    ``tcri.diag.loss`` the curves after the process that fitted it has ended. Each history
    series is kept as its epochs and its values; record entries that are ``None`` are left out,
    because h5ad cannot write them, and read back as missing.
    """
    history = {
        name: {"epoch": np.asarray(frame.index, dtype=np.int64),
               "value": np.asarray(frame.to_numpy(), dtype=float).ravel()}
        for name, frame in (getattr(null, "history_", None) or {}).items()
    }
    record = {k: v for k, v in (getattr(null, "training_record_", None) or {}).items()
              if v is not None}
    return {"history": history, "record": record}


def _null(model, adata, *, kind, within=None, seed=None, key_added=None, **train_kwargs):
    """The shared body: permute, construct, train, write. One per kind, no branching after this."""
    from ..model._model import TCRIModel

    if kind not in KINDS:
        raise ValueError(f"kind={kind!r} is not one of {list(KINDS)}")
    if adata.uns.get(K.METADATA) is None:
        raise ValueError(
            "this AnnData carries no tcri substrate: call model.to_anndata(adata) on the "
            "parent before building a reference against it."
        )

    fit = _fit_name(kind, key_added)
    _check_the_name_is_free(adata, model, fit)

    strata = _permute.resolve_strata(adata, kind, within)
    resolved_seed = int(seed) if seed is not None else int(
        (model.init_params_.get("kwargs", {}).get("seed") or getattr(model, "_seed", None) or 0)
        + OFFSET[kind] + _replicate_index(adata, kind, key_added)
    )
    perm, sizes = _permute.build_permutation(adata, strata, np.random.default_rng(resolved_seed))

    train_args = {**(getattr(model, "train_kwargs_", None) or {}), **train_kwargs}
    if not train_args:
        raise ValueError(
            f"the parent carries no record of the arguments it was trained with "
            f"(`model.train_kwargs_` is empty), so this null cannot be fitted the same way. "
            f"Pass the parent's arguments explicitly, e.g. "
            f"tcri.null.{kind}(model, adata, max_epochs=..., batch_size=...)."
        )

    namespace = f"{model.name}.{fit}"
    null = TCRIModel(adata, name=namespace, permutation=(_permute.AXIS[kind], perm),
                     **_init_params(model))
    null._fit_name = fit
    null.train(**train_args)
    null.to_anndata(adata, fit=fit)

    adata.uns[K.fit_key(K.PERMUTATION, fit)] = perm
    adata.uns[K.fit_key(K.BUFFERS, fit)] = {
        k: v.detach().cpu().numpy() for k, v in null.module.named_buffers() if v is not None
    }
    adata.uns[K.fit_key(K.TRAINING, fit)] = _training_for_uns(null)
    adata.uns[K.fit_key(K.FIT_SETTINGS, fit)] = {
        **adata.uns[K.fit_key(K.FIT_SETTINGS, fit)],
        "kind": kind,
        "axis": _permute.AXIS[kind],
        "strata": list(strata),
        "stratum_sizes": np.asarray(sorted(sizes), dtype=np.int64),
        "n_strata": int(len(sizes)),
        "n_singleton_strata": int(sum(1 for n in sizes if n == 1)),
        "seed": resolved_seed,
        "parent": str(model.name),
        "key_added": "" if key_added is None else str(key_added),
        "namespace": namespace,
    }
    return null


def phenotype(model, adata, *, within=None, seed=None, key_added=None, **train_kwargs):
    """Refit the parent with phenotype labels shuffled within (batch, covariate, replicate).

    The reference for anything that scores how much a clone's phenotype composition departs from
    the repertoire's: clonotypic and phenotypic entropy, mutual information, gene importance.
    Each cell keeps its clone, its condition and its counts, so ``ct_array`` is the parent's cell
    for cell and only the labels move.
    """
    return _null(model, adata, kind="phenotype", within=within, seed=seed,
                 key_added=key_added, **train_kwargs)


def clonotype(model, adata, *, within=None, seed=None, key_added=None, **train_kwargs):
    """Refit the parent with clone ids shuffled within (batch, covariate, replicate).

    Keeps the clone-size distribution and every cell's phenotype; destroys which cells share a
    clone. The complement of :func:`phenotype`: its head still reads the true labels, so a
    metric that moves under this one and not under the phenotype null is reading clonal
    structure rather than phenotype structure.
    """
    return _null(model, adata, kind="clonotype", within=within, seed=seed,
                 key_added=key_added, **train_kwargs)


def condition(model, adata, *, seed=None, key_added=None, **train_kwargs):
    """Refit the parent with the condition label shuffled within (clonotype, batch).

    Each clone keeps its cells and its size; which condition a cell sits at is random. The
    reference for anything that compares conditions -- phenotypic flux and the two deltas. It
    takes no ``within``: its strata are what make it a condition null.
    """
    return _null(model, adata, kind="condition", seed=seed, key_added=key_added, **train_kwargs)


def all(model, adata, *, kinds=KINDS, within=None, seed=None, key_added=None, **train_kwargs):
    """Fit every null in ``kinds`` in sequence and return ``{kind: model}``.

    The common path: one call after ``train()`` and ``to_anndata()`` and the object carries every
    reference the metrics default to. ``condition`` is skipped, with a warning, when the
    covariate has one level -- a condition null needs two conditions to permute between -- and
    never skipped silently.
    """
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise ValueError(f"kinds={kinds!r}: {unknown!r} is not one of {list(KINDS)}")
    if within is not None and "condition" in kinds:
        # Silently dropping it would make the recorded strata differ from the strata the caller
        # asked for, on the one null whose strata are not theirs to choose.
        raise ValueError(
            "the condition null takes no `within`, so passing one to `all` while 'condition' is "
            "in `kinds` would silently drop it. Either drop `within`, or call the kinds "
            "separately, or pass kinds=('phenotype', 'clonotype')."
        )

    n_levels = len(adata.uns[K.COVARIATE_CATEGORIES])
    out = {}
    for kind in kinds:
        if kind == "condition" and n_levels < 2:
            warnings.warn(
                f"skipping the condition null: the covariate has {n_levels} level(s) and a "
                f"condition null permutes cells between conditions. The metrics that default "
                f"to it (phenotypic_flux, the two deltas) are not defined here either.",
                UserWarning, stacklevel=2,
            )
            continue
        out[kind] = _null(model, adata, kind=kind, seed=seed, key_added=key_added,
                          within=(within if kind != "condition" else None), **train_kwargs)
    return out


# `all` shadows the builtin inside this module; nothing here needs it, and the name is worth
# more at the call site (`tcri.null.all(model, adata)`) than the shadowing costs.
assert builtins.all is not all
