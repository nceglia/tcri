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
import uuid
import warnings

import numpy as np
import pyro

from .._state import keys as K
from . import _permute
from ._permute import KINDS

__all__ = ["phenotype", "clonotype", "covariate", "all"]

#: Fits per reference by default. A pooled interval from K nulls carries (K - 1) / K of the
#: variance between them in expectation, so 5 carries 80% of it.
N_NULLS = 5


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


def _namespace(model, fit):
    """A new parameter-store namespace for one fit of ``fit``: ``<parent>.<fit>.<token>``.

    The token is 12 hex digits from ``uuid4``, so every fit, a refit of the same parent and kind
    included, starts from an empty namespace. A refit under the earlier fit's namespace would
    read that fit's guide concentrations, and its networks would not train (see the collision
    check in ``TCRIModel.__init__``). The namespace is recorded with the fit, which is how
    ``rebuild`` finds its entries.
    """
    return f"{model.name}.{fit}.{uuid.uuid4().hex[:12]}"


def _drop_namespace(namespace):
    """Remove every store entry under exactly ``namespace``; a ``key_added`` sibling's stay."""
    from ..model._module import _null_namespace

    store = pyro.get_param_store()
    for key in [k for k in store.keys() if _null_namespace(k) == namespace]:
        del store[key]


def _remove_fit(adata, fit):
    """Remove ``fit`` from ``adata``: its name from the fits list and every key ``fit_key`` built
    for it.

    A key is the fit's when it is ``tcri_<fit>_...`` in ``uns``, ``obs`` or ``obsm``, or
    ``X_tcri_<fit>`` / ``X_tcri_<fit>_...`` in ``obsm``. The separator after the name keeps one
    member's keys apart from a longer name's: ``[1]`` from ``[10]``, ``null.phenotype`` from
    ``null.phenotype.b``.
    """
    prefixes = (f"tcri_{fit}_", f"X_tcri_{fit}_")

    def owned(key):
        return str(key).startswith(prefixes) or str(key) == f"X_tcri_{fit}"

    for store in (adata.uns, adata.obsm):
        for key in [k for k in store.keys() if owned(k)]:
            del store[key]
    columns = [c for c in adata.obs.columns if owned(c)]
    if columns:
        adata.obs.drop(columns=columns, inplace=True)
    meta = dict(adata.uns[K.METADATA])
    remaining = [n for n in K.fits(adata) if n != fit]
    meta.pop(K.FITS, None)
    adata.uns[K.METADATA] = {**meta, **({K.FITS: remaining} if remaining else {})}


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
        f"{_fit_name(fit.split('.')[1].split('[')[0], 'b')!r}).{refit}"
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


def _owned_namespace(record, model):
    """The namespace a fit's record names, when ``model`` is the parent that owns it.

    A fit kept from another parent is that parent's to keep, so its entries are never dropped.
    """
    if record.get("namespace") is None or str(record.get("parent", "")) != str(model.name):
        return None
    return str(record["namespace"])


def _null(model, adata, *, kind, n_nulls=N_NULLS, within=None, seed=None, key_added=None,
          **train_kwargs):
    """The shared body: ``n_nulls`` fits of one kind, each permuted, constructed, trained and
    written. One per kind, no branching after this."""
    from ..model._model import _int_at_least

    if kind not in KINDS:
        raise ValueError(f"kind={kind!r} is not one of {list(KINDS)}")
    n_nulls = _int_at_least("n_nulls", n_nulls, 1)
    if adata.uns.get(K.METADATA) is None:
        raise ValueError(
            "this AnnData carries no tcri substrate: call model.to_anndata(adata) on the "
            "parent before building a reference against it."
        )

    group = _fit_name(kind, key_added)
    names = [K.member_name(group, i) for i in range(n_nulls)]
    earlier = K.members(adata, group)
    for fit in dict.fromkeys(earlier + names):
        _check_the_name_is_free(adata, model, fit)

    strata = _permute.resolve_strata(adata, kind, within)
    base_seed = int(seed) if seed is not None else int(
        model.init_params_.get("kwargs", {}).get("seed") or getattr(model, "_seed", None) or 0)

    train_args = {**(getattr(model, "train_kwargs_", None) or {}), **train_kwargs}
    if not train_args:
        raise ValueError(
            f"the parent carries no record of the arguments it was trained with "
            f"(`model.train_kwargs_` is empty), so this null cannot be fitted the same way. "
            f"Pass the parent's arguments explicitly, e.g. "
            f"tcri.null.{kind}(model, adata, max_epochs=..., batch_size=...)."
        )

    nulls = [_fit_member(model, adata, kind=kind, group=group, index=i, n_nulls=n_nulls,
                         strata=strata, seed=base_seed, key_added=key_added,
                         train_args=train_args)
             for i in range(n_nulls)]
    # A refit with fewer nulls than the group it replaces leaves none of the earlier members
    # behind: a reference read from the records would otherwise pool the old fits with the new.
    for fit in earlier:
        if fit in names:
            continue
        namespace = _owned_namespace(adata.uns.get(K.fit_key(K.FIT_SETTINGS, fit)) or {}, model)
        _remove_fit(adata, fit)
        if namespace is not None:
            _drop_namespace(namespace)
    return nulls


def _fit_member(model, adata, *, kind, group, index, n_nulls, strata, seed, key_added,
                train_args):
    """Fit and write member ``index`` of the reference ``group``, and return it."""
    from ..model._model import TCRIModel

    fit = K.member_name(group, index)
    rng = _permute.permutation_rng(seed, kind, key_added, index)
    perm, sizes = _permute.build_permutation(adata, strata, rng)

    # The fit this one replaces, read before `to_anndata` rewrites its record. Its entries are
    # dropped once the new fit is written.
    replaced_namespace = _owned_namespace(adata.uns.get(K.fit_key(K.FIT_SETTINGS, fit)) or {},
                                          model)

    namespace = _namespace(model, fit)
    null = TCRIModel(adata, name=namespace, permutation=(_permute.AXIS[kind], perm),
                     **_init_params(model))
    null._fit_name = fit
    null.train(**train_args)
    null.to_anndata(adata, fit=fit)
    if replaced_namespace is not None and replaced_namespace != namespace:
        _drop_namespace(replaced_namespace)

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
        "seed": int(seed),
        "spawn_key": np.asarray(_permute.spawn_key(kind, key_added, index), dtype=np.int64),
        "parent": str(model.name),
        "key_added": "" if key_added is None else str(key_added),
        "namespace": namespace,
        "reference": group,
        "index": int(index),
        "n_nulls": int(n_nulls),
    }
    return null


_N_NULLS_DOC = """
    The reference is ``n_nulls`` fits, ``null.<kind>[0]`` to ``null.<kind>[n_nulls - 1]``
    (``null.<kind>.<key_added>[i]`` when named), each on its own permutation and in its own
    parameter namespace. A metric pairs the parent's draw *d* with draw *d* of every one of them
    and summarizes all the pairs, so ``adjusted`` and its interval carry how much the null moves
    from one fit to the next, which one fit cannot show. The cost grows with it: ``n_nulls``
    fits per call (``tcri.null.all`` fits 3 x ``n_nulls``, 15 at the default), and a metric's
    per-draw ``table`` is ``n_nulls`` times longer. Calling again replaces the whole reference.

    A null's permutation is drawn from ``seed`` (default: the parent's seed), spawned by its
    kind, ``key_added`` and index alone, so the same call reproduces the same permutations
    whatever else the object carries. Returns the ``n_nulls`` fitted models, in index order.
"""


def phenotype(model, adata, *, n_nulls=N_NULLS, within=None, seed=None, key_added=None,
              **train_kwargs):
    """Refit the parent with phenotype labels shuffled within (batch, covariate, replicate).

    The reference for anything that scores how much a clone's phenotype composition departs from
    the repertoire's: clonotypic and phenotypic entropy, mutual information, gene importance.
    Each cell keeps its clone, its covariate level and its counts, so ``ct_array`` is the parent's
    cell for cell and only the labels move.
    """
    return _null(model, adata, kind="phenotype", n_nulls=n_nulls, within=within, seed=seed,
                 key_added=key_added, **train_kwargs)


def clonotype(model, adata, *, n_nulls=N_NULLS, within=None, seed=None, key_added=None,
              **train_kwargs):
    """Refit the parent with clone ids shuffled within (batch, covariate, replicate).

    Keeps the clone-size distribution and every cell's phenotype; destroys which cells share a
    clone. The complement of :func:`phenotype`: its head still reads the true labels, so a
    metric that moves under this one and not under the phenotype null is reading clonal
    structure rather than phenotype structure.
    """
    return _null(model, adata, kind="clonotype", n_nulls=n_nulls, within=within, seed=seed,
                 key_added=key_added, **train_kwargs)


def covariate(model, adata, *, n_nulls=N_NULLS, seed=None, key_added=None, **train_kwargs):
    """Refit the parent with the covariate label shuffled within (clonotype, batch).

    Each clone keeps its cells and its size; which covariate level a cell sits at is random. The
    reference for anything that compares covariate levels -- phenotypic flux and the two deltas.
    It takes no ``within``: its strata are what make it a covariate null.
    """
    return _null(model, adata, kind="covariate", n_nulls=n_nulls, seed=seed,
                 key_added=key_added, **train_kwargs)


for _f in (phenotype, clonotype, covariate):
    _f.__doc__ += _N_NULLS_DOC
del _f


def all(model, adata, *, n_nulls=N_NULLS, kinds=KINDS, within=None, seed=None, key_added=None,
        **train_kwargs):
    """Fit every null in ``kinds`` in sequence and return ``{kind: [models]}``.

    The common path: one call after ``train()`` and ``to_anndata()`` and the object carries every
    reference the metrics default to. Each kind is ``n_nulls`` fits, so the default of 5 over
    the three kinds is 15 fits, and a metric's per-draw ``table`` is ``n_nulls`` times longer
    than with one null. ``covariate`` is skipped, with a warning, when the covariate has one
    level -- a covariate null needs two levels to permute between -- and never skipped
    silently.
    """
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise ValueError(f"kinds={kinds!r}: {unknown!r} is not one of {list(KINDS)}")
    if within is not None and "covariate" in kinds:
        # Silently dropping it would make the recorded strata differ from the strata the caller
        # asked for, on the one null whose strata are not theirs to choose.
        raise ValueError(
            "the covariate null takes no `within`, so passing one to `all` while 'covariate' is "
            "in `kinds` would silently drop it. Either drop `within`, or call the kinds "
            "separately, or pass kinds=('phenotype', 'clonotype')."
        )

    n_levels = len(adata.uns[K.COVARIATE_CATEGORIES])
    out = {}
    for kind in kinds:
        if kind == "covariate" and n_levels < 2:
            warnings.warn(
                f"skipping the covariate null: the covariate has {n_levels} level(s) and a "
                f"covariate null permutes cells between its levels. The metrics that default "
                f"to it (phenotypic_flux, the two deltas) are not defined here either.",
                UserWarning, stacklevel=2,
            )
            continue
        out[kind] = _null(model, adata, kind=kind, n_nulls=n_nulls, seed=seed,
                          key_added=key_added,
                          within=(within if kind != "covariate" else None), **train_kwargs)
    return out


# `all` shadows the builtin inside this module; nothing here needs it, and the name is worth
# more at the call site (`tcri.null.all(model, adata)`) than the shadowing costs.
assert builtins.all is not all
