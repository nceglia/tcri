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
import time
import uuid
import warnings

import numpy as np
import pyro

from .. import _output
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
        _output.warn(
            f"fit {fit!r} on this object was written by parent {held!r}. Neither that model nor "
            f"this one ({model.name!r}) was given `name=`, and this one wrote the object's main "
            f"fit, so this is taken as a refit and the fit is rewritten. Give the models `name=` "
            f"at construction if they are different models."
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


def _owned_namespace(record, model):
    """The namespace a fit's record names, when ``model`` is the parent that owns it.

    A fit kept from another parent is that parent's to keep, so its entries are never dropped.
    """
    if record.get("namespace") is None or str(record.get("parent", "")) != str(model.name):
        return None
    return str(record["namespace"])


def _null(model, adata, *, kind, n_nulls=N_NULLS, within=None, seed=None, key_added=None,
          session=None, **train_kwargs):
    """The shared body: ``n_nulls`` fits of one kind, each permuted, constructed, trained and
    written. One per kind, no branching after this. ``session`` collects what the public call
    reports: its bar, and every fit written, with the arguments it ran with."""
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
    if kind == "covariate":
        _permute.check_covariate_movable(adata)
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

    nulls = []
    for i in range(n_nulls):
        bar = session["bar"] if session is not None else None
        if bar is not None:
            bar.describe(K.member_name(group, i))
        null = _fit_member(model, adata, kind=kind, group=group, index=i, n_nulls=n_nulls,
                           strata=strata, seed=base_seed, key_added=key_added,
                           train_args=train_args)
        nulls.append(null)
        if session is not None:
            session["done"].append((K.member_name(group, i), null, train_args))
        if bar is not None:
            bar.advance()
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
    # A null fit is silent: what it would warn about is reported once for the whole set, from
    # its training record (_report_conditions), and its construction repeats the parent's.
    # "ignore", so a caller's "error" filter cannot raise a warning the set does not report.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        null = TCRIModel(adata, name=namespace, permutation=(_permute.AXIS[kind], perm),
                         **_init_params(model))
        null._fit_name = fit
        null.train(**train_args, print=False, enable_progress_bar=False)
    null.to_anndata(adata, fit=fit)
    if replaced_namespace is not None and replaced_namespace != namespace:
        _drop_namespace(replaced_namespace)

    adata.uns[K.fit_key(K.PERMUTATION, fit)] = perm
    adata.uns[K.fit_key(K.BUFFERS, fit)] = {
        k: v.detach().cpu().numpy() for k, v in null.module.named_buffers() if v is not None
    }
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


#: The conditions a set of null fits reports once, in place of one warning per fit: the
#: condition's key, whether it needs early stopping on to be a problem, and how to say it.
_CONDITIONS = (
    ("max_epochs", True,
     "stopped at max_epochs={max_epochs} without converging", "Adjusted values read against "
     "them may be understated; refit the nulls with a larger max_epochs."),
    ("ramp", False,
     "did not complete the KL ramp", "Their fitted prior is scaled by kl_weight below "
     "max_kl_weight, and no checkpoint was selected."),
    ("other limit", True,
     "ended before max_epochs without the early-stopping rule firing (a Lightning limit or an "
     "interrupt)", "They are truncated, not converged."),
    ("non-finite objective", False,
     "stopped because the validation criterion became non-finite", "They diverged; values "
     "read against them are not meaningful."),
)


def _condition_of(record, key) -> bool:
    if key == "ramp":
        return not bool(record.get("ramp_completed", True))
    return record.get("stop_reason") == key


def _conditions(done) -> dict:
    """``{condition: [fit names]}`` over the fits written, from each fit's training record."""
    found = {key: [] for key, *_ in _CONDITIONS}
    for fit, null, args in done:
        record = getattr(null, "training_record_", None) or {}
        for key, needs_stopping, *_ in _CONDITIONS:
            if needs_stopping and not args.get("early_stopping", True):
                continue
            if _condition_of(record, key):
                found[key].append(fit)
    return found


def _report_conditions(model, adata, done) -> None:
    """One warning per condition over the fits written, naming them."""
    if not done:
        return
    n = len(done)
    for key, _needs, what, consequence in _CONDITIONS:
        fits = _conditions(done)[key]
        if fits:
            max_epochs = done[0][2].get("max_epochs")
            _output.warn(f"{len(fits)} of {n} null fits {what.format(max_epochs=max_epochs)}: "
                          f"{', '.join(fits)}. {consequence}")
    parent_batch = (getattr(model, "train_kwargs_", None) or {}).get("batch_size")
    batch = done[0][2].get("batch_size")
    if batch is not None and batch != parent_batch and batch >= adata.n_obs:
        _output.warn(
            f"the null fits ran with batch_size={batch} >= n_obs={adata.n_obs}, which the parent "
            f"did not: each of their epochs is a single optimizer step, so per-epoch overhead "
            f"dominates and max_epochs becomes the number of gradient updates."
        )


def _set_panel(model, done, seconds):
    """The panel a set of null fits prints: each reference with its fits and conditions."""
    from rich.text import Text

    rows = []
    by_reference: dict = {}
    for fit, null, args in done:
        by_reference.setdefault(fit.rsplit("[", 1)[0], []).append((fit, null, args))
    for reference, members in by_reference.items():
        epochs = [int((getattr(n, "training_record_", None) or {}).get("epochs_run", 0))
                  for _f, n, _a in members]
        span = f"{min(epochs):,}" if min(epochs) == max(epochs) else f"{min(epochs):,}–{max(epochs):,}"
        text = Text(f"{len(members)} fit{'s' if len(members) != 1 else ''} · {span} epochs")
        for key, _needs, what, _consequence in _CONDITIONS:
            k = len(_conditions(members)[key])
            if k:
                label = "reached max_epochs" if key == "max_epochs" else (
                    "ramp incomplete" if key == "ramp" else key)
                text.append(" · ")
                text.append(f"{k} {label}", style="bold")
        rows.append((reference, text))
    rows.append(("Time", Text(_output.duration(seconds))))
    return _output.panel(f"null fits · {model.name}", rows)


class NullSet(dict):
    """The null fits one :func:`all` call made: a ``dict`` of ``{kind: [models]}``.

    ``nulls["phenotype"][0]`` is the first phenotype null's fitted model, for a direct look at
    its networks (``predict``, ``tcri.diag.loss``, ``tcri.perturb.gene_importance(null_model=)``).
    The metrics never need this object: every fit is recorded on the AnnData and its parameters
    stay in Pyro's store. Displayed (a notebook cell's last line, ``print``, ``rich.print``), it
    is the set's panel: each reference with its fits, epochs, conditions and time.
    """

    def __init__(self, models, panel=None):
        super().__init__(models)
        self._panel = panel

    def __rich__(self):
        from rich.text import Text

        return self._panel if self._panel is not None else Text(repr(self))

    def _repr_mimebundle_(self, include=None, exclude=None):
        from rich.jupyter import JupyterMixin

        return JupyterMixin._repr_mimebundle_(self, include=include, exclude=exclude)

    def __str__(self) -> str:
        return _output.plain_text(self.__rich__())

    def __repr__(self) -> str:
        kinds = ", ".join(f"{kind}={len(models)}" for kind, models in self.items())
        return f"NullSet({kinds})"


def _fit_set(model, adata, plan, *, n_nulls, seed, key_added, print, enable_progress_bar,
             train_kwargs):
    """Fit each ``(kind, within)`` in ``plan`` under one bar; report once; print the panel.

    The conditions are reported after the panel, and also when the call fails partway, for the
    fits it already wrote.
    """
    from ..model._model import _int_at_least

    n_nulls = _int_at_least("n_nulls", n_nulls, 1)
    session = {"bar": None, "done": []}
    started = time.perf_counter()
    out = {}
    try:
        with _output.set_progress(enable_progress_bar, "null fits",
                                   n_nulls * len(plan)) as bar:
            session["bar"] = bar
            for kind, within in plan:
                out[kind] = _null(model, adata, kind=kind, n_nulls=n_nulls, within=within,
                                  seed=seed, key_added=key_added, session=session,
                                  **train_kwargs)
    except BaseException:
        _report_conditions(model, adata, session["done"])
        raise
    panel = (_set_panel(model, session["done"], time.perf_counter() - started)
             if session["done"] else None)
    if print and panel is not None:
        _output.print_out(panel)
    _report_conditions(model, adata, session["done"])
    return NullSet(out, panel)


_N_NULLS_DOC = """
    The reference is ``n_nulls`` fits, ``null.<kind>[0]`` to ``null.<kind>[n_nulls - 1]``
    (``null.<kind>.<key_added>[i]`` when named), each on its own permutation and in its own
    parameter namespace. A metric pairs the parent's draw *d* with draw *d* of every one of them
    and summarizes all the pairs, so ``adjusted`` and its interval carry how much the null moves
    from one fit to the next, which one fit cannot show. The cost grows with it: ``n_nulls``
    fits per call (``tcri.null.all`` fits 3 x ``n_nulls``, 15 at the default), and a metric's
    per-draw frames (``draws``) are ``n_nulls`` times longer. Calling again replaces the whole reference.

    A null's permutation is drawn from ``seed`` (default: the parent's seed), spawned by its
    kind, ``key_added`` and index alone, so the same call reproduces the same permutations
    whatever else the object carries. Returns the ``n_nulls`` fitted models, in index order.

    The fits are quiet: one bar over them (``enable_progress_bar``), and, with ``print=True``, a
    panel when they are done, each reference with its fits, the epochs they ran and how many
    reached ``max_epochs``, did not complete the KL ramp, ended by another limit or stopped on a
    non-finite criterion. A fit's own warnings are not repeated per fit; each of those conditions
    is reported once, naming the fits, also when the call fails partway.
    ``print`` and ``enable_progress_bar`` are this call's own and are not passed to the fits;
    every other keyword is a ``train()`` argument, replayed over the parent's.
"""


def phenotype(model, adata, *, n_nulls=N_NULLS, within=None, seed=None, key_added=None,
              print=True, enable_progress_bar=True, **train_kwargs):
    """Refit the parent with phenotype labels shuffled within (batch, covariate, replicate).

    The reference for anything that scores how much a clone's phenotype composition departs from
    the repertoire's: clonotypic and phenotypic entropy, mutual information, gene importance.
    Each cell keeps its clone, its covariate level and its counts, so ``ct_array`` is the parent's
    cell for cell and only the labels move.
    """
    return _fit_set(model, adata, [("phenotype", within)], n_nulls=n_nulls, seed=seed,
                    key_added=key_added, print=print, enable_progress_bar=enable_progress_bar,
                    train_kwargs=train_kwargs)["phenotype"]


def clonotype(model, adata, *, n_nulls=N_NULLS, within=None, seed=None, key_added=None,
              print=True, enable_progress_bar=True, **train_kwargs):
    """Refit the parent with clone ids shuffled within (batch, covariate, replicate).

    Keeps the clone-size distribution and every cell's phenotype; destroys which cells share a
    clone. The complement of :func:`phenotype`: its head still reads the true labels, so a
    metric that moves under this one and not under the phenotype null is reading clonal
    structure rather than phenotype structure.
    """
    return _fit_set(model, adata, [("clonotype", within)], n_nulls=n_nulls, seed=seed,
                    key_added=key_added, print=print, enable_progress_bar=enable_progress_bar,
                    train_kwargs=train_kwargs)["clonotype"]


def covariate(model, adata, *, n_nulls=N_NULLS, seed=None, key_added=None, print=True,
              enable_progress_bar=True, **train_kwargs):
    """Refit the parent with the covariate label shuffled within (clonotype, batch).

    Each clone keeps its cells and its size; which covariate level a cell sits at is random. The
    reference for anything that compares covariate levels -- phenotypic flux and every change
    (``cov_from``/``cov_to``).
    It takes no ``within``: its strata are what make it a covariate null.

    Raises ``ValueError`` before fitting when no (clonotype, batch) holds two levels, since no
    shuffle could then move a cell. A metric reading this null blanks the reference of any group
    it cannot move at the levels read, and names it in a warning.
    """
    return _fit_set(model, adata, [("covariate", None)], n_nulls=n_nulls, seed=seed,
                    key_added=key_added, print=print, enable_progress_bar=enable_progress_bar,
                    train_kwargs=train_kwargs)["covariate"]


for _f in (phenotype, clonotype, covariate):
    _f.__doc__ += _N_NULLS_DOC
del _f


def all(model, adata, *, n_nulls=N_NULLS, kinds=KINDS, within=None, seed=None, key_added=None,
        print=True, enable_progress_bar=True, **train_kwargs):
    """Fit every null in ``kinds`` in sequence and return them as a ``NullSet``: a ``dict`` of
    ``{kind: [models]}`` that displays as the set's panel.

    The common path: one call after ``train()`` and ``to_anndata()`` and the object carries every
    reference the metrics default to. Each kind is ``n_nulls`` fits, so the default of 5 over
    the three kinds is 15 fits, and a metric's per-draw frames (``draws``) are ``n_nulls`` times
    longer than with one null. ``covariate`` is skipped, with a warning, when the covariate has one
    level -- a covariate null needs two levels to permute between -- and never skipped
    silently.

    One bar runs over all the fits, and ``print=True`` prints one panel for the set when they are
    done; see :func:`phenotype` for what is reported and how.
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
    if "covariate" in kinds and n_levels >= 2:
        # checked before any kind is fitted, so a covariate null that could move nothing fails
        # before the other references spend their training
        _permute.check_covariate_movable(adata)
    plan = []
    for kind in kinds:
        if kind == "covariate" and n_levels < 2:
            _output.warn(
                f"skipping the covariate null: the covariate has {n_levels} level(s) and a "
                f"covariate null permutes cells between its levels. The metrics that default "
                f"to it (phenotypic_flux and every change between levels) are not defined here "
                f"either."
            )
            continue
        plan.append((kind, within if kind != "covariate" else None))
    return _fit_set(model, adata, plan, n_nulls=n_nulls, seed=seed, key_added=key_added,
                    print=print, enable_progress_bar=enable_progress_bar,
                    train_kwargs=train_kwargs)


# `all` shadows the builtin inside this module; nothing here needs it, and the name is worth
# more at the call site (`tcri.null.all(model, adata)`) than the shadowing costs.
assert builtins.all is not all
