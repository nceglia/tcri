"""The four ``pl``↔``tl`` twins as CACHE RENDERERS, plus ``resolve_colors``.

The load-bearing property here is not "the plot renders". It is that the plot renders the
*stored* result and nothing else: no recompute, no metric arguments of its own, no axis the
``tl`` call did not use. A ``pl`` that calls its ``tl`` twin internally can draw a figure that
disagrees with the frame in the caller's hand, and a ``pl`` that supplies a ``groupby`` of its own
groups the picture by a column the caller never named.
"""
import inspect

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import tcri
from ..conftest import frames
from tcri._state import keys as K
from tcri.plotting._base import BRACKET_LABEL

TWINS = ["mutual_information", "clonotypic_entropy", "phenotypic_entropy", "phenotypic_flux"]

#: The twins with an item view; mutual information has one value per replicate and no items.
ITEM_TWINS = ["clonotypic_entropy", "phenotypic_entropy", "phenotypic_flux"]

#: Anything on this list reaching a `pl` signature means the plot can compute something.
METRIC_ARGS = {
    "covariate", "cov_from", "cov_to", "groupby", "splitby", "n_samples", "temperature",
    "clones", "weighted", "normalized", "normalize_mode", "distance_metric", "random_state",
    "device", "inplace", "key_added",
}


def _cov(adata):
    return list(adata.uns[K.COVARIATE_CATEGORIES])[0]


def _n_points(ax, *, reference=False):
    """Markers drawn by the VALUE mark — the floor of the mark rule.

    The permutation reference is a second collection at the same x positions, labeled
    :data:`tcri.plotting._base.REFERENCE_LABEL`. Counting both would make every "one dot per
    replicate" assertion in this file read double the moment a result carries a reference,
    which is the normal case. ``reference=True`` counts the reference collection instead.
    """
    from tcri.plotting._base import REFERENCE_LABEL

    want = REFERENCE_LABEL if reference else None
    return sum(len(c.get_offsets()) for c in ax.collections
               if (c.get_label() == REFERENCE_LABEL) == reference
               or (want is None and c.get_label() != REFERENCE_LABEL))


def _reference_points(ax):
    """The gray reference markers, as an (n, 2) array of (x, y)."""
    from tcri.plotting._base import REFERENCE_LABEL

    import numpy as _np
    offs = [c.get_offsets() for c in ax.collections if c.get_label() == REFERENCE_LABEL]
    return _np.concatenate(offs) if offs else _np.empty((0, 2))


def _mark_ys(ax, *, reference=False):
    """The y values of the value marks, sorted; ``reference=True`` gives the gray reference
    marks' instead."""
    from tcri.plotting._base import REFERENCE_LABEL

    offs = [c.get_offsets()[:, 1] for c in ax.collections
            if (c.get_label() == REFERENCE_LABEL) == reference]
    return np.sort(np.concatenate(offs)) if offs else np.empty(0)


def _n_violins(ax):
    from matplotlib.collections import PolyCollection

    return sum(isinstance(c, PolyCollection) for c in ax.collections)


def _bracket_stars(ax):
    """The significance brackets, as ``{(x_a, x_b): [text drawn over the bracket's middle]}``.

    A bracket carries :data:`BRACKET_LABEL` and steps up and down in y; the zero rule carries the
    same label and is flat, so it is not a bracket.
    """
    out = {}
    for line in ax.lines:
        ys = np.asarray(line.get_ydata(), dtype=float)
        if line.get_label() != BRACKET_LABEL or len(set(np.round(ys, 12))) < 2:
            continue
        xs = np.asarray(line.get_xdata(), dtype=float)
        mid = (xs.min() + xs.max()) / 2
        out[(float(xs.min()), float(xs.max()))] = [
            t.get_text() for t in ax.texts if np.isclose(t.get_position()[0], mid)]
    return out


def _has_zero_rule(ax):
    """Whether a flat line at y == 0 is drawn, the zero rule of an adjusted or a change axis."""
    return any(line.get_label() == BRACKET_LABEL
               and len(set(np.round(line.get_ydata(), 12))) == 1
               and float(line.get_ydata()[0]) == 0.0
               for line in ax.lines)


def _compute_all(adata, **kw):
    """Run every ``tl`` so the cache is populated for the renderers."""
    cov, *rest = list(adata.uns[K.COVARIATE_CATEGORIES])
    tcri.tl.mutual_information(adata, covariate=cov, **kw)
    tcri.tl.clonotypic_entropy(adata, covariate=cov, **kw)
    tcri.tl.phenotypic_entropy(adata, covariate=cov, **kw)
    tcri.tl.phenotypic_flux(adata, cov_from=cov, cov_to=rest[0] if rest else cov, **kw)


def _views(name):
    """The views a twin draws: the replicate view, and the item view when it has items."""
    return ("replicate", "item") if name in ITEM_TWINS else ("replicate",)


def _change_key(name):
    """The ``key_added`` a change result is stored under by :func:`_compute_changes`.

    A change call replaces the metric's stored result under its own key, so a change kept beside
    a single-covariate result needs a key of its own.
    """
    return f"{name}_change"


def _call(adata, name, mode, **kw):
    """``tl.<name>`` at the first covariate level (``mode="covariate"``), or as the change from
    the first level to the second (``mode="change"``), stored under the metric's own key."""
    cov_from, cov_to = list(adata.uns[K.COVARIATE_CATEGORIES])[:2]
    at = (dict(covariate=cov_from) if mode == "covariate"
          else dict(cov_from=cov_from, cov_to=cov_to))
    getattr(tcri.tl, name)(adata, **at, **kw)


def _compute_changes(adata, **kw):
    """Every twin's change from the first covariate level to the second, each under
    :func:`_change_key`, beside whatever :func:`_compute_all` stored."""
    cov_from, cov_to = list(adata.uns[K.COVARIATE_CATEGORIES])[:2]
    for name in TWINS:
        getattr(tcri.tl, name)(adata, cov_from=cov_from, cov_to=cov_to,
                               key_added=_change_key(name), **kw)
    return cov_from, cov_to


#: Every (twin, view, mode) a stored result is drawn in: each view a twin has, at one covariate
#: and as a change. Flux is always a change.
RENDERED = [(name, view, mode) for name in TWINS for view in _views(name)
            for mode in (("change",) if name == "phenotypic_flux" else ("covariate", "change"))]


# ── the renderer contract ────────────────────────────────────────────────────

@pytest.mark.parametrize("name", TWINS)
def test_pl_takes_no_metric_arguments(name):
    """A ``pl`` twin's signature is ``(adata, key=, view=, display args)`` — nothing computable.

    A metric argument on both sides gives two places to choose it: a ``distance_metric`` on
    the plot that differs from the one the ``tl`` call used makes the axis label and the numbers
    under it describe different quantities. With none on the plot, the distance is chosen once.
    A change result is drawn by the same twin, so ``cov_from``/``cov_to`` stay off it too.
    """
    params = set(inspect.signature(getattr(tcri.pl, name)).parameters)
    assert not (params & METRIC_ARGS), f"pl.{name} can still compute: {sorted(params & METRIC_ARGS)}"
    assert "key" in params, "a twin must be able to render a non-default key_added"
    assert "view" in params, "a twin picks the level it draws with view="


@pytest.mark.parametrize("name", TWINS)
def test_pl_says_which_tool_to_run(name, trained_model):
    """Plotting before computing is the easy mistake, so the error names the call to run."""
    _, adata = trained_model
    adata = adata.copy()
    adata.uns.pop(getattr(K, name.upper()), None)
    with pytest.raises(KeyError, match=f"tcri.tl.{name}"):
        getattr(tcri.pl, name)(adata)


@pytest.mark.parametrize("name", TWINS)
def test_pl_never_recomputes(name, trained_model, monkeypatch):
    """Break the ``tl`` twin, then plot. A renderer that reaches for it fails here."""
    _, adata = trained_model
    adata = adata.copy()
    _compute_all(adata)

    def _boom(*a, **k):
        raise AssertionError(f"pl.{name} recomputed the metric")

    monkeypatch.setattr(tcri.tl, name, _boom)
    monkeypatch.setattr(f"tcri.tools.{name}", _boom, raising=False)
    assert getattr(tcri.pl, name)(adata) is not None


@pytest.mark.parametrize("name", TWINS)
def test_pl_return_df_is_the_cached_result(name, trained_model):
    """``return_df`` hands back the frame ``tcri.get`` does, for each view, at one covariate and
    on a change — not a private copy."""
    _, adata = trained_model
    adata = adata.copy()
    _compute_all(adata)
    import pandas as pd

    payload = tcri.get.result(adata, name)
    pd.testing.assert_frame_equal(
        getattr(tcri.pl, name)(adata, return_df=True), frames(payload, "replicate")[1],
    )
    if payload.get("item") is not None:
        pd.testing.assert_frame_equal(
            getattr(tcri.pl, name)(adata, return_df=True, view="item"), tcri.get.items(adata, name),
        )

    # ...and for a change result, read under its own key
    _compute_changes(adata)
    key = _change_key(name)
    pd.testing.assert_frame_equal(getattr(tcri.pl, name)(adata, key=key, return_df=True),
                                  tcri.get.replicates(adata, name, key=key))
    if name in ITEM_TWINS:
        pd.testing.assert_frame_equal(
            getattr(tcri.pl, name)(adata, key=key, return_df=True, view="item"),
            tcri.get.items(adata, name, key=key),
        )


@pytest.mark.parametrize("name", ["mutual_information", "clonotypic_entropy",
                                  "phenotypic_entropy"])
def test_a_change_frame_carries_its_endpoints(name, cohort):
    """There is no endpoints view: a change's two endpoints are stored beside it, so the frame
    ``return_df`` hands back carries them in each view. Flux, a distance between the two levels,
    has no endpoints and is not listed."""
    _, adata = cohort
    adata = adata.copy()
    _call(adata, name, "change", groupby="patient")
    for view in _views(name):
        frame = getattr(tcri.pl, name)(adata, view=view, return_df=True)
        assert {"value_from", "value_to", "adjusted_from", "adjusted_to"} <= set(frame.columns), (
            f"{name}, {view} view: the endpoints are not in the stored frame")


def test_pl_renders_the_axes_the_tl_call_used(trained_model):
    """The x axis comes from the cached ``params``, not from a plot argument.

    A box plot wants per-unit values, so it is tempting for the renderer to manufacture a
    ``groupby`` from ``batch_col`` when the caller passed none. That groups the figure by a
    column the caller never named and leaves no way to see the ungrouped metric: an ungrouped
    result must render ungrouped.
    """
    _, adata = trained_model
    adata = adata.copy()
    cov = _cov(adata)

    tcri.tl.mutual_information(adata, covariate=cov)
    ungrouped = tcri.pl.mutual_information(adata)
    assert _n_points(ungrouped) == 1, "an ungrouped MI is one number, not a box per patient"

    tcri.tl.mutual_information(adata, covariate=cov, groupby="patient")
    grouped = tcri.pl.mutual_information(adata)
    assert grouped.get_xlabel() == "patient"
    assert {t.get_text() for t in grouped.get_xticklabels()} == set(
        adata.obs["patient"].astype(str).unique()
    )

    # ...and it comes from `params`, not from the registry. `batch_col` happens to BE
    # "patient" in every fixture here, so a renderer that read the registry instead would
    # look identical -- break the registry and the plot must not notice.
    adata.uns[K.METADATA]["batch_col"] = "not_a_column"
    assert tcri.pl.mutual_information(adata).get_xlabel() == "patient"
    tcri.tl.mutual_information(adata, covariate=cov, key_added="mi_flat")
    assert _n_points(tcri.pl.mutual_information(adata, key="mi_flat")) == 1


def test_pl_key_selects_a_non_default_result(trained_model):
    """Two results in one object: ``key_added`` on the tool, ``key`` on the plot."""
    _, adata = trained_model
    adata = adata.copy()
    cov = _cov(adata)
    tcri.tl.mutual_information(adata, covariate=cov, groupby="patient")
    tcri.tl.mutual_information(adata, covariate=cov, key_added="mi_ungrouped")

    assert tcri.pl.mutual_information(adata).get_xlabel() == "patient"
    assert _n_points(tcri.pl.mutual_information(adata, key="mi_ungrouped")) == 1


def test_pl_brackets_the_contrast_only_on_the_split_axis(cohort):
    """Stars are drawn from ``stats``, and only when x IS the split.

    Annotating a contrast over an axis it was not computed on is how a figure comes to claim
    something the numbers never said.
    """
    _, adata = cohort
    adata = adata.copy()
    cov = _cov(adata)

    tcri.tl.mutual_information(adata, covariate=cov, groupby="patient", splitby="disease_status")
    split_ax = tcri.pl.mutual_information(adata)
    assert split_ax.get_xlabel() == "disease_status"

    stats = tcri.get.result(adata, "mutual_information")["stats"]
    # One row per contrast, on the adjusted value, since the result carries a reference. The
    # star drawn is the one for the quantity ON THE AXIS: the default panel draws `adjusted`,
    # so it carries that row's star; the value panel has no row about its quantity, so it
    # draws none rather than a star about a different column.
    assert stats is not None and set(stats["quantity"]) == {"adjusted"}
    assert len(stats) == 1, "one contrast between two arms is one row"
    row = stats.iloc[0]
    assert row["n_a"] == 3 and row["n_b"] == 3, "the contrast is not over the 3 patients per arm"
    assert [t.get_text() for t in split_ax.texts] == [row["stars"] or "ns"]

    value_ax = tcri.pl.mutual_information(adata, quantity="value")
    assert value_ax.get_xlabel() == "disease_status"
    assert not value_ax.texts, "the adjusted value's star was drawn over the value's marks"

    tcri.tl.mutual_information(adata, covariate=cov, groupby="patient")
    plain_ax = tcri.pl.mutual_information(adata)
    assert not plain_ax.texts, "a contrast was annotated on an axis with no contrast"

    # the replicate view puts the split on x, so its contrast is bracketed there
    tcri.tl.clonotypic_entropy(adata, covariate=cov, groupby="patient", splitby="disease_status")
    split_ax = tcri.pl.clonotypic_entropy(adata)
    assert split_ax.get_xlabel() == "disease_status"
    assert split_ax.texts, "the replicate view's contrast was not drawn on the split axis"

    # the case the guard actually exists for: `stats` IS present, but x is the phenotype
    # axis, so a disease-vs-control bracket over phenotype A vs phenotype B would be a claim the
    # numbers never made
    pheno_ax = tcri.pl.clonotypic_entropy(adata, view="item")
    assert pheno_ax.get_xlabel() == "phenotype"
    assert tcri.get.result(adata, "clonotypic_entropy")["stats"] is not None
    assert not pheno_ax.texts, "the cohort contrast was bracketed over the phenotype axis"

    # a change with a split: `stats` holds the covariate rows beside the split row, and the
    # bracket over the split axis is the split row's
    to = list(adata.uns[K.COVARIATE_CATEGORIES])[1]
    tcri.tl.mutual_information(adata, cov_from=cov, cov_to=to, groupby="patient",
                               splitby="disease_status")
    stats = tcri.get.stats(adata, "mutual_information")
    split = stats.loc[stats["comparison"] == "split"]
    assert len(split) == 1 and (stats["comparison"] == "covariate").any()
    change_ax = tcri.pl.mutual_information(adata)
    assert change_ax.get_xlabel() == "disease_status"
    assert _bracket_stars(change_ax).get((0.0, 1.0)) == [split.iloc[0]["stars"] or "ns"]


def test_a_split_named_like_a_covariate_level_gets_its_own_bracket(cohort):
    """A change's covariate rows and its split row can name the same two levels.

    Here the split levels are named as the covariate levels, so a covariate row's ``level_a`` and
    ``level_b`` match the two split positions on x. The bracket between them is the split row's:
    the stars are selected by ``comparison`` as well as by level, so a covariate row is never
    drawn as the contrast between the splits.
    """
    _, adata = cohort
    adata = adata.copy()
    cov_from, cov_to = list(adata.uns[K.COVARIATE_CATEGORIES])[:2]
    adata.obs["arm"] = (adata.obs["disease_status"].astype(str)
                        .map({"disease": cov_from, "control": cov_to}).astype("category"))

    for name in TWINS:
        getattr(tcri.tl, name)(adata, cov_from=cov_from, cov_to=cov_to, groupby="patient",
                               splitby="arm")
        stats = tcri.get.stats(adata, name)
        covariate = stats.loc[stats["comparison"] == "covariate"]
        split = stats.loc[stats["comparison"] == "split"]
        # the fixture: both kinds of row name the two levels on x
        assert len(covariate) == 2 and len(split) == 1, name
        assert {(r["level_a"], r["level_b"]) for _, r in covariate.iterrows()} == {
            (cov_from, cov_to)}, name
        assert {split.iloc[0]["level_a"], split.iloc[0]["level_b"]} == {cov_from, cov_to}, name

        ax = getattr(tcri.pl, name)(adata)
        assert ax.get_xlabel() == "arm"
        assert _ticks(ax) == [cov_from, cov_to], name
        assert _bracket_stars(ax).get((0.0, 1.0)) == [split.iloc[0]["stars"] or "ns"], name


# ── shapes each twin chooses ─────────────────────────────────────────────────

def test_entropy_twins_choose_their_own_x_axis(trained_model):
    """In the item view, clonotypic entropy has a handful of phenotypes (x axis); phenotypic
    entropy has the whole repertoire (a distribution). That difference is the metric's, not a
    style choice."""
    _, adata = trained_model
    adata = adata.copy()
    _compute_all(adata)

    ce = tcri.pl.clonotypic_entropy(adata, view="item")
    assert ce.get_xlabel() == "phenotype"
    assert {t.get_text() for t in ce.get_xticklabels()} == set(
        map(str, adata.uns[K.PHENOTYPE_CATEGORIES])
    )

    n_clones = frames(tcri.get.result(adata, "phenotypic_entropy"))[1]["clonotype"].nunique()
    pe = tcri.pl.phenotypic_entropy(adata, view="item")
    assert len(pe.get_xticklabels()) < n_clones, "one x tick per clone is not a plot"


def test_the_clone_item_view_puts_replicates_on_x_and_the_split_in_hue(cohort):
    """With replicates, phenotypic entropy's item view has one x position per replicate,
    holding that replicate's clones, and the split as the hue: every dot at a position takes
    the stored color of its replicate's arm."""
    from matplotlib.collections import PathCollection
    from tcri.plotting._base import REFERENCE_LABEL

    _, adata = cohort
    adata = adata.copy()
    tcri.tl.phenotypic_entropy(adata, covariate=_cov(adata), groupby="patient",
                               splitby="disease_status")
    ax = tcri.pl.phenotypic_entropy(adata, view="item")

    patients = [str(p) for p in adata.obs["patient"].cat.categories]
    assert ax.get_xlabel() == "patient"
    assert _ticks(ax) == patients

    arm = adata.obs.groupby("patient", observed=True)["disease_status"].first().astype(str)
    stored = dict(zip(adata.obs["disease_status"].cat.categories.astype(str),
                      (c.lower() for c in adata.uns[K.colors("disease_status")])))
    at = {}
    for coll in ax.collections:
        if isinstance(coll, PathCollection) and coll.get_label() != REFERENCE_LABEL:
            for (x, _y), c in zip(coll.get_offsets(), coll.get_facecolor()):
                at.setdefault(patients[int(round(x))], set()).add(
                    matplotlib.colors.to_hex(c).lower())
    assert set(at) == set(patients), f"positions with no clone: {set(patients) - set(at)}"
    for patient, colors in at.items():
        assert colors == {stored[arm[patient]]}, f"{patient}: {colors}"


def test_the_metric_twins_take_a_view(trained_model):
    """``view="replicate"`` draws one value per replicate, ``"item"`` one per replicate and
    item; mutual information has no items and raises for ``"item"``, at one covariate and on a
    change, and any other view raises."""
    _, adata = trained_model
    adata = adata.copy()
    _compute_all(adata)
    for name in TWINS:
        with pytest.raises(ValueError, match="view must be"):
            getattr(tcri.pl, name)(adata, view="items")
    with pytest.raises(ValueError, match="no item view"):
        tcri.pl.mutual_information(adata, view="item")
    for name in ITEM_TWINS:
        assert getattr(tcri.pl, name)(adata, view="item") is not None

    # a change of mutual information has no items either
    _compute_changes(adata)
    with pytest.raises(ValueError, match="no item view"):
        tcri.pl.mutual_information(adata, key=_change_key("mutual_information"), view="item")


def test_flux_labels_the_distance_it_was_computed_with(trained_model):
    """The y label names the distance recorded in ``params``, so it cannot describe a quantity
    other than the one plotted."""
    _, adata = trained_model
    adata = adata.copy()
    covs = list(adata.uns[K.COVARIATE_CATEGORIES])
    if len(covs) < 2:
        pytest.skip("needs >=2 covariates")

    tcri.tl.phenotypic_flux(adata, cov_from=covs[0], cov_to=covs[1], distance_metric="l1")
    assert "l1" in tcri.pl.phenotypic_flux(adata).get_ylabel()
    tcri.tl.phenotypic_flux(adata, cov_from=covs[0], cov_to=covs[1], distance_metric="kl")
    assert "kl" in tcri.pl.phenotypic_flux(adata).get_ylabel()


def test_draws_become_violins_not_a_summary(trained_model):
    """When draws are the sample, show the draws.

    A bar-plus-HDI is a lossy summary of a distribution the package already stored in
    ``table``. With no group and no item axis to vary, the draws ARE the sample, so the mark
    is a violin, for the value and for the adjusted value, whose draws are the paired
    differences. At ``n_samples=0`` there is no distribution and it degrades to a point with
    no interval — NOT a zero-width one, which would state certainty never measured.
    """
    _, adata = trained_model
    adata = adata.copy()
    cov = _cov(adata)
    from matplotlib.container import ErrorbarContainer

    tcri.tl.mutual_information(adata, covariate=cov, n_samples=0)
    plain = tcri.pl.mutual_information(adata)
    assert _n_violins(plain) == 0
    assert _n_points(plain) == 1
    assert not any(isinstance(c, ErrorbarContainer) for c in plain.containers)

    tcri.tl.mutual_information(adata, covariate=cov, n_samples=30, random_state=0)
    drawn = tcri.pl.mutual_information(adata, quantity="value")
    assert _n_violins(drawn) == 1, "the draws were summarized away instead of drawn"
    assert _n_points(drawn, reference=True) == 1, "the null was not drawn behind the violin"

    adjusted = tcri.pl.mutual_information(adata)
    assert _n_violins(adjusted) == 1, "the adjusted draws were summarized away"


def test_a_violin_never_spans_replicates(cohort):
    """Pooling draws across replicates is pseudoreplication drawn as a picture.

    6 patients x 30 draws would read as 180 samples and give a tight violin for a reason
    unrelated to evidence. With a replicate axis present the sample must be replicates.
    """
    _, adata = cohort
    adata = adata.copy()
    cov = _cov(adata)

    tcri.tl.mutual_information(adata, covariate=cov, groupby="patient", splitby="disease_status",
                               n_samples=30, random_state=0)
    ax = tcri.pl.mutual_information(adata)
    assert ax.get_xlabel() == "disease_status"
    assert _n_violins(ax) == 0, "a violin was drawn where replicates vary"
    assert _n_points(ax) == 6, "the dots are not the 6 patients"

    # ...and with the replicate ON x, each patient's own draws are its own violin
    tcri.tl.mutual_information(adata, covariate=cov, groupby="patient",
                               n_samples=30, random_state=0)
    per_patient = tcri.pl.mutual_information(adata, quantity="value")
    assert per_patient.get_xlabel() == "patient"
    assert _n_violins(per_patient) == 6, "one violin per patient over its own draws"

    # ...and so are its adjusted values, over the paired draws
    adjusted = tcri.pl.mutual_information(adata)
    assert adjusted.get_xlabel() == "patient"
    assert _n_violins(adjusted) == 6, "one violin per patient over its paired draws"


# ── colors ──────────────────────────────────────────────────────────────────

def test_resolve_colors_persists_and_is_reused(trained_model):
    """A level keeps its color across figures: the assignment is written to ``uns`` under
    scanpy's own ``<key>_colors`` name and read back from there, so a later figure reuses it
    rather than assigning afresh."""
    _, adata = trained_model
    adata = adata.copy()

    mapping = tcri.pl.resolve_colors(adata, "patient")
    assert set(mapping) == set(adata.obs["patient"].astype("category").cat.categories)
    assert adata.uns[K.colors("patient")] == list(mapping.values())

    # scanpy's own convention, so sc.pl.umap(color="patient") matches
    adata.uns[K.colors("patient")] = ["#000000"] * len(mapping)
    assert set(tcri.pl.resolve_colors(adata, "patient").values()) == {"#000000"}


def test_resolve_colors_palette_forms(trained_model):
    _, adata = trained_model
    adata = adata.copy()
    cats = list(adata.obs["patient"].astype("category").cat.categories)

    one = cats[0]
    partial = tcri.pl.resolve_colors(adata, "patient", palette={one: "#ff0000"})
    assert partial[one] == "#ff0000"
    assert all(v != "#ff0000" for k, v in partial.items() if k != one), (
        "a partial dict must fill its gaps from the canonical cycle, not repeat"
    )

    assert set(tcri.pl.resolve_colors(adata, "patient", palette=["#111111"]).values()) == {"#111111"}
    assert len(set(tcri.pl.resolve_colors(adata, "patient", palette="viridis").values())) == len(cats)

    before = list(adata.uns[K.colors("patient")])
    tcri.pl.resolve_colors(adata, "patient", palette=["#222222"], persist=False)
    assert adata.uns[K.colors("patient")] == before, "persist=False still wrote to uns"


def test_the_palette_has_one_definition():
    """One ``tcri_colors``, and it lives in ``tcri.pl``.

    A second copy in ``tcri.utils._utils`` can disagree on contents and order, so the same
    level draws in a different color depending on which list the caller reached. The entries
    must also be distinct, and the first must be a category color rather than the dark editor
    background (``#272822``) that ships alongside such palettes."""
    from tcri.utils import _utils

    assert not hasattr(_utils, "tcri_colors")
    assert len(tcri.pl.tcri_colors) == len(set(tcri.pl.tcri_colors)), "duplicate color"
    assert tcri.pl.tcri_colors[0] != "#272822", "the background color is not a category color"


def test_plots_route_through_the_shared_palette(trained_model):
    """A plot must not invent its own colors — otherwise the same patient changes color
    between two figures in one notebook.

    Checked on the strip dots rather than the boxes: seaborn bakes ``boxprops.alpha`` into
    the box facecolor, so a box reads back as the palette color blended toward white and
    never compares equal to what was stored.
    """
    _, adata = trained_model
    adata = adata.copy()
    cov = _cov(adata)
    tcri.tl.mutual_information(adata, covariate=cov, groupby="patient")

    ax = tcri.pl.mutual_information(adata)
    assert K.colors("patient") in adata.uns
    stored = {c.lower() for c in adata.uns[K.colors("patient")]}
    drawn = {matplotlib.colors.to_hex(c).lower()
             for coll in ax.collections for c in coll.get_facecolor()}
    assert drawn and drawn <= stored, f"colors off the shared palette: {drawn - stored}"
    

    # and a stored assignment wins on the next figure, which is the whole point of persisting
    fixed = ["#123456", "#654321"][: len(stored)]
    adata.uns[K.colors("patient")] = fixed
    again = tcri.pl.mutual_information(adata)
    redrawn = {matplotlib.colors.to_hex(c).lower()
               for coll in again.collections for c in coll.get_facecolor()}
    assert redrawn <= set(fixed), f"the stored palette did not reach the canvas: {redrawn}"


# ── the mark and the statistic must describe one unit ────────────────────────

@pytest.mark.parametrize(("metric", "mode"),
                         [(m, mode) for m, view, mode in RENDERED if view == "replicate"])
def test_the_dots_are_the_same_unit_as_the_p_value(metric, mode, cohort):
    """The marks and the p-value drawn above them must describe the same unit.

    The item summary of an entropy or of flux carries one row per (replicate, item) — patient x
    clone — so a strip drawn from it would show clones while the bracket from ``stats`` tests
    patients. The default view draws the replicate summary, one mark per patient, the rows the
    contrast is run on, at one covariate and on a change. ``mutual_information`` has no item
    axis and cannot show the difference, which is why the parametrization covers every twin.
    """
    _, adata = cohort
    adata = adata.copy()

    from tcri._compute._tables import collapse_to_replicates

    _call(adata, metric, mode, groupby="patient", splitby="disease_status")
    res = tcri.get.replicates(adata, metric)
    ax = getattr(tcri.pl, metric)(adata)

    # whatever the x axis is, every dot is a replicate -- never an item
    expected = collapse_to_replicates(
        res, groupby="patient",
        keep=[c for c in (ax.get_xlabel(), "disease_status") if c in res.columns],
    )
    assert len(expected) == len(res), f"{metric}: the replicate summary repeats a replicate"
    assert _n_points(ax) == len(expected), (
        f"{metric}: {_n_points(ax)} dots against {len(expected)} replicate rows"
    )
    assert _n_points(ax) % 6 == 0, f"{metric}: dots are not a whole number of patients"


@pytest.mark.parametrize("mode", ["covariate", "change"])
def test_the_plot_draws_the_values_the_contrast_ran_on(mode, cohort):
    """Not merely the same COUNT — the same values: the replicate view's marks are the
    replicate summary, the frame ``build_stats`` tests, so the figure cannot rest on other
    numbers than the contrast drawn over it. On a change the split row tests each replicate's
    change, the value the mark draws.
    """
    _, adata = cohort
    adata = adata.copy()
    _call(adata, "phenotypic_entropy", mode, groupby="patient", splitby="disease_status")
    res = tcri.get.replicates(adata, "phenotypic_entropy")

    # the default panel draws the adjusted value, the column `build_stats` tests
    expected = res.dropna(subset=["adjusted"])
    ax = tcri.pl.phenotypic_entropy(adata)
    assert np.allclose(_mark_ys(ax), np.sort(expected["adjusted"].to_numpy()))
    assert not len(_mark_ys(ax, reference=True)), "a gray mark was drawn on the adjusted panel"
    stats = tcri.get.stats(adata, "phenotypic_entropy")
    assert set(stats["quantity"]) == {"adjusted"}
    split = stats.loc[stats["comparison"] == "split"]
    assert len(split) == 1
    row = split.iloc[0]
    assert len(expected) == row["n_a"] + row["n_b"], (
        "the marks and the contrast rest on different replicate sets")

    # the value panel draws the value and its reference from the same replicate rows
    expected = res.dropna(subset=["value", "null_value"])
    value_ax = tcri.pl.phenotypic_entropy(adata, quantity="value")
    assert np.allclose(_mark_ys(value_ax), np.sort(expected["value"].to_numpy()))
    assert np.allclose(_mark_ys(value_ax, reference=True),
                       np.sort(expected["null_value"].to_numpy()))


@pytest.mark.parametrize("mode", ["covariate", "change"])
def test_the_replicate_marks_are_the_replicate_values_not_the_item_means(mode, cohort):
    """Clonotypic entropy's replicate view draws each replicate's H(C|P), the stored replicate
    summary, which weights each phenotype by its share of the replicate's joint. That is not
    the plain mean of the replicate's item values, and on this fixture the two differ, so the
    marks matching the replicate summary rules out marks drawn from the item means.

    On a change the replicate mark is the difference of the replicate's two values, each over
    every clone at its level, and the item rows are the per-phenotype changes over the
    persistent clones; an average of those item changes is not drawn either.
    """
    _, adata = cohort
    adata = adata.copy()
    _call(adata, "clonotypic_entropy", mode, groupby="patient", splitby="disease_status")
    expected = tcri.get.replicates(adata, "clonotypic_entropy").dropna(subset=["adjusted"])
    item_means = (tcri.get.items(adata, "clonotypic_entropy").dropna(subset=["adjusted"])
                  .groupby("patient", observed=True)["adjusted"].mean())

    ax = tcri.pl.clonotypic_entropy(adata)
    drawn = _mark_ys(ax)
    np.testing.assert_allclose(drawn, np.sort(expected["adjusted"].to_numpy()), atol=1e-12)
    assert not len(_mark_ys(ax, reference=True)), "a gray mark was drawn on the adjusted panel"
    assert len(item_means) == len(drawn)
    assert not np.allclose(drawn, np.sort(item_means.to_numpy())), (
        "the replicate values equal the item means here, so the check above cannot tell them "
        "apart")


@pytest.mark.parametrize("quantity", ["auto", "value"])
@pytest.mark.parametrize("splitby", [None, "disease_status"])
@pytest.mark.parametrize(("metric", "view", "mode"), RENDERED,
                         ids=[f"{m}-{v}-{mode}" for m, v, mode in RENDERED])
def test_the_marks_are_the_stored_summary_of_their_view(metric, view, mode, splitby, quantity,
                                                        cohort):
    """Every mark is a row of the stored summary of its view, and every row is drawn: the
    replicate view draws ``tcri.get.replicates`` and the item view ``tcri.get.items``, for every
    twin at one covariate and on a change, with and without a split.

    Clonotypic entropy's item view collapses to one value per (replicate, phenotype), which on
    the item summary is one row each; the clone item views draw one mark per (replicate, clone)
    row. ``"auto"`` draws the adjusted value with no gray marks; ``"value"`` draws the value
    with its reference, gray, from the same rows.
    """
    _, adata = cohort
    adata = adata.copy()
    _call(adata, metric, mode, groupby="patient", splitby=splitby)
    frame = (tcri.get.items if view == "item" else tcri.get.replicates)(adata, metric)
    ax = getattr(tcri.pl, metric)(adata, view=view, quantity=quantity)

    if quantity == "auto":
        expected = frame.dropna(subset=["adjusted"])
        assert len(expected), "the fixture has no finite adjusted value to draw"
        np.testing.assert_allclose(_mark_ys(ax), np.sort(expected["adjusted"].to_numpy()),
                                   atol=1e-12)
        assert not len(_mark_ys(ax, reference=True)), (
            "a gray mark was drawn on the adjusted panel")
    else:
        expected = frame.dropna(subset=["value", "null_value"])
        np.testing.assert_allclose(_mark_ys(ax), np.sort(expected["value"].to_numpy()),
                                   atol=1e-12)
        np.testing.assert_allclose(_mark_ys(ax, reference=True),
                                   np.sort(expected["null_value"].to_numpy()), atol=1e-12)


def test_nothing_connects_two_x_positions(cohort):
    """A line between x positions claims the two points are the same entity observed twice.

    Only matched data supports that, and these twins render unmatched summaries. Box internals
    (whiskers, caps, medians) stay within one box; a connector spans categories, which are 1.0
    apart — so the span is the discriminator.

    This is a prohibition, not a feature: a connector must be drawn from an identity key, never
    from adjacency, so adding one is a deliberate change to this test rather than a side effect
    of a style change.

    A change result is drawn the same way, in each of its views: a replicate's or an item's
    change is one mark, never its two endpoints joined.
    """
    _, adata = cohort
    adata = adata.copy()
    _compute_all(adata, groupby="patient", splitby="disease_status")
    _compute_changes(adata, groupby="patient", splitby="disease_status")

    panels = [(name, None, "replicate") for name in TWINS] + [
        (name, _change_key(name), view) for name in TWINS for view in _views(name)]
    for name, key, view in panels:
        ax = getattr(tcri.pl, name)(adata, key=key, view=view)
        for line in ax.lines:
            if line.get_label() == BRACKET_LABEL:
                continue      # a significance bracket spans x by design
            xs = np.asarray(line.get_xdata(), dtype=float)
            if xs.size < 2 or not np.isfinite(xs).all():
                continue
            assert xs.max() - xs.min() < 1.0, (
                f"pl.{name}(key={key!r}, view={view!r}) drew a line spanning x positions "
                f"{xs.min()}..{xs.max()}"
            )


def test_a_levels_color_does_not_depend_on_its_position(trained_model):
    """The color is a property of the level, not of where it lands on this figure.

    The renderer sorts x by median, so two panels of the same variable can present its levels
    in different orders. A `resolve_colors` that zipped the stored hex list against the caller's
    display order would then give one level two colors in one figure, so the lookup is per
    level: a reversed order, a subset, and a key with no obs column all keep each level's
    color.
    """
    _, adata = trained_model
    adata = adata.copy()
    cats = list(adata.obs["patient"].astype("category").cat.categories)

    forward = tcri.pl.resolve_colors(adata, "patient", cats)
    reverse = tcri.pl.resolve_colors(adata, "patient", list(reversed(cats)))
    assert forward == reverse, "the color followed the order, not the level"

    # a subset must keep each level's color rather than restarting the cycle
    subset = tcri.pl.resolve_colors(adata, "patient", cats[1:])
    assert all(subset[c] == forward[c] for c in cats[1:])

    # and the same holds for a key with no obs column to appeal to
    a = tcri.pl.resolve_colors(adata, "phenotype", ["x", "y", "z"], persist=False)
    b = tcri.pl.resolve_colors(adata, "phenotype", ["z", "y", "x"], persist=False)
    assert a == b


def test_the_same_level_keeps_its_color_across_panels(cohort):
    """The end-to-end version: two panels of the same split, drawn in different orders."""
    _, adata = cohort
    adata = adata.copy()
    cov, *rest = list(adata.uns[K.COVARIATE_CATEGORIES])

    def _color_by_level(ax):
        out = {}
        for coll in ax.collections:
            for (x, _y), c in zip(coll.get_offsets(), coll.get_facecolor()):
                lvl = ax.get_xticklabels()[int(round(x))].get_text()
                out.setdefault(lvl, set()).add(matplotlib.colors.to_hex(c).lower())
        return out

    tcri.tl.phenotypic_entropy(adata, covariate=cov, groupby="patient", splitby="disease_status")
    first = _color_by_level(tcri.pl.phenotypic_entropy(adata))
    tcri.tl.phenotypic_flux(adata, cov_from=cov, cov_to=rest[0], groupby="patient",
                            splitby="disease_status")
    second = _color_by_level(tcri.pl.phenotypic_flux(adata))

    for level in set(first) & set(second):
        assert first[level] == second[level], (
            f"{level!r} was {first[level]} in one panel and {second[level]} in the next"
        )


# ── changes between covariates ───────────────────────────────────────────────

@pytest.mark.parametrize(("metric", "view"), [(n, v) for n in TWINS for v in _views(n)])
def test_a_change_marks_zero(metric, view, cohort):
    """Zero is a real position on a change axis: an unmarked one hides the direction.

    A change can be negative, so its value panel and its adjusted panel both carry a zero rule,
    in each view. Flux is a distance between the two levels and is never negative, so only its
    adjusted panel, whose zero is the reference, carries one.
    """
    _, adata = cohort
    adata = adata.copy()
    _call(adata, metric, "change", groupby="patient", splitby="disease_status")

    for quantity in ("auto", "adjusted"):
        ax = getattr(tcri.pl, metric)(adata, view=view, quantity=quantity)
        assert _has_zero_rule(ax), f"the adjusted change axis has no zero rule ({quantity!r})"
    value_ax = getattr(tcri.pl, metric)(adata, view=view, quantity="value")
    if metric == "phenotypic_flux":
        assert not _has_zero_rule(value_ax), "a zero rule was drawn on the flux value axis"
    else:
        assert _has_zero_rule(value_ax), "the change's value axis has no zero rule"


@pytest.mark.parametrize("metric", ["mutual_information", "clonotypic_entropy",
                                    "phenotypic_entropy", "phenotypic_flux"])
def test_a_change_axis_says_it_is_a_change(metric, cohort):
    """A change panel of mutual information or an entropy prefixes its y label with Δ, on the
    adjusted and the value panel alike; flux is between two levels by definition and keeps its
    label."""
    _, adata = cohort
    adata = adata.copy()
    _call(adata, metric, "change", groupby="patient", splitby="disease_status")
    adjusted = getattr(tcri.pl, metric)(adata).get_ylabel().replace("\n", " ")
    value = getattr(tcri.pl, metric)(adata, quantity="value").get_ylabel().replace("\n", " ")
    if metric == "phenotypic_flux":
        assert "Δ" not in adjusted and "Δ" not in value, (adjusted, value)
        return
    assert adjusted.startswith("adjusted Δ "), adjusted
    assert value.startswith("Δ "), value
    _call(adata, metric, "covariate", groupby="patient")
    assert "Δ" not in getattr(tcri.pl, metric)(adata).get_ylabel()


def test_the_flux_item_view_draws_each_persistent_clone(trained_model):
    """``pl.phenotypic_flux(view="item")`` draws one mark per persistent clone within each
    replicate: the clones ``pp.clone_persistence`` finds at both levels, each at its replicate's
    x position with its stored item value. A clone seen at one level only has no item row and
    no mark.
    """
    from tcri.plotting._base import REFERENCE_LABEL

    _, adata = trained_model
    adata = adata.copy()
    cov_from, cov_to = list(adata.uns[K.COVARIATE_CATEGORIES])[:2]
    # the fixture's clonotype and covariate columns
    persistence = tcri.pp.clone_persistence(adata, clonotype_key="unique_clone_id",
                                            covariate_key="timepoint", cov_from=cov_from,
                                            cov_to=cov_to, groupby="patient")
    persistence = persistence.set_index("patient")
    assert ((persistence["clones_from_only"] + persistence["clones_to_only"]) > 0).all(), (
        "the fixture has a patient whose every clone persists, so nothing is left out there")
    per_clone = tcri.pp.clone_persistence(adata, clonotype_key="unique_clone_id",
                                          covariate_key="timepoint", cov_from=cov_from,
                                          cov_to=cov_to, groupby="patient", per_clone=True)
    persistent = per_clone.loc[(per_clone[str(cov_from)] > 0) & (per_clone[str(cov_to)] > 0)]

    tcri.tl.phenotypic_flux(adata, cov_from=cov_from, cov_to=cov_to, groupby="patient")
    items = tcri.get.items(adata, "phenotypic_flux")
    for patient, clones in persistent.groupby("patient")["clonotype"]:
        assert set(items.loc[items["patient"].astype(str) == patient, "clonotype"]) == set(
            clones), f"{patient}: the item rows are not the persistent clones"

    ax = tcri.pl.phenotypic_flux(adata, view="item")
    assert ax.get_xlabel() == "patient"
    patients = _ticks(ax)
    drawn = {}
    for coll in ax.collections:
        if coll.get_label() == REFERENCE_LABEL:
            continue
        for x, y in coll.get_offsets():
            drawn.setdefault(patients[int(round(x))], []).append(float(y))
    assert {p: len(ys) for p, ys in drawn.items()} == (
        persistence["persistent_clones"].to_dict()), "not one mark per persistent clone"
    for patient, ys in drawn.items():
        want = items.loc[items["patient"].astype(str) == patient, "adjusted"].to_numpy()
        np.testing.assert_allclose(np.sort(ys), np.sort(want), atol=1e-12, err_msg=patient)


@pytest.mark.parametrize("name", TWINS)
def test_a_change_without_replicates_is_one_position(name, trained_model):
    """With no replicate registered the whole object is one replicate, so its change is one
    value, drawn at one x position labeled ``"cov_from → cov_to"``, with its reference behind it
    on the value panel."""
    _, adata = trained_model
    adata = adata.copy()
    assert adata.uns[K.METADATA].get(K.Config.REPLICATE) is None, (
        "the fixture registers a replicate")
    cov_from, cov_to = list(adata.uns[K.COVARIATE_CATEGORIES])[:2]
    getattr(tcri.tl, name)(adata, cov_from=cov_from, cov_to=cov_to)
    row = tcri.get.replicates(adata, name)
    assert len(row) == 1

    for quantity, column in (("auto", "adjusted"), ("value", "value")):
        ax = getattr(tcri.pl, name)(adata, quantity=quantity)
        assert _ticks(ax) == [f"{cov_from} → {cov_to}"], quantity
        assert _n_points(ax) == 1, quantity
        np.testing.assert_allclose(_mark_ys(ax), row[column].to_numpy(), atol=1e-12)
        if quantity == "value":
            np.testing.assert_allclose(_mark_ys(ax, reference=True),
                                       row["null_value"].to_numpy(), atol=1e-12)


# ── small multiples ──────────────────────────────────────────────────────────

def _ticks(ax):
    return [t.get_text() for t in ax.get_xticklabels()]


def test_panels_given_the_same_categories_share_an_x_order(cohort):
    """The x axis is in category order, never sorted by a panel's own values, so a level sits
    at the same position in every panel of a row; ``order=`` overrides it."""
    _, adata = cohort
    adata = adata.copy()
    a, b = list(adata.uns[K.COVARIATE_CATEGORIES])[:2]
    phenotypes = [str(p) for p in adata.uns[K.PHENOTYPE_CATEGORIES]]
    patients = [str(p) for p in adata.obs["patient"].cat.categories]
    splits = [str(s) for s in adata.obs["disease_status"].cat.categories]

    for cov, key in ((a, "ce_a"), (b, "ce_b")):
        tcri.tl.clonotypic_entropy(adata, covariate=cov, groupby="patient",
                                   splitby="disease_status", key_added=key)
        tcri.tl.mutual_information(adata, covariate=cov, groupby="patient",
                                   splitby="disease_status", key_added=f"mi_{cov}")
        tcri.tl.mutual_information(adata, covariate=cov, groupby="patient",
                                   key_added=f"mi_rep_{cov}")
    for key in ("ce_a", "ce_b"):
        for quantity in ("auto", "value"):
            assert _ticks(tcri.pl.clonotypic_entropy(adata, key=key, quantity=quantity,
                                                     view="item")) == phenotypes
    for cov in (a, b):
        assert _ticks(tcri.pl.mutual_information(adata, key=f"mi_{cov}")) == splits
        assert _ticks(tcri.pl.mutual_information(adata, key=f"mi_rep_{cov}")) == patients

    assert _ticks(tcri.pl.clonotypic_entropy(adata, key="ce_a", view="item",
                                             order=phenotypes[::-1])) == phenotypes[::-1]


def _all_twins(adata, model):
    """Every metric twin and gene importance, computed with replicates and a split, as
    ``name -> callable(**display kwargs)``: each twin's default result, each twin's change in
    every view it has, and gene importance."""
    kw = dict(groupby="patient", splitby="disease_status")
    _compute_all(adata, **kw)
    _compute_changes(adata, **kw)
    tcri.perturb.gene_importance(model, adata, genes=[0, 1, 2], **kw)
    draw = {name: (lambda name=name, **k: getattr(tcri.pl, name)(adata, **k))
            for name in TWINS + ["gene_importance"]}
    for name in TWINS:
        for view in _views(name):
            draw[f"{name}:change:{view}"] = (lambda name=name, view=view, **k: getattr(
                tcri.pl, name)(adata, key=_change_key(name), view=view, **k))
    return draw


def test_legend_false_draws_none(cohort):
    """``legend=False`` draws no legend at all, so a row of panels can carry one."""
    model, adata = cohort
    adata = adata.copy()
    for name, draw in _all_twins(adata, model).items():
        # gene importance draws its one quantity; its legend is the split's
        kw = {} if name == "gene_importance" else {"quantity": "value"}
        assert draw(**kw).get_legend() is not None, f"{name}: no legend to remove"
        assert draw(**kw, legend=False).get_legend() is None, name


def test_replicates_on_an_item_axis_are_colored_by_replicate(cohort):
    """Each phenotype's dots are the replicates, each in its replicate's stored color, under one
    box color: a color per phenotype would make every dot at a position the same color."""
    from matplotlib.collections import PathCollection

    _, adata = cohort
    adata = adata.copy()
    cov = _cov(adata)
    tcri.tl.clonotypic_entropy(adata, covariate=cov, groupby="patient")
    ax = tcri.pl.clonotypic_entropy(adata, view="item")

    stored = dict(zip(adata.obs["patient"].cat.categories,
                      (c.lower() for c in adata.uns[K.colors("patient")])))
    at = {}
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            for (x, _y), c in zip(coll.get_offsets(), coll.get_facecolor()):
                at.setdefault(int(round(x)), set()).add(matplotlib.colors.to_hex(c).lower())
    assert len(at) == len(adata.uns[K.PHENOTYPE_CATEGORIES])
    for x, colors in at.items():
        assert colors == set(stored.values()), f"position {x}: {colors}"

    boxes = {matplotlib.colors.to_hex(p.get_facecolor(), keep_alpha=False)
             for p in ax.patches}
    assert len(boxes) == 1, f"the boxes are colored per phenotype: {boxes}"
    legend = ax.get_legend()
    assert legend is not None and legend.get_title().get_text() == "patient"
    assert [t.get_text() for t in legend.get_texts()] == list(stored)


def test_a_long_y_label_fits_its_axes(cohort):
    """A label longer than its axes is wrapped onto more lines rather than cut off at the
    figure edge; every word is kept."""
    model, adata = cohort
    adata = adata.copy()
    import matplotlib.pyplot as plt

    for name, draw in _all_twins(adata, model).items():
        fig, ax = plt.subplots(figsize=(3, 1.6))
        draw(ax=ax)
        fig.canvas.draw()
        label = ax.yaxis.label
        assert label.get_window_extent().height <= ax.get_window_extent().height + 1, name
        assert label.get_text().split()
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3, 1.6))
    tcri.pl.gene_importance(adata, ax=ax)
    assert "\n" in ax.get_ylabel()
    assert " ".join(ax.get_ylabel().split()) == "adjusted gene importance (L1 shift of the phenotype call)"
    plt.close(fig)
