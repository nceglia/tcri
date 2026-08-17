"""The four ``pl``↔``tl`` twins as CACHE RENDERERS, plus ``resolve_colors``.

The load-bearing property here is not "the plot renders". It is that the plot renders the
*stored* result and nothing else: no recompute, no metric arguments of its own, no axis the
``tl`` call did not use. Before this, every ``pl`` called its ``tl`` twin internally, so the
figure and the frame in the caller's hand could differ — and ``pl.mutual_information`` and
``pl.phenotypic_flux`` invented a ``groupby`` from ``batch_col`` when none was given.
"""
import inspect

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import tcri
from tcri import _keys as K
from tcri.plotting._base import BRACKET_LABEL

TWINS = ["mutual_information", "clonotypic_entropy", "phenotypic_entropy", "phenotypic_flux"]

#: Anything on this list reaching a `pl` signature means the plot can compute something.
METRIC_ARGS = {
    "covariate", "cov_from", "cov_to", "groupby", "splitby", "n_samples", "temperature",
    "clones", "weighted", "normalized", "normalize_mode", "distance_metric", "random_state",
    "device", "inplace", "key_added",
}


def _cov(adata):
    return list(adata.uns[K.COVARIATE_CATEGORIES])[0]


def _n_points(ax):
    """Markers drawn by the point mark — the floor of the mark rule."""
    return sum(len(c.get_offsets()) for c in ax.collections)


def _n_violins(ax):
    from matplotlib.collections import PolyCollection

    return sum(isinstance(c, PolyCollection) for c in ax.collections)


def _compute_all(adata, **kw):
    """Run every ``tl`` so the cache is populated for the renderers."""
    cov, *rest = list(adata.uns[K.COVARIATE_CATEGORIES])
    tcri.tl.mutual_information(adata, covariate=cov, **kw)
    tcri.tl.clonotypic_entropy(adata, covariate=cov, **kw)
    tcri.tl.phenotypic_entropy(adata, covariate=cov, **kw)
    tcri.tl.phenotypic_flux(adata, cov_from=cov, cov_to=rest[0] if rest else cov, **kw)


# ── the renderer contract ────────────────────────────────────────────────────

@pytest.mark.parametrize("name", TWINS)
def test_pl_takes_no_metric_arguments(name):
    """A ``pl`` twin's signature is ``(adata, key=, display args)`` — nothing computable.

    This is the structural fix for the ``tl``/``pl`` ``distance_metric`` disagreement
    (``"kl"`` vs ``"l1"``): with no ``distance_metric`` on the plot there is one place the
    distance is chosen, so the axis label and the numbers under it cannot describe different
    quantities.
    """
    params = set(inspect.signature(getattr(tcri.pl, name)).parameters)
    assert not (params & METRIC_ARGS), f"pl.{name} can still compute: {sorted(params & METRIC_ARGS)}"
    assert "key" in params, "a twin must be able to render a non-default key_added"


@pytest.mark.parametrize("name", TWINS)
def test_pl_says_which_tool_to_run(name, trained_model):
    """Plotting before computing is now the easy mistake, so the error names the call."""
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
    """``return_df`` hands back the same object ``tcri.get`` does — not a private copy."""
    _, adata = trained_model
    adata = adata.copy()
    _compute_all(adata)
    import pandas as pd

    pd.testing.assert_frame_equal(
        getattr(tcri.pl, name)(adata, return_df=True),
        tcri.get.result(adata, name)["result"],
    )


def test_pl_renders_the_axes_the_tl_call_used(trained_model):
    """The x axis comes from the cached ``params``, not from a plot argument.

    ``pl.mutual_information`` used to manufacture ``groupby`` from ``batch_col`` whenever the
    caller passed none, because a box plot needs per-unit values — so the figure was grouped
    by a column the caller never named, and a caller who genuinely wanted the ungrouped MI
    could not get it. Now an ungrouped result renders ungrouped.
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

    tcri.tl.mutual_information(adata, covariate=cov, groupby="patient", splitby="response")
    split_ax = tcri.pl.mutual_information(adata)
    assert split_ax.get_xlabel() == "response"

    stats = tcri.get.result(adata, "mutual_information")["stats"]
    assert stats is not None and len(stats) == 1
    row = stats.iloc[0]
    assert row["n_a"] == 3 and row["n_b"] == 3, "the contrast is not over the 3 patients per arm"
    assert [t.get_text() for t in split_ax.texts] == [row["stars"] or "ns"]

    tcri.tl.mutual_information(adata, covariate=cov, groupby="patient")
    plain_ax = tcri.pl.mutual_information(adata)
    assert not plain_ax.texts, "a contrast was annotated on an axis with no contrast"

    # the case the guard actually exists for: `stats` IS present, but x is the phenotype
    # axis, so an R-vs-NR bracket over phenotype A vs phenotype B would be a claim the
    # numbers never made
    tcri.tl.clonotypic_entropy(adata, covariate=cov, groupby="patient", splitby="response")
    pheno_ax = tcri.pl.clonotypic_entropy(adata)
    assert pheno_ax.get_xlabel() == "phenotype"
    assert tcri.get.result(adata, "clonotypic_entropy")["stats"] is not None
    assert not pheno_ax.texts, "the response contrast was bracketed over the phenotype axis"


# ── shapes each twin chooses ─────────────────────────────────────────────────

def test_entropy_twins_choose_their_own_x_axis(trained_model):
    """Clonotypic entropy has a handful of phenotypes (x axis); phenotypic entropy has the
    whole repertoire (a distribution). That difference is the metric's, not a style choice."""
    _, adata = trained_model
    adata = adata.copy()
    _compute_all(adata)

    ce = tcri.pl.clonotypic_entropy(adata)
    assert ce.get_xlabel() == "phenotype"
    assert {t.get_text() for t in ce.get_xticklabels()} == set(
        map(str, adata.uns[K.PHENOTYPE_CATEGORIES])
    )

    n_clones = tcri.get.result(adata, "phenotypic_entropy")["result"]["clonotype"].nunique()
    pe = tcri.pl.phenotypic_entropy(adata)
    assert len(pe.get_xticklabels()) < n_clones, "one x tick per clone is not a plot"


def test_flux_labels_the_distance_it_was_computed_with(trained_model):
    """``tl`` defaulted to ``kl`` and ``pl`` to ``l1``, so the label could lie about the data."""
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
    is a violin. At ``n_samples=0`` there is no distribution and it degrades to a point with
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
    drawn = tcri.pl.mutual_information(adata)
    assert _n_violins(drawn) == 1, "the draws were summarized away instead of drawn"


def test_a_violin_never_spans_replicates(cohort):
    """Pooling draws across replicates is pseudoreplication drawn as a picture.

    6 patients x 30 draws would read as 180 samples and give a tight violin for a reason
    unrelated to evidence. With a replicate axis present the sample must be replicates.
    """
    _, adata = cohort
    adata = adata.copy()
    cov = _cov(adata)

    tcri.tl.mutual_information(adata, covariate=cov, groupby="patient", splitby="response",
                               n_samples=30, random_state=0)
    ax = tcri.pl.mutual_information(adata)
    assert ax.get_xlabel() == "response"
    assert _n_violins(ax) == 0, "a violin was drawn where replicates vary"
    assert _n_points(ax) == 6, "the dots are not the 6 patients"

    # ...and with the replicate ON x, each patient's own draws are its own violin
    tcri.tl.mutual_information(adata, covariate=cov, groupby="patient",
                               n_samples=30, random_state=0)
    per_patient = tcri.pl.mutual_information(adata)
    assert per_patient.get_xlabel() == "patient"
    assert _n_violins(per_patient) == 6, "one violin per patient over its own draws"


# ── colours ──────────────────────────────────────────────────────────────────

def test_resolve_colors_persists_and_is_reused(trained_model):
    """A level keeps its colour across figures, which ``resolve_palette`` could not promise:
    it had no way to READ an existing assignment, so it reassigned on every call."""
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
    """There were two ``tcri_colors`` — this one and a 30-entry list in ``utils/_utils.py``
    that led with the Monokai *background*, so the first category rendered near-black. They
    disagreed on contents and order, and nothing imported the utils copy."""
    from tcri.utils import _utils

    assert not hasattr(_utils, "tcri_colors")
    assert len(tcri.pl.tcri_colors) == len(set(tcri.pl.tcri_colors)), "duplicate colour"
    assert tcri.pl.tcri_colors[0] != "#272822", "the background colour is not a category colour"


def test_plots_route_through_the_shared_palette(trained_model):
    """A plot must not invent its own colours — otherwise the same patient changes colour
    between two figures in one notebook.

    Checked on the strip dots rather than the boxes: seaborn bakes ``boxprops.alpha`` into
    the box facecolour, so a box reads back as the palette colour blended toward white and
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
    assert drawn and drawn <= stored, f"colours off the shared palette: {drawn - stored}"
    

    # and a stored assignment wins on the next figure, which is the whole point of persisting
    fixed = ["#123456", "#654321"][: len(stored)]
    adata.uns[K.colors("patient")] = fixed
    again = tcri.pl.mutual_information(adata)
    redrawn = {matplotlib.colors.to_hex(c).lower()
               for coll in again.collections for c in coll.get_facecolor()}
    assert redrawn <= set(fixed), f"the stored palette did not reach the canvas: {redrawn}"


# ── the mark and the statistic must describe one unit ────────────────────────

@pytest.mark.parametrize("metric", ["phenotypic_entropy", "clonotypic_entropy",
                                    "mutual_information"])
def test_the_dots_are_the_same_unit_as_the_p_value(metric, cohort):
    """Measured before this fix, on ``phenotypic_entropy(groupby, splitby)``::

        x axis        : response
        result rows   : 47 (one per patient x clone)
        strip dots    : 47
        stats n_a/n_b : 3 / 3   p = 0.1

    The box and strip described 47 clones; the p-value bracketed above them described 6
    patients. That is the pseudoreplication ``build_stats`` collapses away, surviving in the
    marks — and it is invisible on ``mutual_information``, which has no item axis, so the
    parametrization matters.
    """
    _, adata = cohort
    adata = adata.copy()
    cov = _cov(adata)

    from tcri.tools._common import collapse_to_replicates

    res = getattr(tcri.tl, metric)(adata, covariate=cov, groupby="patient",
                                   splitby="response")
    ax = getattr(tcri.pl, metric)(adata)

    # whatever the x axis is, every dot is a replicate -- never an item
    expected = collapse_to_replicates(
        res["result"], groupby="patient",
        keep=[c for c in (ax.get_xlabel(), "response") if c in res["result"].columns],
    )
    assert _n_points(ax) == len(expected), (
        f"{metric}: {_n_points(ax)} dots against {len(expected)} replicate rows "
        f"(result has {len(res['result'])})"
    )
    assert _n_points(ax) % 6 == 0, f"{metric}: dots are not a whole number of patients"


def test_the_plot_uses_the_same_collapse_as_the_contrast(cohort):
    """Not merely the same COUNT — the same values, from the shared helper.

    Two implementations of "average the items to one number per replicate" is how the
    ``tl``/``pl`` ``distance_metric`` disagreement happened. `collapse_to_replicates` has one
    definition and both callers use it.
    """
    from tcri.tools._common import collapse_to_replicates

    _, adata = cohort
    adata = adata.copy()
    cov = _cov(adata)
    res = tcri.tl.phenotypic_entropy(adata, covariate=cov, groupby="patient",
                                     splitby="response")

    expected = collapse_to_replicates(res["result"], groupby="patient", splitby="response")
    ax = tcri.pl.phenotypic_entropy(adata)
    drawn = np.sort(np.concatenate([c.get_offsets()[:, 1] for c in ax.collections]))
    assert np.allclose(drawn, np.sort(expected["value"].to_numpy()))


def test_nothing_connects_two_x_positions(cohort):
    """A line between x positions claims the two points are the same entity observed twice.

    Only matched data supports that, and matched data does not exist on this side of the API
    yet. Box internals (whiskers, caps, medians) stay within one box; a connector spans
    categories, which are 1.0 apart — so the span is the discriminator.

    This is a prohibition, not a feature: when connectors arrive with the delta metrics they
    must be drawn from an identity key, never from adjacency, and this test is what forces
    that to be a deliberate change.
    """
    _, adata = cohort
    adata = adata.copy()
    cov, *rest = list(adata.uns[K.COVARIATE_CATEGORIES])
    _compute_all(adata, groupby="patient", splitby="response")

    for name in TWINS:
        ax = getattr(tcri.pl, name)(adata)
        for line in ax.lines:
            if line.get_label() == BRACKET_LABEL:
                continue      # a significance bracket spans x by design
            xs = np.asarray(line.get_xdata(), dtype=float)
            if xs.size < 2 or not np.isfinite(xs).all():
                continue
            assert xs.max() - xs.min() < 1.0, (
                f"pl.{name} drew a line spanning x positions {xs.min()}..{xs.max()}"
            )


def test_a_levels_colour_does_not_depend_on_its_position(trained_model):
    """The colour is a property of the level, not of where it lands on this figure.

    Found by looking at rendered panels: the renderer sorts x by median, so a `response`
    panel where NR sorted first drew NR purple while the panel beside it, where R sorted
    first, drew R purple. Same variable, same figure, swapped — because `resolve_colors`
    zipped the stored hex list against the caller's display order.
    """
    _, adata = trained_model
    adata = adata.copy()
    cats = list(adata.obs["patient"].astype("category").cat.categories)

    forward = tcri.pl.resolve_colors(adata, "patient", cats)
    reverse = tcri.pl.resolve_colors(adata, "patient", list(reversed(cats)))
    assert forward == reverse, "the colour followed the order, not the level"

    # a subset must keep each level's colour rather than restarting the cycle
    subset = tcri.pl.resolve_colors(adata, "patient", cats[1:])
    assert all(subset[c] == forward[c] for c in cats[1:])

    # and the same holds for a key with no obs column to appeal to
    a = tcri.pl.resolve_colors(adata, "phenotype", ["x", "y", "z"], persist=False)
    b = tcri.pl.resolve_colors(adata, "phenotype", ["z", "y", "x"], persist=False)
    assert a == b


def test_the_same_level_keeps_its_colour_across_panels(cohort):
    """The end-to-end version: two panels of the same split, drawn in different orders."""
    _, adata = cohort
    adata = adata.copy()
    cov, *rest = list(adata.uns[K.COVARIATE_CATEGORIES])

    def _colour_by_level(ax):
        out = {}
        for coll in ax.collections:
            for (x, _y), c in zip(coll.get_offsets(), coll.get_facecolor()):
                lvl = ax.get_xticklabels()[int(round(x))].get_text()
                out.setdefault(lvl, set()).add(matplotlib.colors.to_hex(c).lower())
        return out

    tcri.tl.phenotypic_entropy(adata, covariate=cov, groupby="patient", splitby="response")
    first = _colour_by_level(tcri.pl.phenotypic_entropy(adata))
    tcri.tl.phenotypic_flux(adata, cov_from=cov, cov_to=rest[0], groupby="patient",
                            splitby="response")
    second = _colour_by_level(tcri.pl.phenotypic_flux(adata))

    for level in set(first) & set(second):
        assert first[level] == second[level], (
            f"{level!r} was {first[level]} in one panel and {second[level]} in the next"
        )


# ── the delta twins ──────────────────────────────────────────────────────────

DELTAS = ["delta_clonotypic_entropy", "delta_phenotypic_entropy"]


def _compute_deltas(adata, **kw):
    cov, *rest = list(adata.uns[K.COVARIATE_CATEGORIES])
    to = rest[0] if rest else cov
    for name in DELTAS:
        getattr(tcri.tl, name)(adata, cov_from=cov, cov_to=to, **kw)


@pytest.mark.parametrize("name", DELTAS)
def test_delta_twins_obey_the_renderer_contract(name, cohort):
    """Same rules as every other twin: no metric arguments, and it names the tool to run."""
    _, adata = cohort
    adata = adata.copy()
    params = set(inspect.signature(getattr(tcri.pl, name)).parameters)
    assert not (params & METRIC_ARGS)
    assert {"kind", "key"} <= params

    # `cohort` is session-scoped and other modules compute on it, so a copy inherits whatever
    # they cached. Asserting a key is ABSENT has to clear it first or the test passes or fails
    # on file ordering rather than on behaviour.
    adata.uns.pop(getattr(K, name.upper()), None)
    with pytest.raises(KeyError, match=f"tcri.tl.{name}"):
        getattr(tcri.pl, name)(adata)

    _compute_deltas(adata, groupby="patient")
    with pytest.raises(ValueError, match="kind must be"):
        getattr(tcri.pl, name)(adata, kind="nonsense")


@pytest.mark.parametrize("name", DELTAS)
def test_the_delta_view_marks_zero(name, cohort):
    """Zero is a real position on a delta axis — an unmarked one hides the direction."""
    _, adata = cohort
    adata = adata.copy()
    _compute_deltas(adata, groupby="patient", splitby="response")
    ax = getattr(tcri.pl, name)(adata, kind="delta")
    rules = [l for l in ax.lines
             if l.get_label() == BRACKET_LABEL and len(set(np.round(l.get_ydata(), 12))) == 1
             and float(l.get_ydata()[0]) == 0.0]
    assert rules, "the delta axis has no zero rule"


def test_only_the_clone_metric_connects_its_endpoints(cohort):
    """A line asserts persistence. A clonotype persists; a phenotype is a bin.

    Both metrics can show their endpoints, but only the clone-item one may join them — the
    entity/category distinction, drawn.
    """
    _, adata = cohort
    adata = adata.copy()
    _compute_deltas(adata, groupby="patient")
    from tcri.plotting._base import CONNECTOR_LABEL

    def _connectors(ax):
        return [l for l in ax.lines if l.get_label() == CONNECTOR_LABEL]

    clone_ax = tcri.pl.delta_phenotypic_entropy(adata, kind="endpoints")
    assert len(_connectors(clone_ax)) == adata.obs["patient"].nunique()

    pheno_ax = tcri.pl.delta_clonotypic_entropy(adata, kind="endpoints")
    assert not _connectors(pheno_ax), "a phenotype is a category, not a barcode"

    # and no connector survives into the delta view, where a point IS the pairing
    assert not _connectors(tcri.pl.delta_phenotypic_entropy(adata, kind="delta"))


def test_endpoints_come_from_the_matched_result(cohort):
    """The endpoints view is rendered from the DELTA result, where both sides were computed
    over the intersected clone set — so the unmatched version of this figure is unreachable
    rather than merely discouraged."""
    _, adata = cohort
    adata = adata.copy()
    cov, *rest = list(adata.uns[K.COVARIATE_CATEGORIES])
    tcri.tl.delta_phenotypic_entropy(adata, cov_from=cov, cov_to=rest[0], groupby="patient")

    res = tcri.get.result(adata, "delta_phenotypic_entropy")["result"]
    assert {"value_from", "value_to"} <= set(res.columns), (
        "the endpoints are not in the payload, so the view would need a second result"
    )
    ax = tcri.pl.delta_phenotypic_entropy(adata, kind="endpoints")
    assert [t.get_text() for t in ax.get_xticklabels()] == [str(cov), str(rest[0])]

    drawn = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections])
    from tcri.tools._common import collapse_to_replicates
    for col in ("value_from", "value_to"):
        want = collapse_to_replicates(res, groupby="patient", value=col)[col].to_numpy()
        assert all(np.isclose(drawn, w).any() for w in want), f"{col} is not on the figure"


def test_dot_area_is_the_matched_clone_count(cohort):
    """`n_matched` varies per replicate, so it is encoded per point rather than in a title.

    And a replicate's two endpoint dots must be the SAME size: the matched clone set is the
    same on both sides, so a size difference would mean the intersection did not hold.
    """
    _, adata = cohort
    adata = adata.copy()
    cov, *rest = list(adata.uns[K.COVARIATE_CATEGORIES])
    tcri.tl.delta_phenotypic_entropy(adata, cov_from=cov, cov_to=rest[0], groupby="patient")
    res = tcri.get.result(adata, "delta_phenotypic_entropy")["result"]
    counts = res.groupby("patient", observed=True)["clonotype"].nunique()

    ax = tcri.pl.delta_phenotypic_entropy(adata, kind="endpoints")
    per_patient = [c.get_sizes() for c in ax.collections if len(c.get_offsets()) == 2]
    assert len(per_patient) == len(counts)
    for sizes in per_patient:
        assert len(set(np.round(sizes, 9))) == 1, "a patient's two endpoints differ in size"

    if counts.nunique() > 1:
        flat = sorted(float(s[0]) for s in per_patient)
        assert flat[0] < flat[-1], "the matched count is not encoded in the area"
    assert ax.get_legend() is not None, "no size legend to read the areas against"


def test_only_the_entity_matched_metric_sizes_by_matched_clones(cohort):
    """The size legend says "clones matched", so it must be clones.

    For `delta_clonotypic_entropy` the item is a phenotype and the matched clone count is not
    in `result` — those clones were summed over inside H(c|phi). Counting item rows there
    would count PHENOTYPES: measured on a 4-phenotype fixture the legend read
    "clones matched: 4", a different number about a different thing.
    """
    _, adata = cohort
    adata = adata.copy()
    _compute_deltas(adata, groupby="patient")

    sized = tcri.pl.delta_phenotypic_entropy(adata, kind="endpoints")
    assert sized.get_legend() is not None

    unsized = tcri.pl.delta_clonotypic_entropy(adata, kind="endpoints")
    assert unsized.get_legend() is None, "a phenotype count was labelled as clones matched"
    areas = {round(float(s), 6) for c in unsized.collections for s in c.get_sizes()}
    assert len(areas) == 1, "the phenotype panel encoded a count in the dot area"
