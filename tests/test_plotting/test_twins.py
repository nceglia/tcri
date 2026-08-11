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

TWINS = ["mutual_information", "clonotypic_entropy", "phenotypic_entropy", "phenotypic_flux"]

#: Anything on this list reaching a `pl` signature means the plot can compute something.
METRIC_ARGS = {
    "covariate", "cov_from", "cov_to", "groupby", "splitby", "n_samples", "temperature",
    "clones", "weighted", "normalized", "normalize_mode", "distance_metric", "random_state",
    "device", "inplace", "key_added",
}


def _cov(adata):
    return list(adata.uns[K.COVARIATE_CATEGORIES])[0]


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
    assert len(ungrouped.patches) == 1, "an ungrouped MI is one number, not a box per patient"

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
    assert len(tcri.pl.mutual_information(adata, key="mi_flat").patches) == 1


def test_pl_key_selects_a_non_default_result(trained_model):
    """Two results in one object: ``key_added`` on the tool, ``key`` on the plot."""
    _, adata = trained_model
    adata = adata.copy()
    cov = _cov(adata)
    tcri.tl.mutual_information(adata, covariate=cov, groupby="patient")
    tcri.tl.mutual_information(adata, covariate=cov, key_added="mi_ungrouped")

    assert tcri.pl.mutual_information(adata).get_xlabel() == "patient"
    assert len(tcri.pl.mutual_information(adata, key="mi_ungrouped").patches) == 1


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


def test_posterior_draws_become_error_bars(trained_model):
    """An ungrouped metric with draws has an HDI; a bar with no interval hides it."""
    _, adata = trained_model
    adata = adata.copy()
    cov = _cov(adata)

    from matplotlib.container import ErrorbarContainer

    tcri.tl.clonotypic_entropy(adata, covariate=cov, n_samples=0)
    plain = tcri.pl.clonotypic_entropy(adata)
    assert not any(isinstance(c, ErrorbarContainer) for c in plain.containers)

    tcri.tl.clonotypic_entropy(adata, covariate=cov, n_samples=8, random_state=0)
    drawn = tcri.pl.clonotypic_entropy(adata)
    bars = [c for c in drawn.containers if not isinstance(c, ErrorbarContainer)][0]
    assert bars.errorbar is not None, "the posterior HDI is not on the figure"


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
