"""``pl.joint_distribution(view="sankey")`` draws the phenotype-flow Sankey of the stored joint.

The geometry is pinned on a toy small enough to compute by hand, since column totals alone cannot
catch a transposed flow or a band stacked at the wrong end of its node. The rest reads the drawn
bars back off the axes and compares them with counts taken straight from ``obs``.
"""
import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import tcri
from tcri._state import keys as K
from tcri.plotting._sankey import _sankey_layout

from .test_twins import METRIC_ARGS

COV, CLONE, PHENOTYPE = "timepoint", "unique_clone_id", "phenotype_col"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def adata(trained_model):
    """A copy of the fitted fixture with the joint stored at every covariate value."""
    _, fitted = trained_model
    a = fitted.copy()
    tcri.tl.joint_distribution(a)
    return a


def _columns(ax):
    """The bar heights drawn at each x position, in phenotype order."""
    columns = {}
    for rect in ax.patches:
        center = rect.get_x() + rect.get_width() / 2
        columns.setdefault(int(round(center)), []).append(rect.get_height())
    return {x: np.array(heights) for x, heights in sorted(columns.items())}


def _cells(adata, value, clones):
    """Cells at covariate ``value`` whose clone is in ``clones``, counted from ``obs``."""
    obs = adata.obs
    keep = (obs[COV].astype(str) == value) & obs[CLONE].astype(str).isin(set(clones))
    return int(keep.sum())


def _stored_clones(adata, value, key=None):
    summary = tcri.get.result(adata, "joint_distribution", key=key)["summary"]
    return [str(c) for c in summary.xs(value, level="covariate").index]


def test_layout_matches_the_hand_computed_geometry():
    """Two values, two phenotypes, three clones. ``c1`` is at both values and carries all the
    flow; ``c2`` is at the first only and ``c3`` at the second only."""
    mass = [
        {"c1": np.array([2.0, 0.0]), "c2": np.array([1.0, 1.0])},
        {"c1": np.array([1.0, 1.0]), "c3": np.array([0.0, 2.0])},
    ]
    layout = _sankey_layout(mass, 2)

    np.testing.assert_allclose(layout["nodes"][0], [[0, 3], [3, 1]])
    np.testing.assert_allclose(layout["nodes"][1], [[0, 1], [1, 3]])
    origin, destination = layout["bands"][0]
    # c1's two cells of A at the first value split half to A, half to B
    np.testing.assert_allclose(origin, [[[0, 1], [1, 1]], [[3, 0], [3, 0]]])
    # at the second value each band end is c1's own cell there, and the band into B sits on top
    # of c3's two cells, which arrive from nowhere
    np.testing.assert_allclose(destination, [[[0, 1], [3, 1]], [[1, 0], [4, 0]]])


def test_sankey_draws_bars_and_bands(adata):
    ax = tcri.pl.joint_distribution(adata)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["T1", "T2"]
    assert ax.get_xlabel() == COV
    assert sorted(_columns(ax)) == [0, 1]
    assert len(ax.images) > 0, "no flow band was drawn"


def test_normalized_bars_sum_to_one(adata):
    for heights in _columns(tcri.pl.joint_distribution(adata)).values():
        assert heights.sum() == pytest.approx(1.0)


def test_counts_are_the_cells_of_the_stored_clones(adata):
    columns = _columns(tcri.pl.joint_distribution(adata, normalize=False))
    for x, value in enumerate(["T1", "T2"]):
        assert columns[x].sum() == pytest.approx(_cells(adata, value, _stored_clones(adata, value)))


def test_a_stored_subset_draws_only_its_clones(adata):
    clones = _stored_clones(adata, "T1")[:5]
    tcri.tl.joint_distribution(adata, clones=clones, key_added="sub")
    columns = _columns(tcri.pl.joint_distribution(adata, key="sub", normalize=False))
    for x, value in enumerate(["T1", "T2"]):
        assert columns[x].sum() == pytest.approx(_cells(adata, value, clones))


def test_a_single_covariate_joint_draws_one_bar(adata):
    tcri.tl.joint_distribution(adata, covariate="T1", key_added="t1")
    ax = tcri.pl.joint_distribution(adata, key="t1")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["T1"]
    assert len(ax.images) == 0
    with pytest.raises(ValueError, match="order names"):
        tcri.pl.joint_distribution(adata, key="t1", order=["T1", "T2"])


def test_order_draws_the_named_values(adata):
    ax = tcri.pl.joint_distribution(adata, order=["T2", "T1"])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["T2", "T1"]
    with pytest.raises(ValueError, match="order names"):
        tcri.pl.joint_distribution(adata, order=["T1", "nope"])


def test_plotting_before_computing_names_the_call(trained_model):
    _, fitted = trained_model
    a = fitted.copy()
    a.uns.pop(K.JOINT_DISTRIBUTION, None)
    with pytest.raises(KeyError, match="tcri.tl.joint_distribution"):
        tcri.pl.joint_distribution(a)


def test_a_null_fit_joint_is_refused(adata):
    """A null reassigns cells by permutation, so the ``obs`` counts are not its clones'."""
    null = next(f for f in tcri.get.fits(adata) if K.fit_key(K.PERMUTATION, f) in adata.uns)
    tcri.tl.joint_distribution(adata, fit=null)
    with pytest.raises(ValueError, match="permutation-null"):
        tcri.pl.joint_distribution(adata, key=K.fit_key(K.JOINT_DISTRIBUTION, null))


def test_pl_never_recomputes(adata, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("pl.joint_distribution recomputed the joint")

    monkeypatch.setattr(tcri.tl, "joint_distribution", _boom)
    monkeypatch.setattr(tcri, "joint_distribution", _boom)
    monkeypatch.setattr("tcri.tools._joint.joint_distribution", _boom)
    monkeypatch.setattr("tcri.tools._joint._engine_blocks", _boom)
    assert tcri.pl.joint_distribution(adata) is not None


def test_return_df_is_the_stored_summary(adata):
    pd.testing.assert_frame_equal(tcri.pl.joint_distribution(adata, return_df=True),
                                  tcri.get.result(adata, "joint_distribution")["summary"])


def test_signature_takes_no_compute_argument():
    params = set(inspect.signature(tcri.pl.joint_distribution).parameters)
    computes = set(inspect.signature(tcri.tl.joint_distribution).parameters) - {"adata"}
    assert not params & (computes | METRIC_ARGS)
    assert {"key", "view"} <= params


def test_phenotype_colors_are_the_stored_ones(adata):
    ax = tcri.pl.joint_distribution(adata)
    phenotypes = [str(p) for p in adata.uns[K.PHENOTYPE_CATEGORIES]]
    stored = dict(zip(adata.obs[PHENOTYPE].cat.categories.astype(str),
                      adata.uns[K.colors(PHENOTYPE)]))
    first = ax.patches[:len(phenotypes)]
    assert [to_hex(r.get_facecolor()) for r in first] == [stored[p] for p in phenotypes]

    ax = tcri.pl.joint_distribution(adata, palette={phenotypes[0]: "#000000"})
    assert to_hex(ax.patches[0].get_facecolor()) == "#000000"


def test_arguments(adata):
    with pytest.raises(ValueError, match="view must be 'sankey'"):
        tcri.pl.joint_distribution(adata, view="heatmap")
    _, given = plt.subplots()
    assert tcri.pl.joint_distribution(adata, ax=given) is given
    assert tcri.pl.joint_distribution(adata, legend=False).get_legend() is None
    assert tcri.pl.joint_distribution(adata).get_legend() is not None
