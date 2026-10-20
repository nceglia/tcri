"""The ``pl`` twins of the ``pp`` repertoire summaries draw what the summaries stored.

Each test reads the drawn values and colors back off the axes and compares them with the stored
frames, so a plot cannot show a number, or a band, the frames do not hold.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgba
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import tcri
from tcri._compute._repertoire import _record_derivation
from tcri.plotting._repertoire import (GREEN, RED, YELLOW, _GROUP_BANDS, _PAIR_BANDS,
                                       _SPREAD_BANDS, _band, _shades)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _adata():
    """Four patients at three timepoints, two of them responders. Clonotype ``pub`` is carried
    by P1, P2 and P3; P4 shares nothing and has cells at t0 only."""
    rng = np.random.default_rng(3)
    rows = []
    for patient in ("P1", "P2", "P3"):
        for timepoint in ("t0", "t1", "t2"):
            for clone in rng.integers(0, 6, size=12):
                rows.append((f"{patient}_c{clone}", patient, timepoint))
            rows.append(("pub", patient, timepoint))
    rows += [(f"P4_c{k}", "P4", "t0") for k in (0, 0, 1)]
    obs = pd.DataFrame(rows, columns=["clone_id", "patient", "timepoint"])
    obs["response"] = obs["patient"].map({"P1": "R", "P2": "NR", "P3": "R", "P4": "NR"})
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    return AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs)


SUMMARY = dict(clonotype_key="clone_id", groupby="patient", splitby="response")
PATIENTS = ["P1", "P2", "P3", "P4"]


def _hex(color):
    return to_hex(color, keep_alpha=False)


def _split_hex(adata):
    return dict(zip(["NR", "R"], adata.uns["response_colors"]))


@pytest.mark.parametrize("name", ["repertoire_summary", "clonotype_sharing",
                                  "clone_persistence"])
def test_a_twin_names_the_call_that_fills_it(name):
    with pytest.raises(KeyError, match=f"tcri.pp.{name}"):
        getattr(tcri.pl, name)(_adata())


def test_band_edges():
    """Green below the low edge, yellow from it up to the high edge, red above."""
    assert [_band(v, 1.0, 2.0) for v in (0.0, 0.99, 1.0, 2.0, 2.01)] == [
        GREEN, GREEN, YELLOW, YELLOW, RED]
    assert [_band(n, *_SPREAD_BANDS) for n in (2, 3, 4, 5, 6, 9)] == [
        GREEN, GREEN, YELLOW, YELLOW, RED, RED]


# ── repertoire_summary ────────────────────────────────────────────────────────────────────────

def test_repertoire_summary_draws_every_panel_from_the_stored_table():
    """Each panel's bars are its column of the table, one bar per group and level, the levels
    in shades of green, and the group labels take the split colors."""
    adata = _adata()
    tcri.pp.repertoire_summary(adata, **SUMMARY, covariate_key="timepoint")
    table = tcri.get.result(adata, "repertoire_summary")
    axes = tcri.pl.repertoire_summary(adata)
    columns = ["n_cells", "n_clones", "singleton_fraction", "clonality"]
    assert [ax.get_title() for ax in axes] == ["Cells", "Clones", "Singleton fraction",
                                               "Clonality"]
    shades = dict(zip(["t0", "t1", "t2"], _shades(3)))
    for ax, column in zip(axes, columns):
        assert [label.get_text() for label in ax.get_xticklabels()] == PATIENTS
        assert ax.get_xlabel() == "patient"
        bars = [bar for bar in ax.patches if not bar.get_hatch()]
        for k, level in enumerate(["t0", "t1", "t2"]):
            drawn = [bar.get_height() for bar in bars[4 * k:4 * (k + 1)]]
            expected = (table[table["covariate"] == level].set_index("patient")[column]
                        .reindex(PATIENTS).to_numpy(float))
            np.testing.assert_allclose(drawn, expected, equal_nan=True, err_msg=column)
            assert {_hex(bar.get_facecolor()) for bar in bars[4 * k:4 * (k + 1)]} \
                == {shades[level]}
    split = _split_hex(adata)
    assert [_hex(label.get_color()) for label in axes[1].get_xticklabels()] == [
        split["R"], split["NR"], split["R"], split["NR"]]


def test_repertoire_summary_shades_the_cells_in_pools():
    """On a pooled column, each cell bar carries a hatched bar the height of its cells in
    pools."""
    adata = _adata()
    tcri.pp.pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
    tcri.pp.repertoire_summary(adata, clonotype_key="clone_id_pooled", groupby="patient",
                               covariate_key="timepoint")
    table = tcri.get.result(adata, "repertoire_summary")
    assert table["cells_in_pools"].sum() > 0, "the fixture has no pooled cell"
    cells = tcri.pl.repertoire_summary(adata)[0]
    hatched = [bar.get_height() for bar in cells.patches if bar.get_hatch()]
    expected = np.concatenate([
        table[table["covariate"] == level].set_index("patient")["cells_in_pools"]
        .reindex(PATIENTS).to_numpy(float) for level in ("t0", "t1", "t2")])
    np.testing.assert_allclose(hatched, expected, equal_nan=True)


def test_repertoire_summary_without_a_covariate_or_a_group():
    """Without a covariate every bar is green, apart from the split colors of the labels;
    with ``groupby=None`` the whole dataset is one group labeled ``all``."""
    adata = _adata()
    tcri.pp.repertoire_summary(adata, **SUMMARY)
    axes = tcri.pl.repertoire_summary(adata)
    assert {_hex(bar.get_facecolor()) for bar in axes[1].patches} == {GREEN}
    tcri.pp.repertoire_summary(adata, clonotype_key="clone_id", covariate_key="timepoint",
                               key_added="whole")
    axes = tcri.pl.repertoire_summary(adata, key="whole")
    assert [label.get_text() for label in axes[0].get_xticklabels()] == ["all"]
    assert tcri.pl.repertoire_summary(adata, key="whole", return_df=True).equals(
        tcri.get.result(adata, "repertoire_summary", key="whole"))


# ── clonotype_sharing ─────────────────────────────────────────────────────────────────────────

def _pair_percent(pairs, groups):
    percent = pd.DataFrame(np.nan, index=groups, columns=groups)
    for row in pairs.itertuples(index=False):
        percent.loc[row.patient_a, row.patient_b] = 100 * row.shared_fraction_a
        percent.loc[row.patient_b, row.patient_a] = 100 * row.shared_fraction_b
    return percent.to_numpy()


def test_clonotype_sharing_draws_the_bands_of_the_stored_frames():
    """The heatmap cell (a, b) is in the band of 100 × a's shared fraction with b and prints
    it; the middle bars are each group's shared fraction in its band; the right bars count the
    shared clonotypes by the number of groups carrying them, in its band."""
    adata = _adata()
    tcri.pp.clonotype_sharing(adata, **SUMMARY)
    stored = tcri.get.result(adata, "clonotype_sharing")
    left, middle, right = tcri.pl.clonotype_sharing(adata)

    percent = _pair_percent(stored["pairs"], PATIENTS)
    image = left.images[0].get_array()
    for i in range(4):
        for j in range(4):
            expected = (to_rgba("white") if i == j
                        else to_rgba(_band(percent[i, j], *_PAIR_BANDS)))
            np.testing.assert_allclose(image[i, j], expected, err_msg=f"cell {i}, {j}")
    printed = {(round(t.get_position()[1]), round(t.get_position()[0])): float(t.get_text())
               for t in left.texts}
    for (i, j), value in printed.items():
        assert value == pytest.approx(percent[i, j], abs=0.005)
    assert len(printed) == 12

    shared = stored["groups"].set_index("patient").loc[PATIENTS, "shared_fraction"] * 100
    np.testing.assert_allclose([bar.get_height() for bar in middle.patches], shared)
    assert [_hex(bar.get_facecolor()) for bar in middle.patches] == [
        _band(v, *_GROUP_BANDS) for v in shared]
    assert middle.get_ylim()[0] == 0

    spread = stored["clonotypes"]["n_groups"].value_counts()
    carriers = [int(bar.get_x() + bar.get_width() / 2) for bar in right.patches]
    assert carriers == [2, 3, 4]
    assert [bar.get_height() for bar in right.patches] == [int(spread.get(n, 0))
                                                          for n in carriers]
    assert [_hex(bar.get_facecolor()) for bar in right.patches] == [
        _band(n, *_SPREAD_BANDS) for n in carriers]

    split = _split_hex(adata)
    assert [_hex(label.get_color()) for label in left.get_yticklabels()] == [
        split["R"], split["NR"], split["R"], split["NR"]]


def test_clonotype_sharing_with_nothing_shared_is_green_from_zero():
    """When no two groups share a clonotype every cell is green, and the bar axes start at 0
    rather than spanning negative values."""
    adata = _adata()
    adata = adata[adata.obs["clone_id"] != "pub"].copy()
    tcri.pp.clonotype_sharing(adata, clonotype_key="clone_id", groupby="patient")
    left, middle, right = tcri.pl.clonotype_sharing(adata)
    image = left.images[0].get_array()
    off_diagonal = image[~np.eye(4, dtype=bool)]
    np.testing.assert_allclose(off_diagonal, np.tile(to_rgba(GREEN), (12, 1)))
    assert middle.get_ylim()[0] == 0 and right.get_ylim()[0] == 0


def test_long_y_labels_are_wrapped():
    adata = _adata()
    tcri.pp.clonotype_sharing(adata, clonotype_key="clone_id", groupby="patient")
    _, middle, _ = tcri.pl.clonotype_sharing(adata, figsize=(9, 2.4))
    assert "\n" in middle.get_ylabel()


def test_order_selects_and_orders_the_groups():
    adata = _adata()
    tcri.pp.clonotype_sharing(adata, clonotype_key="clone_id", groupby="patient")
    left, _, _ = tcri.pl.clonotype_sharing(adata, order=["P3", "P1"])
    assert [label.get_text() for label in left.get_xticklabels()] == ["P3", "P1"]
    with pytest.raises(ValueError, match="does not have"):
        tcri.pl.clonotype_sharing(adata, order=["P9"])


# ── clone_persistence ─────────────────────────────────────────────────────────────────────────

def test_persistence_draws_one_heatmap_per_group_from_the_frame():
    """One panel per group: cell (a, b) is the share of the row level's clones also seen at the
    column level, red where none is shared, the x axis named after the covariate and the title
    in the group's split color. P4, with clones at one level only, has no pair and no panel."""
    adata = _adata()
    tcri.pp.clone_persistence(adata, clonotype_key="clone_id", covariate_key="timepoint",
                              groupby="patient", splitby="response")
    pairs = tcri.get.result(adata, "clone_persistence")
    axes = tcri.pl.clone_persistence(adata)
    assert [ax.get_title() for ax in axes] == [f"Clone overlap - {p}" for p in PATIENTS[:3]]
    levels = ["t0", "t1", "t2"]
    split = _split_hex(adata)
    for ax, patient, response in zip(axes[:3], PATIENTS, ["R", "NR", "R"]):
        sub = pairs[pairs["patient"] == patient]
        expected = np.full((3, 3), np.nan)
        for row in sub.itertuples(index=False):
            a, b = levels.index(row.covariate_a), levels.index(row.covariate_b)
            expected[a, b], expected[b, a] = row.shared_fraction_a, row.shared_fraction_b
        np.testing.assert_allclose(np.ma.filled(ax.images[0].get_array().astype(float),
                                                np.nan), expected)
        assert ax.get_xlabel() == "timepoint"
        assert _hex(ax.title.get_color()) == split[response]


def test_persistence_marks_levels_that_share_no_clone_in_red():
    obs = pd.DataFrame({"clone_id": ["a", "a", "b", "c"], "patient": ["P1"] * 4,
                        "timepoint": ["t0", "t1", "t0", "t2"]})
    obs.index = [f"cell_{i}" for i in range(4)]
    adata = AnnData(X=np.zeros((4, 1), dtype=np.float32), obs=obs)
    tcri.pp.clone_persistence(adata, clonotype_key="clone_id", covariate_key="timepoint",
                              groupby="patient")
    ax = tcri.pl.clone_persistence(adata)[0]
    red = np.isfinite(np.ma.filled(ax.images[1].get_array().astype(float), np.nan))
    # t0-t1 share a; t0-t2 and t1-t2 share nothing
    assert red.tolist() == [[False, False, True], [False, False, True], [True, True, False]]
    assert ax.images[1].get_cmap()(1.0)[:3] == pytest.approx(to_rgba(RED)[:3])


def test_persistence_of_the_whole_dataset_is_one_panel():
    adata = _adata()
    tcri.pp.clone_persistence(adata, clonotype_key="clone_id", covariate_key="timepoint")
    axes = tcri.pl.clone_persistence(adata)
    assert [ax.get_title() for ax in axes] == ["Clone overlap - all"]
