"""The ``pl`` twins of the ``pp`` repertoire summaries draw what the summaries stored.

Each test reads a drawn value back off the axes and compares it with the stored table, so a
plot cannot show a number the table does not hold.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import tcri
from tcri.plotting._repertoire import NONE_COLOR


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


def _image(ax, k=0):
    return np.ma.filled(ax.images[k].get_array().astype(float), np.nan)


@pytest.mark.parametrize("name", ["repertoire_summary", "clonotype_sharing",
                                  "clone_persistence"])
def test_a_twin_names_the_call_that_fills_it(name):
    with pytest.raises(KeyError, match=f"tcri.pp.{name}"):
        getattr(tcri.pl, name)(_adata())


def test_repertoire_summary_draws_the_stored_table():
    """Four panels; each bar is the table's value for its group and level, in the level
    colors stored for the covariate, and the group labels take the split colors."""
    adata = _adata()
    tcri.pp.repertoire_summary(adata, **SUMMARY, covariate_key="timepoint")
    table = tcri.get.result(adata, "repertoire_summary")
    axes = tcri.pl.repertoire_summary(adata)
    assert [ax.get_title() for ax in axes] == ["Cells", "Clones", "Singleton fraction",
                                               "Clonality"]
    groups = [label.get_text() for label in axes[1].get_xticklabels()]
    assert groups == ["P1", "P2", "P3", "P4"]
    level_colors = dict(zip(["t0", "t1", "t2"], adata.uns["timepoint_colors"]))
    bars = axes[1].patches
    for k, level in enumerate(["t0", "t1", "t2"]):
        drawn = [bar.get_height() for bar in bars[4 * k:4 * (k + 1)]]
        expected = (table[table["covariate"] == level].set_index("patient")["n_clones"]
                    .reindex(groups).fillna(0).tolist())
        np.testing.assert_allclose(np.nan_to_num(drawn), expected)
        assert to_hex(bars[4 * k].get_facecolor()) == level_colors[level]
    split = dict(zip(["NR", "R"], adata.uns["response_colors"]))
    assert [to_hex(label.get_color()) for label in axes[1].get_xticklabels()] == [
        split["R"], split["NR"], split["R"], split["NR"]]
    assert tcri.pl.repertoire_summary(adata, return_df=True).equals(table)


def test_repertoire_summary_of_the_whole_dataset_is_one_group():
    adata = _adata()
    tcri.pp.repertoire_summary(adata, clonotype_key="clone_id", covariate_key="timepoint")
    axes = tcri.pl.repertoire_summary(adata)
    assert [label.get_text() for label in axes[0].get_xticklabels()] == ["all"]


def test_clonotype_sharing_draws_the_stored_pairs():
    """The heatmap holds 100 × ``shared_fraction`` of each ordered pair, blank on the diagonal,
    red where two groups share none; the bars are each group's shared fraction."""
    adata = _adata()
    tcri.pp.clonotype_sharing(adata, **SUMMARY)
    stored = tcri.get.result(adata, "clonotype_sharing")
    left, right = tcri.pl.clonotype_sharing(adata)
    groups = ["P1", "P2", "P3", "P4"]
    expected = (stored["pairs"].pivot(index="patient_a", columns="patient_b",
                                      values="shared_fraction")
                .reindex(index=groups, columns=groups).to_numpy() * 100)
    np.testing.assert_allclose(_image(left), expected)
    none = stored["pairs"].pivot(index="patient_a", columns="patient_b",
                                 values="shared_clonotypes").reindex(index=groups, columns=groups)
    assert np.array_equal(np.isfinite(_image(left, 1)), (none == 0).to_numpy())
    assert np.isfinite(_image(left, 1)[3, :3]).all()   # P4 shares nothing
    assert left.images[1].get_cmap()(1.0)[:3] == pytest.approx(
        matplotlib.colors.to_rgb(NONE_COLOR))
    table = stored["table"].set_index("patient")
    np.testing.assert_allclose([bar.get_height() for bar in right.patches],
                               table.loc[groups, "shared_fraction"] * 100)


def test_persistence_draws_one_heatmap_per_group_from_the_pairs():
    """One panel per group: the share of the row level's clones also seen at the column level,
    red where none is shared. A group with clones at one level only has nothing to draw."""
    adata = _adata()
    tcri.pp.clone_persistence(adata, clonotype_key="clone_id", covariate_key="timepoint",
                              groupby="patient", splitby="response")
    pairs = tcri.get.result(adata, "clone_persistence")["pairs"]
    axes = tcri.pl.clone_persistence(adata)
    assert [ax.get_title() for ax in axes] == [f"Clone overlap - {p}"
                                               for p in ("P1", "P2", "P3", "P4")]
    levels = ["t0", "t1", "t2"]
    for ax, patient in zip(axes[:3], ("P1", "P2", "P3")):
        sub = pairs[pairs["patient"] == patient]
        expected = (sub.pivot(index="covariate_a", columns="covariate_b",
                              values="shared_fraction")
                    .reindex(index=levels, columns=levels).to_numpy())
        np.testing.assert_allclose(_image(ax), expected)
    assert not axes[3].images


@pytest.mark.parametrize(("name", "flag"), [("clonotype_sharing", "per_clonotype"),
                                            ("clone_persistence", "per_clone")])
def test_a_result_per_clone_raises(name, flag):
    adata = _adata()
    arguments = dict(clonotype_key="clone_id", groupby="patient", **{flag: True})
    if name == "clone_persistence":
        arguments["covariate_key"] = "timepoint"
    getattr(tcri.pp, name)(adata, **arguments)
    with pytest.raises(ValueError, match=f"{flag}=True"):
        getattr(tcri.pl, name)(adata)


def test_order_selects_and_orders_the_groups():
    adata = _adata()
    tcri.pp.clonotype_sharing(adata, clonotype_key="clone_id", groupby="patient")
    left, _ = tcri.pl.clonotype_sharing(adata, order=["P3", "P1"])
    assert [label.get_text() for label in left.get_xticklabels()] == ["P3", "P1"]
    with pytest.raises(ValueError, match="does not have"):
        tcri.pl.clonotype_sharing(adata, order=["P9"])
