"""The replicate and item levels of mutual information and the two entropies at one covariate.

Each metric stores ``{replicate, item, stats}``, a level holding ``draws`` and ``summary``, and
returns ``None``; ``inplace=False`` returns that result instead. These tests pin the row sets
(every replicate has a replicate row), the error for a normalizer of 0, the case with no
replicate, and the ``tcri.get`` views, which refuse a result not stored by level
(``joint_distribution``). The levels of a change between two covariate levels, flux's
included, are pinned in ``test_changes.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tcri
from tcri._state import keys as K

METRICS = ["mutual_information", "clonotypic_entropy", "phenotypic_entropy"]
ENTROPIES = ["clonotypic_entropy", "phenotypic_entropy"]


def _cov(adata):
    return str(adata.uns[K.COVARIATE_CATEGORIES][0])


def _clones_at(adata, cov, patients):
    """The clones of ``patients`` with cells at ``cov``, in category order."""
    obs = adata.obs
    at = obs.loc[(obs["covariate"].astype(str) == cov) & obs["patient"].isin(patients),
                 "clone_id"]
    present = set(at.astype(str))
    return [c for c in adata.uns[K.CLONOTYPE_CATEGORIES] if c in present]


def _patients(adata):
    return [str(p) for p in adata.obs["patient"].cat.categories]


def _call(metric, adata, **kw):
    return getattr(tcri.tl, metric)(adata, covariate=_cov(adata), inplace=False, **kw)


@pytest.mark.parametrize("metric", METRICS)
def test_replicates_is_the_stored_replicate_summary(metric, cohort):
    """``tcri.get.replicates`` is the stored replicate summary, one row per replicate."""
    _, adata = cohort
    adata = adata.copy()
    assert getattr(tcri.tl, metric)(adata, covariate=_cov(adata), groupby="patient") is None
    replicates = tcri.get.replicates(adata, metric)
    assert sorted(replicates["patient"].astype(str)) == sorted(_patients(adata))
    pd.testing.assert_frame_equal(replicates,
                                  tcri.get.result(adata, metric)["replicate"]["summary"])


def test_replicates_refuses_a_result_not_stored_by_level(cohort):
    _, adata = cohort
    adata = adata.copy()
    tcri.tl.joint_distribution(adata, covariate=_cov(adata))
    with pytest.raises(KeyError, match="not stored by level"):
        tcri.get.replicates(adata, "joint_distribution")


@pytest.mark.parametrize("metric", METRICS)
def test_a_replicate_with_no_clone_left_is_a_nan_row_outside_n(metric, cohort):
    """A replicate whose clones ``clones=`` leaves out keeps its replicate row, as NaN, and
    leaves the test's ``n``. Its item rows: one NaN row per phenotype for clonotypic entropy,
    none for phenotypic entropy."""
    _, adata = cohort
    cov = _cov(adata)
    kept = _clones_at(adata, cov, [p for p in _patients(adata) if p != "P0"])
    res = _call(metric, adata, groupby="patient", splitby="disease_status", clones=kept)

    rep = res["replicate"]["summary"].set_index("patient")
    assert len(rep) == len(_patients(adata))
    assert np.isnan(rep.loc["P0", "value"])
    assert np.isfinite(rep.drop(index="P0")["value"].to_numpy(dtype=float)).all()
    assert {"P0"} == set(res["replicate"]["draws"].loc[
        lambda d: d["value"].isna(), "patient"].astype(str))

    row = res["stats"].iloc[0]
    arms = {row["level_a"]: row["n_a"], row["level_b"]: row["n_b"]}
    assert arms == {"disease": 2, "control": 3}

    if metric == "clonotypic_entropy":
        items = res["item"]["summary"]
        p0 = items.loc[items["patient"].astype(str) == "P0"]
        assert sorted(p0["phenotype"].astype(str)) == sorted(
            map(str, adata.uns[K.PHENOTYPE_CATEGORIES]))
        assert p0["value"].isna().all()
    elif metric == "phenotypic_entropy":
        items = res["item"]["summary"]
        assert not (items["patient"].astype(str) == "P0").any()


def test_an_arm_with_no_finite_value_keeps_its_stats_row(cohort):
    """Every pair of split levels with a replicate has its ``stats`` row: an arm whose
    replicates are all NaN gives ``n`` 0, NaN statistics and blank stars."""
    _, adata = cohort
    cov = _cov(adata)
    controls = [p for p in _patients(adata)
                if adata.obs.loc[adata.obs["patient"] == p, "disease_status"].iloc[0]
                == "control"]
    res = _call("mutual_information", adata, groupby="patient", splitby="disease_status",
                clones=_clones_at(adata, cov, controls))
    assert len(res["stats"]) == 1
    row = res["stats"].iloc[0]
    assert row["comparison"] == "split"
    assert {row["level_a"]: row["n_a"], row["level_b"]: row["n_b"]} == {"disease": 0,
                                                                        "control": 3}
    assert np.isnan(row["p"]) and np.isnan(row["stat"]) and np.isnan(row["delta"])
    assert row["stars"] == ""


@pytest.mark.parametrize("metric", ["mutual_information", "clonotypic_entropy"])
def test_a_replicate_with_one_clone_raises_naming_it(metric, cohort):
    """A normalized replicate value whose normalizer is 0 raises, naming the metric, the
    replicate and the covariate level; the same call in bits returns."""
    _, adata = cohort
    cov = _cov(adata)
    others = [p for p in _patients(adata) if p != "P1"]
    kept = _clones_at(adata, cov, ["P1"])[:1] + _clones_at(adata, cov, others)
    with pytest.raises(ValueError, match=rf"{metric}: patient='P1' at covariate '{cov}'"):
        _call(metric, adata, groupby="patient", clones=kept)
    res = _call(metric, adata, groupby="patient", clones=kept, normalized=False)
    rep = res["replicate"]["summary"].set_index("patient")
    assert rep.loc["P1", "value"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("metric", METRICS)
def test_with_no_replicate_the_object_is_one_replicate(metric, cohort):
    """No replicate registered and no ``groupby``: one replicate row with its draws, null value,
    adjusted value and interval; nothing is compared."""
    _, adata = cohort
    bare = adata.copy()
    bare.uns[K.METADATA] = {**bare.uns[K.METADATA], "replicate": None}
    res = _call(metric, bare, n_samples=3, random_state=0)

    summary, draws = res["replicate"]["summary"], res["replicate"]["draws"]
    assert len(summary) == 1 and "patient" not in summary.columns
    assert {"value", "null_value", "adjusted", "hdi_low", "hdi_high",
            "adjusted_hdi_low", "adjusted_hdi_high"} <= set(summary.columns)
    assert np.isfinite(summary[["value", "null_value", "adjusted"]].to_numpy(dtype=float)).all()
    assert len(draws) == 3 * draws["null"].nunique()
    assert res["stats"] is None


@pytest.mark.parametrize("metric, level", [("clonotypic_entropy", "replicate"),
                                           ("phenotypic_entropy", "replicate"),
                                           ("clonotypic_entropy", "item")])
def test_a_reference_of_k_fits_gives_k_rows_per_draw(metric, level, cohort):
    """Draws computed against a reference of K null fits carry a ``null`` column with K
    values, and K times the rows of the same call with ``null_model=None``."""
    _, adata = cohort
    n_nulls = len(K.members(adata, "null.phenotype"))
    assert n_nulls > 1, "a reference of one fit cannot tell K rows from one"
    kw = dict(groupby="patient", splitby="disease_status", n_samples=3, random_state=0)
    bare = _call(metric, adata, null_model=None, **kw)[level]["draws"]
    pooled = _call(metric, adata, **kw)[level]["draws"]
    assert pooled["null"].nunique() == n_nulls
    assert len(pooled) == n_nulls * len(bare)


def test_the_replicate_phenotypic_entropy_of_a_row_normalized_joint_is_the_item_mean(cohort):
    """At ``weighted=False`` every clone has the same share of the joint, so the replicate value
    is the plain mean of the clones' values, on every draw."""
    _, adata = cohort
    res = _call("phenotypic_entropy", adata, groupby="patient", n_samples=3, random_state=0,
                null_model=None, weighted=False)
    items = res["item"]["draws"].dropna(subset=["value"])
    mean = items.groupby(["patient", "draw"], observed=True)["value"].mean()
    rep = res["replicate"]["draws"].set_index(["patient", "draw"])["value"]
    pd.testing.assert_series_equal(rep.loc[mean.index], mean, check_names=False, atol=1e-12)


def test_the_get_views_read_the_stored_levels(cohort, tmp_path):
    """``items``, ``stats`` and ``draws`` are the stored frames, before and after an ``.h5ad``
    round trip; a metric with no items raises on the item views."""
    import anndata as ad

    _, adata = cohort
    adata = adata.copy()
    cov = _cov(adata)
    kw = dict(covariate=cov, groupby="patient", splitby="disease_status", n_samples=2,
              random_state=0)
    for metric in METRICS:
        getattr(tcri.tl, metric)(adata, **kw)
    adata.write_h5ad(tmp_path / "levels.h5ad")
    reread = ad.read_h5ad(tmp_path / "levels.h5ad")

    for metric in METRICS:
        stored = tcri.get.result(adata, metric)
        assert set(stored) == ({"replicate", "stats"} if metric == "mutual_information"
                               else {"replicate", "item", "stats"}), metric
        for obj in (adata, reread):
            pd.testing.assert_frame_equal(tcri.get.stats(obj, metric), stored["stats"],
                                          check_dtype=False)
            pd.testing.assert_frame_equal(tcri.get.draws(obj, metric),
                                          stored["replicate"]["draws"], check_dtype=False)
            pd.testing.assert_frame_equal(tcri.get.result(obj, metric)["replicate"]["summary"],
                                          stored["replicate"]["summary"], check_dtype=False)
            if metric == "mutual_information":
                with pytest.raises(KeyError, match="no item level"):
                    tcri.get.items(obj, metric)
                with pytest.raises(KeyError, match="no item level"):
                    tcri.get.draws(obj, metric, items=True)
            else:
                pd.testing.assert_frame_equal(tcri.get.items(obj, metric),
                                              stored["item"]["summary"], check_dtype=False)
                pd.testing.assert_frame_equal(tcri.get.draws(obj, metric, items=True),
                                              stored["item"]["draws"], check_dtype=False)


def test_the_level_views_refuse_a_result_stored_otherwise(cohort):
    """``joint_distribution`` stores ``{summary, draws}``: the level views refuse it, and
    ``tcri.get.draws`` reads its one draws frame and has no item level to give."""
    _, adata = cohort
    adata = adata.copy()
    tcri.tl.joint_distribution(adata, covariate=_cov(adata), n_samples=2, random_state=0)
    for view in (tcri.get.items, tcri.get.replicates):
        with pytest.raises(KeyError, match="not stored by level"):
            view(adata, "joint_distribution")
    pd.testing.assert_frame_equal(tcri.get.draws(adata, "joint_distribution"),
                                  tcri.get.result(adata, "joint_distribution")["draws"])
    with pytest.raises(KeyError, match="has no item level"):
        tcri.get.draws(adata, "joint_distribution", items=True)


@pytest.mark.parametrize("mode", ["covariate", "change"])
def test_a_generator_random_state_is_recorded_as_its_integer_seed(mode, cohort, tmp_path):
    """A numpy ``Generator`` as ``random_state`` runs on one integer seed, which ``params``
    records: the result writes to ``.h5ad``, and the recorded seed reruns the same draws."""
    _, adata = cohort
    adata = adata.copy()
    a, b = list(adata.uns[K.COVARIATE_CATEGORIES])[:2]
    where = dict(covariate=a) if mode == "covariate" else dict(cov_from=a, cov_to=b)
    tcri.tl.phenotypic_entropy(adata, **where, n_samples=3,
                               random_state=np.random.default_rng(7))
    seed = tcri.get.params(adata, "phenotypic_entropy")["random_state"]
    assert isinstance(seed, (int, np.integer)), type(seed)
    adata.write_h5ad(tmp_path / "generator.h5ad")
    again = tcri.tl.phenotypic_entropy(adata, **where, n_samples=3, random_state=int(seed),
                                       inplace=False)
    pd.testing.assert_frame_equal(again["replicate"]["draws"],
                                  tcri.get.draws(adata, "phenotypic_entropy"), check_dtype=False)
