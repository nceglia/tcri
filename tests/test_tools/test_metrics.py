"""The four engine-backed metric twins and the group contrast: shapes, bits/normalization,
and the group-comparison math.

Every ``tl`` stores its result under its ``uns`` key and returns ``None``; with
``inplace=False`` it stores nothing and returns that result. The four metrics hold
``{replicate, item, stats}`` (mutual information has no ``item``), each level ``draws`` (one
row per draw, never reduced) and ``summary`` (reduced over draws). ``stats`` holds the tests:
the covariate rows of a change and the between-split contrast, ``None`` at one covariate level
without ``splitby``. These tests read the result rather than a bare float/Series, so there is
one shape to learn rather than a return type per combination of axes.
"""
import contextlib
import io
import logging
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import tcri
from ..conftest import frames
from tcri._state import keys as K


def _cov(adata):
    return list(adata.uns[K.COVARIATE_CATEGORIES])[0]


def _one(summary):
    """The single value of a metric computed with no group."""
    return float(summary["value"].iloc[0])


def test_clonotypic_entropy_shapes_and_range(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    res = tcri.tl.clonotypic_entropy(adata, covariate=cov, n_samples=0, inplace=False)
    assert set(res) == {"replicate", "item", "stats"}
    assert res["stats"] is None, "no splitby -> no contrast"

    out = res["item"]["summary"]
    assert set(out["phenotype"]) == set(adata.uns[K.PHENOTYPE_CATEGORIES]), (
        "H(c|phi) is one value per PHENOTYPE"
    )
    for frame in (out, res["replicate"]["summary"]):
        finite = frame["value"].dropna().to_numpy()
        assert (finite >= -1e-9).all() and (finite <= 1 + 1e-9).all()  # normalized, in [0,1]
    assert len(res["replicate"]["summary"]) == 1, "no group -> one replicate value"

    sampled = tcri.tl.clonotypic_entropy(adata, covariate=cov, n_samples=8, random_state=0,
                                         inplace=False)
    for level in ("replicate", "item"):
        draws, summary = sampled[level]["draws"], sampled[level]["summary"]
        assert {"value", "sd", "hdi_low", "hdi_high"}.issubset(summary.columns)
        assert len(draws) == 8 * draws["null"].nunique() * len(summary), (
            "`draws` keeps every draw, paired with every null; `summary` is what reduces it"
        )


def test_phenotypic_entropy_shapes_and_range(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    res = tcri.tl.phenotypic_entropy(adata, covariate=cov, n_samples=0, inplace=False)
    out = res["item"]["summary"]
    assert "clonotype" in out.columns, "H(phi|c) is one value per CLONE"
    for frame in (out, res["replicate"]["summary"]):
        finite = frame["value"].dropna().to_numpy()
        assert (finite >= -1e-9).all() and (finite <= 1 + 1e-9).all()


def test_mutual_information_scalar_and_draws(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    point = tcri.tl.mutual_information(adata, covariate=cov, n_samples=0,
                                       inplace=False)["replicate"]["summary"]
    assert len(point) == 1, "no group and no item axis -> a single row"
    assert -1e-9 <= _one(point) <= 1 + 1e-9  # normalized (min) in [0,1]

    summ = tcri.tl.mutual_information(adata, covariate=cov, n_samples=8, random_state=0,
                                      inplace=False)
    assert "item" not in summ, "mutual information has one value per joint and no items"
    assert {"value", "sd", "hdi_low", "hdi_high"}.issubset(summ["replicate"]["summary"].columns)
    draws = summ["replicate"]["draws"]
    assert len(draws) == 8 * draws["null"].nunique()


def test_mutual_information_unnormalized_is_bits(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    mi_bits = _one(tcri.tl.mutual_information(adata, covariate=cov, n_samples=0,
                                              normalized=False,
                                              inplace=False)["replicate"]["summary"])
    assert mi_bits >= -1e-9  # raw MI in bits, non-negative


def test_metric_groupby_tidy(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    mi = frames(tcri.tl.mutual_information(adata, covariate=cov, groupby="patient",
                                    n_samples=0, inplace=False))[1]
    assert {"patient", "value"}.issubset(mi.columns)
    assert len(mi) == adata.obs["patient"].nunique()

    ce = frames(tcri.tl.clonotypic_entropy(adata, covariate=cov, groupby="patient",
                                    n_samples=0, inplace=False))[1]
    assert {"patient", "phenotype", "value"}.issubset(ce.columns)


def _assert_same_result(stored, returned, obj):
    """``stored`` and ``returned`` hold the same slots and, level by level, the same frames."""
    assert set(stored) == set(returned), obj
    for slot, value in returned.items():
        if value is None:
            assert stored[slot] is None, f"{obj}.{slot}"
        elif isinstance(value, dict):
            _assert_same_result(stored[slot], value, f"{obj}.{slot}")
        else:
            pd.testing.assert_frame_equal(stored[slot], value, check_dtype=False,
                                          obj=f"{obj}.{slot}")


def test_every_tool_stores_what_inplace_false_returns(trained_model):
    """The store-once invariant, on every stored tool: a call stores its result and returns
    ``None``, and ``inplace=False`` returns that same result and stores nothing.

    ``tcri.get`` is then the one reader of a stored result. If ``pl.*`` recomputed the metric
    from ``adata`` instead of reading the cache, the plot and the frame in the caller's hand
    could disagree — a different ``n_samples``, a different draw, a ``distance_metric`` that
    differed between ``tl`` and ``pl``.
    """
    model, adata = trained_model
    cov, *rest = list(adata.uns[K.COVARIATE_CATEGORIES])
    assert rest, "the fixture needs two covariate levels for the change calls"
    pair = dict(cov_from=cov, cov_to=rest[0])

    def importance(obj, **kwargs):
        return tcri.perturb.gene_importance(model, obj, **kwargs)

    # (tool, call, uns key, arguments): a metric that takes both forms is called at one level
    # and on a change, each stored under the same key
    calls = [
        ("joint_distribution", tcri.tl.joint_distribution, K.JOINT_DISTRIBUTION,
         dict(covariate=cov)),
        ("mutual_information", tcri.tl.mutual_information, K.MUTUAL_INFORMATION,
         dict(covariate=cov)),
        ("mutual_information", tcri.tl.mutual_information, K.MUTUAL_INFORMATION, pair),
        ("clonotypic_entropy", tcri.tl.clonotypic_entropy, K.CLONOTYPIC_ENTROPY,
         dict(covariate=cov)),
        ("clonotypic_entropy", tcri.tl.clonotypic_entropy, K.CLONOTYPIC_ENTROPY, pair),
        ("phenotypic_entropy", tcri.tl.phenotypic_entropy, K.PHENOTYPIC_ENTROPY,
         dict(covariate=cov)),
        ("phenotypic_entropy", tcri.tl.phenotypic_entropy, K.PHENOTYPIC_ENTROPY, pair),
        ("phenotypic_flux", tcri.tl.phenotypic_flux, K.PHENOTYPIC_FLUX, pair),
        ("gene_importance", importance, K.GENE_IMPORTANCE,
         dict(genes=list(adata.var_names[:2]), covariate=cov)),
    ]

    for name, fn, key, kwargs in calls:
        label = f"{name}({', '.join(kwargs)})"
        fresh = adata.copy()
        fresh.uns.pop(key, None)
        before = set(fresh.uns)
        returned = fn(fresh, **kwargs, inplace=False)
        assert key not in fresh.uns, f"{label}: inplace=False stored its result"
        assert set(fresh.uns) == before, f"{label}: inplace=False wrote to uns"

        assert fn(adata, **kwargs) is None, f"{label}: a storing call returned its result"
        _assert_same_result(tcri.get.result(adata, name), returned, label)

        params = tcri.get.params(adata, name)
        for param, value in kwargs.items():
            assert params[param] == value, f"{label}: params lost {param}"


def test_phenotypic_flux_over_common_clones(trained_model):
    """Flux is one value per clone at both levels, and per replicate their mean at
    ``weighted=False``; it has no endpoints."""
    _, adata = trained_model
    covs = list(adata.uns[K.COVARIATE_CATEGORIES])
    if len(covs) < 2:
        pytest.skip("needs >=2 covariates")
    res = tcri.tl.phenotypic_flux(adata, cov_from=covs[0], cov_to=covs[1], n_samples=0,
                                  distance_metric="l1", inplace=False)
    out = res["item"]["summary"]
    assert {"cov_from", "cov_to", "clonotype", "value"}.issubset(out.columns)
    assert not {"value_from", "value_to"} & set(out.columns)
    v = out["value"].dropna().to_numpy()
    assert (v >= -1e-9).all() and (v <= 2 + 1e-9).all()  # l1 on the simplex is bounded [0,2]

    obs = adata.obs
    at = {lv: set(obs.loc[obs["timepoint"] == lv, "unique_clone_id"].astype(str))
          for lv in covs[:2]}
    assert set(out["clonotype"].astype(str)) == at[covs[0]] & at[covs[1]]
    rep = res["replicate"]["summary"]
    assert len(rep) == 1, "no replicate registered: the whole object is one replicate"
    assert float(rep["value"].iloc[0]) == pytest.approx(v.mean(), rel=1e-12)


@pytest.mark.parametrize("distance_metric", ["jsd", "js"])
def test_phenotypic_flux_rejects_jensen_shannon(trained_model, distance_metric):
    """Flux is the KL divergence, and Jensen–Shannon is not one of its distances: the name
    raises, and the error lists the ones that are."""
    _, adata = trained_model
    cov = _cov(adata)
    with pytest.raises(ValueError, match="distance_metric") as err:
        tcri.tl.phenotypic_flux(adata, cov_from=cov, cov_to=cov, distance_metric=distance_metric)
    message = str(err.value)
    assert all(f"'{name}'" in message for name in ("kl", "dkl", "l1")), message


# ── the contrast: internal, and reached through `splitby` ────────────────────

def test_compare_groups_is_not_public():
    """The contrast is not a public step: ``splitby`` produces it as part of the metric.

    A public function would have to be called on the right frame, with the replicate unit
    picked by the caller. It stays importable internally, for the metrics to use."""
    assert "compare_groups" not in tcri.tl.__all__
    assert not hasattr(tcri.tl, "compare_groups")
    from tcri._stats import compare_groups   # importable internally, not a public step

    assert callable(compare_groups)


def test_build_stats_delegates_to_the_one_contrast(cohort):
    """``stats`` is not a second Mann-Whitney implementation — it IS ``compare_groups``.

    Two copies of "rank-test two levels and star the p" would be free to drift apart. This
    asserts they cannot: the delta and p in ``stats`` equal what ``compare_groups`` returns on
    the per-group frame of the column ``stats`` says it contrasted (``quantity``: the adjusted
    value, since the fixture carries a reference).
    """
    from tcri._stats import compare_groups

    _, adata = cohort
    cov = list(adata.uns[K.COVARIATE_CATEGORIES])[0]
    res = tcri.tl.mutual_information(adata, covariate=cov, groupby="patient",
                                     splitby="disease_status", inplace=False)
    stats, result = res["stats"], res["replicate"]["summary"]

    row = stats.iloc[0]
    quantity = row["quantity"]
    assert quantity == "adjusted"
    per_group = (result.groupby(["patient", "disease_status"], observed=True)[quantity]
                 .mean().reset_index())
    direct = compare_groups(per_group, value=quantity, splitby="disease_status",
                            replicate="patient").iloc[0]
    assert row["p"] == pytest.approx(float(direct["p"]))
    assert row["delta"] == pytest.approx(float(direct["delta"]))
    assert row["stat"] == pytest.approx(float(direct["stat"]))


def test_stats_carries_the_between_replicate_spread(cohort):
    """``ci_*`` (across patients) sits beside ``hdi_*`` (across draws, within a patient).

    They are different quantities and are named apart on purpose. Reporting them under one
    name would make them indistinguishable on sight — and a between-patient interval read as
    a posterior would badly overstate what the model claims to know.
    """
    _, adata = cohort
    cov = list(adata.uns[K.COVARIATE_CATEGORIES])[0]
    res = tcri.tl.mutual_information(adata, covariate=cov, groupby="patient",
                                     splitby="disease_status", n_samples=8, random_state=0,
                                     inplace=False)
    row = res["stats"].iloc[0]

    assert row["replicate_unit"] == "patient"
    assert row["n_a"] == 3 and row["n_b"] == 3
    for suffix in ("a", "b"):
        assert row[f"ci_low_{suffix}"] <= row[f"mean_{suffix}"] <= row[f"ci_high_{suffix}"]
        assert row[f"sd_{suffix}"] > 0

    assert {"hdi_low", "hdi_high"} <= set(res["replicate"]["summary"].columns)
    assert not {"hdi_low", "hdi_high"} & set(res["stats"].columns), (
        "a within-group posterior interval leaked into the between-replicate slot"
    )


@pytest.mark.parametrize("metric,item_col", [
    ("phenotypic_entropy", "clonotype"),
    ("clonotypic_entropy", "phenotype"),
    ("phenotypic_flux", "clonotype"),
    ("mutual_information", None),
])
def test_the_contrast_counts_groups_even_when_the_metric_has_items(metric, item_col, cohort):
    """n is the number of PATIENTS, on every metric — including the ones with an item axis.

    The item-bearing metrics are where this bites: the item summary holds one row per (patient,
    clone), so handing it to a rank test would compare hundreds of clones and report a star off
    six patients. Checked here on all four because the same assertion on `mutual_information`
    alone proves nothing — MI has no item axis. Flux is always a change, so its ``stats`` also
    holds covariate rows; the split row is the one read here.
    """
    _, adata = cohort
    covs = list(adata.uns[K.COVARIATE_CATEGORIES])
    kwargs = dict(groupby="patient", splitby="disease_status")
    if metric == "phenotypic_flux":
        kwargs.update(cov_from=covs[0], cov_to=covs[1])
    else:
        kwargs["covariate"] = covs[0]
    res = getattr(tcri.tl, metric)(adata, **kwargs, inplace=False)

    split = res["stats"].loc[res["stats"]["comparison"] == "split"]
    assert len(split) == 1
    row = split.iloc[0]
    assert row["n_a"] == 3 and row["n_b"] == 3, (
        f"{metric}: n={row['n_a']}/{row['n_b']} is not the 3 patients per arm"
    )
    if item_col is not None:
        n_rows = len(res["item"]["summary"])
        assert n_rows > 6, "fixture is too small for this test to discriminate"
        assert row["n_a"] + row["n_b"] < n_rows, (
            f"{metric}: the contrast saw {row['n_a'] + row['n_b']} of {n_rows} item rows -- "
            f"items were not collapsed to groups"
        )


def test_compare_groups_unpaired():
    """Mann–Whitney contrast on a per-replicate frame (R vs NR), equal to scipy's."""
    from scipy.stats import mannwhitneyu

    from tcri._stats import compare_groups

    df = pd.DataFrame({
        "patient": [f"p{i}" for i in range(8)],
        "response": ["R"] * 4 + ["NR"] * 4,
        "MI": [0.8, 0.75, 0.82, 0.79, 0.4, 0.35, 0.45, 0.5],
    })
    out = compare_groups(df, value="MI", splitby="response", replicate="patient")
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["group_a"], row["group_b"]) == ("NR", "R")
    nr, r = df.loc[df.response == "NR", "MI"], df.loc[df.response == "R", "MI"]
    U, p = mannwhitneyu(nr, r, alternative="two-sided")
    assert row["stat"] == pytest.approx(U) and row["p"] == pytest.approx(p)
    assert row["delta"] == pytest.approx(r.mean() - nr.mean())
    assert (row["n_a"], row["n_b"]) == (4, 4)


def test_compare_groups_signed_rank_on_paired_replicates():
    """Paired: the signed-rank test on each replicate's difference, level b minus level a,
    equal to scipy's; a zero difference leaves the test and ``n``."""
    from scipy.stats import wilcoxon

    from tcri._stats import compare_groups

    a = [0.10, 0.20, 0.15, 0.30, 0.25, 0.40]
    b = [0.30, 0.45, 0.20, 0.55, 0.25, 0.70]          # p5's difference is exactly 0
    df = pd.DataFrame({"patient": [f"p{i}" for i in range(6)] * 2,
                       "timepoint": ["pre"] * 6 + ["post"] * 6, "v": a + b})
    row = compare_groups(df, value="v", splitby="timepoint", replicate="patient",
                         paired=True).iloc[0]
    assert (row["group_a"], row["group_b"]) == ("post", "pre")
    diffs = np.array(a) - np.array(b)                # pre - post: level b minus level a
    W, p = wilcoxon(diffs[diffs != 0])
    assert row["stat"] == pytest.approx(W) and row["p"] == pytest.approx(p)
    assert (row["n_a"], row["n_b"]) == (5, 5), "the zero difference counted as a replicate"


@pytest.mark.parametrize("b", [[0.10, 0.20], [0.30, 0.20]], ids=["no-change", "one-change"])
def test_compare_groups_signed_rank_needs_two_differences(b):
    """Fewer than two non-zero differences: the row is kept with NaN p and blank stars, and the
    test is not run (scipy would return p = 1.0, or raise on a single zero)."""
    from tcri._stats import compare_groups

    df = pd.DataFrame({"patient": ["p0", "p1"] * 2, "timepoint": ["pre"] * 2 + ["post"] * 2,
                       "v": [0.10, 0.20] + b})
    row = compare_groups(df, value="v", splitby="timepoint", replicate="patient",
                         paired=True).iloc[0]
    assert np.isnan(row["p"]) and np.isnan(row["stat"]) and row["stars"] == ""


def test_compare_groups_refuses_rows_that_are_not_replicates():
    """Two rows for one replicate at one level are items or draws, not replicates: refused,
    so a per-clone frame can never be tested as if each clone were a patient."""
    from tcri._stats import compare_groups

    df = pd.DataFrame({"patient": ["p0", "p0", "p1", "p2"],
                       "response": ["R", "R", "NR", "NR"], "v": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="replicate 'p0' has 2 rows"):
        compare_groups(df, value="v", splitby="response", replicate="patient")


def test_compare_groups_refuses_a_non_numeric_value():
    from tcri._stats import compare_groups

    df = pd.DataFrame({"patient": ["p0", "p1"], "response": ["R", "NR"], "v": ["a", "b"]})
    with pytest.raises(ValueError, match="not numeric"):
        compare_groups(df, value="v", splitby="response", replicate="patient")


def test_build_result_survives_an_empty_table():
    """An empty table has NO columns, which is what makes this a crash rather than a no-op.

    `build_result` derives its grouping keys from `table.columns`. When the table is empty
    those columns are a bare RangeIndex, so `keys` is empty, the scalar branch runs, and
    `table["value"]` raises KeyError('value') from inside pandas.

    This is a real path, not a hypothetical: `phenotypic_flux` legitimately produces no item
    rows when no replicate has clones at both covariate levels — which is always true if the
    covariate is constant within replicate.
    """
    import pandas as pd
    from tcri._compute._tables import build_result

    out = build_result(pd.DataFrame())
    assert out is not None and len(out) == 0, "an empty table must reduce to an empty result"
    assert "value" in out.columns, "the result must keep its schema so downstream code can read it"


@pytest.fixture(scope="module")
def constant_within():
    """Two fitted patients, each with cells at one covariate level only: the covariate is
    constant within every replicate, as a between-group label (treatment arm, response) is."""
    from tcri.datasets import simulate_tcri
    from tcri.model._model import TCRIModel

    parts = []
    for i, (patient, level) in enumerate((("P0", "cov_0"), ("P1", "cov_1"))):
        block = simulate_tcri(n_clones=8, n_phenotypes=4, n_genes=25, n_cells=150,
                              n_covariates=1, omega_concentration=0.3, seed=i)
        block.obs["clone_id"] = block.obs["clone_id"].astype(str) + "@" + patient
        block.obs["covariate"] = level
        block.obs["patient"] = patient
        block.obs_names = [f"{patient}_{j}" for j in range(block.n_obs)]
        parts.append(block)
    adata = ad.concat(parts, join="outer", label=None)
    for col in ("clone_id", "phenotype", "covariate", "patient"):
        adata.obs[col] = adata.obs[col].astype("category")
    adata.layers["counts"] = adata.X.copy()

    logging.disable(logging.INFO)
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                                    phenotype_key="phenotype", covariate_key="covariate",
                                    batch_key="patient", replicate="patient")
            model = TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1,
                              classifier_n_layers=1, classifier_hidden=16, K=4, seed=0,
                              name="constant_within")
            model.train(max_epochs=10, batch_size=128, n_epochs_kl_warmup=4,
                        accelerator="cpu", enable_progress_bar=False,
                        enable_model_summary=False)
            model.to_anndata(adata)
    finally:
        logging.disable(logging.NOTSET)
    return adata


@pytest.mark.parametrize("metric", ["phenotypic_flux", "mutual_information",
                                    "clonotypic_entropy", "phenotypic_entropy"])
def test_a_change_explains_a_covariate_constant_within_every_replicate(metric,
                                                                       constant_within):
    """A change that finds no clone at both levels must say WHY, not return silently empty.

    The outcome is legitimate -- no clone present at both covariate levels within any replicate
    -- and is exactly what a covariate that is constant within replicate produces. A user who
    points a change at a between-group label gets NaN back, and without the warnings has no way
    to tell that their covariate is the problem rather than their data. Two warnings, from the
    caller's line: how many replicates have no clone at one of the levels, and that none has a
    clone at both, naming both levels and the within-replicate rule, and, on a metric with
    items, that there are no item rows.

    ``null_model=None``: no covariate null is fitted on this object, and the warnings are about
    the call's inputs, not its reference.
    """
    adata = constant_within
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = getattr(tcri.tl, metric)(adata, cov_from="cov_0", cov_to="cov_1",
                                       null_model=None, inplace=False)

    mine = [w for w in caught if str(w.message).startswith(f"{metric}:")]
    messages = [str(w.message) for w in mine]
    assert len(messages) == 2, messages
    assert messages[0] == (
        f"{metric}: 2 of 2 patient groups have no counted clone at 'cov_0' or at 'cov_1'; "
        f"their values are NaN and they leave the tests.")
    has_items = metric != "mutual_information"
    assert messages[1].startswith(
        f"{metric}: no patient has a clone at both 'cov_0' and 'cov_1'"
        + (", so there are no item rows. " if has_items else ". ")), messages[1]
    if not has_items:
        assert "item rows" not in messages[1], "mutual information has no items to be without"
    assert "taken within each patient" in messages[1], "the warning must name the rule"
    assert "varies inside a single patient" in messages[1]
    assert all(w.filename == __file__ for w in mine), "a warning names a line in the package"

    rep = res["replicate"]["summary"]
    assert sorted(rep["patient"].astype(str)) == ["P0", "P1"]
    assert rep["value"].isna().all(), "a replicate with one level has no change"
    if metric == "clonotypic_entropy":
        items = res["item"]["summary"]
        assert len(items) == 2 * len(adata.uns[K.PHENOTYPE_CATEGORIES])
        assert items["value"].isna().all()
    elif metric != "mutual_information":
        items = res["item"]["summary"]
        assert len(items) == 0 and "value" in items.columns, "no item rows, and no crash"

    row = res["stats"].loc[res["stats"]["comparison"] == "covariate"].iloc[0]
    if metric != "phenotypic_flux":
        # a flux row without a usable reference is pinned by its missing test alone
        assert row["n_a"] == row["n_b"] == 0
    assert np.isnan(row["p"]) and row["stars"] == ""
