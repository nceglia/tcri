"""The four engine-backed metric twins + compare_groups (PR6). Shapes, bits/normalization,
and the group-comparison math.

Every ``tl`` now returns the same three payload keys — ``table`` (one row per draw, never
reduced), ``result`` (reduced over draws), ``stats`` (the between-split contrast, ``None``
without ``splitby``) — and stores that object under its ``uns`` key. These tests read the
payload rather than a bare float/Series, which is the whole point of the migration: there is
one shape to learn instead of four return types that depended on which axes you passed.
"""
import numpy as np
import pandas as pd
import pytest

import tcri
from tcri import _keys as K


def _cov(adata):
    return list(adata.uns[K.COVARIATE_CATEGORIES])[0]


def _one(res):
    """The single value of a metric computed with no group and no item axis."""
    return float(res["result"]["value"].iloc[0])


def test_clonotypic_entropy_shapes_and_range(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    res = tcri.tl.clonotypic_entropy(adata, covariate=cov, n_samples=0)
    assert set(res) == {"table", "result", "stats"}
    assert res["stats"] is None, "no splitby -> no contrast"

    out = res["result"]
    assert set(out["phenotype"]) == set(adata.uns[K.PHENOTYPE_CATEGORIES]), (
        "H(c|phi) is one value per PHENOTYPE"
    )
    finite = out["value"].dropna().to_numpy()
    assert (finite >= -1e-9).all() and (finite <= 1 + 1e-9).all()  # normalized bits in [0,1]

    sampled = tcri.tl.clonotypic_entropy(adata, covariate=cov, n_samples=8, random_state=0)
    assert {"value", "sd", "hdi_low", "hdi_high"}.issubset(sampled["result"].columns)
    assert len(sampled["table"]) == 8 * len(sampled["result"]), (
        "`table` keeps every draw; `result` is what reduces it"
    )


def test_phenotypic_entropy_shapes_and_range(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    out = tcri.tl.phenotypic_entropy(adata, covariate=cov, n_samples=0)["result"]
    assert "clonotype" in out.columns, "H(phi|c) is one value per CLONE"
    finite = out["value"].dropna().to_numpy()
    assert (finite >= -1e-9).all() and (finite <= 1 + 1e-9).all()


def test_mutual_information_scalar_and_draws(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    point = tcri.tl.mutual_information(adata, covariate=cov, n_samples=0)
    assert len(point["result"]) == 1, "no group and no item axis -> a single row"
    assert -1e-9 <= _one(point) <= 1 + 1e-9  # normalized (min) in [0,1]

    summ = tcri.tl.mutual_information(adata, covariate=cov, n_samples=8, random_state=0)
    assert {"value", "sd", "hdi_low", "hdi_high"}.issubset(summ["result"].columns)
    assert len(summ["table"]) == 8


def test_mutual_information_unnormalized_is_bits(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    mi_bits = _one(tcri.tl.mutual_information(adata, covariate=cov, n_samples=0,
                                              normalized=False))
    assert mi_bits >= -1e-9  # raw MI in bits, non-negative


def test_metric_groupby_tidy(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    mi = tcri.tl.mutual_information(adata, covariate=cov, groupby="patient",
                                    n_samples=0)["result"]
    assert {"patient", "value"}.issubset(mi.columns)
    assert len(mi) == adata.obs["patient"].nunique()

    ce = tcri.tl.clonotypic_entropy(adata, covariate=cov, groupby="patient",
                                    n_samples=0)["result"]
    assert {"patient", "phenotype", "value"}.issubset(ce.columns)


def test_every_metric_returns_what_it_cached(trained_model):
    """The store-once invariant, on all five tools: what you get back IS what is in ``uns``.

    Before this, each ``pl.*`` recomputed the metric from ``adata``, so the plot and the frame
    in the caller's hand could disagree — different ``n_samples``, a different draw, a
    ``distance_metric`` default that differed between ``tl`` and ``pl``.
    """
    _, adata = trained_model
    cov, *rest = list(adata.uns[K.COVARIATE_CATEGORIES])

    calls = {
        "joint_distribution": dict(covariate=cov),
        "mutual_information": dict(covariate=cov),
        "clonotypic_entropy": dict(covariate=cov),
        "phenotypic_entropy": dict(covariate=cov),
    }
    if rest:
        calls["phenotypic_flux"] = dict(cov_from=cov, cov_to=rest[0])

    for name, kwargs in calls.items():
        returned = getattr(tcri.tl, name)(adata, **kwargs)
        cached = tcri.get.result(adata, name)
        assert set(cached) == set(returned), name
        for slot, frame in returned.items():
            if frame is None:
                assert cached[slot] is None, f"{name}.{slot}"
            else:
                pd.testing.assert_frame_equal(cached[slot], frame, check_dtype=False,
                                              obj=f"{name}.{slot}")
        params = tcri.get.params(adata, name)
        for key, value in kwargs.items():
            assert params[key] == value, f"{name}: params lost {key}"


def test_phenotypic_flux_over_common_clones(trained_model):
    _, adata = trained_model
    covs = list(adata.uns[K.COVARIATE_CATEGORIES])
    if len(covs) < 2:
        pytest.skip("needs >=2 covariates")
    res = tcri.tl.phenotypic_flux(adata, cov_from=covs[0], cov_to=covs[1], n_samples=0,
                                  distance_metric="l1")
    out = res["result"]
    assert {"cov_from", "cov_to", "clonotype", "value"}.issubset(out.columns)
    v = out["value"].dropna().to_numpy()
    assert (v >= -1e-9).all() and (v <= 2 + 1e-9).all()  # l1 on the simplex is bounded [0,2]


# ── the contrast: internal, and reached through `splitby` ────────────────────

def test_compare_groups_is_not_public():
    """It was ``tl.compare_groups``: a second function you had to remember to call, on the
    right frame, having picked the replicate unit yourself. ``splitby`` now produces the
    contrast as part of the metric, so the separate step has nothing left to do."""
    assert "compare_groups" not in tcri.tl.__all__
    assert not hasattr(tcri.tl, "compare_groups")
    from tcri.tools._compare import compare_groups   # still there, just not a public step

    assert callable(compare_groups)


def test_build_stats_delegates_to_the_one_contrast(cohort):
    """``stats`` is not a second Mann-Whitney implementation — it IS ``compare_groups``.

    Two copies of "rank-test two levels and star the p" is exactly how the ``tl``/``pl``
    ``distance_metric`` disagreement happened. This asserts they cannot drift: the delta and
    p in ``stats`` equal what ``compare_groups`` returns on the per-group frame.
    """
    from tcri.tools._compare import compare_groups

    _, adata = cohort
    cov = list(adata.uns[K.COVARIATE_CATEGORIES])[0]
    res = tcri.tl.mutual_information(adata, covariate=cov, groupby="patient",
                                     splitby="response")
    stats, result = res["stats"], res["result"]

    per_group = (result.groupby(["patient", "response"], observed=True)["value"]
                 .mean().reset_index())
    direct = compare_groups(per_group, value="value", splitby="response").iloc[0]
    row = stats.iloc[0]
    assert row["p"] == pytest.approx(float(direct["p"]))
    assert row["delta"] == pytest.approx(float(direct["delta"]))
    assert row["stat"] == pytest.approx(float(direct["U"]))


def test_stats_carries_the_between_replicate_spread(cohort):
    """``ci_*`` (across patients) sits beside ``hdi_*`` (across draws, within a patient).

    They are different quantities and are named apart on purpose. Reporting them under one
    name would make them indistinguishable on sight — and a between-patient interval read as
    a posterior would badly overstate what the model claims to know.
    """
    _, adata = cohort
    cov = list(adata.uns[K.COVARIATE_CATEGORIES])[0]
    res = tcri.tl.mutual_information(adata, covariate=cov, groupby="patient",
                                     splitby="response", n_samples=8, random_state=0)
    row = res["stats"].iloc[0]

    assert row["replicate_unit"] == "patient"
    assert row["n_a"] == 3 and row["n_b"] == 3
    for suffix in ("a", "b"):
        assert row[f"ci_low_{suffix}"] <= row[f"mean_{suffix}"] <= row[f"ci_high_{suffix}"]
        assert row[f"sd_{suffix}"] > 0

    assert {"hdi_low", "hdi_high"} <= set(res["result"].columns)
    assert not {"hdi_low", "hdi_high"} & set(res["stats"].columns), (
        "a within-group posterior interval leaked into the between-replicate slot"
    )


def test_compare_groups_unpaired():
    """Mann–Whitney contrast on a tidy per-unit frame (R vs NR)."""
    from tcri.tools._compare import compare_groups

    df = pd.DataFrame({
        "patient": [f"p{i}" for i in range(8)],
        "response": ["R"] * 4 + ["NR"] * 4,
        "MI": [0.8, 0.75, 0.82, 0.79, 0.4, 0.35, 0.45, 0.5],
    })
    out = compare_groups(df, value="MI", splitby="response", reference="NR")
    assert len(out) == 1
    row = out.iloc[0]
    assert {row["group_a"], row["group_b"]} == {"R", "NR"}
    assert row["mean_b"] > row["mean_a"] or row["mean_a"] > row["mean_b"]
    assert "p" in out.columns and "stars" in out.columns and "delta" in out.columns


def test_compare_groups_paired_direction():
    """Paired posterior-draw contrast emits p_gt + HDI via prob_direction.

    NOTE: this branch has no producer — it wants a frame whose cells are draw VECTORS, and
    no ``tl`` emits that shape. It is kept rather than deleted because ``table`` (one row per
    group/item/draw) makes a paired posterior contrast genuinely reachable now, and which
    estimand that should be is a question for the authors. Tracked as an issue.
    """
    from tcri.tools._compare import compare_groups

    rng = np.random.default_rng(0)
    rows = []
    for u in range(5):
        rows.append({"unit": u, "arm": "A", "v": rng.normal(0.0, 0.1, size=200)})
        rows.append({"unit": u, "arm": "B", "v": rng.normal(0.5, 0.1, size=200)})  # B > A
    df = pd.DataFrame(rows)
    out = compare_groups(df, value="v", splitby="arm", reference="A", paired=True,
                         pair_on="unit")
    assert len(out) == 1
    assert out.iloc[0]["delta"] > 0 and out.iloc[0]["p_gt"] > 0.9  # B-A positive, high direction prob
    assert "hdi_low" in out.columns
