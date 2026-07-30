"""The four engine-backed metric twins + compare_groups (PR6). Shapes, bits/normalization,
and the group-comparison math."""
import numpy as np
import pandas as pd
import pytest

import tcri
from tcri import _keys as K


def _cov(adata):
    return list(adata.uns[K.COVARIATE_CATEGORIES])[0]


def test_clonotypic_entropy_shapes_and_range(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    s = tcri.tl.clonotypic_entropy(adata, covariate=cov, n_samples=0)
    assert isinstance(s, pd.Series)
    assert list(s.index) == list(adata.uns[K.PHENOTYPE_CATEGORIES])
    finite = s.dropna().to_numpy()
    assert (finite >= -1e-9).all() and (finite <= 1 + 1e-9).all()  # normalized bits in [0,1]
    df = tcri.tl.clonotypic_entropy(adata, covariate=cov, n_samples=8, random_state=0)
    assert set(["mean", "sd", "hdi_low", "hdi_high"]).issubset(df.columns)


def test_phenotypic_entropy_shapes_and_range(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    s = tcri.tl.phenotypic_entropy(adata, covariate=cov, n_samples=0)
    assert isinstance(s, pd.Series)
    finite = s.dropna().to_numpy()
    assert (finite >= -1e-9).all() and (finite <= 1 + 1e-9).all()


def test_mutual_information_scalar_and_fastpath(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    mi = tcri.tl.mutual_information(adata, covariate=cov, n_samples=0)
    assert isinstance(mi, float) and -1e-9 <= mi <= 1 + 1e-9  # normalized (min) in [0,1]
    jd = tcri.joint_distribution(adata, covariate=cov, n_samples=0)
    assert np.isclose(tcri.tl.mutual_information(jd, n_samples=0), mi)  # fast path
    summ = tcri.tl.mutual_information(adata, covariate=cov, n_samples=8, random_state=0)
    assert set(["mean", "sd", "hdi_low", "hdi_high"]) == set(summ)


def test_mutual_information_unnormalized_is_bits(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    mi_bits = tcri.tl.mutual_information(adata, covariate=cov, n_samples=0, normalized=False)
    assert mi_bits >= -1e-9  # raw MI in bits, non-negative


def test_metric_groupby_tidy(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    df = tcri.tl.mutual_information(adata, covariate=cov, groupby="patient", n_samples=0)
    assert "patient" in df.columns and "MI" in df.columns
    ce = tcri.tl.clonotypic_entropy(adata, covariate=cov, groupby="patient", n_samples=0)
    assert set(["patient", "phenotype", "clonotypic_entropy"]).issubset(ce.columns)


def test_phenotypic_flux_over_common_clones(trained_model):
    _, adata = trained_model
    covs = list(adata.uns[K.COVARIATE_CATEGORIES])
    if len(covs) < 2:
        pytest.skip("needs >=2 covariates")
    fx = tcri.tl.phenotypic_flux(adata, cov_from=covs[0], cov_to=covs[1], n_samples=0, distance_metric="l1")
    assert isinstance(fx, pd.Series)
    v = fx.dropna().to_numpy()
    assert (v >= -1e-9).all() and (v <= 2 + 1e-9).all()  # l1 on the simplex is bounded [0,2]


def test_compare_groups_unpaired():
    """Mann–Whitney contrast on a tidy per-unit frame (R vs NR)."""
    df = pd.DataFrame({
        "patient": [f"p{i}" for i in range(8)],
        "response": ["R"] * 4 + ["NR"] * 4,
        "MI": [0.8, 0.75, 0.82, 0.79, 0.4, 0.35, 0.45, 0.5],
    })
    out = tcri.tl.compare_groups(df, value="MI", splitby="response", reference="NR")
    assert len(out) == 1
    row = out.iloc[0]
    assert {row["group_a"], row["group_b"]} == {"R", "NR"}
    assert row["mean_b"] > row["mean_a"] or row["mean_a"] > row["mean_b"]
    assert "p" in out.columns and "stars" in out.columns and "delta" in out.columns


def test_compare_groups_paired_direction():
    """Paired posterior-draw contrast emits p_gt + HDI via prob_direction."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(5):
        rows.append({"unit": u, "arm": "A", "v": rng.normal(0.0, 0.1, size=200)})
        rows.append({"unit": u, "arm": "B", "v": rng.normal(0.5, 0.1, size=200)})  # B > A
    df = pd.DataFrame(rows)
    out = tcri.tl.compare_groups(df, value="v", splitby="arm", reference="A", paired=True, pair_on="unit")
    assert len(out) == 1
    assert out.iloc[0]["delta"] > 0 and out.iloc[0]["p_gt"] > 0.9  # B-A positive, high direction prob
    assert "hdi_low" in out.columns
