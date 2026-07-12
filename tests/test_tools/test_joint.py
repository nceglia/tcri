"""The Phase-5 engine gate — ``tcri.joint_distribution`` identities (§7.1).

Comparisons use the *frozen* canonical keys written by ``to_anndata`` (``uns[P_CT]``,
``obsm[X_LOGITS]``, ``obsm[X_PROBABILITIES]``) rather than a live ``predict()`` call,
so the identities do not depend on the process-global pyro param store (§5.2).
"""
import json

import numpy as np
import pandas as pd

import tcri
from tcri import _keys as K


def _first_covariate(adata):
    return list(adata.uns[K.COVARIATE_CATEGORIES])[0]


def test_top_level_reexport():
    assert hasattr(tcri, "joint_distribution")
    assert tcri.joint_distribution is tcri.tools.joint_distribution


def test_identity_ct_table_equals_p_ct(trained_model):
    """use_logits=False, n_samples=0, T=1  ==  uns[P_CT] restricted to the covariate."""
    _, adata = trained_model
    cov = _first_covariate(adata)
    df = tcri.joint_distribution(adata, covariate=cov, use_logits=False, n_samples=0, temperature=1.0)

    cov_i = list(adata.uns[K.COVARIATE_CATEGORIES]).index(cov)
    rows = np.where(np.asarray(adata.uns[K.CT_TO_COV]) == cov_i)[0]
    expected = np.asarray(adata.uns[K.P_CT])[rows]  # one ct row per clone
    np.testing.assert_allclose(df.values, expected, atol=1e-6)


def test_identity_use_logits_equals_predict_aggregation(trained_model):
    """use_logits=True, n_samples=0, T=1  ==  per-clone mean of predict() (== frozen X_PROBABILITIES)."""
    _, adata = trained_model
    cov = _first_covariate(adata)
    df = tcri.joint_distribution(adata, covariate=cov, use_logits=True, n_samples=0, temperature=1.0)

    cov_i = list(adata.uns[K.COVARIATE_CATEGORIES]).index(cov)
    mask = np.asarray(adata.uns[K.COV_ARRAY]) == cov_i
    clone_col = adata.uns[K.METADATA]["clone_col"]
    probs = adata.obsm[K.X_PROBABILITIES][mask]
    agg = pd.DataFrame(probs, columns=list(adata.uns[K.PHENOTYPE_CATEGORIES]))
    agg["c"] = adata.obs[clone_col].values[mask]
    agg = agg.groupby("c")[list(adata.uns[K.PHENOTYPE_CATEGORIES])].mean().reindex(df.index)
    np.testing.assert_allclose(df.values, agg.values, atol=1e-4)


def test_point_estimate_is_deterministic(trained_model):
    """n_samples=0 is deterministic and bit-identical across repeated calls."""
    _, adata = trained_model
    cov = _first_covariate(adata)
    a = tcri.joint_distribution(adata, covariate=cov, n_samples=0)
    b = tcri.joint_distribution(adata, covariate=cov, n_samples=0)
    assert np.array_equal(a.values, b.values)


def test_sampling_is_seeded_reproducible(trained_model):
    """n_samples>0 is reproducible under a fixed random_state, varies otherwise, and
    carries a (clonotype, sample_id) MultiIndex."""
    _, adata = trained_model
    cov = _first_covariate(adata)
    s1 = tcri.joint_distribution(adata, covariate=cov, n_samples=8, random_state=0)
    s2 = tcri.joint_distribution(adata, covariate=cov, n_samples=8, random_state=0)
    s3 = tcri.joint_distribution(adata, covariate=cov, n_samples=8, random_state=1)
    np.testing.assert_allclose(s1.values, s2.values)
    assert not np.allclose(s1.values, s3.values)
    assert list(s1.index.names) == ["clonotype", "sample_id"]
    # draws are valid simplices
    np.testing.assert_allclose(s1.values.sum(axis=1), 1.0, atol=1e-5)


def test_sampling_mean_approaches_tempered_base(trained_model):
    """Many draws average to the Dirichlet mean == the (tempered) base — validates the
    draw is Dirichlet(clamp(local_scale·base))."""
    _, adata = trained_model
    cov = _first_covariate(adata)
    base = tcri.joint_distribution(adata, covariate=cov, use_logits=False, n_samples=0)
    draws = tcri.joint_distribution(adata, covariate=cov, use_logits=False, n_samples=400, random_state=0)
    mean = draws.groupby(level="clonotype", sort=False).mean().reindex(base.index)
    np.testing.assert_allclose(mean.values, base.values, atol=0.05)


def test_weighted_scales_rows_by_ct_cell_count(trained_model):
    """weighted=True scales each clone row by its (ct-keyed) cell count; weighted=False
    is a per-clone simplex."""
    _, adata = trained_model
    cov = _first_covariate(adata)
    w0 = tcri.joint_distribution(adata, covariate=cov, use_logits=False, n_samples=0, weighted=False)
    w1 = tcri.joint_distribution(adata, covariate=cov, use_logits=False, n_samples=0, weighted=True)
    np.testing.assert_allclose(w0.values.sum(axis=1), 1.0, atol=1e-6)          # unweighted rows sum to 1
    assert not np.allclose(w0.values, w1.values)
    # weighted row sum == that clone's cell count at this covariate
    cov_i = list(adata.uns[K.COVARIATE_CATEGORIES]).index(cov)
    mask = np.asarray(adata.uns[K.COV_ARRAY]) == cov_i
    clone_col = adata.uns[K.METADATA]["clone_col"]
    counts = adata.obs[clone_col].values[mask]
    expected = pd.Series(counts).value_counts().reindex(w1.index).astype(float)
    np.testing.assert_allclose(w1.values.sum(axis=1), expected.values, atol=1e-6)


def test_shared_draw_invariant_across_covariates(trained_model):
    """covariate=None draws once and shares it: the slice for a covariate equals the
    per-covariate call at the same random_state (draw-count == n_samples, not ×#covariates)."""
    _, adata = trained_model
    cov = _first_covariate(adata)
    allcov = tcri.joint_distribution(adata, covariate=None, n_samples=6, random_state=7)
    percov = tcri.joint_distribution(adata, covariate=cov, n_samples=6, random_state=7)
    assert "covariate" in allcov.index.names
    sl = allcov.xs(cov, level="covariate")
    np.testing.assert_allclose(sl.values, percov.values, atol=1e-6)
    assert allcov.attrs["params"]["n_draws"] == 6  # one draw block, not ×#covariates


def test_provenance_is_json_serializable(trained_model):
    _, adata = trained_model
    cov = _first_covariate(adata)
    df = tcri.joint_distribution(adata, covariate=cov, n_samples=4, random_state=0)
    json.dumps(df.attrs["params"])  # must not raise
    assert df.attrs["params"]["use_logits"] is True
    assert df.attrs["params"]["n_draws"] == 4


def test_groupby_deferred(trained_model):
    _, adata = trained_model
    cov = _first_covariate(adata)
    import pytest
    with pytest.raises(NotImplementedError, match="groupby"):
        tcri.joint_distribution(adata, covariate=cov, groupby="patient")
