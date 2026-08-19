"""The Phase-5 engine gate — ``tcri.joint_distribution`` identities (§7.1).

Comparisons use the *frozen* canonical keys written by ``to_anndata`` (``uns[P_CT]``,
``obsm[X_LOGITS]``, ``obsm[X_PROBABILITIES]``) rather than a live ``predict()`` call,
so the identities do not depend on the process-global pyro param store (§5.2).
"""
import json

import numpy as np
import pandas as pd

import tcri
from tcri._state import keys as K


def _first_covariate(adata):
    return list(adata.uns[K.COVARIATE_CATEGORIES])[0]


def test_top_level_reexport():
    assert hasattr(tcri, "joint_distribution")
    assert tcri.joint_distribution is tcri.tools.joint_distribution


def test_identity_ct_table_equals_p_ct(trained_model):
    """use_logits=False, n_samples=0, T=1  ==  uns[P_CT] restricted to the covariate."""
    _, adata = trained_model
    cov = _first_covariate(adata)
    df = tcri.joint_distribution(adata, covariate=cov, use_logits=False, n_samples=0, temperature=1.0)['table']

    cov_i = list(adata.uns[K.COVARIATE_CATEGORIES]).index(cov)
    rows = np.where(np.asarray(adata.uns[K.CT_TO_COV]) == cov_i)[0]
    expected = np.asarray(adata.uns[K.P_CT])[rows]  # one ct row per clone
    np.testing.assert_allclose(df.values, expected, atol=1e-6)


def test_identity_use_logits_equals_predict_aggregation(trained_model):
    """use_logits=True, n_samples=0, T=1  ==  per-clone mean of predict() (== frozen X_PROBABILITIES)."""
    _, adata = trained_model
    cov = _first_covariate(adata)
    df = tcri.joint_distribution(adata, covariate=cov, use_logits=True, n_samples=0, temperature=1.0)['table']

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
    a = tcri.joint_distribution(adata, covariate=cov, n_samples=0)['table']
    b = tcri.joint_distribution(adata, covariate=cov, n_samples=0)['table']
    assert np.array_equal(a.values, b.values)


def test_sampling_is_seeded_reproducible(trained_model):
    """n_samples>0 is reproducible under a fixed random_state, varies otherwise, and
    carries a (clonotype, sample_id) MultiIndex."""
    _, adata = trained_model
    cov = _first_covariate(adata)
    s1 = tcri.joint_distribution(adata, covariate=cov, n_samples=8, random_state=0)['table']
    s2 = tcri.joint_distribution(adata, covariate=cov, n_samples=8, random_state=0)['table']
    s3 = tcri.joint_distribution(adata, covariate=cov, n_samples=8, random_state=1)['table']
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
    base = tcri.joint_distribution(adata, covariate=cov, use_logits=False, n_samples=0)['table']
    draws = tcri.joint_distribution(adata, covariate=cov, use_logits=False, n_samples=400, random_state=0)['table']
    mean = draws.groupby(level="clonotype", sort=False).mean().reindex(base.index)
    np.testing.assert_allclose(mean.values, base.values, atol=0.05)


def test_weighted_scales_rows_by_ct_cell_count(trained_model):
    """weighted=True scales each clone row by its (ct-keyed) cell count; weighted=False
    is a per-clone simplex."""
    _, adata = trained_model
    cov = _first_covariate(adata)
    w0 = tcri.joint_distribution(adata, covariate=cov, use_logits=False, n_samples=0, weighted=False)['table']
    w1 = tcri.joint_distribution(adata, covariate=cov, use_logits=False, n_samples=0, weighted=True)['table']
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
    allcov = tcri.joint_distribution(adata, covariate=None, n_samples=6, random_state=7)['table']
    percov = tcri.joint_distribution(adata, covariate=cov, n_samples=6, random_state=7)['table']
    assert "covariate" in allcov.index.names
    sl = allcov.xs(cov, level="covariate")
    np.testing.assert_allclose(sl.values, percov.values, atol=1e-6)
    assert tcri.get.params(adata, 'joint_distribution')["n_draws"] == 6  # one draw block, not ×#covariates


def test_provenance_is_json_serializable(trained_model):
    _, adata = trained_model
    cov = _first_covariate(adata)
    df = tcri.joint_distribution(adata, covariate=cov, n_samples=4, random_state=0)['table']
    json.dumps(tcri.get.params(adata, 'joint_distribution'))  # must not raise
    assert tcri.get.params(adata, 'joint_distribution')["use_logits"] is True
    assert tcri.get.params(adata, 'joint_distribution')["n_draws"] == 4


def test_groupby_is_not_a_joint_distribution_argument(trained_model):
    """``groupby`` was declared in the first contract freeze (7599959) and never implemented --
    it raised ``NotImplementedError``. Removed rather than implemented: a joint IS the
    covariate object, and grouping belongs to the metrics that reduce it."""
    import inspect

    import pytest

    _, adata = trained_model
    cov = _first_covariate(adata)
    assert "groupby" not in inspect.signature(tcri.joint_distribution).parameters
    with pytest.raises(TypeError, match="groupby"):
        tcri.joint_distribution(adata, covariate=cov, groupby="patient")


def test_gate_aware_combine_direct():
    """use_logits=True with a numeric gate_prob uses g·logits + (1-g)·log(base), then
    mean per clone — exercised directly (the trained_model fixture has gate_prob=None)."""
    from tcri._compute._joint import _joint_draws

    p_ct = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=float)  # 2 ct rows == 2 clones
    ct_to_cov = np.array([0, 0]); ct_to_c = np.array([0, 1])
    ct_array = np.array([0, 0, 1, 1]); cov_array = np.array([0, 0, 0, 0])
    logits = np.array([[2.0, -1.0], [1.0, 0.0], [-1.0, 2.0], [0.0, 1.0]], dtype=float)
    g = 0.5
    blocks, _ = _joint_draws(
        p_ct, ct_to_cov, ct_to_c, ct_array, cov_array, local_scale=1.0, n_samples=0,
        temperature=1.0, use_logits=True, covariate_idx=0, logits=logits, gate_prob=g,
    )
    _, _, J = blocks[0]  # [1, 2, 2]

    eps = 1e-8
    def _sm(x):
        e = np.exp(x - x.max()); return e / e.sum()
    exp = np.zeros((2, 2))
    for c in range(2):
        cells = [i for i in range(4) if ct_to_c[ct_array[i]] == c]
        exp[c] = np.mean([_sm(g * logits[i] + (1 - g) * np.log(p_ct[ct_array[i]] + eps)) for i in cells], 0)
    np.testing.assert_allclose(J[0], exp, atol=1e-10)


def test_temperature_tempers_base_on_ct_path(trained_model):
    """use_logits=False, T!=1 == softmax(log(P_CT+1e-8)/T) restricted to the covariate."""
    from scipy.special import softmax
    _, adata = trained_model
    cov = _first_covariate(adata)
    df = tcri.joint_distribution(adata, covariate=cov, use_logits=False, n_samples=0, temperature=2.0)['table']
    cov_i = list(adata.uns[K.COVARIATE_CATEGORIES]).index(cov)
    rows = np.where(np.asarray(adata.uns[K.CT_TO_COV]) == cov_i)[0]
    expected = softmax(np.log(np.asarray(adata.uns[K.P_CT])[rows] + 1e-8) / 2.0, axis=1)
    np.testing.assert_allclose(df.values, expected, atol=1e-5)


def test_temperature_changes_use_logits_joint(trained_model):
    """T!=1 tempers the cell-informed joint too (behavior lock, per §7.1)."""
    _, adata = trained_model
    cov = _first_covariate(adata)
    a = tcri.joint_distribution(adata, covariate=cov, use_logits=True, n_samples=0, temperature=1.0)['table']
    b = tcri.joint_distribution(adata, covariate=cov, use_logits=True, n_samples=0, temperature=2.0)['table']
    assert not np.allclose(a.values, b.values)


def test_subset_anndata_raises(trained_model):
    """A filtered/sliced AnnData (full-space uns vs subset obsm) is refused, not silently
    misaligned."""
    import pytest
    _, adata = trained_model
    sub = adata[: adata.n_obs // 2].copy()  # uns per-cell arrays stay full-space
    with pytest.raises(ValueError, match="n_obs|filtered|subset"):
        tcri.joint_distribution(sub, covariate=_first_covariate(adata))


def test_missing_local_scale_raises_only_when_sampling(trained_model):
    """n_samples>0 without uns[LOCAL_SCALE] errors (no silent 1.0 fallback); n_samples=0 is fine."""
    import pytest
    _, adata = trained_model
    a2 = adata.copy()
    del a2.uns[K.LOCAL_SCALE]
    with pytest.raises(RuntimeError, match="LOCAL_SCALE|local_scale"):
        tcri.joint_distribution(a2, covariate=_first_covariate(adata), n_samples=4)
    tcri.joint_distribution(a2, covariate=_first_covariate(adata), n_samples=0)  # ok
