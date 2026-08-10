"""tcri.diag smoke — each PPC returns a DataFrame; the two plots return axes."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import tcri


def test_joint_distribution_ppc(trained_model):
    _, adata = trained_model
    df = tcri.diag.joint_distribution_ppc(adata, distance_metric="l1")
    assert isinstance(df, pd.DataFrame)
    if len(df):
        assert {"covariate", "clonotype", "distance"}.issubset(df.columns)


def test_phenotype_calibration(trained_model):
    _, adata = trained_model
    df = tcri.diag.phenotype_calibration(adata, n_bins=5)
    assert {"bin", "mean_pred", "emp_freq", "count"}.issubset(df.columns)
    assert "ECE" in df.attrs and df.attrs["ECE"] >= 0


def test_reconstruction_ppc(trained_model):
    model, adata = trained_model
    df = tcri.diag.reconstruction_ppc(model, adata, n_sims=1, random_state=0)
    assert {"statistic", "observed", "simulated", "discrepancy"}.issubset(df.columns)
    assert (df["discrepancy"] >= 0).all()


def test_permutation_null(trained_model):
    _, adata = trained_model
    df = tcri.diag.permutation_null(adata, n_perm=30, random_state=0)
    assert {"covariate", "observed", "null_mean", "null_sd", "z", "p"}.issubset(df.columns)
    assert ((df["p"] >= 0) & (df["p"] <= 1)).all()


def test_loss_and_archetypes(trained_model):
    model, _ = trained_model
    assert tcri.diag.loss(model) is not None
    assert tcri.diag.archetypes(model) is not None


def test_reconstruction_ppc_n_sims_is_wired(trained_model):
    """NEW-3: ``n_sims`` is declared in the frozen contract and was never read.

    The body drew exactly one replicate per cell regardless, so ``n_sims=1`` and
    ``n_sims=1000`` returned bit-identical frames in identical wall-clock — a user tightening
    the check got no more precision and no warning that the knob was inert.

    Averaging more posterior-predictive draws must move the simulated statistics. This asserts
    the knob changes the OUTPUT, not that the value arrives somewhere: an argument that is
    merely connected is the failure mode this repo keeps rediscovering.
    """
    model, adata = trained_model

    one = tcri.diag.reconstruction_ppc(model, adata, n_sims=1, random_state=0)
    many = tcri.diag.reconstruction_ppc(model, adata, n_sims=8, random_state=0)

    assert list(one["statistic"]) == list(many["statistic"])
    # observed data does not depend on the number of draws
    assert np.allclose(one["observed"].to_numpy(), many["observed"].to_numpy())
    assert not np.allclose(one["simulated"].to_numpy(), many["simulated"].to_numpy()), (
        "n_sims does not change the simulated statistics, so it is still inert"
    )

    with pytest.raises(ValueError, match="n_sims must be >= 1"):
        tcri.diag.reconstruction_ppc(model, adata, n_sims=0)


def _perfectly_coupled_adata(n_clones=6, per_clone=30):
    """Clone k is ALWAYS phenotype k. No shuffle can reach the observed MI, so the unfloored
    estimator returns exactly 0.0 — which is what makes this fixture able to see the floor.
    Built by hand rather than fitted: permutation_null is model-free and reads only obs + uns.
    """
    import anndata as ad
    import pandas as pd

    from tcri import _keys as K

    clones = np.repeat([f"clone_{i}" for i in range(n_clones)], per_clone)
    phenos = np.repeat([f"phen_{i}" for i in range(n_clones)], per_clone)
    n = len(clones)
    obs = pd.DataFrame({"clone_id": clones, "phenotype": phenos, "covariate": ["cov_0"] * n},
                       index=[f"cell_{i}" for i in range(n)])
    adata = ad.AnnData(X=np.zeros((n, 2), dtype="float32"), obs=obs)
    adata.uns[K.METADATA] = {"clone_col": "clone_id", "phenotype_col": "phenotype",
                             "covariate_col": "covariate", "batch_col": "covariate"}
    adata.uns[K.PHENOTYPE_CATEGORIES] = [f"phen_{i}" for i in range(n_clones)]
    adata.uns[K.COVARIATE_CATEGORIES] = ["cov_0"]
    return adata


@pytest.mark.parametrize("R", [50, 200])
def test_permutation_null_p_value_is_floored(R):
    """DE-15: a permutation p-value estimated from R shuffles can never be 0.

    The observed statistic is itself one realisation under the null, so the estimator is
    (1 + #{null >= obs}) / (1 + R), not the raw fraction (Phipson & Smyth 2010). The unfloored
    version returned exactly 0.0 whenever no shuffle beat the observation — claiming infinite
    evidence from a finite sample, and giving -inf to anyone who logged it.

    Uses a perfectly coupled fixture on purpose: on ordinary data some shuffle usually ties the
    observation, the raw fraction is already nonzero, and the floor is invisible.
    """
    adata = _perfectly_coupled_adata()
    df = tcri.diag.permutation_null(adata, n_perm=R, random_state=0)

    assert (df["p"] > 0).all(), f"p of exactly 0 from {R} permutations: {df['p'].tolist()}"
    assert float(df["p"].iloc[0]) == pytest.approx(1.0 / (R + 1)), (
        f"p={float(df['p'].iloc[0]):.6f} on a perfectly coupled fixture where no shuffle can "
        f"beat the observation; the floored estimator must give exactly 1/(R+1) = "
        f"{1/(R+1):.6f}"
    )


def test_permutation_null_honours_normalize_mode(trained_model):
    """DE-15: the null must be on the same scale as the statistic it is a null for.

    ``mode="min"`` was hardcoded, so a caller working in ``"average"`` compared their number
    against a null computed on a different normalizer. That is not a weaker null — it is not a
    null for their statistic at all.
    """
    model, adata = trained_model
    lo = tcri.diag.permutation_null(adata, n_perm=50, normalize_mode="min", random_state=0)
    hi = tcri.diag.permutation_null(adata, n_perm=50, normalize_mode="average", random_state=0)

    assert not np.allclose(lo["observed"].to_numpy(), hi["observed"].to_numpy()), (
        "normalize_mode does not change the statistic, so it is still hardcoded"
    )
    with pytest.raises(ValueError, match="normalize_mode must be"):
        tcri.diag.permutation_null(adata, n_perm=10, normalize_mode="nonsense")


def test_permutation_null_groupby_matches_the_metric_surface(trained_model):
    """DE-15: ``groupby`` was accepted and never read — passing it returned a bit-identical
    frame. It is now implemented rather than removed.

    Every ``tl.*`` metric takes ``groupby``, and this is the null FOR those metrics, so without
    it a per-patient MI had no per-patient null. Cells are restricted to the group and
    phenotypes permuted within each (covariate, group) stratum, so the null conditions on what
    the reported statistic conditions on.
    """
    import inspect

    model, adata = trained_model
    assert "groupby" in inspect.signature(tcri.diag.permutation_null).parameters

    flat = tcri.diag.permutation_null(adata, n_perm=30, random_state=0)
    grouped = tcri.diag.permutation_null(adata, groupby="patient", n_perm=30, random_state=0)

    assert "patient" in grouped.columns, "the group label is not carried into the result"
    n_groups = adata.obs["patient"].nunique()
    assert len(grouped) == len(flat) * n_groups, (
        f"expected one row per (covariate, group): {len(flat)} x {n_groups}, got {len(grouped)}"
    )
    assert not np.allclose(
        grouped["observed"].to_numpy()[: len(flat)], flat["observed"].to_numpy()
    ) or n_groups == 1, "grouping did not change the statistic, so groupby is still inert"

    with pytest.raises(ValueError, match="not a column"):
        tcri.diag.permutation_null(adata, groupby="no_such_column", n_perm=5)
