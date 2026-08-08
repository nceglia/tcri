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
