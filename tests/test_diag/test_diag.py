"""tcri.diag smoke — each PPC returns a DataFrame; the two plots return axes."""
import matplotlib

matplotlib.use("Agg")

import pandas as pd

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
