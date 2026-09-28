"""``simulate_from_fit_params`` — synthetic data whose structure comes from a real fit.

The generating ``(pi, omega)`` are given rather than drawn from a symmetric Dirichlet, so the
clone-phenotype coupling of a real cohort can be reproduced without the cohort. These tests call
the function end to end: its label spaces are built from the shapes of the parameters it was
given, and a shape the caller supplies has to survive into ``obs``.
"""
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from tcri.datasets import simulate_from_fit_params


def _params(n_clones=5, P=3, L=4, D=12, seed=0):
    """A minimal fit-parameter dict: the five keys the function reads."""
    rng = np.random.default_rng(seed)
    pi = rng.dirichlet(np.ones(n_clones))
    omega = rng.dirichlet(np.ones(P) * 0.5, size=n_clones)
    gamma_params = {f"phen_{p}": {"alpha": rng.uniform(1.5, 3.0, size=L),
                                  "beta": rng.uniform(1.0, 2.0, size=L)} for p in range(P)}
    return {"pi": pi, "omega": omega, "gamma_params": gamma_params,
            "V": rng.uniform(0.1, 1.0, size=(L, D)), "L": L}


def test_it_returns_an_anndata_shaped_by_the_parameters():
    params = _params()
    adata = simulate_from_fit_params(params, n_cells=150, seed=1)

    assert isinstance(adata, AnnData)
    assert adata.shape == (150, params["V"].shape[1])
    assert "counts" in adata.layers
    np.testing.assert_array_equal(adata.layers["counts"], adata.X)


def test_the_phenotype_label_space_comes_from_omega():
    """The number of phenotypes is ``omega.shape[1]``; nothing else declares it."""
    adata = simulate_from_fit_params(_params(P=6), n_cells=120, seed=2)

    assert list(adata.obs["phenotype"].cat.categories) == [f"phen_{p}" for p in range(6)]
    assert list(adata.obs["true_phenotype"].cat.categories) == [f"phen_{p}" for p in range(6)]
    assert not adata.obs["phenotype"].isna().any()


def test_clone_names_from_the_fit_survive_into_obs():
    params = _params(n_clones=4)
    params["clone_levels"] = ["CASSIRSSYEQYF", "CASSLGQAYF", "CASSPGTGDSNQPQHF", "CASSFSTCSANYGYTF"]

    adata = simulate_from_fit_params(params, n_cells=100, seed=3)

    assert list(adata.obs["clone_id"].cat.categories) == params["clone_levels"]
    assert set(adata.obs["clone_id"]) <= set(params["clone_levels"])


def test_the_truth_block_records_the_generating_structure():
    adata = simulate_from_fit_params(_params(), n_cells=200, seed=4)
    truth = adata.uns["tcri_truth"]

    for key in ("true_mi", "true_nmi_min", "true_nmi_average", "empirical_mi"):
        assert np.isfinite(truth[key])
    assert truth["settings"]["source"] == "empirical fit"
    assert truth["settings"]["n_cells"] == 200


def test_the_seed_fixes_the_draw():
    params = _params()
    a = simulate_from_fit_params(params, n_cells=80, seed=7)
    b = simulate_from_fit_params(params, n_cells=80, seed=7)
    c = simulate_from_fit_params(params, n_cells=80, seed=8)

    np.testing.assert_array_equal(a.X, b.X)
    assert not np.array_equal(a.X, c.X)


def test_missing_parameters_are_reported_by_name():
    params = _params()
    del params["omega"]

    with pytest.raises(KeyError, match="omega"):
        simulate_from_fit_params(params, n_cells=10)
