"""simulate_from_fit_params must run end-to-end (regression for #179)."""

import numpy as np
import pytest

from tcri.datasets import simulate_from_fit_params


def _minimal_fit_params(*, n_clones=3, n_phenotypes=2, n_factors=2, n_genes=4):
    """Build a tiny params dict matching the shape fit_params.pkl stores."""
    rng = np.random.default_rng(0)
    pi = np.full(n_clones, 1.0 / n_clones)
    omega = np.full((n_clones, n_phenotypes), 1.0 / n_phenotypes)
    gamma_params = {
        f"phen_{p}": {
            "alpha": rng.uniform(1.5, 3.0, size=n_factors),
            "beta": rng.uniform(1.0, 2.0, size=n_factors),
        }
        for p in range(n_phenotypes)
    }
    V = rng.gamma(2.0, 1.0, size=(n_factors, n_genes))
    return {
        "pi": pi,
        "omega": omega,
        "gamma_params": gamma_params,
        "V": V,
        "L": n_factors,
    }


def test_simulate_from_fit_params_runs_without_nameerror():
    """Public API must not raise NameError on n_phenotypes (#179).

    simulate_from_fit_params binds the phenotype count as P from omega.shape
    but previously built phen_levels with the undefined name n_phenotypes.
    """
    params = _minimal_fit_params()
    adata = simulate_from_fit_params(params, n_cells=20, seed=0)
    assert adata.n_obs == 20
    assert list(adata.obs["phenotype"].cat.categories) == ["phen_0", "phen_1"]
    assert adata.uns["tcri_truth"]["settings"]["n_phenotypes"] == 2
