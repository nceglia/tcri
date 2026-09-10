"""The fitted hierarchy depends on which cells sit in which clone x covariate group.

Until 2026-09 the hierarchy had no per-cell data term (the surrogate's target is detached
and nothing else in the data plate touches q(p_ct)), so the fitted p_ct was BITWISE invariant
to permuting the covariate labels within each clone (same per-clone label counts, so the same
initialisation; no per-cell path). The noisy-label readout gives it one. This is the
acceptance test: the same permutation now moves the fit, and with the readout off it must not.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pyro
import pytest

warnings.filterwarnings("ignore")


def _cohort(seed=0):
    from tcri.datasets import simulate_cohort

    return simulate_cohort(
        n_patients=4, conditions=("pre", "post"), disease_fraction=0.5, n_clones=(14, 24),
        n_phenotypes=4, n_genes=40, n_cells_per_sample=200,
        clone_size_distribution="powerlaw", clone_size_exponent=2.0,
        disease_enrichment=12.0, control_enrichment=1.1, seed=seed,
    )


def _shuffle_covariate_within_clone(adata, *, clone_key, cov_key, seed):
    """Permute covariate labels among the cells of each clone. Per-clone label counts and
    per-(clone, covariate) group sizes are preserved, so the initialisation is identical;
    only which cells fall in which group changes."""
    rng = np.random.default_rng(seed)
    cov = adata.obs[cov_key].astype(str).to_numpy().copy()
    for _, idx in adata.obs.groupby(clone_key, observed=True).indices.items():
        if len(idx) > 1:
            cov[idx] = cov[rng.permutation(idx)]
    out = adata.copy()
    out.obs[cov_key] = pd.Categorical(cov, categories=adata.obs[cov_key].astype("category").cat.categories)
    return out


def _fit_p_ct(adata, **model_kwargs):
    from tcri.model._model import TCRIModel

    pyro.clear_param_store()
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="condition",
                            batch_key="patient")
    m = TCRIModel(adata, n_latent=8, n_hidden=32, n_layers=1, classifier_n_layers=1,
                  classifier_hidden=32, K=4, seed=0, **model_kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        m.train(max_epochs=80, batch_size=256, accelerator="cpu",
                enable_progress_bar=False, enable_model_summary=False)
    p = m.get_p_ct()
    # rows are (clone, covariate) groups in the module's order; both fits share the same
    # group set and order because the shuffle preserves every (clone, covariate) pair
    keys = list(zip(m.module.ct_to_c.tolist(), m.module.ct_to_cov.tolist()))
    pyro.clear_param_store()
    return p, keys


def test_hierarchy_moves_with_the_covariate_shuffle():
    ad = _cohort()
    shuffled = _shuffle_covariate_within_clone(ad, clone_key="clone_id", cov_key="condition", seed=0)
    changed = (ad.obs["condition"].astype(str).to_numpy() != shuffled.obs["condition"].astype(str).to_numpy()).mean()
    assert changed > 0.2, f"the shuffle is nearly a no-op ({changed:.2%} of cells relabelled)"

    p0, k0 = _fit_p_ct(ad)
    p1, k1 = _fit_p_ct(shuffled)
    assert k0 == k1, "the shuffle changed the (clone, covariate) group set; it must not"

    l1 = np.abs(p0 - p1).sum(1)
    assert float(l1.mean()) > 1e-3, (
        f"the fitted p_ct is invariant to which cells carry which covariate label (mean L1 "
        f"{l1.mean():.2e} over {len(l1)} groups): no per-cell information reaches the "
        f"hierarchy -- is the label readout on?"
    )


def test_hierarchy_is_invariant_with_the_readout_off():
    """The switch restores the label-free model exactly: with ``label_error_rate=None`` the
    same shuffle leaves the fitted p_ct bitwise identical (the pre-2026-09 behaviour)."""
    ad = _cohort()
    shuffled = _shuffle_covariate_within_clone(ad, clone_key="clone_id", cov_key="condition", seed=0)
    p0, _ = _fit_p_ct(ad, label_error_rate=None)
    p1, _ = _fit_p_ct(shuffled, label_error_rate=None)
    assert np.array_equal(p0, p1), (
        f"with the readout off the hierarchy still moved with the shuffle (mean L1 "
        f"{np.abs(p0 - p1).sum(1).mean():.2e}); the off switch does not restore the label-free model"
    )
