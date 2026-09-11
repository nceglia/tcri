"""Does ``perturb.gene_importance`` rank the genes that actually carry phenotype?

``tests/test_perturbation.py`` pins structure: identities that hold for any fitted model. This
file asks the accuracy question, which needs an oracle. :func:`tcri.datasets.simulate_tcri`
generates expression as ``x_i ~ Poisson(U_i @ V)`` with ``U_i ~ Gamma(alpha[phi_i],
1/beta[phi_i])``, so a gene's expected expression given phenotype is known in closed form,

    m_{phi,j} = sum_f (alpha_{phi,f} / beta_{phi,f}) V_{f,j},

and its true discriminativeness is the spread of ``m_{.,j}`` across phenotypes.

What was measured before these assertions were written (2026-09-10; 60 genes, 8 factors,
1500 cells, 4 phenotypes, one-layer networks of width 32, 60 epochs, seeds 11-14):

* the fitted head calls the phenotype at 0.86-0.91 accuracy;
* the rank correlation between importance and the closed-form discriminativeness is weak
  and seed-dependent (rho -0.11 to 0.32), and does not improve with 150 or 300 epochs;
* against a naive-Bayes reader of the true programs given the same knockout it is 0.28-0.54;
* with 20 phenotype-independent noise genes appended at matched count scale, the importance
  ranks informative genes above noise only moderately (AUROC 0.61-0.75; 0.52-0.73 at 300-600
  epochs or width 128), and within the noise genes it tracks the gene's mean count
  (rho 0.26-0.63).

So on this fixture the importance is only partly about information: silencing a gene is an
intervention whose size scales with the gene's counts, and the fitted encoder responds to
that regardless of what the gene says about phenotype. Gene-level truth is also not
identifiable here -- eight dense factors drive sixty genes, so any reader may lean on any
subset. The assertions below are the floors those numbers clear with margin: informative
genes outrank noise on average over seeds, and an expression matrix that carries no
phenotype gives a flat importance. They are not a claim that the ranking recovers the
programs gene for gene; the measured values above are the record of how far it does.
"""
from __future__ import annotations

import contextlib
import io
import itertools

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import tcri
from tcri.datasets import simulate_tcri

_SIM = dict(n_clones=20, n_phenotypes=4, n_genes=60, n_cells=1500, n_factors=8,
            omega_concentration=0.3)


def _true_discriminativeness(adata):
    """Relative spread across phenotypes of each gene's expected expression."""
    t = adata.uns["tcri_truth"]
    alpha, beta, V = t["gamma_params"]["alpha"], t["gamma_params"]["beta"], t["V"]
    m = (alpha / beta) @ V                                   # [P, G]
    return (m.max(axis=0) - m.min(axis=0)) / m.mean(axis=0)  # [G]


def _with_noise_genes(adata, n_noise, seed):
    """Append ``n_noise`` Poisson genes whose rate is drawn from the real genes' means and
    does not depend on phenotype: the one gene set whose importance has an unambiguous
    truth, whatever the redundancy among the real genes."""
    rng = np.random.default_rng(seed)
    X = np.asarray(adata.X, dtype=np.float32)
    lam = rng.choice(X.mean(axis=0), size=n_noise, replace=True)
    noise = rng.poisson(lam[None, :], size=(adata.n_obs, n_noise)).astype(np.float32)
    var = pd.DataFrame(index=list(adata.var_names) + [f"noise_{k}" for k in range(n_noise)])
    out = ad.AnnData(X=np.concatenate([X, noise], axis=1), obs=adata.obs.copy(), var=var,
                     uns=dict(adata.uns))
    out.layers["counts"] = out.X.copy()
    return out, np.r_[np.ones(adata.n_vars, bool), np.zeros(n_noise, bool)]


# ── the oracle itself ────────────────────────────────────────────────────────

def test_truth_carries_the_expression_programs():
    a = simulate_tcri(n_genes=30, n_factors=5, n_phenotypes=3, n_cells=200, seed=0)
    t = a.uns["tcri_truth"]
    assert t["V"].shape == (5, 30)
    assert t["gamma_params"]["alpha"].shape == (3, 5) == t["gamma_params"]["beta"].shape
    d = _true_discriminativeness(a)
    assert d.shape == (30,) and np.isfinite(d).all() and (d >= 0).all()


def test_fuzziness_collapses_the_true_discriminativeness():
    """At ``fuzziness=1`` the programs are identical, so no gene can tell phenotypes apart;
    the oracle must say so."""
    sharp = _true_discriminativeness(simulate_tcri(fuzziness=0.0, n_cells=200, seed=1))
    flat = _true_discriminativeness(simulate_tcri(fuzziness=1.0, n_cells=200, seed=1))
    assert sharp.max() > 0.5
    np.testing.assert_allclose(flat, 0.0, atol=1e-9)


def test_the_oracle_is_visible_in_the_realised_counts():
    """Genes the oracle calls discriminative separate the true phenotype labels in the data
    itself -- otherwise no estimator could be expected to find them."""
    a = simulate_tcri(n_genes=60, n_factors=8, n_phenotypes=4, n_cells=3000, seed=2)
    d = _true_discriminativeness(a)
    X = np.asarray(a.X, dtype=float)
    phi = a.obs["true_phenotype"].cat.codes.to_numpy()
    means = np.stack([X[phi == p].mean(axis=0) for p in range(4)])   # [P, G]
    spread = (means.max(axis=0) - means.min(axis=0)) / (means.mean(axis=0) + 1e-9)
    rho = spearmanr(d, spread).statistic
    assert rho > 0.8, f"the oracle does not describe the realised counts: rho={rho:.3f}"


# ── the fitted model ─────────────────────────────────────────────────────────

#: One namespace per FIT, not per seed. These tests fit several models with the same `seed`
#: (it seeds the network init, not the identity of the fit), and two fits sharing a namespace
#: means the second continues the first -- which is what `clear_param_store` used to prevent
#: here, and what made the flat-vs-sharp separation collapse from 25x to 1.8x when the clear
#: was removed without this.
_FIT_COUNTER = itertools.count()


def _fit(adata, *, seed=0, max_epochs=60):
    from tcri.model._model import TCRIModel

    # NAMED (0.12), and no longer clearing the store: these fits used to wipe the session
    # fixtures' parameters, which is why this module carried an autouse save/restore fixture.
    # A namespace per fit makes that unnecessary -- the fits simply do not collide.
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="covariate",
                            batch_key="batch")
    model = TCRIModel(adata, n_latent=16, n_hidden=32, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=32, K=4, seed=seed,
                      name=f"recovery{next(_FIT_COUNTER)}")
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=max_epochs, batch_size=256, accelerator="cpu",
                    enable_progress_bar=False, enable_model_summary=False)
        model.to_anndata(adata)
    return model


@pytest.mark.slow
def test_informative_genes_outrank_matched_noise_genes_on_average():
    """Over three seeds, the mean AUROC of importance for informative-vs-noise genes is above
    chance, and the ten most important genes are mostly informative.

    Measured 2026-09-10 (see the module docstring): AUROC 0.747 / 0.704 / 0.656 for seeds
    11 / 12 / 13 (mean 0.70), 0 / 1 / 0 noise genes in the top ten. The floors are 0.55 and 3.
    """
    aurocs, noise_in_top = [], []
    for seed in (11, 12, 13):
        a = simulate_tcri(fuzziness=0.0, seed=seed, **_SIM)
        b, is_info = _with_noise_genes(a, 20, seed)
        model = _fit(b)
        imp = tcri.perturb.gene_importance(model, b, use_gate=False,
                                           inplace=False)["result"]["value"].to_numpy()
        aurocs.append(roc_auc_score(is_info, imp))
        noise_in_top.append(int((~is_info[np.argsort(-imp)[:10]]).sum()))
    report = f"AUROC per seed {np.round(aurocs, 3).tolist()}, noise genes in top ten {noise_in_top}"
    assert float(np.mean(aurocs)) > 0.55, f"importance does not favour informative genes -- {report}"
    assert max(noise_in_top) <= 3, f"the top of the ranking is noise -- {report}"


@pytest.mark.slow
def test_importance_is_flat_when_expression_carries_no_phenotype():
    """At ``fuzziness=1`` no gene can inform the phenotype, so the head-alone importance
    must be small; at ``fuzziness=0`` on the same seed the largest importance must be
    substantially larger. Difficulty moved, not the truth."""
    flat = simulate_tcri(fuzziness=1.0, seed=12, **_SIM)
    sharp = simulate_tcri(fuzziness=0.0, seed=12, **_SIM)
    i_flat = tcri.perturb.gene_importance(_fit(flat), flat, use_gate=False,
                                          inplace=False)["result"]["value"].to_numpy()
    i_sharp = tcri.perturb.gene_importance(_fit(sharp), sharp, use_gate=False,
                                           inplace=False)["result"]["value"].to_numpy()
    assert i_sharp.max() > 3 * i_flat.max(), (
        f"no separation between informative and uninformative expression: "
        f"max importance sharp={i_sharp.max():.4f} flat={i_flat.max():.4f}"
    )
