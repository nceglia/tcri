"""Guardrails on TCRIModel construction/training defaults.

These pin fixes for footguns found in the performance audit: a hard crash on small
datasets, silent cross-model contamination via Pyro's process-global param store,
a pathological batch size, and Trainer knobs the caller could not override.
"""
import contextlib
import io

import numpy as np
import pandas as pd
import pyro
import pytest
from anndata import AnnData

from tcri.model._model import TCRIModel


@pytest.fixture(autouse=True)
def _isolate_param_store():
    """Own the process-global Pyro param store for this module.

    These tests train small models, leaving shape-specific params (``q_p_c_raw``
    etc.) behind; without clearing on teardown the next model test in the suite
    reuses a stale-shaped param and fails. Module-local (not a conftest autouse) so
    it never wipes the session-scoped ``trained_model`` fixture mid-suite.
    """
    pyro.clear_param_store()
    yield
    pyro.clear_param_store()


@pytest.fixture
def tiny_adata():
    """5 clonotypes — fewer than the default K=10."""
    n_clones, n_per, n_genes = 5, 40, 5
    rows, clone, phen, cov = [], [], [], []
    for c in range(n_clones):
        for j in range(2):
            for _ in range(n_per):
                v = np.zeros(n_genes, dtype="float32")
                v[c % n_genes] = 100.0
                rows.append(v)
                clone.append(f"clone_{c}")
                phen.append(f"phen_{c}")
                cov.append(f"cond_{j}")
    ad = AnnData(
        X=np.asarray(rows),
        obs=pd.DataFrame({"clone_id": clone, "true_phenotype": phen,
                          "covariate": cov, "patient": "P1"}),
        var=pd.DataFrame(index=[f"g{g}" for g in range(n_genes)]),
    )
    ad.layers["counts"] = ad.X.copy()
    TCRIModel.setup_anndata(
        ad, layer="counts", clonotype_key="clone_id", phenotype_key="true_phenotype",
        covariate_key="covariate", batch_key="patient",
    )
    return ad


def _model(adata, **kw):
    kw.setdefault("n_latent", 8)
    kw.setdefault("n_hidden", 16)
    kw.setdefault("n_layers", 1)
    kw.setdefault("classifier_n_layers", 1)
    kw.setdefault("classifier_hidden", 16)
    return TCRIModel(adata, **kw)


def test_K_clamped_to_n_clonotypes(tiny_adata):
    """K > n_clonotypes used to raise from sklearn KMeans; it now clamps + warns."""
    pyro.clear_param_store()
    with pytest.warns(UserWarning, match="archetype"):
        model = _model(tiny_adata)  # K=10 default, only 5 clones
    assert model.centers.shape[0] == 5


def test_second_model_warns_about_shared_param_store(tiny_adata):
    """Pyro's param store is process-global — a 2nd model silently continues the 1st fit."""
    pyro.clear_param_store()
    m1 = _model(tiny_adata, K=5)
    with contextlib.redirect_stdout(io.StringIO()):
        m1.train(max_epochs=5, batch_size=256,
                 enable_progress_bar=False, enable_model_summary=False)
    with pytest.warns(UserWarning, match="param store"):
        _model(tiny_adata, K=5)


def test_batch_size_at_or_above_n_obs_warns(tiny_adata):
    """batch_size >= n_obs => 1 optimizer step/epoch (the 9-hour-notebook pathology)."""
    pyro.clear_param_store()
    model = _model(tiny_adata, K=5)
    with pytest.warns(UserWarning, match="SINGLE"):
        with contextlib.redirect_stdout(io.StringIO()):
            model.train(max_epochs=2, batch_size=10_000,
                        enable_progress_bar=False, enable_model_summary=False)


def test_lr_and_weight_decay_reach_pyros_optimizer(tiny_adata):
    """The optimizer settings must configure SVI, not a side optimizer.

    ``UnifiedTrainingPlan`` used to override ``configure_optimizers`` with a real
    torch Adam over every module parameter. That replaced scvi's deliberate no-op
    shim and ran *after* ``SVI.step()`` had already zeroed the gradients, so it
    only ever applied a scale-free ~lr*sign(p) shrink — and ``lr`` never reached
    the optimizer that actually descends the ELBO (Pyro stayed at scvi's 1e-3).
    """
    from tcri.model._training import UnifiedTrainingPlan

    model = _model(tiny_adata, K=5)
    plan = UnifiedTrainingPlan(
        module=model.module, n_steps_kl_warmup=10, reconstruction_loss_scale=1e-3,
        optimizer_config={"lr": 0.05, "betas": (0.9, 0.999), "eps": 1e-5,
                          "weight_decay": 1e-4},
    )
    args = plan.optim.pt_optim_args
    assert args["lr"] == 0.05, f"lr did not reach Pyro's SVI optimizer: {args}"
    assert args["weight_decay"] == 1e-4, f"weight_decay did not reach Pyro: {args}"

    # and the Lightning-facing optimizer must be scvi's dummy shim, not the module
    opt = plan.configure_optimizers()
    opt = opt["optimizer"] if isinstance(opt, dict) else opt
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_module = sum(p.numel() for p in model.module.parameters())
    assert n_opt == 1 < n_module, (
        f"configure_optimizers covers {n_opt} params (module has {n_module}); it must "
        "stay scvi's single-dummy-param shim so nothing steps on zeroed gradients."
    )


def test_trainer_knobs_are_overridable(tiny_adata):
    """These were hard-coded keywords: passing them raised 'got multiple values'."""
    pyro.clear_param_store()
    model = _model(tiny_adata, K=5)
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(  # would previously raise TypeError
            max_epochs=6, batch_size=256,
            early_stopping_patience=1, check_val_every_n_epoch=1,
            enable_progress_bar=False, enable_model_summary=False,
        )
    assert model.trainer.current_epoch <= 6
