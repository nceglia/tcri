"""Guardrails on TCRIModel construction and training defaults.

Each test asserts that a configuration which would otherwise produce a quietly bad fit is
corrected with a warning or rejected outright: K above the clonotype count, a second model
sharing Pyro's process-global param store, a batch size that leaves one optimizer step per
epoch, and Trainer keywords the caller must be able to override.
"""
import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pyro
import pytest
import torch
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


def _assert_store_untouched(call, match):
    """``call()`` raises ``ValueError`` matching ``match`` and leaves the Pyro store as it was:
    the same keys, each holding the same tensor object with the same values.

    The store's own (unconstrained) tensors are compared, not values recomputed through their
    constraints, and NaN counts as equal to NaN: a tiny unseeded fit can leave NaN in a guide
    concentration, and that is not a change to the store.
    """
    before = pyro.get_param_store().get_state()["params"]
    values = {k: v.detach().clone() for k, v in before.items()}

    with pytest.raises(ValueError, match=match):
        call()

    after = pyro.get_param_store().get_state()["params"]
    assert set(after) == set(before)
    for key, value in values.items():
        assert after[key] is before[key], f"{key} was replaced"
        torch.testing.assert_close(after[key].detach(), value, rtol=0, atol=0, equal_nan=True,
                                   msg=lambda m, key=key: f"{key} changed: {m}")


def _model(adata, **kw):
    kw.setdefault("n_latent", 8)
    kw.setdefault("n_hidden", 16)
    kw.setdefault("n_layers", 1)
    kw.setdefault("classifier_n_layers", 1)
    kw.setdefault("classifier_hidden", 16)
    return TCRIModel(adata, **kw)


def test_K_clamped_to_n_clonotypes(tiny_adata):
    """K above the clonotype count cannot be met by KMeans, so the constructor clamps and
    warns rather than raising."""
    pyro.clear_param_store()
    with pytest.warns(UserWarning, match="archetype"):
        model = _model(tiny_adata)  # K=10 default, only 5 clones
    assert model.centers.shape[0] == 5


def test_a_second_model_under_a_held_name_raises(tiny_adata):
    """Pyro's param store is process-global, so a 2nd model built under a name the store already
    holds would read the first fit's guide concentrations, and its networks would not train.
    Construction raises and leaves the store as it was. A generated name never collides, so only
    a `name=` passed twice can."""
    pyro.clear_param_store()
    m1 = _model(tiny_adata, K=5, name="shared")
    with contextlib.redirect_stdout(io.StringIO()):
        m1.train(max_epochs=5, batch_size=256,
                 enable_progress_bar=False, enable_model_summary=False)
    _assert_store_untouched(lambda: _model(tiny_adata, K=5, name="shared"), "already holds")


_ONE_EPOCH = dict(max_epochs=1, batch_size=256, accelerator="cpu",
                  enable_progress_bar=False, enable_model_summary=False)


def _ddp_strategy():
    from lightning.pytorch.strategies import DDPStrategy
    return DDPStrategy()


@pytest.mark.parametrize("request_more", [
    {"devices": 2}, {"devices": [0, 1]}, {"devices": -1}, {"devices": "-1"},
    {"devices": "0,1"}, {"num_nodes": 2}, {"strategy": "ddp"}, {"strategy": "ddp_spawn"},
    {"strategy": "fsdp"}, {"strategy": "deepspeed"}, {"strategy": _ddp_strategy},
], ids=lambda r: ",".join(f"{k}={getattr(v, '__name__', v)}" for k, v in r.items()))
def test_more_than_one_device_raises(tiny_adata, request_more):
    """A distributed fit leaves each process with its own guide concentrations and networks,
    neither synchronized, so train() refuses more than one device before anything is fitted."""
    model = _model(tiny_adata, K=5)
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(**_ONE_EPOCH)
    request_more = {k: v() if callable(v) else v for k, v in request_more.items()}
    _assert_store_untouched(lambda: model.train(**{**_ONE_EPOCH, **request_more}),
                            "more than one device")


@pytest.mark.parametrize("one", [{}, {"devices": "auto"}, {"devices": 1},
                                 {"devices": "1"}])
def test_one_device_trains(tiny_adata, one):
    """The default, and an explicit single device, still train."""
    model = _model(tiny_adata, K=5)
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(**{**_ONE_EPOCH, **one})
    assert model.training_record_["epochs_run"] == 1


def test_batch_size_at_or_above_n_obs_warns(tiny_adata):
    """batch_size >= n_obs leaves ONE optimizer step per epoch, so a large max_epochs buys
    almost no optimization; the caller must be told."""
    pyro.clear_param_store()
    model = _model(tiny_adata, K=5)
    with pytest.warns(UserWarning, match="SINGLE"):
        with contextlib.redirect_stdout(io.StringIO()):
            model.train(max_epochs=2, batch_size=10_000,
                        enable_progress_bar=False, enable_model_summary=False)


def test_lr_and_weight_decay_reach_pyros_optimizer(tiny_adata):
    """The optimizer settings must configure SVI, not a side optimizer.

    Pyro's optimizer is the one that descends the ELBO; scvi's Lightning-facing
    ``configure_optimizers`` is a deliberate no-op shim over a single dummy parameter. A real
    torch optimizer there runs *after* ``SVI.step()`` has already zeroed the gradients, so it
    applies only a scale-free shrink to every module parameter while ``lr`` never reaches the
    optimizer doing the work. Both halves are asserted: the per-parameter settings Pyro
    resolves, and the size of what Lightning is handed.
    """
    from tcri.model._training import UnifiedTrainingPlan

    model = _model(tiny_adata, K=5)
    plan = UnifiedTrainingPlan(
        module=model.module, n_epochs_kl_warmup=4, reconstruction_loss_scale=1e-3,
        optimizer_config={"lr": 0.05, "betas": (0.9, 0.999), "eps": 1e-5,
                          "weight_decay": 1e-4},
    )
    # Pyro resolves per-parameter settings through a callable keyed by the normalized
    # param-store name: module parameters arrive as "scvi.<path>", the two guide
    # concentrations as their bare names.
    args = plan.optim.pt_optim_args
    assert callable(args), f"expected a per-parameter optim_args callable, got {args!r}"
    net = args("scvi.encoder.fc_layers.0.weight")
    assert net["lr"] == 0.05, f"lr did not reach Pyro's SVI optimizer: {net}"
    assert net["weight_decay"] == 1e-4, f"weight_decay did not reach Pyro: {net}"
    guide = args("q_p_ct_raw")
    assert guide["lr"] == 0.05 and guide["weight_decay"] == 0.0, (
        f"the guide concentrations must share lr but carry NO weight decay: {guide}"
    )

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
    """A Trainer keyword train() also sets must be forwarded once, so passing it explicitly
    overrides the default instead of colliding with it."""
    pyro.clear_param_store()
    model = _model(tiny_adata, K=5)
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(
            max_epochs=6, batch_size=256,
            early_stopping_patience=1, check_val_every_n_epoch=1,
            enable_progress_bar=False, enable_model_summary=False,
        )
    assert model.trainer.current_epoch <= 6


def _fitted_for_binding():
    from tcri.datasets import simulate_tcri

    adata = simulate_tcri(n_clones=6, n_phenotypes=4, n_genes=30, n_cells=200,
                          omega_concentration=0.4, fuzziness=0.1, seed=0)
    pyro.clear_param_store()
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="covariate",
                            batch_key="batch")
    model = TCRIModel(adata, n_latent=6, n_hidden=12, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=12, K=4, seed=0)
    model.train(max_epochs=3, batch_size=64, accelerator="cpu",
                enable_progress_bar=False, enable_model_summary=False)
    return model, adata


@pytest.mark.parametrize("view", ["reversed", "tail", "shuffled"])
def test_predict_binds_p_ct_by_cell_not_by_loader_position(view):
    """A cell's prediction must not depend on what else was passed alongside it.

    ``predict`` binds each cell to its clonotype x covariate prior. Bound instead by a running
    loader offset, that offset is the right index only when the passed object is a contiguous
    prefix of the training data in its original order; any other subset, a reordered view, or a
    per-patient slice -- all legal under ``governance/API_CONTRACT.md`` -- gives cell *i* the
    prior of the *i*-th TRAINING cell.

    A prefix view cannot see that, which is why ``tail`` and ``shuffled`` are in the parametrize
    list: the same cells must get the same probabilities in any arrangement.
    """
    model, adata = _fitted_for_binding()
    full = model.predict(adata)

    if view == "reversed":
        sub = adata[::-1].copy()
    elif view == "tail":
        sub = adata[100:].copy()
    else:
        sub = adata[np.random.default_rng(1).permutation(adata.n_obs)].copy()

    got = model.predict(sub)
    ref = full.loc[got.index]
    delta = float(np.abs(ref.to_numpy() - got.to_numpy()).max())
    assert delta < 1e-6, (
        f"predicting the {view} view changed the same cells' probabilities by {delta:.4f}. "
        f"p_ct is being bound by position in the loader rather than by each cell's own "
        f"clonotype x covariate group (NEW-1)."
    )


def test_to_anndata_binds_p_ct_by_cell_not_by_loader_position():
    """The same per-cell binding in ``to_anndata``'s logit/prior loop: a reversed view must
    leave every cell's log-posterior where ``predict`` put it."""
    from tcri._state.keys import X_LOGPOSTERIOR

    model, adata = _fitted_for_binding()
    full = model.to_anndata(adata.copy())
    rev = model.to_anndata(adata[::-1].copy())

    order = [list(rev.obs_names).index(n) for n in full.obs_names]
    delta = float(np.abs(full.obsm[X_LOGPOSTERIOR] - rev.obsm[X_LOGPOSTERIOR][order]).max())
    assert delta < 1e-5, (
        f"to_anndata on a reversed view moved the log-posterior by {delta:.4f} for the same "
        f"cells; the prior is bound by loader position rather than by cell (NEW-1)."
    )


def _labeled(clone, phen):
    """An unregistered AnnData with one cell per label pair, two covariate levels and one patient."""
    n = len(clone)
    X = np.random.default_rng(0).poisson(5.0, size=(n, 5)).astype("float32")
    ad = AnnData(
        X=X,
        obs=pd.DataFrame({"clone_id": clone, "true_phenotype": phen,
                          "covariate": [f"cond_{i % 2}" for i in range(n)], "patient": "P1"}),
        var=pd.DataFrame(index=[f"g{g}" for g in range(5)]),
    )
    ad.layers["counts"] = ad.X.copy()
    return ad


def _setup(ad):
    TCRIModel.setup_anndata(
        ad, layer="counts", clonotype_key="clone_id", phenotype_key="true_phenotype",
        covariate_key="covariate", batch_key="patient",
    )


def _small_category_warnings(record):
    return [str(w.message) for w in record if "has fewer than 3 cells" in str(w.message)]


def test_small_clones_do_not_warn_at_setup():
    """scvi warns when a registered category has fewer than 3 cells. A one-cell clone is the
    normal state of a repertoire and the per-cell ``indices`` column has one cell per category,
    so neither warns; every phenotype here has enough cells, so nothing warns at all."""
    clone = ["big"] * 30 + ["single"]
    phen = ["A", "B", "C"] * 10 + ["A"]
    ad = _labeled(clone, phen)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _setup(ad)

    assert _small_category_warnings(record) == []


def test_a_small_phenotype_still_warns_at_setup():
    """The warning is hidden for the clonotype and ``indices`` columns only: a phenotype with
    fewer than 3 cells is a real problem and still warns."""
    clone = ["big"] * 30 + ["single"]
    phen = ["A", "B"] * 15 + ["rare"]
    ad = _labeled(clone, phen)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _setup(ad)

    found = _small_category_warnings(record)
    assert len(found) == 1 and "_scvi_phenotype_col_in_registry" in found[0], found


def test_small_clones_do_not_warn_when_setup_is_transferred():
    """A model scoring an AnnData it was not set up on transfers its setup to it, which runs
    scvi's check again; the clonotype and ``indices`` columns stay quiet there too."""
    clone = ["big"] * 30 + ["single"]
    phen = ["A", "B", "C"] * 10 + ["A"]
    ad = _labeled(clone, phen)
    _setup(ad)
    model = _model(ad, K=2)
    fresh = _labeled(clone, phen)
    fresh.obs["indices"] = list(range(fresh.n_obs))

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model._validate_anndata(fresh)

    assert _small_category_warnings(record) == []
