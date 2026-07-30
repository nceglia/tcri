"""Phenotype-classifier recovery test.

Locks in the two coupled fixes that make ``f_cls`` (the phenotype classifier head,
methods eq. 4) actually train:

  1. The classifier enters the ELBO through ``pyro.factor("phenotype_alignment", ...)``
     in ``TCRIModule.model()`` (the surrogate KL objective, methods "Inference
     Details"). Without it ``cls_logits`` never touches the log-joint and f_cls
     gets no gradient (weight change == 0).
  2. The per-cell alignment target ``phi = p_ct[ct_idx]`` is indexed with the
     GLOBAL cell indices, not the local pyro data-plate index. Indexing with the
     local index silently scrambles each cell's target across shuffled minibatches,
     which trains the classifier on the wrong labels and collapses it to a constant
     (recovery == chance).

The dataset is "perfect": each clonotype expresses one unique marker gene and maps
to one phenotype, so a correctly-trained classifier recovers the phenotype from gene
expression alone (gate_prob=1.0, the pure-classifier path).
"""
import contextlib
import io

import numpy as np
import pandas as pd
import pyro
import pytest
import torch
from anndata import AnnData

from tcri.model._model import TCRIModel


@pytest.fixture(autouse=True)
def _isolate_param_store():
    """Own the process-global Pyro param store for this module's tests.

    Each test here trains a small model, leaving shape-specific params
    (``q_p_c_raw`` etc.) in the global store; without clearing on teardown the
    next model test in the suite reuses a stale-shaped param and crashes. Scoped
    to THIS module (not a conftest autouse) so it never wipes the session-scoped
    ``trained_model`` fixture's params mid-suite.
    """
    pyro.clear_param_store()
    yield
    pyro.clear_param_store()


def _perfect_adata(n_clones=5, n_per=60, n_genes=6, seed=0):
    """One marker gene per clone, one phenotype per clone (n_conditions=1)."""
    rng = np.random.default_rng(seed)
    rows, clone, phen = [], [], []
    for c in range(n_clones):
        for _ in range(n_per):
            v = rng.poisson(0.2, size=n_genes).astype("float32")
            v[c % n_genes] = 100.0  # unique hot gene marks the clone
            rows.append(v)
            clone.append(f"clone_{c}")
            phen.append(f"phen_{c}")
    obs = pd.DataFrame(
        {
            "clone_id": clone,
            "true_phenotype": phen,
            "covariate": "c0",
            "patient": "P1",
        }
    )
    ad = AnnData(
        X=np.asarray(rows),
        obs=obs,
        var=pd.DataFrame(index=[f"g{g}" for g in range(n_genes)]),
    )
    ad.layers["counts"] = ad.X.copy()
    return ad


@pytest.mark.parametrize("gate_prob", [1.0, 0.5])
def test_classifier_perfect_recovery(gate_prob):
    """f_cls recovers the phenotype on a linearly-separable dataset.

    gate_prob=1.0 is the strict test (phenotype comes from the classifier alone);
    gate_prob=0.5 is the methods default (classifier + clonotype prior).
    """
    np.random.seed(0)
    torch.manual_seed(0)
    pyro.set_rng_seed(0)  # param store cleared by the autouse fixture
    import scvi

    scvi.settings.seed = 0

    ad = _perfect_adata()
    truth = ad.obs["true_phenotype"].to_numpy()

    TCRIModel.setup_anndata(
        ad,
        layer="counts",
        clonotype_key="clone_id",
        phenotype_key="true_phenotype",
        covariate_key="covariate",
        batch_key="patient",
    )
    model = TCRIModel(
        ad,
        n_latent=8,
        n_hidden=16,
        n_layers=1,
        classifier_n_layers=1,
        classifier_hidden=16,
        K=5,
        n_pseudo_obs=3,
        gate_prob=gate_prob,
    )

    # snapshot classifier weights to prove the head actually trains
    w0 = {k: v.detach().clone() for k, v in model.module.classifier.state_dict().items()}

    with contextlib.redirect_stdout(io.StringIO()):
        model.train(
            max_epochs=200,
            batch_size=128,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    w1 = model.module.classifier.state_dict()
    dw = sum((w1[k] - w0[k]).pow(2).sum().item() for k in w0) ** 0.5
    assert dw > 1e-3, f"classifier head did not train (ΔL2={dw:.2e}); is it in the ELBO?"

    pred = model.predict(ad)
    recovery = (pred.columns[pred.values.argmax(1)] == truth).mean()
    assert recovery >= 0.9, f"phenotype recovery {recovery:.3f} (gate={gate_prob}); chance=0.2"
