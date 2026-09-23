"""Phenotype-classifier recovery test.

Two coupled requirements make ``f_cls``, the phenotype classifier head, train at all:

  1. The classifier enters the ELBO through ``pyro.factor("phenotype_alignment", ...)`` in
     ``TCRIModule.model()`` -- the alignment surrogate of ``governance/MODEL_CONTRACT.md``
     eq 7. Without it ``cls_logits`` never touches the log-joint and f_cls gets no gradient
     (weight change == 0).
  2. The per-cell alignment target ``phi = p_ct[ct_idx]`` is indexed with the GLOBAL cell
     indices, not the local pyro data-plate index. The local index scrambles each cell's
     target across shuffled minibatches, which trains the classifier on the wrong labels and
     collapses it to a constant (recovery == chance).

The dataset is "perfect": each clonotype expresses one unique marker gene and maps to one
phenotype, so a correctly-trained classifier recovers the phenotype from gene expression alone
(gate_prob=1.0, the pure-classifier path).
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


def test_head_is_not_constant_after_a_short_fit():
    """The head varies across cells after a short fit on a realistic (noisy) cohort.

    Trained on a SAMPLE of z it does not: on a noisy cohort the encoder's posterior scale dwarfs
    the spread of the posterior mean across cells, so the head's input is almost all posterior
    noise, a constant is the optimum, and its hidden ReLUs die. The head must therefore read the
    posterior mean (``governance/MODEL_CONTRACT.md``, "The head reads the posterior mean").

    ``test_classifier_perfect_recovery`` below cannot see this: its marker gene at 100 counts
    puts the posterior mean far outside the posterior width, so even a sample-trained head
    recovers the phenotype there. The assertion here is on the logit spread across cells, which
    a collapsed head cannot clear.
    """
    from tcri.datasets import simulate_cohort
    from tcri.model._priors import encoder_posterior

    ad = simulate_cohort(
        n_patients=4, conditions=("pre", "post"), disease_fraction=0.5, n_clones=(14, 24),
        n_phenotypes=4, n_genes=40, n_cells_per_sample=200,
        clone_size_distribution="powerlaw", clone_size_exponent=2.0,
        disease_enrichment=12.0, control_enrichment=1.1, seed=0,
    )
    TCRIModel.setup_anndata(
        ad, layer="counts", clonotype_key="clone_id", phenotype_key="phenotype",
        covariate_key="condition", batch_key="patient",
    )
    model = TCRIModel(ad, n_latent=8, n_hidden=32, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=32, K=4, seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=80, batch_size=256, accelerator="cpu",
                    enable_progress_bar=False, enable_model_summary=False)

    mod = model.module
    mod.eval()
    X = ad.layers["counts"]
    x = torch.as_tensor(X.toarray() if hasattr(X, "toarray") else np.asarray(X), dtype=torch.float32)
    b = torch.as_tensor(ad.obs["patient"].astype("category").cat.codes.values,
                        dtype=torch.long).view(-1, 1)
    with torch.no_grad():
        z_mean, _ = encoder_posterior(mod.encoder, x, b)
        logits = mod.classifier(z_mean)
    logit_sd = float(logits.std(0).mean())
    n_classes = int(len(torch.unique(logits.argmax(1))))
    assert logit_sd > 0.02, (
        f"the head is (near-)constant across cells: logit sd {logit_sd:.2e}, {n_classes} "
        f"argmax class(es). It is being trained on a sample of z rather than the posterior "
        f"mean, so its input is almost all posterior noise."
    )


@pytest.mark.parametrize("gate_prob", [1.0, 0.5])
def test_classifier_perfect_recovery(gate_prob):
    """f_cls recovers the phenotype on a linearly-separable dataset.

    gate_prob=1.0 is the strict test (the phenotype comes from the classifier alone);
    gate_prob=0.5 is the constructor default (classifier + clonotype prior).
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
