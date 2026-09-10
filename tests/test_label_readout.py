"""The noisy-label readout: y_i | z^phi_i ~ Cat(C[z^phi_i, .]), summed out under eq 4.

It is the one observation downstream of the latent phenotype, so it is what gives the clone x
covariate hierarchy a per-cell data term. ``label_error_rate=None`` removes it and restores
the label-free model exactly. Each test names the mutation it catches.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pyro
import pyro.poutine as poutine
import pytest
import torch
from anndata import AnnData

from tcri.model._model import TCRIModel
from tcri.model._priors import encoder_posterior

warnings.filterwarnings("ignore")


def _fixture(seed=0, n_cells=60, n_genes=6):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(n_cells, n_genes)).astype("float32")
    obs = pd.DataFrame({
        "clone": [f"c{i % 5}" for i in range(n_cells)],
        "phen": [f"p{i % 3}" for i in range(n_cells)],
        "cov": ["a", "b"] * (n_cells // 2),
        "pt": ["P1"] * n_cells,
    })
    ad = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    ad.layers["counts"] = ad.X.copy()
    return ad


def _model(ad, **kw):
    pyro.clear_param_store()
    TCRIModel.setup_anndata(ad, layer="counts", clonotype_key="clone", phenotype_key="phen",
                            covariate_key="cov", batch_key="pt")
    return TCRIModel(ad, n_latent=4, n_hidden=8, n_layers=1, classifier_n_layers=1,
                     classifier_hidden=8, K=3, n_pseudo_obs=2, seed=0, **kw)


def _traces(m, batch_size=32):
    mod = m.module
    mod.eval()
    loader = m._make_data_loader(adata=m.adata, batch_size=batch_size, shuffle=False)
    args, kwargs = mod._get_fn_args_from_batch(next(iter(loader)))
    with contextlib.redirect_stdout(io.StringIO()):
        g = poutine.trace(mod.guide).get_trace(*args, **kwargs)
        t = poutine.trace(poutine.replay(mod.model, trace=g)).get_trace(*args, **kwargs)
    t.compute_log_prob()
    return t, args


def _grad_to_hierarchy(node_log_prob):
    lam = pyro.param("q_p_ct_raw").unconstrained()
    return torch.autograd.grad(node_log_prob.sum(), lam, retain_graph=True, allow_unused=True)[0]


def test_off_switch_restores_the_label_free_model():
    """``None``: no readout site, and the hierarchy gets no per-cell gradient (the shipped
    pre-2026-09 behaviour). Catches a readout that ignores the switch."""
    m = _model(_fixture(), label_error_rate=None)
    t, _ = _traces(m)
    assert "phenotype_label" not in t.nodes
    g = _grad_to_hierarchy(t.nodes["phenotype_alignment"]["log_prob"])
    assert g is None or float(g.abs().sum()) == 0.0
    pyro.clear_param_store()


def test_readout_is_the_hierarchy_data_term():
    """On: the site is present and observed, its gradient reaches q(p_ct), the head and the
    encoder, and the surrogate factor is still detached from q(p_ct). Catches (a) the readout
    being computed from a detached phi, (b) the surrogate's target being made live."""
    m = _model(_fixture(), label_error_rate=0.1)
    t, _ = _traces(m)
    node = t.nodes["phenotype_label"]
    assert node["type"] == "sample" and node["is_observed"]
    lp = node["log_prob"]

    g_h = _grad_to_hierarchy(lp)
    assert g_h is not None and float(g_h.abs().sum()) > 0.0, (
        "the readout carries no gradient to q(p_ct); phi must enter it live"
    )
    w_head = m.module.classifier.mlp[-1].weight
    w_enc = next(p for _, p in m.module.encoder.named_parameters() if p.ndim == 2)
    for name, w in (("head", w_head), ("encoder", w_enc)):
        g = torch.autograd.grad(lp.sum(), w, retain_graph=True, allow_unused=True)[0]
        assert g is not None and float(g.abs().sum()) > 0.0, f"the readout does not train the {name}"

    g_s = _grad_to_hierarchy(t.nodes["phenotype_alignment"]["log_prob"])
    assert g_s is None or float(g_s.abs().sum()) == 0.0, (
        "the surrogate's target is live; it must stay detached (a live target collapses the "
        "head and the hierarchy onto one constant)"
    )
    pyro.clear_param_store()


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.3])
def test_readout_probability_is_the_closed_form(eps):
    """p(y_i) = (softmax(l_i) @ C)[y_i] with l_i the gated logits on a LIVE phi and C the
    confusion matrix; eps = 0 is the hard-label limit p(y_i) = softmax(l_i)[y_i]. Catches a
    wrong confusion matrix, a detached or missing gate, or the head read from a sample."""
    m = _model(_fixture(), label_error_rate=eps)
    mod = m.module
    t, args = _traces(m)
    live = t.nodes["phenotype_label"]["fn"].probs.detach()
    with torch.no_grad():
        z_mean, _ = encoder_posterior(mod.encoder, args[0], args[1])
        logits = mod.classifier(z_mean)
        phi = t.nodes["p_ct"]["value"][mod.ct_array[args[3]]]
        ell = mod.gate_prob * logits + (1.0 - mod.gate_prob) * torch.log(phi + 1e-8)
        q = torch.softmax(ell, -1)
        P = q.shape[1]
        e = max(eps, 1e-6)
        C = torch.full((P, P), e / (P - 1)); C.fill_diagonal_(1.0 - e)
        expected = q @ C
    torch.testing.assert_close(live, expected, rtol=1e-5, atol=1e-6)
    if eps == 0.0:
        torch.testing.assert_close(live, q, rtol=1e-4, atol=1e-5)
    y = t.nodes["phenotype_label"]["value"]
    assert torch.equal(y, mod._target_phenotypes[args[3]]), "the readout scores the wrong cells' labels"
    pyro.clear_param_store()


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2.0 / 3.0])
def test_label_error_rate_is_validated(bad):
    """P = 3 here, so 1 - 1/P = 2/3 is already uninformative and must be rejected."""
    with pytest.raises(ValueError):
        _model(_fixture(), label_error_rate=bad)
    pyro.clear_param_store()


def test_label_error_rate_survives_save_and_load(tmp_path):
    """The knob rides on init_params_, so a saved session reloads with the same model."""
    ad = _fixture()
    m = _model(ad, label_error_rate=0.25)
    m.save(str(tmp_path), overwrite=True, save_anndata=False)
    pyro.clear_param_store()
    TCRIModel.setup_anndata(ad, layer="counts", clonotype_key="clone", phenotype_key="phen",
                            covariate_key="cov", batch_key="pt")
    m2 = TCRIModel.load(str(tmp_path), adata=ad)
    assert m2.module.label_error_rate == pytest.approx(0.25)
    pyro.clear_param_store()
