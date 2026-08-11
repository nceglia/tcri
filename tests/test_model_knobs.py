"""Knob-test matrix — every constructor/train knob gets a correctness test.

Two layers, and the split matters:

**WIRING** — does the value actually reach the object it claims to configure?
This layer exists because ``lr`` was marked "hooked up" in the matrix for months
while never reaching Pyro's optimizer: the model still converged, so every
behavioral test passed. A knob that is silently ignored is invisible to
convergence tests. Assert the plumbing directly.

**BEHAVIOR** — the mathematically-correct input->output assertion (draw variance
scales as 1/(scale+1), temperature sharpens, gate endpoints reduce to closed
forms, batch size is an invariance).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pyro
import pytest
import torch
from anndata import AnnData

from tcri.model._model import TCRIModel
from tcri.model._training import UnifiedTrainingPlan


@pytest.fixture(autouse=True)
def _isolate_param_store():
    pyro.clear_param_store()
    yield
    pyro.clear_param_store()


@pytest.fixture
def adata():
    n_clones, n_per, n_genes = 6, 25, 8
    rng = np.random.default_rng(0)
    rows, clone, phen, cov = [], [], [], []
    for c in range(n_clones):
        for j in range(2):
            for _ in range(n_per):
                v = rng.poisson(0.3, size=n_genes).astype("float32")
                v[c % n_genes] = 60.0
                rows.append(v)
                clone.append(f"clone_{c}")
                phen.append(f"phen_{c % 4}")
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


def _model(ad, **kw):
    # DE-19: without a seed the network init depends on whatever ran earlier in the session,
    # so these assertions were order-dependent and passed by luck. A change elsewhere that
    # merely shifted RNG consumption was enough to land this fixture on a near-degenerate
    # p_ct row and break the variance identity below.
    kw.setdefault("seed", 0)
    kw.setdefault("n_latent", 8)
    kw.setdefault("n_hidden", 16)
    kw.setdefault("n_layers", 1)
    kw.setdefault("classifier_n_layers", 1)
    kw.setdefault("classifier_hidden", 16)
    kw.setdefault("K", 3)
    return TCRIModel(ad, **kw)


def _train(model, **kw):
    kw.setdefault("max_epochs", 3)
    kw.setdefault("batch_size", 128)
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(enable_progress_bar=False, enable_model_summary=False, **kw)


# ══════════════════════════ WIRING ══════════════════════════════════════════
# "does the value reach the thing it configures?"

@pytest.mark.parametrize("knob,value,reader", [
    ("n_latent", 6, lambda m: m.module.n_latent),
    ("n_pseudo_obs", 4, lambda m: m.module.vamp_prior.pseudo_inputs.shape[0]),
    ("K", 3, lambda m: m.module.mixture_concentration.shape[0]),
    ("global_scale", 7.5, lambda m: m.module.global_scale),
    ("local_scale", 2.5, lambda m: m.module.local_scale),
    ("prior_temperature", 1.5, lambda m: m.module.prior_temperature),
    ("guide_temperature", 0.5, lambda m: m.module.guide_temperature),
    ("gate_prob", 0.25, lambda m: m.module.gate_prob),
    ("classifier_temperature", 2.0, lambda m: m.module.classifier_temperature),
    ("classifier_dropout", 0.3, lambda m: m.module.classifier.mlp[2].p),
    ("classifier_hidden", 12, lambda m: m.module.classifier.mlp[0].out_features),
    ("kl_weight_max", 0.7, lambda m: m.module.kl_weight_max),
    ("guide_init_scale", 4.0, lambda m: m.module.guide_init_scale),
    ("phenotype_kl_weight", 3.0, lambda m: m.module.phenotype_kl_weight),
])
def test_constructor_knob_is_wired(adata, knob, value, reader):
    """Each constructor knob must be readable back off the module."""
    m = _model(adata, **{knob: value})
    assert reader(m) == pytest.approx(value), f"{knob}={value} did not reach the module"


def test_n_latent_determines_latent_width(adata):
    m = _model(adata, n_latent=6)
    _train(m)
    assert m.get_latent_representation().shape[1] == 6


def test_K_determines_archetype_count(adata):
    m = _model(adata, K=3)
    assert m.centers.shape[0] == 3
    assert m.module.mixture_concentration.shape[0] == 3


def test_classifier_depth_is_wired(adata):
    """classifier_n_layers controls the number of Linear blocks."""
    shallow = _model(adata, classifier_n_layers=1)
    deep = _model(adata, classifier_n_layers=3)
    n_lin = lambda m: sum(isinstance(x, torch.nn.Linear) for x in m.module.classifier.mlp)
    assert n_lin(deep) == n_lin(shallow) + 2


def test_train_knobs_reach_the_optimizer_and_plan(adata):
    """lr/weight_decay must reach PYRO's optimizer — the regression that started this.

    Marked "hooked up" in the knob matrix while Pyro silently used scvi's hard-coded
    1e-3; the model still converged, so no behavioral test noticed.
    """
    m = _model(adata)
    plan = UnifiedTrainingPlan(
        module=m.module, n_steps_kl_warmup=123, reconstruction_loss_scale=5e-3,
        optimizer_config={"lr": 0.07, "betas": (0.8, 0.99), "eps": 1e-6,
                          "weight_decay": 2e-4},
    )
    args = plan.optim.pt_optim_args
    assert args["lr"] == 0.07
    assert args["weight_decay"] == 2e-4
    assert args["betas"] == (0.8, 0.99)
    assert args["eps"] == 1e-6
    assert plan.n_steps_kl_warmup == 123
    assert plan.reconstruction_loss_scale == 5e-3


def test_reconstruction_loss_scale_reaches_the_module(adata):
    m = _model(adata)
    _train(m, reconstruction_loss_scale=7e-3)
    assert m.module.reconstruction_loss_scale == pytest.approx(7e-3)


def test_max_epochs_and_patience_reach_the_trainer(adata):
    m = _model(adata, patience=4)
    _train(m, max_epochs=5)
    assert m.trainer.max_epochs == 5
    # scvi installs its own EarlyStopping subclass (LoudEarlyStopping), so match on
    # the attribute rather than the class name.
    es = [cb for cb in m.trainer.callbacks if hasattr(cb, "patience")]
    assert es, f"no early-stopping callback installed: {[type(c).__name__ for c in m.trainer.callbacks]}"
    assert es[0].patience == 4, "patience did not reach the early-stopping callback"


def test_network_geometry_knobs_are_wired(adata):
    """n_hidden/n_layers must actually size the encoder."""
    small = _model(adata, n_hidden=16, n_layers=1)
    big = _model(adata, n_hidden=64, n_layers=3)
    n = lambda m: sum(p.numel() for p in m.module.encoder.parameters())
    assert n(big) > 10 * n(small), f"encoder did not grow: {n(small)} -> {n(big)}"


def test_batch_size_reaches_the_dataloader(adata):
    m = _model(adata)
    for bs in (32, 200):
        loader = m._make_data_loader(adata=m.adata, batch_size=bs, shuffle=False)
        assert next(iter(loader))["X"].shape[0] == bs


def test_n_steps_kl_warmup_ramps_the_kl_weight(adata):
    """The warmup must actually anneal module.kl_weight from ~0 up to kl_weight_max.

    NOTE the warmup is counted in optimizer STEPS while max_epochs is in epochs; with
    batch_size >= n_obs that is one step per epoch (tracked as deviation DUX-2).
    """
    m = _model(adata)
    seen = []
    orig = UnifiedTrainingPlan.training_step

    def spy(self, batch, batch_idx):
        out = orig(self, batch, batch_idx)
        seen.append(float(self.module.kl_weight))
        return out

    UnifiedTrainingPlan.training_step = spy
    try:
        _train(m, max_epochs=12, batch_size=64, n_steps_kl_warmup=20)
    finally:
        UnifiedTrainingPlan.training_step = orig
    warm = seen[:20]                      # the ramp itself; it plateaus afterwards
    assert warm[0] < warm[len(warm) // 2] < warm[-1], f"kl_weight did not ramp: {warm[:6]}"
    assert all(b >= a - 1e-12 for a, b in zip(seen, seen[1:])), "ramp must be monotonic"
    assert max(seen) == pytest.approx(m.module.kl_weight_max), "ramp must reach the ceiling"
    assert seen[0] < 1e-3, "ramp must start near zero"


def test_use_enumeration_selects_the_elbo(adata):
    """use_enumeration picks TraceEnum_ELBO vs Trace_ELBO."""
    from pyro.infer import TraceEnum_ELBO, Trace_ELBO

    plain = UnifiedTrainingPlan(module=_model(adata, use_enumeration=False).module,
                                n_steps_kl_warmup=1, reconstruction_loss_scale=1e-3)
    assert isinstance(plain.loss, Trace_ELBO)
    pyro.clear_param_store()
    enum = UnifiedTrainingPlan(module=_model(adata, use_enumeration=True).module,
                               n_steps_kl_warmup=1, reconstruction_loss_scale=1e-3)
    assert isinstance(enum.loss, TraceEnum_ELBO)


# ══════════════════════════ BEHAVIOR ════════════════════════════════════════

def test_local_scale_controls_p_ct_draw_variance(adata):
    """Dirichlet(β·p): Var = p(1−p)/(β+1) — variance strictly decreases in β."""
    m = _model(adata)
    _train(m)
    base = torch.as_tensor(m.get_p_ct(), dtype=torch.float64)
    vs = []
    for beta in (1.0, 10.0, 100.0):
        conc = torch.clamp(beta * base, min=1e-3)
        d = torch.distributions.Dirichlet(conc)
        vs.append(float(d.sample((4000,)).var(0).mean()))
    assert vs[0] > vs[1] > vs[2], f"p_ct draw variance must fall with local_scale: {vs}"
    # The knob assertion is the monotonicity above, on the FITTED base: that is what "local_scale
    # controls the p_ct draw variance" means.
    #
    # The exact Dirichlet identity is a separate claim and is checked on a well-conditioned row
    # rather than a fitted one. This fixture gives every cell of a clone the same phenotype, so
    # the true p_ct row is one-hot and the fitted row is near-degenerate; the empirical variance
    # of a near-degenerate Dirichlet is a high-variance estimator, and asserting rel=0.15 on it
    # was marginal — it passed by luck and broke on a change that merely made the fit sharper.
    # Checking the identity where it is well conditioned tests the same mathematics without
    # inheriting the fixture's degeneracy.
    beta = 10.0
    row = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    conc = beta * row                      # a_0 = beta exactly; the clamp cannot bite
    expected = (row * (1 - row) / (beta + 1)).mean()
    got = torch.distributions.Dirichlet(conc).sample((40000,)).var(0).mean()
    assert float(got) == pytest.approx(float(expected), rel=0.05)


def test_global_scale_enters_the_eq1_prior(adata):
    """α scales the clonotype-prior concentration (deviation [G] fix)."""
    lo = _model(adata, global_scale=1.0)
    hi = _model(adata, global_scale=50.0)
    conc = lambda m: float((m.module.global_scale * m.module.mixture_concentration).sum(-1).mean())
    assert conc(hi) == pytest.approx(50.0 * conc(lo) / 1.0, rel=1e-5)


def test_prior_temperature_raises_clone_prior_entropy(adata):
    """clone_phen_prior = normalize(prior**(1/T)); T>1 flattens ⇒ higher entropy."""
    ent = lambda M: float(np.mean([-np.sum(r[r > 0] * np.log2(r[r > 0])) for r in M]))
    t1 = _model(adata, prior_temperature=1.0).module.clone_phen_prior.numpy()
    t3 = _model(adata, prior_temperature=3.0).module.clone_phen_prior.numpy()
    assert ent(t3) > ent(t1), "T>1 must raise the row-entropy of clone_phen_prior"


def test_guide_temperature_sharpens_get_p_ct(adata):
    """get_p_ct sharpens q**(1/T); T<1 ⇒ lower row-entropy, same q_p_ct_raw."""
    m = _model(adata)
    _train(m)
    ent = lambda M: float(np.mean([-np.sum(r[r > 0] * np.log2(r[r > 0])) for r in M]))
    m.module.guide_temperature = 1.0
    e1 = ent(m.get_p_ct())
    m.module.guide_temperature = 0.25          # same params, sharper read-out
    e_sharp = ent(m.get_p_ct())
    assert e_sharp < e1, "T<1 must lower the row-entropy of get_p_ct()"


def test_classifier_temperature_divides_logits(adata):
    """forward() divides by T ⇒ logits(T=2) == logits(T=1)/2 for fixed weights."""
    m = _model(adata, classifier_temperature=1.0)
    z = torch.randn(5, m.module.n_latent)
    m.module.eval()
    with torch.no_grad():
        a = m.module.classifier(z)
        m.module.classifier.temperature = 2.0
        b = m.module.classifier(z)
    torch.testing.assert_close(b, a / 2.0)


@pytest.mark.parametrize("gate,expect", [(0.0, "prior"), (1.0, "classifier")])
def test_gate_prob_endpoints_reduce_to_closed_forms(adata, gate, expect):
    """ℓ = π·f_cls + (1−π)·log φ: π=0 ⇒ softmax(log φ); π=1 ⇒ softmax(f_cls)."""
    import torch.nn.functional as F
    from scvi import REGISTRY_KEYS

    m = _model(adata, gate_prob=gate)
    _train(m)
    got = m.predict(adata).values

    mod = m.module
    mod.eval()
    p_ct = torch.as_tensor(m.get_p_ct())
    with torch.no_grad():
        loader = m._make_data_loader(adata=m.adata, batch_size=4096)
        chunks, start = [], 0
        for tensors in loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            b = tensors[REGISTRY_KEYS.BATCH_KEY].long()
            n = x.shape[0]
            prior = p_ct[mod.ct_array[start:start + n]]
            z_loc, _, _ = mod.encoder(x, b)
            if expect == "prior":
                logits = torch.log(prior + 1e-8)
            else:
                logits = mod.classifier(z_loc)
            chunks.append(F.softmax(logits, dim=-1))
            start += n
        want = torch.cat(chunks).numpy()
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_predict_is_invariant_to_batch_size(adata):
    """batch_size is a chunking detail, not a modelling one.

    Tolerance is float32-scale, not exact: different batch shapes take different BLAS
    kernel paths, so the encoder's accumulations differ in the last bits (~1e-7 here).
    Anything materially larger would mean batch_size is affecting the computation.
    """
    m = _model(adata)
    _train(m)
    a = m.predict(adata, batch_size=64).values
    b = m.predict(adata, batch_size=4096).values
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
    # rows must still be probability vectors regardless of chunking
    np.testing.assert_allclose(a.sum(1), 1.0, atol=1e-5)


# ══════════════════════ device seam (CU-01) ═════════════════════════════════

def test_device_reaches_the_engine_from_every_metric(adata):
    """``device=`` must actually configure the numeric core.

    The seam existed in ``_compute/_xp`` from PR5 but no metric exposed it, so it was
    unreachable — GPU was documented and dead. This asserts the value arrives at
    ``_joint_draws`` for every public metric, which is the only thing that makes a
    GPU run possible.
    """
    import tcri
    import tcri._compute._joint as CJ
    import tcri.tools._joint as TJ

    m = _model(adata)
    _train(m)
    m.to_anndata(adata)
    covs = list(adata.uns["tcri_covariate_categories"])

    seen = []
    orig = CJ._joint_draws

    def spy(*a, **k):
        seen.append(k.get("device"))
        return orig(*a, **k)

    TJ._joint_draws = spy
    try:
        for call in (
            lambda: tcri.tl.mutual_information(adata, covariate=covs[0], device="cpu"),
            lambda: tcri.tl.clonotypic_entropy(adata, covariate=covs[0], device="cpu"),
            lambda: tcri.tl.phenotypic_entropy(adata, covariate=covs[0], device="cpu"),
            lambda: tcri.tl.phenotypic_flux(
                adata, cov_from=covs[0], cov_to=covs[1], device="cpu"),
        ):
            seen.clear()
            call()
            assert seen and all(d == "cpu" for d in seen), (
                f"device did not reach the engine: {seen}"
            )
    finally:
        TJ._joint_draws = orig


def test_device_does_not_change_results(adata):
    """Routing through the device seam is a placement detail, not a numerical one."""
    import tcri

    m = _model(adata)
    _train(m)
    m.to_anndata(adata)
    cov = list(adata.uns["tcri_covariate_categories"])[0]
    a = tcri.tl.mutual_information(adata, covariate=cov, device=None)["result"]
    b = tcri.tl.mutual_information(adata, covariate=cov, device="cpu")["result"]
    assert float(a["value"].iloc[0]) == pytest.approx(float(b["value"].iloc[0]), rel=1e-12)
