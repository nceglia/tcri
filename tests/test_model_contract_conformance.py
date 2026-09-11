"""Model-contract conformance — the guardrail for the model's *mathematics*.

``governance/MODEL_CONTRACT.md`` is the definition; its ```python contract``` block lists every
stochastic site with its family, plate, observed flag and event dim. This test traces the live
``TCRIModule.model``/``.guide`` and asserts they match that block exactly — declared sites
present, **no undeclared sites**, the guide's variational family intact — and then the
invariants the structure alone cannot pin (α and β scaling, the surrogate's sign and detached
target, the gate rule, global alignment indices, one σ, the plate scaling).

If you changed the model and this test fails, the contract file changes in the same PR,
deliberately. Do not loosen the block to match new code.
"""
import contextlib
import io

import numpy as np
import pandas as pd
import pyro
import pyro.poutine as poutine
import pytest
import torch
from anndata import AnnData

from tcri.model._model import TCRIModel
from tcri.model._priors import encoder_posterior
from tests._governance import contract_namespace

MC = contract_namespace("MODEL_CONTRACT.md")
GENERATIVE_SITES = MC["GENERATIVE_SITES"]
GUIDE_SITES = MC["GUIDE_SITES"]
GUIDE_PARAMS = MC["GUIDE_PARAMS"]
PLATES = MC["PLATES"]
FORBIDDEN_GUIDE_SITES = MC["FORBIDDEN_GUIDE_SITES"]
FORBIDDEN_MODEL_SITES = MC["FORBIDDEN_MODEL_SITES"]

# pyro wrappers that carry no model semantics — unwrapped before comparing.
_WRAPPERS = ("Independent", "ExpandedDistribution", "MaskedDistribution")


@pytest.fixture(scope="module")
def traced():
    """Trace model() and guide() once on a tiny fixture (no training needed)."""
    pyro.clear_param_store()
    np.random.seed(0)
    torch.manual_seed(0)
    pyro.set_rng_seed(0)

    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 6
    X = rng.poisson(1.0, size=(n_cells, n_genes)).astype("float32")
    obs = pd.DataFrame(
        {
            "clone": [f"c{i % 5}" for i in range(n_cells)],
            "phen": [f"p{i % 3}" for i in range(n_cells)],
            "cov": ["a", "b"] * (n_cells // 2),
            "pt": ["P1"] * n_cells,
        }
    )
    ad = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    ad.layers["counts"] = ad.X.copy()

    TCRIModel.setup_anndata(
        ad, layer="counts", clonotype_key="clone", phenotype_key="phen",
        covariate_key="cov", batch_key="pt",
    )
    model = TCRIModel(
        ad, n_latent=4, n_hidden=8, n_layers=1, classifier_n_layers=1,
        classifier_hidden=8, K=3, n_pseudo_obs=2,
    )

    loader = model._make_data_loader(adata=model.adata, batch_size=32, shuffle=False)
    batch = next(iter(loader))
    args, kwargs = model.module._get_fn_args_from_batch(batch)

    with contextlib.redirect_stdout(io.StringIO()):
        m_trace = poutine.trace(model.module.model).get_trace(*args, **kwargs)
        g_trace = poutine.trace(model.module.guide).get_trace(*args, **kwargs)

    yield model, m_trace, g_trace, args, kwargs
    pyro.clear_param_store()


def _unwrap(d):
    """Strip Independent/Expanded wrappers to the semantic distribution class."""
    seen = 0
    while type(d).__name__ in _WRAPPERS and seen < 8:
        d = getattr(d, "base_dist", None) or getattr(d, "base_distribution", None)
        if d is None:
            return None
        seen += 1
    return d


def _stochastic(trace):
    """{name: node} for real stochastic sites (drops plate _Subsample bookkeeping)."""
    out = {}
    for name, node in trace.nodes.items():
        if node["type"] != "sample":
            continue
        if type(node["fn"]).__name__ == "_Subsample":
            continue
        out[name] = node
    return out


def _plates(node):
    return {f.name for f in node.get("cond_indep_stack", ())}


# ── structure: the generative program ────────────────────────────────────────

def test_generative_sites_match_contract(traced):
    _, m_trace, _, _, _ = traced
    live = _stochastic(m_trace)

    for spec in GENERATIVE_SITES:
        name = spec["name"]
        assert name in live, (
            f"model() is missing the declared site '{name}' (contract eq {spec['eq']}). "
            "If you removed it, change governance/MODEL_CONTRACT.md in the same PR."
        )
        node = live[name]
        base = _unwrap(node["fn"])
        assert base is not None and type(base).__name__ == spec["dist"], (
            f"site '{name}' (eq {spec['eq']}) is "
            f"{type(base).__name__ if base is not None else None}, contract says {spec['dist']}."
        )
        if spec["plate"] is not None:
            assert spec["plate"] in _plates(node), (
                f"site '{name}' must live in plate '{spec['plate']}' "
                f"({PLATES.get(spec['plate'], '')}); found {_plates(node)}."
            )
        assert bool(node.get("is_observed")) == spec["observed"], (
            f"site '{name}' observed={bool(node.get('is_observed'))}, "
            f"contract says observed={spec['observed']}."
        )
        if spec["event_dim"] is not None:
            assert len(node["fn"].event_shape) == spec["event_dim"], (
                f"site '{name}' event_dim={len(node['fn'].event_shape)}, "
                f"contract says {spec['event_dim']}."
            )


def test_no_undeclared_generative_sites(traced):
    """An EXTRA stochastic site changes the joint distribution — the contract must say so."""
    _, m_trace, _, _, _ = traced
    live = set(_stochastic(m_trace))
    declared = {s["name"] for s in GENERATIVE_SITES}
    extra = live - declared
    assert not extra, (
        f"model() has undeclared stochastic site(s): {sorted(extra)}. Every site "
        "changes the joint. Declare it in the contract block of governance/MODEL_CONTRACT.md "
        "and describe it in the prose."
    )


# ── structure: the variational family ────────────────────────────────────────

def test_guide_family_matches_contract(traced):
    _, _, g_trace, _, _ = traced
    live = _stochastic(g_trace)

    for spec in GUIDE_SITES:
        name = spec["name"]
        assert name in live, f"guide() is missing q({name}) (eq 6)."
        base = _unwrap(live[name]["fn"])
        assert base is not None and type(base).__name__ == spec["dist"], (
            f"q({name}) is {type(base).__name__ if base is not None else None}, "
            f"contract says {spec['dist']}."
        )

    declared = {s["name"] for s in GUIDE_SITES}
    extra = set(live) - declared
    assert not extra, (
        f"guide() has undeclared site(s): {sorted(extra)}. This changes the "
        "variational family (eq 6) and therefore the ELBO."
    )


def test_guide_registers_variational_params(traced):
    """Both variational parameters are registered, under whatever namespace the module has.

    Since 0.12 a module registers under ``f"{name}."`` so two models can share the process, so
    the contract names the PARAMETERS and the test compares the tail of each traced site. The
    fixture here is unnamed, and the second half asserts the tail match is doing real work
    rather than passing because the tail happens to be the whole name.
    """
    _, _, g_trace, _, _ = traced
    params = {n for n, nd in g_trace.nodes.items() if nd["type"] == "param"}
    tails = {n.rsplit(".", 1)[-1] for n in params}
    for p in GUIDE_PARAMS:
        assert p in tails, (
            f"guide() must register the learnable variational parameter '{p}' "
            f"(λ_c / λ'_m of eq 6); without it the Dirichlet posteriors are not learned. "
            f"Traced param sites: {sorted(params)}"
        )


def test_guide_params_live_under_the_module_namespace():
    """A named module registers the same two parameters under its own prefix.

    The contract's `GUIDE_PARAMS` is a list of parameter names, not of store keys; this is
    what makes that distinction real rather than a wording choice.
    """
    import pyro

    from tcri.datasets import simulate_tcri
    from tcri.model._model import TCRIModel

    store = pyro.get_param_store()
    saved = store.get_state()
    try:
        store.clear()
        adata = simulate_tcri(n_clones=5, n_phenotypes=3, n_genes=15, n_cells=60, seed=0)
        adata.layers["counts"] = adata.X.copy()
        TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                                phenotype_key="phenotype", covariate_key="covariate",
                                batch_key="batch")
        model = TCRIModel(adata, n_latent=6, n_hidden=12, n_layers=1, classifier_n_layers=1,
                          classifier_hidden=12, K=3, name="ns")
        args, kwargs = model.module._get_fn_args_from_batch(
            next(iter(model._make_data_loader(adata=adata, batch_size=16)))
        )
        model.module.guide(*args, **kwargs)
        for p in GUIDE_PARAMS:
            assert f"ns.{p}" in store, f"named module did not register ns.{p}: {sorted(store.keys())[:6]}"
            assert p not in store, f"named module also registered the bare {p}"
    finally:
        store.clear()
        store.set_state(saved)


def test_discrete_phenotype_latent_is_not_sampled(traced):
    """z^ϕ is replaced by the surrogate; a q(z^ϕ) site would change the objective."""
    _, _, g_trace, _, _ = traced
    live = set(_stochastic(g_trace))
    for forbidden in FORBIDDEN_GUIDE_SITES:
        assert forbidden not in live, (
            f"guide() samples '{forbidden}'. z^ϕ has no variational factor: it is summed out in "
            "the label readout (eq 8) and stands behind the surrogate. A q(z^ϕ) site changes "
            "the objective; change the contract in the same PR."
        )


def test_discrete_phenotype_latent_is_not_observed_directly(traced):
    """z^ϕ is LATENT: never sampled in the guide and never conditioned on directly.

    The input label enters as a noisy READOUT of z^ϕ (eq 8, site ``phenotype_label``), with
    z^ϕ summed out. Observing z^ϕ itself would hard-wire every cell to its label; the readout
    keeps the phenotype latent and lets ε say how far the model may move from the label.
    """
    _, m_trace, _, _, _ = traced
    observed = {
        name for name, site in m_trace.nodes.items()
        if site["type"] == "sample" and site.get("is_observed")
    }
    for forbidden in FORBIDDEN_MODEL_SITES:
        assert forbidden not in observed, (
            f"model() conditions on '{forbidden}' directly. z^ϕ is latent; the labels enter "
            f"through the phenotype_label readout (eq 8) with error rate ε, not as z^ϕ itself."
        )
    assert {"obs", "phenotype_label"} <= observed, (
        "the model observes expression (eq 5) and the label readout (eq 8); one is missing"
    )


# ── semantics: invariants the structure alone cannot pin ─────────────────────

def test_alpha_scales_the_clonotype_prior(traced):
    """eq 1: p_c prior concentration must scale with α (global_scale)."""
    model, m_trace, _, args, kwargs = traced
    conc = _unwrap(m_trace.nodes["p_c"]["fn"]).concentration
    alpha = float(model.module.global_scale)
    archetypes = model.module.mixture_concentration  # rows sum to ~1
    expected_total = alpha * float(archetypes.sum(-1).mean())
    live_total = float(conc.sum(-1).mean())
    assert live_total == pytest.approx(expected_total, rel=1e-4), (
        f"p_c concentration totals {live_total:.4f}, expected ≈{expected_total:.4f} "
        f"(α={alpha} × archetype). eq 1: the p_c prior concentration must scale with α; without "
        f"it the prior is Dir(ψ_b), U-shaped with mass at the simplex corners."
    )


def test_beta_scales_the_covariate_prior(traced):
    """eq 2: p_ct concentration must be β·ω_h(m) — the SAMPLED ω under ct_to_c.

    Asserted as an elementwise identity against the same trace, which pins three
    things at once: the scale (β), the source tensor (the sampled ``p_c``, not the
    static empirical prior), and the index map (``ct_to_c`` = h(m)). A scalar
    "totals ≈ β" check cannot do this — every simplex row totals 1, so any tensor
    under any permutation would satisfy it while the hierarchy is severed.
    """
    model, m_trace, _, _, _ = traced
    mod = model.module
    conc = _unwrap(m_trace.nodes["p_ct"]["fn"]).concentration
    omega = m_trace.nodes["p_c"]["value"]  # the sampled ω_c from THIS trace
    beta = float(mod.local_scale)

    expected = torch.clamp(beta * (omega[mod.ct_to_c] + mod.eps), min=1e-3)
    assert torch.allclose(conc, expected, rtol=1e-5, atol=1e-6), (
        "p_ct concentration is not β·ω_h(m) built from the sampled p_c under "
        f"ct_to_c (max|diff|={float((conc - expected).abs().max()):.3e}). "
        "eq 2 is hierarchical: p_ct's concentration is β·ω_h(m) built from the SAMPLED ω_c under "
        "ct_to_c, not from the static empirical prior and not under another index map."
    )


def test_alignment_factor_is_a_negative_kl(traced):
    """Inference Details: the surrogate must ENTER as −γ·KL (a penalty), never +γ·KL."""
    _, m_trace, _, _, _ = traced
    node = m_trace.nodes["phenotype_alignment"]
    val = node["fn"].log_factor if hasattr(node["fn"], "log_factor") else node["value"]
    val = torch.as_tensor(val)
    assert torch.all(val <= 1e-6), (
        f"phenotype_alignment carries a positive log-factor (max={float(val.max()):.4e}). "
        "the surrogate is −γ·KL(probs‖ϕ) ≤ 0: SVI maximises the log-joint and the KL is a penalty."
    )
    # and it must be non-trivial (a zero factor trains nothing)
    assert float(val.abs().sum()) > 0, (
        "phenotype_alignment is identically zero — f_cls would receive no gradient "
        "(this is exactly the bug the surrogate exists to fix)."
    )


def test_alignment_target_uses_global_indices(traced):
    """The ϕ target must be indexed by GLOBAL cell indices, not the local plate index.

    Checked *behaviorally*: trace a minibatch whose global indices differ from the
    local plate positions (0..B−1), then recompute the surrogate from the global
    map and compare to the traced factor. Under the local-index bug the two differ.
    A source-text assertion cannot do this — it is defeated by any rename or by
    routing the same wrong lookup through ``index_select``.
    """
    model, _, _, _, _ = traced
    mod = model.module

    # a batch whose global indices are NOT 0..B-1 (so local != global)
    loader = model._make_data_loader(adata=model.adata, batch_size=16, shuffle=False)
    batches = list(loader)
    assert len(batches) >= 2, "need >1 batch for local-vs-global to differ"
    args, kwargs = mod._get_fn_args_from_batch(batches[1])
    global_idx = args[3]
    assert not torch.equal(
        global_idx, torch.arange(global_idx.numel(), device=global_idx.device)
    ), "fixture batch must have global indices != local positions"

    # eval mode: classifier dropout is stochastic in train mode, which would make
    # the recomputation below irreproducible. Restored afterwards.
    was_training = mod.training
    mod.eval()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tr = poutine.trace(mod.model).get_trace(*args, **kwargs)
    finally:
        if was_training:
            mod.train()

    z = tr.nodes["latent"]["value"]
    p_ct = tr.nodes["p_ct"]["value"]
    live = torch.as_tensor(tr.nodes["phenotype_alignment"]["fn"].log_factor).detach()
    # the head reads the posterior MEAN, not the sampled latent (see model()); the
    # sample-based recompute is kept below as the mutation guard for that
    with torch.no_grad():
        mod.eval()
        z_mean, _ = encoder_posterior(mod.encoder, args[0], args[1])
        if was_training:
            mod.train()

    def _surrogate(ct_index, z_in=z_mean):
        phi = p_ct[ct_index].detach()
        log_phi = torch.log(phi + 1e-8)
        with torch.no_grad():
            mod.eval()
            logits = mod.classifier(z_in)
            if was_training:
                mod.train()
        ell = (
            mod.gate_prob * logits + (1.0 - mod.gate_prob) * log_phi
            if mod.gate_prob is not None
            else logits + log_phi
        )
        probs = torch.softmax(ell, dim=-1)
        kl = (probs * (torch.log(probs + 1e-8) - log_phi)).sum(-1)
        return (-mod.phenotype_kl_weight * kl).detach()

    expected_global = _surrogate(mod.ct_array[global_idx])
    local_idx = torch.arange(global_idx.numel(), device=global_idx.device)
    expected_local = _surrogate(mod.ct_array[local_idx])
    expected_from_sample = _surrogate(mod.ct_array[global_idx], z_in=z)
    assert not torch.allclose(live, expected_from_sample, rtol=1e-4, atol=1e-5), (
        "the phenotype_alignment factor was computed from the SAMPLED latent; the head must "
        "read the posterior mean (encoder_posterior), as predict()/to_anndata() do -- on a "
        "sample its input is ~1% signal and it collapses to a constant"
    )

    # the local-index variant must be a genuinely different target, or this
    # fixture cannot discriminate and the test would be vacuous
    assert not torch.allclose(expected_global, expected_local, rtol=1e-4, atol=1e-6), (
        "fixture cannot distinguish global from local indexing — strengthen it."
    )
    assert torch.allclose(live, expected_global, rtol=1e-4, atol=1e-5), (
        "the phenotype_alignment target does not match the GLOBAL-index mapping "
        f"(max|diff| vs global={float((live - expected_global).abs().max()):.3e}, "
        f"vs local={float((live - expected_local).abs().max()):.3e}). "
        "the target ϕ_g(i) must be indexed by GLOBAL cell indices, never the local plate index, "
        "which scrambles targets across shuffled minibatches."
    )


@pytest.mark.parametrize(
    "gate,expect", [(1.0, "classifier"), (0.0, "prior")]
)
def test_gate_rule_endpoints(gate, expect):
    """eq 4: π=1 ⇒ predict is the pure classifier; π=0 ⇒ the pure clonotype prior."""
    import torch.nn.functional as F

    pyro.clear_param_store()
    np.random.seed(0)
    torch.manual_seed(0)
    pyro.set_rng_seed(0)

    rng = np.random.default_rng(0)
    n_cells, n_genes = 40, 5
    X = rng.poisson(1.0, size=(n_cells, n_genes)).astype("float32")
    obs = pd.DataFrame(
        {
            "clone": [f"c{i % 4}" for i in range(n_cells)],
            "phen": [f"p{i % 3}" for i in range(n_cells)],
            "cov": ["a"] * n_cells,
            "pt": ["P1"] * n_cells,
        }
    )
    ad = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    ad.layers["counts"] = ad.X.copy()
    TCRIModel.setup_anndata(
        ad, layer="counts", clonotype_key="clone", phenotype_key="phen",
        covariate_key="cov", batch_key="pt",
    )
    m = TCRIModel(
        ad, n_latent=4, n_hidden=8, n_layers=1, classifier_n_layers=1,
        classifier_hidden=8, K=3, n_pseudo_obs=2, gate_prob=gate,
    )
    # run the guide once so q_p_ct_raw exists in the global param store (get_p_ct
    # reads it); no training needed — this is a formula identity, not a fit.
    loader = m._make_data_loader(adata=m.adata, batch_size=32, shuffle=False)
    g_args, g_kwargs = m.module._get_fn_args_from_batch(next(iter(loader)))
    with contextlib.redirect_stdout(io.StringIO()):
        poutine.trace(m.module.guide).get_trace(*g_args, **g_kwargs)
    m.module.eval()

    probs = m.predict(ad).values
    x = torch.tensor(ad.layers["counts"])
    b = torch.zeros(x.shape[0], 1)
    with torch.no_grad():
        z_loc, _, _ = m.module.encoder(x, b)
        cls = F.softmax(m.module.classifier(z_loc), dim=-1).numpy()
        p_ct = m.module.get_p_ct()
        prior = F.softmax(
            torch.log(p_ct[m.module.ct_array] + 1e-8), dim=-1
        ).numpy()

    target = cls if expect == "classifier" else prior
    np.testing.assert_allclose(probs, target, atol=1e-4, err_msg=(
        f"gate_prob={gate} must reduce predict() to the pure {expect}. "
        "eq 4: ℓ_i = π·f_cls + (1−π)·log ϕ; π=1 is the pure head, π=0 the pure group prior."
    ))
    pyro.clear_param_store()


def test_latent_scale_is_the_encoder_std(traced):
    """eq 6 / eq 3: one σ. The guide's q(z|x) scale and every VampPrior component scale must be
    the square root of the encoder's variance output, under the same clamp.

    scvi's ``Encoder`` returns ``(mean, VARIANCE, sample)``. The guide used the variance as the
    Normal scale (q(z|x) = N(μ, σ⁴)) and the VampPrior used ``sqrt(exp(variance))`` (never below
    1), so eq 3's "mixture of encoder posteriors" was not built from the encoder posterior and
    the latent KL compared two parameterisations. Both fail on the parent commit. Traced in
    eval mode so encoder dropout does not make the comparison stochastic.
    """
    model, _, _, args, kwargs = traced
    mod = model.module
    x, batch_idx = args[0], args[1]
    why = (
        "q(z|x) = N(μ, diag(σ²)) and the VampPrior p(z) = (1/B) Σ_k q(z|u_k) share ONE σ. "
        "scvi's Encoder returns (mean, VARIANCE, sample), so σ = sqrt(variance) in the guide "
        "AND in every VampPrior component, through the same clamp (`encoder_posterior`). "
        "Using the variance as the scale makes q(z|x) = N(μ, σ⁴); sqrt(exp(variance)) in the "
        "prior makes a scale that can never fall below 1. Either way the prior is not the "
        "mixture of encoder posteriors, and the latent KL compares two parameterisations."
    )

    was_training = mod.training
    mod.eval()
    try:
        with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
            g_trace = poutine.trace(mod.guide).get_trace(*args, **kwargs)
            live = _unwrap(g_trace.nodes["latent"]["fn"])
            loc, var, _ = mod.encoder(x, batch_idx)
            expected = torch.clamp(var.sqrt(), min=1e-3, max=10.0)
            torch.testing.assert_close(live.loc, loc, msg=f"guide latent loc is not the encoder mean. {why}")
            torch.testing.assert_close(live.scale, expected, msg=f"guide latent scale is not sqrt(encoder variance). {why}")

            u = mod.vamp_prior.pseudo_inputs
            dummy = torch.zeros(u.shape[0], 1, dtype=torch.long, device=u.device)
            p_loc, p_var, _ = mod.encoder(u, dummy)
            comp = _unwrap(mod.vamp_prior.get_mixture().component_distribution)
            torch.testing.assert_close(comp.loc, p_loc, msg=f"VampPrior component loc is not the encoder mean at the pseudo-inputs. {why}")
            torch.testing.assert_close(
                comp.scale, torch.clamp(p_var.sqrt(), min=1e-3, max=10.0),
                msg=f"VampPrior component scale is not sqrt(encoder variance) under the guide's clamp. {why}",
            )
    finally:
        if was_training:
            mod.train()


def test_data_plate_is_scaled_to_the_dataset(traced):
    """eq 7 sums the per-cell terms over N cells and the two Dirichlet KLs once, so a
    minibatch is an unbiased estimate only if the data plate carries ``size = N`` with the
    batch as its subsample. Checked on a live trace of a batch with ``B < N``: the per-cell
    sites carry scale ``N/B`` (times their own poutine scale), the global sites carry 1.

    Declared at ``size = B`` every site read scale 1 and the assertion below fails at the
    first line. The second pass sets ``n_obs_training`` to a smaller number and asserts the
    scale follows it, which is what makes the training split -- not the whole object -- the
    reference.
    """
    model, _, _, _, _ = traced
    mod = model.module
    n = int(mod.n_cells)
    loader = model._make_data_loader(adata=model.adata, batch_size=16, shuffle=False)
    args, kwargs = mod._get_fn_args_from_batch(next(iter(loader)))
    b = int(args[0].shape[0])
    assert 0 < b < n

    for size in (None, n - 6):
        prev = mod.n_obs_training
        mod.n_obs_training = size
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mt = poutine.trace(mod.model).get_trace(*args, **kwargs)
                gt = poutine.trace(mod.guide).get_trace(*args, **kwargs)
        finally:
            mod.n_obs_training = prev
        expect = (size or n) / b
        msg = ("the data plate must carry size = N (the cells being fit) with the minibatch as "
               "its subsample, so per-cell sites are scaled by N/B and p_c/p_ct keep scale 1; "
               "otherwise the Dirichlet KLs are counted once per step instead of once per pass")
        assert float(mt.nodes["phenotype_alignment"]["scale"]) == pytest.approx(expect), msg
        assert float(mt.nodes["obs"]["scale"]) == pytest.approx(
            expect * mod.reconstruction_loss_scale), msg
        assert float(mt.nodes["latent"]["scale"]) == pytest.approx(expect * mod.kl_weight), msg
        assert float(gt.nodes["latent"]["scale"]) == pytest.approx(expect * mod.kl_weight), msg
        for name in ("p_c", "p_ct"):
            assert float(mt.nodes[name]["scale"]) == pytest.approx(1.0), msg
            assert float(gt.nodes[name]["scale"]) == pytest.approx(1.0), msg
