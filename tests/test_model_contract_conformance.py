"""Model-contract conformance — the guardrail for the model's *mathematics*.

``tests/contracts/model.py`` freezes the probabilistic structure of
Supplementary Note 1 (prose: ``docs/contract/MODEL_CONTRACT.md``). This test traces
the live ``TCRIModule.model``/``.guide`` and asserts they match that manifest
exactly — declared sites present with the right distribution family / plate /
event-dim, **no undeclared sites**, the guide's variational family intact, and the
semantic invariants (α and β scaling, the surrogate's sign, the gate rule, global
alignment indices) holding.

If you changed the model and this test fails: **update the contract first**
(``tests/contracts/model.py`` + ``MODEL_CONTRACT.md``, citing the note), then make the
code agree. Do not loosen the manifest to match new code — that silently rewrites
the model the package claims to implement.

Sibling of ``test_contract_conformance.py`` (which does the same for the public API).
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
from tests.contracts import model as MC

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

    for spec in MC.GENERATIVE_SITES:
        assert spec.name in live, (
            f"model() is missing the declared site '{spec.name}' (note eq {spec.eq}). "
            "If you removed it, update tests/contracts/model.py + "
            "docs/contract/MODEL_CONTRACT.md first."
        )
        node = live[spec.name]
        base = _unwrap(node["fn"])
        assert base is not None and type(base).__name__ == spec.dist, (
            f"site '{spec.name}' (eq {spec.eq}) is "
            f"{type(base).__name__ if base is not None else None}, contract says "
            f"{spec.dist}. {spec.note}"
        )
        if spec.plate is not None:
            assert spec.plate in _plates(node), (
                f"site '{spec.name}' must live in plate '{spec.plate}' "
                f"({MC.PLATES.get(spec.plate, '')}); found {_plates(node)}."
            )
        assert bool(node.get("is_observed")) == spec.observed, (
            f"site '{spec.name}' observed={bool(node.get('is_observed'))}, "
            f"contract says observed={spec.observed}."
        )
        if spec.event_dim is not None:
            assert len(node["fn"].event_shape) == spec.event_dim, (
                f"site '{spec.name}' event_dim={len(node['fn'].event_shape)}, "
                f"contract says {spec.event_dim}."
            )


def test_no_undeclared_generative_sites(traced):
    """An EXTRA stochastic site changes the joint distribution — the contract must say so."""
    _, m_trace, _, _, _ = traced
    live = set(_stochastic(m_trace))
    declared = {s.name for s in MC.GENERATIVE_SITES}
    extra = live - declared
    assert not extra, (
        f"model() has undeclared stochastic site(s): {sorted(extra)}. Every site "
        "changes the joint p(Ω,Φ,z,x). Declare it in tests/contracts/model.py "
        "(with its note equation) and document it in docs/contract/MODEL_CONTRACT.md."
    )


# ── structure: the variational family ────────────────────────────────────────

def test_guide_family_matches_contract(traced):
    _, _, g_trace, _, _ = traced
    live = _stochastic(g_trace)

    for spec in MC.GUIDE_SITES:
        assert spec.name in live, (
            f"guide() is missing q({spec.name}) (eq 6). {spec.note}"
        )
        base = _unwrap(live[spec.name]["fn"])
        assert base is not None and type(base).__name__ == spec.dist, (
            f"q({spec.name}) is {type(base).__name__ if base is not None else None}, "
            f"contract says {spec.dist}. {spec.note}"
        )

    declared = {s.name for s in MC.GUIDE_SITES}
    extra = set(live) - declared
    assert not extra, (
        f"guide() has undeclared site(s): {sorted(extra)}. This changes the "
        "variational family (eq 6) and therefore the ELBO."
    )


def test_guide_registers_variational_params(traced):
    _, _, g_trace, _, _ = traced
    params = {n for n, nd in g_trace.nodes.items() if nd["type"] == "param"}
    for p in MC.GUIDE_PARAMS:
        assert p in params, (
            f"guide() must register the learnable variational parameter '{p}' "
            "(λ_c / λ'_m of eq 6); without it the Dirichlet posteriors are not learned."
        )


def test_discrete_phenotype_latent_is_not_sampled(traced):
    """z^ϕ is replaced by the surrogate; a q(z^ϕ) site would change the objective."""
    _, _, g_trace, _, _ = traced
    live = set(_stochastic(g_trace))
    for forbidden in MC.FORBIDDEN_GUIDE_SITES:
        assert forbidden not in live, (
            f"guide() samples '{forbidden}', but the note's Inference Details replace "
            "the discrete z^ϕ with the phenotype_alignment surrogate. Re-introducing "
            "it changes the optimized objective — update the contract first."
        )


def test_discrete_phenotype_latent_is_not_observed_either(traced):
    """z^ϕ is LATENT. Not sampled in the guide, and not conditioned on in the model.

    The hierarchical branch (ω_c -> ϕ_m -> z^ϕ) is a prior over phenotype composition that
    never sees x directly; data reaches it only through z. DE-18 read that absence as a
    missing data term and conditioned the site on the input phenotype labels, which makes
    the model supervised and every metric partly a readout of its own input. Withdrawn.
    """
    _, m_trace, _, _, _ = traced
    observed = {
        name for name, site in m_trace.nodes.items()
        if site["type"] == "sample" and site.get("is_observed")
    }
    for forbidden in MC.FORBIDDEN_MODEL_SITES:
        assert forbidden not in observed, (
            f"model() conditions on '{forbidden}'. z^ϕ is latent — the note replaces it "
            f"with the phenotype_alignment surrogate rather than observing the input "
            f"labels. See DE-18 (WITHDRAWN) in DEFECTS.md before re-adding this."
        )
    assert "obs" in observed, "the ZINB likelihood is the model's only observation (eq 5)"


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
        f"(α={alpha} × archetype). {MC.SEMANTIC_INVARIANTS['alpha_scales_clonotype_prior']}"
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
        f"{MC.SEMANTIC_INVARIANTS['beta_scales_covariate_prior']} "
        f"{MC.SEMANTIC_INVARIANTS['hierarchy_ct_depends_on_c']}"
    )


def test_alignment_factor_is_a_negative_kl(traced):
    """Inference Details: the surrogate must ENTER as −γ·KL (a penalty), never +γ·KL."""
    _, m_trace, _, _, _ = traced
    node = m_trace.nodes["phenotype_alignment"]
    val = node["fn"].log_factor if hasattr(node["fn"], "log_factor") else node["value"]
    val = torch.as_tensor(val)
    assert torch.all(val <= 1e-6), (
        f"phenotype_alignment carries a positive log-factor (max={float(val.max()):.4e}). "
        f"{MC.SEMANTIC_INVARIANTS['factor_is_negative_kl']}"
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

    def _surrogate(ct_index):
        phi = p_ct[ct_index].detach()
        log_phi = torch.log(phi + 1e-8)
        with torch.no_grad():
            mod.eval()
            logits = mod.classifier(z)
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

    # the local-index variant must be a genuinely different target, or this
    # fixture cannot discriminate and the test would be vacuous
    assert not torch.allclose(expected_global, expected_local, rtol=1e-4, atol=1e-6), (
        "fixture cannot distinguish global from local indexing — strengthen it."
    )
    assert torch.allclose(live, expected_global, rtol=1e-4, atol=1e-5), (
        "the phenotype_alignment target does not match the GLOBAL-index mapping "
        f"(max|diff| vs global={float((live - expected_global).abs().max()):.3e}, "
        f"vs local={float((live - expected_local).abs().max()):.3e}). "
        f"{MC.SEMANTIC_INVARIANTS['alignment_target_uses_global_indices']}"
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
        f"{MC.SEMANTIC_INVARIANTS['gate_mixes_classifier_and_prior']}"
    ))
    pyro.clear_param_store()


# ── the contract must stay in sync with its prose ────────────────────────────

def test_sanctioned_deviations_are_documented():
    """Every accepted departure from the note must be spelled out in the prose contract."""
    from pathlib import Path

    import tcri

    doc = Path(tcri.__file__).parent.parent / "docs" / "contract" / "MODEL_CONTRACT.md"
    assert doc.exists(), f"missing prose model contract: {doc}"
    text = doc.read_text()
    for key in MC.SANCTIONED_DEVIATIONS:
        assert key in text, (
            f"sanctioned deviation '{key}' is declared in tests/contracts/model.py but not "
            f"documented in {doc.name}. Keep the machine contract and the prose in sync."
        )
