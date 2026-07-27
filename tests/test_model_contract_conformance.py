"""Model-contract conformance — the guardrail for the model's *mathematics*.

``tcri/model/_model_contract.py`` freezes the probabilistic structure of
Supplementary Note 1 (prose: ``docs/contract/MODEL_CONTRACT.md``). This test traces
the live ``TCRIModule.model``/``.guide`` and asserts they match that manifest
exactly — declared sites present with the right distribution family / plate /
event-dim, **no undeclared sites**, the guide's variational family intact, and the
semantic invariants (α and β scaling, the surrogate's sign, the gate rule, global
alignment indices) holding.

If you changed the model and this test fails: **update the contract first**
(``_model_contract.py`` + ``MODEL_CONTRACT.md``, citing the note), then make the
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
from tcri.model import _model_contract as MC

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
            "If you removed it, update tcri/model/_model_contract.py + "
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
        "changes the joint p(Ω,Φ,z,x). Declare it in tcri/model/_model_contract.py "
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
    """eq 2: p_ct prior concentration must scale with β (local_scale)."""
    model, m_trace, _, _, _ = traced
    conc = _unwrap(m_trace.nodes["p_ct"]["fn"]).concentration
    beta = float(model.module.local_scale)
    live_total = float(conc.sum(-1).mean())
    assert live_total == pytest.approx(beta, rel=0.05), (
        f"p_ct concentration totals {live_total:.4f}, expected ≈β={beta}. "
        f"{MC.SEMANTIC_INVARIANTS['beta_scales_covariate_prior']}"
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


def test_alignment_target_uses_global_indices():
    """The ϕ target must be indexed by global cell indices, not the local plate index."""
    import inspect

    from tcri.model._module import TCRIModule

    src = inspect.getsource(TCRIModule.model)
    assert "self.ct_array[indices]" in src, (
        "model() must map cells to their clonotype×covariate group with the GLOBAL "
        f"`indices`. {MC.SEMANTIC_INVARIANTS['alignment_target_uses_global_indices']}"
    )
    assert "self.ct_array[idx]" not in src, (
        "model() indexes ct_array with the LOCAL plate index `idx`. "
        f"{MC.SEMANTIC_INVARIANTS['alignment_target_uses_global_indices']}"
    )
    # the batch must actually carry them
    fn_src = inspect.getsource(TCRIModule._get_fn_args_from_batch)
    assert "indices" in fn_src, (
        "_get_fn_args_from_batch must pass global cell indices through to model()/guide()."
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
            f"sanctioned deviation '{key}' is declared in _model_contract.py but not "
            f"documented in {doc.name}. Keep the machine contract and the prose in sync."
        )
