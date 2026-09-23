"""Two models in one process, and the store keys that keep them apart.

Pyro's parameter store is process-global, so two `TCRIModule`s registered under the same names
share one fit: the second overwrites the first, and the first goes on predicting from
parameters it never learned. `name` is what separates them -- a model owns exactly the store
keys under `f"{name}."`, and `""` is the unnamed layout that a session saved before namespaces
existed restores into.

These are the tests a null rests on, since a permutation reference is a second fit of the same
model living beside its parent (`governance/MODEL_CONTRACT.md`, "A null is this model").

Fits are deliberately tiny -- the assertions are about names and isolation, not accuracy.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pyro
import pytest
import torch

from tcri.datasets import simulate_tcri
from tcri.model._model import TCRIModel, _owns_param


def _adata(seed):
    a = simulate_tcri(n_clones=6, n_phenotypes=3, n_genes=20, n_cells=120,
                      n_covariates=2, omega_concentration=0.4, seed=seed)
    a.layers["counts"] = a.X.copy()
    return a


def _fit(adata, name, *, seed=0, max_epochs=3):
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="covariate",
                            batch_key="batch")
    model = TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=16, K=3, seed=seed, name=name)
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=max_epochs, batch_size=64, n_steps_kl_warmup=4,
                    accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    return model


@pytest.fixture(autouse=True)
def _own_the_store():
    """These tests are about the store, so they take it and give it back."""
    store = pyro.get_param_store()
    saved = store.get_state()
    store.clear()
    yield
    store.clear()
    store.set_state(saved)


def _keys():
    return set(pyro.get_param_store().keys())


# ── the guarantee ────────────────────────────────────────────────────────────

def test_two_models_coexist():
    """Fit `a`, fit `b`, and `a` still predicts what it predicted. No `clear_param_store`.

    This is the whole point of `name`. Sharing store keys, `b`'s guide writes `q_p_ct_raw` over
    `a`'s and `a.predict()` returns numbers from `b`'s fit -- loudly, as an `IndexError`, when
    the two have different phenotype counts, and silently when they do not.
    """
    a_data, b_data = _adata(0), _adata(1)
    a = _fit(a_data, "a", seed=0)
    before = a.predict(a_data).to_numpy()

    b = _fit(b_data, "b", seed=1)
    after = a.predict(a_data).to_numpy()

    np.testing.assert_array_equal(before, after)
    assert b.predict(b_data).shape[0] == b_data.n_obs

    keys = _keys()
    assert keys, "the store is empty; the models registered nothing"
    assert all(k.startswith(("a.", "b.")) for k in keys), sorted(keys)[:5]
    for owner in ("a", "b"):
        assert any(_owns_param(owner, k) for k in keys), f"{owner} owns nothing"


def test_fitting_b_does_not_move_a():
    """The gradient-leak test: every leaf `a` owns is bitwise unchanged by `b`'s fit.

    `predict()` compares an output and could in principle agree by luck; this compares the
    parameters themselves, and it is the assertion a shared encoder or a double
    `pyro.module` registration would fail.
    """
    a_data, b_data = _adata(0), _adata(1)
    a = _fit(a_data, "a", seed=0)

    store = pyro.get_param_store()
    before = {n: p.detach().clone() for n, p in store.named_parameters() if _owns_param("a", n)}
    assert before, "a registered no parameters"
    a_state = {k: v.detach().clone() for k, v in a.module.state_dict().items()}

    _fit(b_data, "b", seed=1)

    for n, p in store.named_parameters():
        if _owns_param("a", n):
            assert torch.equal(p.detach(), before[n]), f"b's fit moved {n}"
    for k, v in a.module.state_dict().items():
        assert torch.equal(v, a_state[k]), f"b's fit moved a.module.{k}"


def test_unnamed_is_the_legacy_layout():
    """`name=""` registers exactly the names a session saved before namespaces existed was
    written under.

    Pinned as a literal set rather than a property, because this is a compatibility claim about
    files on disk: such a session restores by key, so a rename here makes every one of them
    silently load into nothing.
    """
    model = _fit(_adata(0), "", seed=0)
    keys = _keys()

    assert "q_p_c_raw" in keys and "q_p_ct_raw" in keys
    assert all(k in ("q_p_c_raw", "q_p_ct_raw") or k.startswith("scvi$$$") for k in keys), (
        f"unnamed model registered something outside the legacy layout: "
        f"{sorted(k for k in keys if k not in ('q_p_c_raw', 'q_p_ct_raw') and not k.startswith('scvi$$$'))}"
    )
    assert model.name == "" and model.module.pname("q_p_ct_raw") == "q_p_ct_raw"


def test_ownership_is_not_a_bare_prefix_test():
    """`""` must not own every named model's keys.

    Written as a unit test on the predicate because the obvious implementation --
    `key.startswith(name)` -- is correct for every named model and wrong only for the
    unnamed one, which is the default. It would make the constructor warn on every second
    model and make the best-weight snapshot restore another fit's concentrations.
    """
    assert _owns_param("", "q_p_ct_raw") and _owns_param("", "scvi$$$encoder.weight")
    assert not _owns_param("", "a.q_p_ct_raw"), "the unnamed model claimed a named model's key"
    assert _owns_param("a", "a.q_p_ct_raw")
    assert not _owns_param("a", "q_p_ct_raw")
    assert not _owns_param("a", "ab.q_p_ct_raw"), "prefix match without the separator"

    # ...and it must not claim DOWNWARD either. A null's namespace is `f"{parent}.null.{kind}"`,
    # so plain `key.startswith(f"{name}.")` -- which is what the unnamed case above requires --
    # makes a parent own its own nulls' parameters and warn that it is about to continue its
    # own fit.
    assert not _owns_param("a", "a.null.phenotype.q_p_ct_raw"), "a parent claimed its null"
    assert not _owns_param("", "null.phenotype.q_p_ct_raw"), "an unnamed parent claimed its null"
    assert _owns_param("a.null.phenotype", "a.null.phenotype.q_p_ct_raw")
    assert _owns_param("null.phenotype", "null.phenotype.scvi$$$px_r")


def test_session_round_trip_keeps_the_namespace(tmp_path):
    """Save and load a named model: its keys come back under its own namespace, and the
    prediction is bitwise identical."""
    from tcri.utils import load_tcri_session, save_tcri_session

    adata = _adata(0)
    model = _fit(adata, "a", seed=0)
    model.to_anndata(adata)
    before = model.predict(adata).to_numpy()

    with contextlib.redirect_stdout(io.StringIO()):
        save_tcri_session(model, adata, str(tmp_path / "run"))
    pyro.clear_param_store()
    with contextlib.redirect_stdout(io.StringIO()):
        loaded, loaded_adata = load_tcri_session(str(tmp_path / "run"), map_location="cpu")

    assert loaded.name == "a"
    assert all(k.startswith("a.") for k in _keys()), sorted(_keys())[:5]
    np.testing.assert_allclose(loaded.predict(loaded_adata).to_numpy(), before, atol=1e-6)


def test_the_snapshot_restores_only_its_own_concentrations():
    """`BestObjectiveSnapshot` keys on the module's namespace, not on "everything but the
    networks".

    Filtering by `not name.startswith("scvi$$$")` -- the whole store minus the networks -- is
    indistinguishable from the right answer while one model exists per process, and a silent
    cross-fit restore the moment two do.
    """
    from tcri.model._callbacks import _is_own_guide_param

    class _M:
        name = "a"

    m = _M()
    assert _is_own_guide_param(m, "a.q_p_ct_raw") and _is_own_guide_param(m, "a.q_p_c_raw")
    assert not _is_own_guide_param(m, "b.q_p_ct_raw"), "would restore another fit's parameters"
    assert not _is_own_guide_param(m, "a.scvi$$$encoder.weight"), "a network parameter is not a concentration"

    m.name = ""
    assert _is_own_guide_param(m, "q_p_ct_raw")
    assert not _is_own_guide_param(m, "a.q_p_ct_raw")
    assert not _is_own_guide_param(m, "scvi$$$encoder.weight")


def test_the_weight_decay_exemption_survives_a_namespace():
    """B8's exemption matches the parameter's tail, so a named model keeps it.

    An exact match against `{"q_p_c_raw", "q_p_ct_raw"}` passes every test in the suite that
    builds unnamed models and silently reinstates the flat-Dirichlet prior for every named one
    -- the effect `governance/TRAINING_CONTRACT.md` B8 exists to rule out.
    """
    from tcri.model._training import _per_param_optim_args

    args = _per_param_optim_args({"lr": 1e-3, "weight_decay": 1e-4})
    for name in ("q_p_ct_raw", "q_p_c_raw", "a.q_p_ct_raw", "a.null.phenotype.q_p_c_raw"):
        assert args(name)["weight_decay"] == 0.0, f"{name} lost the exemption"
    for name in ("scvi.encoder.weight", "a.scvi.encoder.weight", "a.px_r"):
        assert args(name)["weight_decay"] == 1e-4, f"{name} wrongly exempted"
