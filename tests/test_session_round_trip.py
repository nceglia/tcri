"""Session save/load round-trip, and the split between what ``setup_anndata`` and
``to_anndata`` may write.

Locks three things:
  1. ``to_anndata`` writes exactly the canonical key set (latent / logits / probabilities /
     gate / classifier-temperature / local-scale), and no ``tcri_manager`` stash.
  2. ``setup_anndata`` is registration-only — no analysis/label ``obs`` mutation.
  3. save → load reproduces ``p_ct`` + latent + ``predict`` from the *reloaded model*, checked
     against the reloaded model rather than against the saved file, with uniform rows of
     ``q_p_ct_raw`` caught via row variance.
"""
import contextlib
import io
import json
import warnings

import anndata
import numpy as np
import pyro
import pytest
import torch

import tcri
from tcri._state import keys as K
from tcri.model._model import TCRIModel
from tcri.utils._utils import SESSION_FORMAT_VERSION
from tcri.utils._utils import load_tcri_session, save_tcri_session


@pytest.fixture
def fresh_trained_model(synthetic_adata):
    """A model trained fresh inside this test so it OWNS the process-global pyro
    param store for the save: loading or training a second model in one process overwrites
    ``q_p_ct_raw``, so the shared session-scoped ``trained_model`` store is unsafe for a
    round-trip that recomputes from the reloaded model."""
    from tcri.model._model import TCRIModel

    pyro.clear_param_store()
    adata = synthetic_adata.copy()
    TCRIModel.setup_anndata(
        adata, clonotype_key="unique_clone_id", phenotype_key="phenotype_col",
        covariate_key="timepoint", batch_key="patient",
    )
    model = TCRIModel(
        adata, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
        classifier_hidden=16, K=3, n_pseudo_obs=3,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=50, batch_size=64, enable_progress_bar=False,
                    enable_model_summary=False)
        model.to_anndata(adata)
    return model, adata

CANONICAL_UNS = [
    K.METADATA, K.P_CT, K.CT_TO_COV, K.CT_TO_C, K.CT_ARRAY, K.COV_ARRAY,
    K.LOCAL_SCALE, K.GATE_PROB, K.CLASSIFIER_TEMPERATURE,
    K.COVARIATE_CATEGORIES, K.CLONOTYPE_CATEGORIES, K.PHENOTYPE_CATEGORIES,
]
CANONICAL_OBSM = [K.X_TCRI, K.X_LOGITS, K.X_LOGPOSTERIOR, K.X_PROBABILITIES]


def test_to_anndata_writes_canonical_set(trained_model):
    """to_anndata writes the full canonical key set, with the scalar knobs equal to the
    model's configured values; no manager stash."""
    model, adata = trained_model
    for k in CANONICAL_UNS:
        assert k in adata.uns, f"to_anndata did not write uns[{k}]"
    for k in CANONICAL_OBSM:
        assert k in adata.obsm, f"to_anndata did not write obsm[{k}]"
    assert K.PHENOTYPE in adata.obs, "to_anndata did not write the hard-label obs"
    assert "tcri_manager" not in adata.uns, "manager stash was not retired"
    # the scalar knobs: written == the model's configured value (not just finite)
    assert adata.uns[K.LOCAL_SCALE] == float(model.module.local_scale)
    assert adata.uns[K.CLASSIFIER_TEMPERATURE] == float(model.module.classifier_temperature)
    gp = model.module.gate_prob
    if gp is None:
        assert np.isnan(adata.uns[K.GATE_PROB])  # no gate -> nan sentinel (serializable)
    else:
        assert adata.uns[K.GATE_PROB] == float(gp)


def test_setup_anndata_leaves_analysis_obs_untouched(synthetic_adata):
    """setup_anndata is registration-only: it may add the 'indices' glue column but
    writes no analysis/label obs (that is exclusively to_anndata's job)."""
    from tcri.model._model import TCRIModel

    adata = synthetic_adata.copy()
    obs_before = set(adata.obs.columns)
    TCRIModel.setup_anndata(
        adata, clonotype_key="unique_clone_id", phenotype_key="phenotype_col",
        covariate_key="timepoint", batch_key="patient",
    )
    new_cols = set(adata.obs.columns) - obs_before
    # registration glue is allowed ('indices' + scvi's internal '_scvi_*' columns);
    # any OTHER new obs column would be analysis/label leakage.
    non_glue = {c for c in new_cols if c != "indices" and not c.startswith("_scvi")}
    assert not non_glue, f"setup_anndata mutated analysis obs: {non_glue}"
    assert K.PHENOTYPE not in adata.obs, "setup_anndata must not write hard labels"
    assert "tcri_manager" not in adata.uns


def test_session_round_trip(fresh_trained_model, tmp_path):
    model, adata = fresh_trained_model

    out_dir = tmp_path / "session"
    save_tcri_session(model, adata, str(out_dir))

    pyro.clear_param_store()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        loaded_model, loaded = load_tcri_session(str(out_dir))

    # 1) serialization: the saved AnnData survives the h5ad round-trip
    np.testing.assert_array_equal(
        adata.obsm[K.X_PROBABILITIES], loaded.obsm[K.X_PROBABILITIES]
    )
    np.testing.assert_allclose(adata.obsm[K.X_TCRI], loaded.obsm[K.X_TCRI], atol=1e-6)
    np.testing.assert_allclose(adata.uns[K.P_CT], loaded.uns[K.P_CT], atol=1e-6)
    assert "tcri_manager" not in loaded.uns

    # 2) pyro store restored, not silently re-initialized to a uniform prior
    store = pyro.get_param_store()
    key = loaded_model.module.pname("q_p_ct_raw")
    assert key in store, f"{key} missing from pyro store after load"
    q = store[key].detach().cpu().numpy()
    assert q.var(axis=-1).mean() > 1e-6, (
        "q_p_ct_raw rows are uniform; pyro load silently failed (weights_only regression)"
    )

    # 3) the RELOADED model reproduces what to_anndata wrote before save
    np.testing.assert_allclose(adata.uns[K.P_CT], loaded_model.get_p_ct(), atol=1e-4)
    np.testing.assert_allclose(
        adata.obsm[K.X_TCRI], loaded_model.get_latent_representation(loaded), atol=1e-4
    )
    np.testing.assert_allclose(
        adata.obsm[K.X_PROBABILITIES], loaded_model.predict(loaded).values, atol=1e-4
    )

    # 4) the model scalars carried only in init_params_ (not in the AnnData) survive
    #    the save/load — after a load they are reachable nowhere else.
    assert loaded_model.module.phenotype_kl_weight == model.module.phenotype_kl_weight
    assert loaded_model.module.gate_prob == model.module.gate_prob
    assert loaded_model.module.classifier_dropout == model.module.classifier_dropout


# ── the session format version ───────────────────────────────────────────────

def test_a_saved_session_records_its_format_and_the_tcri_that_wrote_it(fresh_trained_model, tmp_path):
    model, adata = fresh_trained_model
    out_dir = tmp_path / "session"

    save_tcri_session(model, adata, str(out_dir))

    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["format_version"] == SESSION_FORMAT_VERSION
    assert meta["versions"]["tcri"] == tcri.__version__


def test_a_session_from_a_newer_tcri_is_refused(tmp_path):
    """Refused before anything is read: a newer session may need files this tcri knows nothing of."""
    (tmp_path / "meta.json").write_text(json.dumps({
        "format_version": SESSION_FORMAT_VERSION + 1,
        "versions": {"tcri": "99.0.0"},
    }))

    with pytest.raises(ValueError, match=r"format v\d+.*99\.0\.0.*reads up to"):
        load_tcri_session(str(tmp_path))


def test_a_session_written_before_format_versions_is_not_refused(tmp_path):
    """A `meta.json` with no `format_version` predates the field; it is read, not refused.

    The load gets past the version check and stops at the missing files of this empty directory,
    which is what proves the check let it through.
    """
    (tmp_path / "meta.json").write_text(json.dumps({"train_kwargs": {"max_epochs": 3}}))

    with pytest.raises(FileNotFoundError):
        load_tcri_session(str(tmp_path))


# ── a load restores the loaded model, and only it ───────────────────────────
#
# Every model below is built WITHOUT `name=`: the default path is the one these guard. Each test
# runs through both savers, `TCRIModel.save` with `TCRIModel.load` and the session pair.

SAVERS = ("scvi", "session")


def _registered(adata, replicate=None):
    adata = adata.copy()
    TCRIModel.setup_anndata(adata, clonotype_key="unique_clone_id", phenotype_key="phenotype_col",
                            covariate_key="timepoint", batch_key="patient", replicate=replicate)
    return adata


def _fit(adata, seed):
    model = TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=16, K=3, n_pseudo_obs=3, seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=20, batch_size=64, n_epochs_kl_warmup=2,
                    enable_progress_bar=False, enable_model_summary=False)
    return model


def _save(model, adata, path, how):
    with contextlib.redirect_stdout(io.StringIO()):
        if how == "scvi":
            model.save(str(path), overwrite=True, save_anndata=False)
            adata.write_h5ad(path / "adata.h5ad")
        else:
            save_tcri_session(model, adata, str(path))


def _load(path, how):
    with contextlib.redirect_stdout(io.StringIO()):
        if how == "scvi":
            adata = anndata.read_h5ad(path / "adata.h5ad")
            return TCRIModel.load(str(path), adata=adata), adata
        return load_tcri_session(str(path))


def _edit_saved_store(path, how, edit, *, name=None):
    """Apply ``edit`` to every Pyro store saved under ``path``; ``name`` rewrites the model's
    recorded name."""
    blob = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    edit(blob["model_state_dict"]["pyro_param_store"])
    if name is not None:
        blob["attr_dict"]["init_params_"]["non_kwargs"]["name"] = name
    torch.save(blob, path / "model.pt")
    if how == "session":
        state = torch.load(path / "pyro_params.pt", map_location="cpu", weights_only=False)
        edit(state)
        torch.save(state, path / "pyro_params.pt")


def _readout(model, adata):
    store = pyro.get_param_store()
    out = {
        "p_ct": model.get_p_ct(),
        "q_p_c_raw": store[model.module.pname("q_p_c_raw")].detach().cpu().numpy(),
        "q_p_ct_raw": store[model.module.pname("q_p_ct_raw")].detach().cpu().numpy(),
        "latent": model.get_latent_representation(adata),
        "predict": model.predict(adata).to_numpy(),
    }
    ad = adata.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        model.to_anndata(ad)
    for level in ad.obs["timepoint"].cat.categories:
        res = tcri.tl.mutual_information(ad, covariate=level, null_model=None, inplace=False)
        out[f"mi_{level}"] = np.asarray(res["result"]["value"], dtype=float)
    return out


@pytest.mark.parametrize("how", SAVERS)
def test_a_reload_reproduces_the_fit(synthetic_adata, tmp_path, how):
    """save → clear → load reproduces the posteriors, the latent, ``predict`` and a metric
    exactly, under the name the model was given at construction."""
    pyro.clear_param_store()
    adata = _registered(synthetic_adata)
    model = _fit(adata, seed=0)
    before = _readout(model, adata)
    _save(model, adata, tmp_path / "m", how)
    pyro.clear_param_store()

    loaded, _ = _load(tmp_path / "m", how)

    assert loaded.name == model.name
    after = _readout(loaded, adata)
    for k, v in before.items():
        np.testing.assert_array_equal(after[k], v, err_msg=k)


@pytest.mark.parametrize("how", SAVERS)
def test_loading_a_second_model_leaves_the_first_unchanged(synthetic_adata, tmp_path, how):
    """Two models loaded into one process: the second load leaves the first one's posteriors
    bit-identical, rather than replacing them with its own."""
    adata = _registered(synthetic_adata)
    for tag, seed in (("a", 0), ("b", 1)):
        pyro.clear_param_store()
        _save(_fit(adata, seed=seed), adata, tmp_path / tag, how)
    pyro.clear_param_store()
    a, _ = _load(tmp_path / "a", how)
    expected = a.get_p_ct().copy()

    b, _ = _load(tmp_path / "b", how)

    assert not np.array_equal(b.get_p_ct(), expected), "the two fits must differ to be told apart"
    np.testing.assert_array_equal(a.get_p_ct(), expected)
    assert a.name != b.name


@pytest.mark.parametrize("how", SAVERS)
def test_training_after_a_load_moves_the_networks(synthetic_adata, tmp_path, how):
    """Training a loaded model steps the module's own parameters: the store's network entries
    are those parameters, not copies of them, and the encoder moves."""
    pyro.clear_param_store()
    adata = _registered(synthetic_adata)
    _save(_fit(adata, seed=0), adata, tmp_path / "m", how)
    pyro.clear_param_store()
    model, _ = _load(tmp_path / "m", how)
    before = {n: p.detach().clone() for n, p in model.module.named_parameters()
              if n.startswith("encoder.")}

    with contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=2, batch_size=64, n_epochs_kl_warmup=0, early_stopping=False,
                    enable_progress_bar=False, enable_model_summary=False)

    stored = dict(pyro.get_param_store().named_parameters())
    prefix = f"{model.module.pname('scvi')}$$$"
    copies = [n for n, p in model.module.named_parameters() if stored.get(prefix + n) is not p]
    assert not copies, f"store entries that are not the module's parameters: {copies[:3]}"
    moved = [n for n, p in model.module.named_parameters()
             if n in before and not torch.equal(before[n], p.detach())]
    assert moved, "no encoder parameter moved"


@pytest.mark.parametrize("how", SAVERS)
def test_a_load_touches_only_its_family(synthetic_adata, tmp_path, how):
    """A load replaces the loaded model's entries and its nulls', and nothing else.

    Straight after the load the store holds the other models' entries, unchanged, plus the
    file's family without the model's own network entries. The next guide call registers those
    from the module, and the store is then exactly the other models' entries plus the file's
    family.
    """
    pyro.clear_param_store()
    adata = _registered(synthetic_adata)
    parent = _fit(adata, seed=0)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parent.to_anndata(adata)
        tcri.null.phenotype(parent, adata)
    _save(parent, adata, tmp_path / "p", how)
    family = set(pyro.get_param_store().keys())
    own_networks = {k for k in family if k.startswith(f"{parent.name}.scvi$$$")}
    assert any(k.startswith(f"{parent.name}.null.") for k in family)
    pyro.clear_param_store()
    _fit(_registered(synthetic_adata), seed=1)
    others = dict(pyro.get_param_store().named_parameters())
    values = {k: v.detach().clone() for k, v in others.items()}

    loaded, _ = _load(tmp_path / "p", how)

    after = dict(pyro.get_param_store().named_parameters())
    assert set(after) == set(others) | (family - own_networks)
    assert all(after[k] is v and torch.equal(v.detach(), values[k]) for k, v in others.items())
    batch = next(iter(loaded._make_data_loader(loaded.adata, batch_size=8)))
    args, kwargs = loaded.module._get_fn_args_from_batch(batch)
    with torch.no_grad():
        loaded.module.guide(*args, **kwargs)
    assert set(pyro.get_param_store().keys()) == set(others) | family


@pytest.mark.parametrize("how", SAVERS)
def test_a_bare_layout_file_is_refused(synthetic_adata, tmp_path, how):
    """A store saved under bare keys, the layout of a model built without a name before every
    model had one, is refused with a message naming that layout; the store is untouched."""
    pyro.clear_param_store()
    adata = _registered(synthetic_adata)
    model = _fit(adata, seed=0)
    _save(model, adata, tmp_path / "m", how)
    prefix = f"{model.name}."

    def bare(state):
        for part in ("params", "constraints"):
            state[part] = {k[len(prefix):] if k.startswith(prefix) else k: v
                           for k, v in state[part].items()}

    _edit_saved_store(tmp_path / "m", how, bare, name="")
    pyro.clear_param_store()
    _fit(adata, seed=1)
    keys = set(pyro.get_param_store().keys())

    with pytest.raises(ValueError, match="bare names"):
        _load(tmp_path / "m", how)
    assert set(pyro.get_param_store().keys()) == keys


@pytest.mark.parametrize("how", SAVERS)
def test_a_fitted_model_without_its_posteriors_is_refused(synthetic_adata, tmp_path, how):
    """A trained model whose saved store lacks its posteriors raises, rather than loading with
    the guide's initialization in their place."""
    pyro.clear_param_store()
    adata = _registered(synthetic_adata)
    model = _fit(adata, seed=0)
    _save(model, adata, tmp_path / "m", how)
    drop = {model.module.pname("q_p_c_raw"), model.module.pname("q_p_ct_raw")}

    def strip(state):
        for part in ("params", "constraints"):
            state[part] = {k: v for k, v in state[part].items() if k not in drop}

    _edit_saved_store(tmp_path / "m", how, strip)
    pyro.clear_param_store()

    with pytest.raises(ValueError, match="holds no posteriors"):
        _load(tmp_path / "m", how)
    assert not list(pyro.get_param_store().keys())


def test_an_unreadable_session_store_is_refused(synthetic_adata, tmp_path):
    """A ``pyro_params.pt`` that cannot be read fails the load before anything is loaded,
    rather than leaving a model with uniform posteriors."""
    pyro.clear_param_store()
    adata = _registered(synthetic_adata)
    _save(_fit(adata, seed=0), adata, tmp_path / "m", "session")
    f = tmp_path / "m" / "pyro_params.pt"
    f.write_bytes(f.read_bytes()[: f.stat().st_size // 2])
    pyro.clear_param_store()

    with pytest.raises(RuntimeError, match="could not read the Pyro store"):
        _load(tmp_path / "m", "session")
    assert not list(pyro.get_param_store().keys())


# ── the registered replicate ─────────────────────────────────────────────────

def test_a_session_records_and_restores_the_registered_replicate(synthetic_adata, tmp_path,
                                                                  monkeypatch):
    """``setup.json`` records the registered replicate, ``load_tcri_session`` registers it again,
    and a metric's default ``groupby`` gives the same groups after the round trip.

    ``TCRIModel.load`` re-runs ``setup_anndata`` from the arguments saved in ``model.pt``, so the
    reloaded registry has the replicate whatever ``load_tcri_session`` passes. The spy checks the
    setup call ``load_tcri_session`` makes itself, from ``setup.json``.
    """
    pyro.clear_param_store()
    adata = _registered(synthetic_adata, replicate="patient")
    model = _fit(adata, seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        model.to_anndata(adata)
    level = adata.obs["timepoint"].cat.categories[0]
    before = tcri.tl.mutual_information(adata, covariate=level, null_model=None, inplace=False)
    _save(model, adata, tmp_path / "m", "session")
    calls = []
    setup_anndata = TCRIModel.setup_anndata

    def spy(adata, **kwargs):
        calls.append(kwargs)
        return setup_anndata(adata, **kwargs)

    monkeypatch.setattr(TCRIModel, "setup_anndata", spy)
    pyro.clear_param_store()

    loaded_model, loaded = _load(tmp_path / "m", "session")

    assert json.loads((tmp_path / "m" / "setup.json").read_text())["replicate"] == "patient"
    assert [c.get("replicate") for c in calls if "source_registry" not in c] == ["patient"]
    assert loaded_model.adata_manager.registry[K.Config.REPLICATE] == "patient"
    with contextlib.redirect_stdout(io.StringIO()):
        loaded_model.to_anndata(loaded)
    after = tcri.tl.mutual_information(loaded, covariate=level, null_model=None, inplace=False)
    assert after["result"]["patient"].tolist() == before["result"]["patient"].tolist()


def test_a_session_without_a_recorded_replicate_loads_without_one(synthetic_adata, tmp_path):
    """A ``setup.json`` with no ``replicate`` key predates the key; it loads, with no replicate
    registered."""
    pyro.clear_param_store()
    adata = _registered(synthetic_adata)
    _save(_fit(adata, seed=0), adata, tmp_path / "m", "session")
    setup_file = tmp_path / "m" / "setup.json"
    setup = json.loads(setup_file.read_text())
    del setup["replicate"]
    setup_file.write_text(json.dumps(setup))
    pyro.clear_param_store()

    loaded_model, _ = _load(tmp_path / "m", "session")

    assert loaded_model.adata_manager.registry[K.Config.REPLICATE] is None

