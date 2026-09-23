"""Session save/load round-trip, and the split between what ``setup_anndata`` and
``to_anndata`` may write.

Locks three things:
  1. ``to_anndata`` writes exactly the canonical key set (latent / logits / probabilities /
     gate / classifier-temperature / local-scale), and no ``tcri_manager`` stash.
  2. ``setup_anndata`` is registration-only — no analysis/label ``obs`` mutation.
  3. save → load reproduces ``p_ct`` + latent + ``predict`` from the *reloaded model*. A pyro
     param-store load can fail without raising, which leaves ``q_p_ct_raw`` re-initialised to
     a uniform 1/P matrix, so the round-trip is checked against the reloaded model rather than
     against the saved file, and uniform rows are caught via row variance.
"""
import contextlib
import io
import json

import numpy as np
import pyro
import pytest

import tcri
from tcri._state import keys as K
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
    assert "q_p_ct_raw" in store, "q_p_ct_raw missing from pyro store after load"
    q = store["q_p_ct_raw"].detach().cpu().numpy()
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
