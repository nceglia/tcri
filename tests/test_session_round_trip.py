"""Session save/load round-trip + the Phase-4 model→AnnData streamline contract.

Locks three things:
  1. ``to_anndata`` writes exactly the canonical key set, including the Phase-4
     additions (logits / gate / classifier-temperature / local-scale), and no
     ``tcri_manager`` stash.
  2. ``setup_anndata`` is registration-only — no analysis/label ``obs`` mutation.
  3. save → load reproduces ``p_ct`` + latent + ``predict`` from the *reloaded model*
     (this also guards the PyTorch ``weights_only=True`` regression that silently
     broke pyro param-store loading; a failed load re-inits ``q_p_ct_raw`` to a
     uniform 1/P matrix, which we detect via row variance).
"""
import contextlib
import io

import numpy as np
import pyro
import pytest

from tcri import _keys as K
from tcri.utils._utils import load_tcri_session, save_tcri_session


@pytest.fixture
def fresh_trained_model(synthetic_adata):
    """A model trained fresh inside this test so it OWNS the process-global pyro
    param store for the save (§5.2 — loading/training a second model in one process
    clobbers ``q_p_ct_raw``, so the shared session-scoped ``trained_model`` store is
    unsafe for a round-trip that recomputes from the reloaded model)."""
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
    """to_anndata writes the full canonical key set incl. the Phase-4 additions,
    with the scalar knobs equal to the model's configured values; no manager stash."""
    model, adata = trained_model
    for k in CANONICAL_UNS:
        assert k in adata.uns, f"to_anndata did not write uns[{k}]"
    for k in CANONICAL_OBSM:
        assert k in adata.obsm, f"to_anndata did not write obsm[{k}]"
    assert K.PHENOTYPE in adata.obs, "to_anndata did not write the hard-label obs"
    assert "tcri_manager" not in adata.uns, "manager stash was not retired"
    # Phase-4 scalar knobs: written == the model's configured value (not just finite)
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
