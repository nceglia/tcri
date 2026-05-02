"""Round-trip save/load test (Notion T4 + T11).

Guards against the PyTorch 2.6+ weights_only=True regression that silently
broke pyro param store loading. If load fails silently, _ensure_pyro_posterior_params
re-initializes q_p_ct_raw to a uniform 1/P matrix; we detect that here.
"""
import contextlib
import io

import numpy as np
import pyro

from tcri.utils._utils import load_tcri_session, save_tcri_session


def test_session_round_trip(trained_model, tmp_path):
    model, adata = trained_model

    out_dir = tmp_path / "session"
    save_tcri_session(model, adata, str(out_dir))

    pyro.clear_param_store()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, loaded = load_tcri_session(str(out_dir))

    np.testing.assert_allclose(
        adata.obsm["X_tcri"], loaded.obsm["X_tcri"], atol=1e-5
    )
    np.testing.assert_array_equal(
        adata.obsm["X_tcri_probabilities"], loaded.obsm["X_tcri_probabilities"]
    )
    np.testing.assert_allclose(
        adata.uns["tcri_p_ct"], loaded.uns["tcri_p_ct"], atol=1e-5
    )

    store = pyro.get_param_store()
    assert "q_p_ct_raw" in store, "q_p_ct_raw missing from pyro store after load"
    q = store["q_p_ct_raw"].detach().cpu().numpy()
    row_var = q.var(axis=-1)
    assert row_var.mean() > 1e-6, (
        f"q_p_ct_raw rows are uniform (mean row var {row_var.mean():.2e}); "
        "pyro load silently failed (PyTorch weights_only regression)"
    )
