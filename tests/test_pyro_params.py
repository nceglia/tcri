"""Tests for the _ensure_pyro_posterior_params fallback warning (Notion #24).

When the Pyro param store has no 'q_p_ct_raw', the helper re-initializes it to a
uniform 1/P simplex. That silent fallback used to mask failed param-store loads, so
downstream posterior metrics ran on an uninformative prior with no signal to the
caller. It must now emit a warning. The param store is global state, so each test
snapshots and restores it to avoid leaking into other tests.
"""

import warnings

import pyro

from tcri.utils._utils import _ensure_pyro_posterior_params


def test_ensure_pyro_params_warns_on_empty_store(trained_model):
    """An empty param store triggers the uniform fallback, which must warn."""
    model, adata = trained_model
    store = pyro.get_param_store()
    saved = store.get_state()
    try:
        store.clear()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _ensure_pyro_posterior_params(model, adata)

        runtime = [w for w in rec if issubclass(w.category, RuntimeWarning)]
        assert runtime, "expected a RuntimeWarning on the uniform fallback"
        assert "q_p_ct_raw" in str(runtime[0].message)
        # fallback still populates the param so callers can proceed
        assert "q_p_ct_raw" in store
    finally:
        store.set_state(saved)


def test_ensure_pyro_params_silent_when_present(trained_model):
    """When 'q_p_ct_raw' already exists the helper early-returns without warning."""
    model, adata = trained_model
    store = pyro.get_param_store()
    saved = store.get_state()
    try:
        _ensure_pyro_posterior_params(model, adata)  # guarantee it is present
        assert "q_p_ct_raw" in store

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _ensure_pyro_posterior_params(model, adata)

        assert not any(issubclass(w.category, RuntimeWarning) for w in rec)
    finally:
        store.set_state(saved)
