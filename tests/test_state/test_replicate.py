"""``setup_anndata(replicate=)`` — the registered independent unit.

``replicate`` names the column a metric will use when ``groupby`` is left implicit, so it is
declared once at setup rather than retyped at every call.

It is deliberately NOT derived from ``batch_key``. scvi's ``batch_key`` is one-hot encoded into
every hidden layer of the encoder and decoder (``n_cat_list=[n_batch]``), which is a modelling
decision about what to correct for. ``replicate`` is a claim about what counts as an
independent observation for statistics. They coincide when batches are patients and diverge the
moment they are sequencing runs — at which point deriving one from the other gives a silently
wrong n.
"""
from __future__ import annotations

import contextlib
import io
import logging
import warnings

import numpy as np
import pyro
import pytest

warnings.filterwarnings("ignore")

from tcri._state import keys as K
from tcri.datasets import simulate_tcri
from tcri.model._model import TCRIModel


def _adata():
    adata = simulate_tcri(n_clones=6, n_phenotypes=3, n_genes=20, n_cells=120, seed=0)
    adata.obs["patient"] = np.where(np.arange(adata.n_obs) % 2 == 0, "P0", "P1")
    adata.obs["run"] = np.where(np.arange(adata.n_obs) % 3 == 0, "R1", "R2")
    return adata


def _setup(adata, **kw):
    pyro.clear_param_store()
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="covariate", **kw)


def test_replicate_reaches_the_registry_and_the_metadata():
    """The registry alone is not enough — a metric must resolve it from a SAVED AnnData, after
    the model object is gone, which means it has to live in ``uns``."""
    logging.disable(logging.INFO)
    adata = _adata()
    _setup(adata, batch_key="run", replicate="patient")
    model = TCRIModel(adata, n_latent=4, n_hidden=8, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=8, K=3, seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=2, batch_size=64, accelerator="cpu",
                    enable_progress_bar=False, enable_model_summary=False)
        model.to_anndata(adata)
    logging.disable(logging.NOTSET)

    meta = adata.uns[K.METADATA]
    assert meta[K.Config.REPLICATE] == "patient"
    assert meta[K.Config.LAYER] == "counts"
    # and it is NOT the batch column -- the whole point of the separation
    assert meta[K.Config.BATCH_COL] == "run"
    assert meta[K.Config.REPLICATE] != meta[K.Config.BATCH_COL]


def test_replicate_is_optional_and_defaults_to_none():
    logging.disable(logging.INFO)
    adata = _adata()
    _setup(adata, batch_key="patient")
    logging.disable(logging.NOTSET)
    from tcri.model._model import TCRIModel as M
    assert M._get_most_recent_anndata_manager(adata).registry[K.Config.REPLICATE] is None


def test_a_replicate_that_is_not_a_column_raises_at_setup():
    """Caught at registration, not at the first metric call three steps later."""
    adata = _adata()
    with pytest.raises(ValueError, match="replicate='no_such_column' is not a column"):
        _setup(adata, batch_key="patient", replicate="no_such_column")
