"""Runtime smoke for the model: construct -> train (2 epochs) -> latent / p_ct / predict.

Walks the whole construct/train/query path across the pieces a fit is assembled from
(``TCRIModule`` model/guide, ``UnifiedTrainingPlan``, ``build_archetypes``,
``MixtureDirichlet``, ``VampPrior``), so a break anywhere along it surfaces here rather than
only inside a metric test that happens to depend on it.
"""
import contextlib
import io

import numpy as np

from tcri.model._model import TCRIModel, build_archetypes


def test_model_construct_train_predict(synthetic_adata):
    adata = synthetic_adata.copy()
    TCRIModel.setup_anndata(
        adata,
        clonotype_key="unique_clone_id",
        phenotype_key="phenotype_col",
        covariate_key="timepoint",
        batch_key="patient",
    )
    model = TCRIModel(
        adata,
        n_latent=8,
        n_hidden=16,
        n_layers=1,
        classifier_n_layers=1,
        classifier_hidden=16,
        K=3,
        n_pseudo_obs=3,
    )

    # the clone x phenotype prior is exposed as clone_phenotype_prior, and under no other name.
    assert hasattr(model, "clone_phenotype_prior")
    assert not hasattr(model, "c2p_mat")
    n_clones, P = model.clone_phenotype_prior.shape

    # build_archetypes returns centers AND labels.
    centers, labels = build_archetypes(model.clone_phenotype_prior, K=3)
    assert centers.shape == (3, P)
    assert labels.shape == (n_clones,)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model.train(
            max_epochs=2,
            batch_size=64,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    n_cells = adata.n_obs
    z = model.get_latent_representation()
    assert z.shape == (n_cells, 8)

    p_ct = model.get_p_ct()
    assert p_ct.ndim == 2 and p_ct.shape[1] == P

    probs = model.predict()
    assert probs.shape == (n_cells, P)
    assert list(probs.index) == list(adata.obs_names)  # order-preserving, labelled
    np.testing.assert_allclose(probs.values.sum(axis=1), 1.0, atol=1e-4)
