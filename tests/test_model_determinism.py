"""A seeded fit is reproducible; an unseeded one is not (DE-19).

Network init and minibatch order were unseeded: `seed` reached the simulator and the metric
draw only, never the model. Two fits with the same nominal seed differed by ~1.8e-3 in reported
NMI — larger than the effect of several defects this stack fixes, which made those effects
unmeasurable from a paired fit.

The negative control matters as much as the positive one. Asserting only that a seeded fit
reproduces would pass on a model that is deterministic for some unrelated reason (a collapsed
network, a degenerate objective), so this also asserts that WITHOUT the seed the same two fits
differ. If both arms became deterministic the positive test alone would go on passing while
telling you nothing.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

pytestmark = pytest.mark.slow


def _fit(adata, seed, epochs=8):
    """One short fit; returns the learned clone x phenotype table."""
    import pyro

    from tcri.model._model import TCRIModel

    pyro.clear_param_store()
    TCRIModel.setup_anndata(
        adata, layer="counts", clonotype_key="clone_id", phenotype_key="phenotype",
        covariate_key="covariate", batch_key="batch",
    )
    model = TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1,
                      classifier_n_layers=1, classifier_hidden=16, K=5, seed=seed)
    model.train(max_epochs=epochs, batch_size=128, accelerator="cpu",
                enable_progress_bar=False, enable_model_summary=False)
    return np.asarray(model.module.get_p_ct().detach().cpu(), dtype=float)


@pytest.fixture(scope="module")
def _adata():
    from tcri.datasets import simulate_tcri

    return simulate_tcri(n_clones=8, n_phenotypes=5, n_genes=40, n_cells=300,
                         omega_concentration=0.4, fuzziness=0.1, seed=0)


def test_seeded_fits_are_identical(_adata):
    """Same seed, same data -> bit-identical p_ct."""
    a = _fit(_adata.copy(), seed=1234)
    b = _fit(_adata.copy(), seed=1234)
    assert np.array_equal(a, b), (
        f"seeded fits differ: max |delta| = {np.abs(a - b).max():.3e}"
    )


def test_unseeded_fits_differ(_adata):
    """Negative control: without a seed the same two fits do NOT agree.

    Without this, `test_seeded_fits_are_identical` would keep passing on a model that had
    become deterministic for an unrelated reason, and would stop testing the seed.
    """
    a = _fit(_adata.copy(), seed=None)
    b = _fit(_adata.copy(), seed=None)
    assert not np.array_equal(a, b), (
        "unseeded fits are bit-identical, so the seeded test proves nothing about seeding — "
        "either the training became deterministic on its own, or the fit is degenerate"
    )


def test_different_seeds_differ(_adata):
    """A seed selects a fit rather than pinning one universal answer."""
    a = _fit(_adata.copy(), seed=1234)
    b = _fit(_adata.copy(), seed=5678)
    assert not np.array_equal(a, b), "different seeds produced identical fits"
