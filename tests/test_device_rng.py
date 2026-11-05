"""Seeded blocks restore the generators they seed, on every device.

``torch.manual_seed`` reseeds every device's generator. A seeded block that restores only the
CPU leaves a CUDA or MPS generator reseeded afterwards; inside the validation check that
restarts training's random stream from the evaluation seed every epoch. The MPS tests run only
where MPS is available; the generator mapping is checked everywhere.
"""
from __future__ import annotations

import warnings

import pyro
import pytest
import torch

from tcri._compute._joint import _torch_seed
from tcri._compute._xp import rng_device, seeded_rng

warnings.filterwarnings("ignore")

needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


@pytest.mark.parametrize("device, expected", [
    ("cpu", ("cpu", None)),
    (torch.device("cpu"), ("cpu", None)),
    ("cuda:1", ("cuda", 1)),
    (torch.device("cuda", 0), ("cuda", 0)),
    ("mps", ("mps", None)),
    (torch.device("mps", 0), ("mps", None)),
])
def test_rng_device_names_the_generator_a_draw_uses(device, expected):
    assert rng_device(device) == expected


def test_seeded_rng_on_cpu_matches_manual_seed_and_restores():
    torch.manual_seed(11)
    before = torch.get_rng_state().clone()
    with seeded_rng(3, "cpu"):
        inside = torch.randn(4)
    assert torch.equal(torch.get_rng_state(), before)
    torch.manual_seed(3)
    assert torch.equal(inside, torch.randn(4))


def test_torch_seed_restores_the_cpu_generator():
    torch.manual_seed(11)
    before = torch.get_rng_state().clone()
    with _torch_seed(5):
        a = torch.randn(3)
    with _torch_seed(5):
        b = torch.randn(3)
    assert torch.equal(a, b)
    assert torch.equal(torch.get_rng_state(), before)


@needs_mps
def test_seeded_rng_on_mps_matches_manual_seed_and_restores():
    torch.manual_seed(11)
    torch.randn(2, device="mps")
    before = torch.mps.get_rng_state().clone()
    with seeded_rng(3, "mps"):
        inside = torch.randn(4, device="mps")
    assert torch.equal(torch.mps.get_rng_state(), before), "the MPS generator stayed reseeded"
    torch.manual_seed(3)
    assert torch.equal(inside, torch.randn(4, device="mps"))


@needs_mps
@pytest.mark.parametrize("enter", [
    lambda: seeded_rng(3, "cpu"),
    lambda: _torch_seed(3),
], ids=["seeded_rng", "_torch_seed"])
def test_a_cpu_seeded_block_leaves_the_mps_generator_alone(enter):
    torch.manual_seed(11)
    torch.randn(2, device="mps")
    before = torch.mps.get_rng_state().clone()
    with enter():
        torch.randn(4)
    assert torch.equal(torch.mps.get_rng_state(), before)


@pytest.fixture
def small_model():
    from tcri.datasets import simulate_tcri
    from tcri.model._model import TCRIModel

    adata = simulate_tcri(n_clones=8, n_phenotypes=5, n_genes=40, n_cells=300,
                          omega_concentration=0.4, fuzziness=0.1, seed=0)
    pyro.clear_param_store()
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="covariate",
                            batch_key="batch")
    model = TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=16, K=5, seed=0)
    return model, adata


class _EpochStarts:
    """Records the MPS generator at each epoch's first training batch and counts validations."""

    def __init__(self):
        self.states, self.validations = [], 0

    def __call__(self):
        import lightning.pytorch as pl

        outer = self

        class _CB(pl.Callback):
            def on_train_batch_start(self, trainer, module, batch, batch_idx):
                if batch_idx == 0:
                    outer.states.append(torch.mps.get_rng_state().clone())

            def on_validation_end(self, trainer, module):
                outer.validations += 1

        return _CB()


@needs_mps
def test_consecutive_mps_epochs_start_from_different_random_states(small_model):
    model, _ = small_model
    rec = _EpochStarts()
    model.train(max_epochs=3, early_stopping=False, accelerator="mps", callbacks=[rec()],
                enable_progress_bar=False, enable_model_summary=False)
    assert rec.validations >= 2, "no validation check ran between epochs; nothing is tested"
    assert len(rec.states) == 3
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.equal(rec.states[i], rec.states[j]), (
                f"epochs {i} and {j} start from the same MPS generator state: the validation "
                f"check's seed outlived it"
            )


def test_reconstruction_ppc_seeds_the_models_device(small_model, monkeypatch):
    """The PPC seeds through ``seeded_rng`` with the device its model is on. Its ZINB draw
    needs a Poisson sampler, which MPS does not have, so this runs on the CPU and checks the
    device passed."""
    import tcri
    from tcri._compute import _xp

    model, adata = small_model
    model.train(max_epochs=2, early_stopping=False, accelerator="cpu",
                enable_progress_bar=False, enable_model_summary=False)
    calls = []
    real = _xp.seeded_rng

    def spy(seed, device="cpu"):
        calls.append((seed, torch.device(device)))
        return real(seed, device)

    monkeypatch.setattr(_xp, "seeded_rng", spy)
    tcri.diag.reconstruction_ppc(model, adata, n_sims=2, random_state=7, plot=False,
                                 print=False)
    assert calls == [(7, next(model.module.parameters()).device)]
