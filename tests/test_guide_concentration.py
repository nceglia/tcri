"""The variational concentration is a free parameter, not a pinned constant (DE-5, eq 6).

Note 1's eq 6 and its notation table give ``λ'_m, λ_c ∈ ℝ^P_{>0}`` — free variational
parameters in Λ, the set being optimised — while ``α, β`` are scalar PRIOR hyperparameters
appearing only in ``p(· | x; α, β, {ψ_b}, {u_k})``. The guide used the scalar prior
hyperparameter as the variational parameter's total, which conflates two rows of the note's own
notation table and pins every group's posterior width to the same constant.

The consequence is not abstract: a 3-cell clone and a 3000-cell clone got the same posterior
width, so no credible interval reported at ``n_samples>0`` was data-informed, and comparisons
between groups of very different size were the worst case.

This is a structural test — it asserts the concentration is not a constant multiple of a
simplex, which is the defect. Whether the learned magnitudes track group size is a question
about a *fitted* model and belongs with the benchmark, not here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import pyro
import torch

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def fitted():
    from tcri.datasets import simulate_tcri
    from tcri.model._model import TCRIModel

    ad = simulate_tcri(n_clones=10, n_phenotypes=5, n_genes=60, n_cells=600,
                       omega_concentration=0.4, fuzziness=0.1, seed=0)
    pyro.clear_param_store()
    TCRIModel.setup_anndata(
        ad, layer="counts", clonotype_key="clone_id", phenotype_key="phenotype",
        covariate_key="covariate", batch_key="batch",
    )
    m = TCRIModel(ad, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
                  classifier_hidden=16, K=5, seed=0)
    m.train(max_epochs=40, batch_size=128, accelerator="cpu",
            enable_progress_bar=False, enable_model_summary=False)
    return m


def _guide_concentrations(model):
    """Trace the guide and pull the Dirichlet concentrations actually used."""
    import pyro.poutine as poutine

    module = model.module
    batch = {
        "X": torch.as_tensor(np.asarray(model.adata.layers["counts"][:64], dtype="float32")),
        "indices": torch.arange(64).view(-1, 1),
        "batch": torch.zeros(64, 1, dtype=torch.long),
    }
    args, kwargs = module._get_fn_args_from_batch(batch)
    tr = poutine.trace(module.guide).get_trace(*args, **kwargs)
    return {name: tr.nodes[name]["fn"].base_dist.concentration
            if hasattr(tr.nodes[name]["fn"], "base_dist")
            else tr.nodes[name]["fn"].concentration
            for name in ("p_c", "p_ct")}


@pytest.mark.slow
@pytest.mark.parametrize("site,scale_attr", [("p_c", "global_scale"), ("p_ct", "local_scale")])
def test_concentration_total_is_not_pinned_to_the_prior_scale(fitted, site, scale_attr):
    """Eq 6: the magnitude is learned. If every row totals the prior scale, it is not."""
    conc = _guide_concentrations(fitted)[site].detach().cpu().numpy()
    totals = conc.sum(axis=-1).ravel()
    pinned = float(getattr(fitted.module, scale_attr))

    assert totals.size > 1, "need more than one group to test this"
    spread = totals.max() / max(totals.min(), 1e-12)
    assert spread > 1.01, (
        f"q({site}) totals are effectively identical across groups "
        f"(min {totals.min():.4f}, max {totals.max():.4f}). That is the signature of "
        f"concentration = {scale_attr} x (a row summing to 1), which pins every group's "
        f"posterior width to a constant regardless of how much data it has."
    )
    assert not np.allclose(totals, pinned, rtol=1e-6), (
        f"every q({site}) total equals {scale_attr}={pinned}. {scale_attr} is a scalar PRIOR "
        f"hyperparameter; using it as the variational total is the DE-5 defect."
    )


# ── DE-5b: the freed concentration must reach the metric draw ────────────────

def test_guide_concentration_reaches_the_metric_draw(fitted):
    """DE-5b: credible intervals must come from the fitted posterior, not a reconstruction.

    DE-5 freed lambda'_m per eq 6 — but only inside ``guide()``. Every interval the metrics
    report is drawn in ``_compute/_joint.py``, which rebuilt its own Dirichlet as
    ``local_scale * p_ct``: a fixed pseudo-count, identical for every group no matter how much
    data supported it. So DE-5 changed no reported interval at all, and its own test passed
    while the user-visible number was untouched.

    Two assertions, because either alone is passable by accident: the concentration must be
    EXPORTED (it reaches ``uns``), and it must be USED (dropping it changes the draw).
    """
    import numpy as np
    import tcri
    from tcri._state import keys as K

    model = fitted
    adata = model.adata
    model.to_anndata(adata)

    assert K.CONC_CT in adata.uns, (
        "to_anndata does not export the guide concentration, so the metric engine has nothing "
        "to draw from and silently falls back to local_scale * p_ct (DE-5b)"
    )
    conc = np.asarray(adata.uns[K.CONC_CT], dtype=float)
    assert conc.shape == np.asarray(adata.uns[K.P_CT]).shape
    assert np.isfinite(conc).all() and (conc > 0).all()

    # it must be the GUIDE's, not local_scale rebuilt
    totals = conc.sum(1)
    local_scale = float(adata.uns[K.LOCAL_SCALE])
    assert not np.allclose(totals, local_scale, rtol=1e-3), (
        f"concentration totals all equal local_scale ({local_scale}); this is the "
        f"reconstruction DE-5b removes, not the fitted posterior"
    )

    def _mean_width(use_guide):
        b = adata.copy()
        if not use_guide:
            del b.uns[K.CONC_CT]
        jd = tcri.tl.joint_distribution(b, covariate=None, n_samples=120, random_state=0)
        # `table` is the unreduced substrate -- one block of rows per draw. `result` has
        # already averaged them, and an interval width cannot be read off a posterior mean.
        v = jd["table"].to_numpy(dtype=float).reshape(120, -1)
        return float(np.mean(np.percentile(v, 97, 0) - np.percentile(v, 3, 0)))

    assert not np.isclose(_mean_width(True), _mean_width(False), rtol=1e-3), (
        "dropping the guide concentration does not change the interval width, so the draw is "
        "still using local_scale and DE-5b is inert"
    )
