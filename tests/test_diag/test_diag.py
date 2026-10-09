"""tcri.diag smoke — each PPC returns a DataFrame; the two plots return axes."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import tcri


def test_joint_distribution_ppc(trained_model):
    _, adata = trained_model
    df = tcri.diag.joint_distribution_ppc(adata, distance_metric="l1")
    assert isinstance(df, pd.DataFrame)
    if len(df):
        assert {"covariate", "clonotype", "distance"}.issubset(df.columns)


def test_phenotype_calibration(trained_model):
    _, adata = trained_model
    df = tcri.diag.phenotype_calibration(adata, n_bins=5)
    assert {"bin", "mean_pred", "emp_freq", "count"}.issubset(df.columns)
    assert "ECE" in df.attrs and df.attrs["ECE"] >= 0


def test_reconstruction_ppc(trained_model):
    model, adata = trained_model
    df = tcri.diag.reconstruction_ppc(model, adata, n_sims=1, random_state=0)
    assert {"statistic", "observed", "simulated", "discrepancy"}.issubset(df.columns)
    assert (df["discrepancy"] >= 0).all()


def test_permutation_null(trained_model):
    _, adata = trained_model
    df = tcri.diag.permutation_null(adata, n_perm=30, random_state=0)
    assert {"covariate", "observed", "null_mean", "null_sd", "z", "p"}.issubset(df.columns)
    assert ((df["p"] >= 0) & (df["p"] <= 1)).all()


def test_loss_and_archetypes(trained_model):
    model, _ = trained_model
    assert tcri.diag.loss(model) is not None
    assert tcri.diag.archetypes(model) is not None


def test_loss_draws_the_validation_objective_on_its_own_axes(trained_model):
    """The monitored series is per cell and the training loss is summed, so they get separate
    axes, and the monitored one is labeled as the per-cell objective, not an ELBO."""
    import matplotlib.pyplot as plt

    model, _ = trained_model
    train_ax = tcri.diag.loss(model)
    fig = train_ax.figure
    val = [a for a in fig.axes if "Validation objective" in a.get_title()]
    assert len(val) == 1 and val[0] is not train_ax, [a.get_title() for a in fig.axes]
    assert "per cell" in val[0].get_ylabel()
    expected = np.asarray(model.history_["objective_validation_percell"].values, dtype=float).ravel()
    np.testing.assert_allclose(val[0].lines[0].get_ydata(), expected)
    labels = [line.get_label() for a in fig.axes for line in a.lines]
    assert not [lab for lab in labels if "ELBO" in lab and "val" in lab.lower()], labels
    selected = model.training_record_["selected_epoch"]
    marks = [line for line in val[0].lines if line.get_label().startswith("selected epoch")]
    if selected is None:
        assert not marks, "a selected epoch was marked although the fit selected none"
    else:
        assert len(marks) == 1 and list(marks[0].get_xdata()) == [selected, selected]
    plt.close(fig)


def _curves(record, n_before=0):
    """A stand-in for a fitted model: the three curves ``diag.loss`` draws and a record.

    The last six epochs are one ``train()`` call whose KL ramp ends at its fourth epoch: the
    first three sit far above, and after the ramp each curve moves by a few thousandths.
    ``n_before`` epochs of an earlier call come first, on the same high level.
    """
    from types import SimpleNamespace

    curves = {
        "elbo_train": [5000.0] * n_before + [5000.0, 3000.0, 1500.0, 900.0, 899.0, 898.0],
        "objective_validation_percell": [34.0] * n_before + [34.0, 20.0, 8.0, 3.650, 3.645, 3.640],
        "kl_divergence_with_prior_train_epoch": [50.0] * n_before + [50.0, 30.0, 10.0, 2.0, 2.01,
                                                                     2.02],
    }
    history = {name: pd.DataFrame({name: values},
                                  index=pd.Index(range(len(values)), name="epoch"))
               for name, values in curves.items()}
    return SimpleNamespace(history_=history, training_record_=record), curves


RAMPED = dict(epochs_run=6, ramp_completes_at_epoch=3, ramp_completed=True, selected_epoch=None)


@pytest.mark.parametrize("log_scale", [False, True], ids=["linear", "log"])
@pytest.mark.parametrize("n_before", [0, 4], ids=["one_call", "two_calls"])
def test_loss_y_range_spans_the_epochs_after_the_ramp(log_scale, n_before):
    """Each panel's y range spans the epochs after the KL ramp, so their change fills most of the
    axis and the first epochs run off the top. With two ``train()`` calls the ramp is located in
    the last one, whose epochs are the last of the history."""
    import matplotlib.pyplot as plt

    model, curves = _curves(RAMPED, n_before)
    fig = tcri.diag.loss(model, log_scale=log_scale).figure
    titles = {"Training loss": "elbo_train",
              "Validation objective (per cell)": "objective_validation_percell",
              "prior KL": "kl_divergence_with_prior_train_epoch"}
    for ax in fig.axes:
        after = np.asarray(curves[titles[ax.get_title()]][n_before + 3:])
        lo, hi = ax.get_ylim()
        assert lo <= after.min() and after.max() <= hi, ax.get_title()
        assert hi < curves[titles[ax.get_title()]][0], ax.get_title()
        span = (np.log10(after.max() / after.min()) / np.log10(hi / lo) if log_scale
                else (after.max() - after.min()) / (hi - lo))
        assert span > 0.85, (ax.get_title(), span)
    plt.close(fig)


def test_loss_y_range_spans_every_epoch_without_a_completed_ramp():
    """A ramp that never completed leaves no epoch after it: every epoch is in range."""
    import matplotlib.pyplot as plt

    model, curves = _curves({**RAMPED, "ramp_completed": False})
    fig = tcri.diag.loss(model).figure
    val = [ax for ax in fig.axes if "Validation objective" in ax.get_title()][0]
    assert val.get_ylim()[1] >= curves["objective_validation_percell"][0]
    plt.close(fig)


def test_loss_marks_the_selected_epoch_after_two_train_calls():
    """A second ``train()`` call appends its epochs to the history and counts the epoch it
    selected within itself: the marker sits at that epoch after the first call's epochs."""
    import contextlib
    import io
    import warnings

    import matplotlib.pyplot as plt

    from tcri.datasets import simulate_tcri
    from tcri.model._model import TCRIModel

    a = simulate_tcri(n_clones=8, n_phenotypes=3, n_genes=20, n_cells=200, n_covariates=2,
                      seed=0)
    TCRIModel.setup_anndata(a, clonotype_key="clone_id", phenotype_key="phenotype",
                            covariate_key="covariate", batch_key="batch")
    model = TCRIModel(a, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=16, K=3, seed=0)
    train = dict(batch_size=64, n_epochs_kl_warmup=2, accelerator="cpu",
                 enable_progress_bar=False, enable_model_summary=False)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(max_epochs=4, **train)
        first = len(model.history_["objective_validation_percell"])
        model.train(max_epochs=3, **train)
    selected = model.training_record_["selected_epoch"]
    assert first == 4 and len(model.history_["objective_validation_percell"]) == 7
    assert selected is not None, "the second call selected no epoch"

    fig = tcri.diag.loss(model).figure
    val = [ax for ax in fig.axes if "Validation objective" in ax.get_title()][0]
    marks = [line for line in val.lines if line.get_label().startswith("selected epoch")]
    assert len(marks) == 1 and list(marks[0].get_xdata()) == [first + selected] * 2
    plt.close(fig)


def test_reconstruction_ppc_n_sims_is_wired(trained_model):
    """``n_sims`` must change the result, not merely be accepted.

    ``reconstruction_ppc(..., n_sims=)`` is part of the diagnostics surface pinned in
    ``governance/API_CONTRACT.md``. A body that draws one replicate per cell whatever the
    caller passes returns bit-identical frames for ``n_sims=1`` and ``n_sims=8``: a user
    tightening the check gets no more precision and no warning that the knob is inert.

    Averaging more posterior-predictive draws must move the simulated statistics while leaving
    the observed ones alone, which is what this asserts — the value arriving in the body is not
    enough.
    """
    model, adata = trained_model

    one = tcri.diag.reconstruction_ppc(model, adata, n_sims=1, random_state=0)
    many = tcri.diag.reconstruction_ppc(model, adata, n_sims=8, random_state=0)

    assert list(one["statistic"]) == list(many["statistic"])
    # observed data does not depend on the number of draws
    assert np.allclose(one["observed"].to_numpy(), many["observed"].to_numpy())
    assert not np.allclose(one["simulated"].to_numpy(), many["simulated"].to_numpy()), (
        "n_sims does not change the simulated statistics, so it is still inert"
    )

    with pytest.raises(ValueError, match="n_sims must be >= 1"):
        tcri.diag.reconstruction_ppc(model, adata, n_sims=0)


def _perfectly_coupled_adata(n_clones=6, per_clone=30):
    """Clone k is ALWAYS phenotype k. No shuffle can reach the observed MI, so the raw
    exceedance fraction is exactly 0 and the floor is the only thing that can lift the p-value
    off zero — which is what makes this fixture able to see the floor.
    Built by hand rather than fitted: permutation_null is model-free and reads only obs + uns.
    """
    import anndata as ad
    import pandas as pd

    from tcri._state import keys as K

    clones = np.repeat([f"clone_{i}" for i in range(n_clones)], per_clone)
    phenos = np.repeat([f"phen_{i}" for i in range(n_clones)], per_clone)
    n = len(clones)
    obs = pd.DataFrame({"clone_id": clones, "phenotype": phenos, "covariate": ["cov_0"] * n},
                       index=[f"cell_{i}" for i in range(n)])
    adata = ad.AnnData(X=np.zeros((n, 2), dtype="float32"), obs=obs)
    adata.uns[K.METADATA] = {"clone_col": "clone_id", "phenotype_col": "phenotype",
                             "covariate_col": "covariate", "batch_col": "covariate"}
    adata.uns[K.PHENOTYPE_CATEGORIES] = [f"phen_{i}" for i in range(n_clones)]
    adata.uns[K.COVARIATE_CATEGORIES] = ["cov_0"]
    return adata


@pytest.mark.parametrize("R", [50, 200])
def test_permutation_null_p_value_is_floored(R):
    """A permutation p-value estimated from R shuffles can never be 0.

    The observed statistic is itself one realisation under the null, so the estimator is
    (1 + #{null >= obs}) / (1 + R), not the raw fraction (Phipson & Smyth 2010). The raw
    fraction is exactly 0.0 whenever no shuffle beats the observation — claiming infinite
    evidence from a finite sample, and giving -inf to anyone who logs it.

    Uses a perfectly coupled fixture on purpose: on ordinary data some shuffle usually ties the
    observation, the raw fraction is already nonzero, and the floor is invisible.
    """
    adata = _perfectly_coupled_adata()
    df = tcri.diag.permutation_null(adata, n_perm=R, random_state=0)

    assert (df["p"] > 0).all(), f"p of exactly 0 from {R} permutations: {df['p'].tolist()}"
    assert float(df["p"].iloc[0]) == pytest.approx(1.0 / (R + 1)), (
        f"p={float(df['p'].iloc[0]):.6f} on a perfectly coupled fixture where no shuffle can "
        f"beat the observation; the floored estimator must give exactly 1/(R+1) = "
        f"{1/(R+1):.6f}"
    )


def test_permutation_null_honours_normalize_mode(trained_model):
    """The null must be on the same scale as the statistic it is a null for.

    If the normalizer were fixed inside the null, a caller working in ``"average"`` would
    compare their number against a null computed on a different one. That is not a weaker null
    — it is not a null for their statistic at all.
    """
    model, adata = trained_model
    lo = tcri.diag.permutation_null(adata, n_perm=50, normalize_mode="min", random_state=0)
    hi = tcri.diag.permutation_null(adata, n_perm=50, normalize_mode="average", random_state=0)

    assert not np.allclose(lo["observed"].to_numpy(), hi["observed"].to_numpy()), (
        "normalize_mode does not change the statistic, so it is still hardcoded"
    )
    with pytest.raises(ValueError, match="normalize_mode must be"):
        tcri.diag.permutation_null(adata, n_perm=10, normalize_mode="nonsense")


def test_permutation_null_groupby_matches_the_metric_surface(trained_model):
    """``groupby`` must change the null, not merely be accepted.

    Every ``tl.*`` metric takes ``groupby``, and this is the null FOR those metrics, so a
    per-patient MI needs a per-patient null. Cells are restricted to the group and phenotypes
    permuted within each (covariate, group) stratum, so the null conditions on what the
    reported statistic conditions on.
    """
    import inspect

    model, adata = trained_model
    assert "groupby" in inspect.signature(tcri.diag.permutation_null).parameters

    flat = tcri.diag.permutation_null(adata, n_perm=30, random_state=0)
    grouped = tcri.diag.permutation_null(adata, groupby="patient", n_perm=30, random_state=0)

    assert "patient" in grouped.columns, "the group label is not carried into the result"
    n_groups = adata.obs["patient"].nunique()
    assert len(grouped) == len(flat) * n_groups, (
        f"expected one row per (covariate, group): {len(flat)} x {n_groups}, got {len(grouped)}"
    )
    assert not np.allclose(
        grouped["observed"].to_numpy()[: len(flat)], flat["observed"].to_numpy()
    ) or n_groups == 1, "grouping did not change the statistic, so groupby is still inert"

    with pytest.raises(ValueError, match="not a column"):
        tcri.diag.permutation_null(adata, groupby="no_such_column", n_perm=5)
