"""Behavioural tests for the metric comparison axes.

Every metric takes the same family of arguments — ``covariate``, ``groupby``, ``splitby``,
``clones``, ``weighted``, ``normalize_mode``, ``n_samples``, ``temperature`` — and until now
nothing checked that any of them reaches a code path. That is the defect class in issue #64:
an argument can be declared, accepted, and completely inert while every test stays green. It
has already happened four times in this package (``n_sims``, ``permutation_null(groupby=)``,
``random_state``, ``hue_order``).

So these tests assert that each axis CHANGES THE ANSWER, and — where an exact identity exists —
that it changes it to the right thing.

The fixture combines independent synthetic blocks with deliberately different clone→phenotype
coupling into one AnnData. What that buys:

* **Exact** assertions for routing. ``groupby="patient"`` restricted to a block equals the same
  metric called with ``clones=<that block's clones>``, to full float precision, because clone
  ids are disjoint across patients.
* **Ordinal** assertions for accuracy. The model ranks the blocks as the oracle does.

What it deliberately does NOT claim: equality against the oracle's absolute value. The metrics
read the model's ``p_ct``, not the observed crosstab, and the two differ (issue #59). Asserting
an absolute value here would be pinning the fit, not the metric.
"""
from __future__ import annotations

import contextlib
import io
import logging
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pyro
import pytest

warnings.filterwarnings("ignore")

import tcri
from tcri.datasets import mi_from_joint_oracle, simulate_tcri
from tcri.model._model import TCRIModel

#: (patient, response, omega_concentration). Low omega => sharp clone→phenotype coupling.
#: The blocks MUST differ: if every block scored the same, a `groupby` that silently ignored
#: its argument would still pass every test here.
BLOCKS = (("P0", "R", 0.15), ("P1", "NR", 1.5))


def _build_blocks(n_cells=250, seed=0):
    parts = []
    for i, (patient, response, omega) in enumerate(BLOCKS):
        block = simulate_tcri(n_clones=8, n_phenotypes=4, n_genes=25, n_cells=n_cells,
                              n_covariates=2, omega_concentration=omega, seed=seed + i)
        # patient-scoped clone ids. This is what makes `groupby` legal — the metric restricts
        # by clone, so a clone spanning groups would let one group absorb another's cells.
        block.obs["clone_id"] = block.obs["clone_id"].astype(str) + "@" + patient
        block.obs["patient"] = patient
        block.obs["response"] = response
        block.obs_names = [f"{patient}_{j}" for j in range(block.n_obs)]
        parts.append(block)

    adata = ad.concat(parts, join="outer", label=None)
    for col in ("clone_id", "phenotype", "covariate", "patient", "response"):
        adata.obs[col] = adata.obs[col].astype("category")
    adata.layers["counts"] = adata.X.copy()
    return adata, parts


@pytest.fixture(scope="module")
def blocks():
    """Combined AnnData, fitted once, plus the per-block oracle truth from obs alone."""
    logging.disable(logging.INFO)
    adata, parts = _build_blocks()

    truth = {}
    for (patient, _resp, _omega), part in zip(BLOCKS, parts):
        crosstab = pd.crosstab(part.obs["clone_id"], part.obs["phenotype"]).to_numpy(float)
        truth[patient] = mi_from_joint_oracle(crosstab)

    pyro.clear_param_store()
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="covariate",
                            batch_key="patient")
    model = TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=16, K=4, seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=10, batch_size=128, n_steps_kl_warmup=8, accelerator="cpu",
                    enable_progress_bar=False, enable_model_summary=False)
        model.to_anndata(adata)
    logging.disable(logging.NOTSET)
    return model, adata, truth


def _clones_of(adata, patient):
    return adata.obs.loc[adata.obs["patient"] == patient, "clone_id"].unique().tolist()


# ── the fixture's own preconditions ──────────────────────────────────────────

def test_fixture_blocks_are_disjoint_and_distinguishable(blocks):
    """Both properties the rest of the file rests on, asserted rather than assumed.

    Disjoint clone ids are what make ``groupby`` legal at all. Blocks that differ are what
    make an inert ``groupby`` detectable — with identical blocks, every assertion below would
    pass on a function that ignored the argument entirely.
    """
    _model, adata, truth = blocks

    spans = adata.obs.groupby("clone_id", observed=True)["patient"].nunique()
    assert int((spans > 1).sum()) == 0, "clone ids are not disjoint across patients"

    values = [truth[p]["nmi_min"] for p, _r, _o in BLOCKS]
    assert max(values) - min(values) > 0.2, (
        f"blocks score {values}; too similar to detect an axis that is ignored"
    )


# ── axis liveness: every argument must change the answer ─────────────────────

@pytest.mark.parametrize("metric", ["mutual_information", "clonotypic_entropy",
                                    "phenotypic_entropy"])
def test_covariate_axis_is_live(blocks, metric):
    """Different covariate levels must give different results."""
    _model, adata, _truth = blocks
    fn = getattr(tcri.tl, metric)
    a = np.asarray(fn(adata, covariate="cov_0"), dtype=float).ravel()
    b = np.asarray(fn(adata, covariate="cov_1"), dtype=float).ravel()
    assert not np.allclose(a, b, equal_nan=True), (
        f"{metric} returns the same values for cov_0 and cov_1; the covariate axis is inert"
    )


def test_groupby_is_live_and_routes_exactly(blocks):
    """``groupby`` must both change the shape AND route each group to its own clones.

    The equality is exact, not approximate: ``groupby`` restricts by clone id, so a group's
    value must be bit-identical to calling the metric with that group's clones directly. This
    is what an inert ``groupby`` — accepted and ignored — cannot produce.
    """
    _model, adata, _truth = blocks

    grouped = tcri.tl.mutual_information(adata, covariate="cov_0", groupby="patient",
                                         normalize_mode="average")
    assert len(grouped) == adata.obs["patient"].nunique()
    assert grouped["MI"].nunique() > 1, "every group scored the same; groupby may be inert"

    for _, row in grouped.iterrows():
        direct = float(tcri.tl.mutual_information(
            adata, covariate="cov_0", clones=_clones_of(adata, row["patient"]),
            normalize_mode="average",
        ))
        assert float(row["MI"]) == pytest.approx(direct, rel=1e-12), (
            f"groupby value for {row['patient']} ({row['MI']}) differs from the same metric "
            f"restricted to that patient's clones ({direct}); groupby is not routing by clone"
        )


def test_splitby_labels_the_groups(blocks):
    """``splitby`` attaches a group-level label. It must not change the values."""
    _model, adata, _truth = blocks

    plain = tcri.tl.mutual_information(adata, covariate="cov_0", groupby="patient")
    split = tcri.tl.mutual_information(adata, covariate="cov_0", groupby="patient",
                                       splitby="response")

    assert "response" in split.columns
    assert np.allclose(plain["MI"].to_numpy(), split["MI"].to_numpy()), (
        "splitby changed the values; it is a labelling axis, not a computation axis"
    )
    expected = {p: r for p, r, _o in BLOCKS}
    for _, row in split.iterrows():
        assert row["response"] == expected[row["patient"]], "splitby carried the wrong label"


def test_clones_restriction_is_live(blocks):
    """``clones=`` must actually restrict, and the result must depend on which clones."""
    _model, adata, _truth = blocks
    p0, p1 = _clones_of(adata, "P0"), _clones_of(adata, "P1")

    a = float(tcri.tl.mutual_information(adata, covariate="cov_0", clones=p0))
    b = float(tcri.tl.mutual_information(adata, covariate="cov_0", clones=p1))
    both = float(tcri.tl.mutual_information(adata, covariate="cov_0", clones=p0 + p1))
    assert a != b, "clones= does not change the result"
    assert both not in (a, b), "restricting to both blocks equals one of them alone"

    jd = tcri.tl.joint_distribution(adata, covariate="cov_0", clones=p0)
    assert set(jd.index) <= set(p0), "joint_distribution returned clones outside `clones=`"


def test_weighted_is_live(blocks):
    """``weighted`` selects the clone marginal, so it must change the number."""
    _model, adata, _truth = blocks
    unweighted = float(tcri.tl.mutual_information(adata, covariate="cov_0", weighted=False))
    weighted = float(tcri.tl.mutual_information(adata, covariate="cov_0", weighted=True))
    assert unweighted != weighted, (
        "weighted=True and weighted=False agree; the clone marginal is not being applied"
    )


def test_normalize_mode_is_live(blocks):
    """``min`` and ``average`` are different normalizers (metrics doc eq 6 vs the default)."""
    _model, adata, _truth = blocks
    lo = float(tcri.tl.mutual_information(adata, covariate="cov_0", normalize_mode="min"))
    av = float(tcri.tl.mutual_information(adata, covariate="cov_0", normalize_mode="average"))
    assert lo != av
    assert lo >= av - 1e-12, "min(H) <= mean(H), so NMI_min must be >= NMI_average"


def test_temperature_is_live(blocks):
    """Analysis-time temperature tempers the mean table, so it must move the metric."""
    _model, adata, _truth = blocks
    base = float(tcri.tl.mutual_information(adata, covariate="cov_0", temperature=1.0))
    hot = float(tcri.tl.mutual_information(adata, covariate="cov_0", temperature=3.0))
    assert base != hot, "temperature does not reach the computation"


def test_n_samples_returns_a_posterior_summary(blocks):
    """``n_samples>0`` switches from a scalar to a mean/sd/HDI summary that brackets itself."""
    _model, adata, _truth = blocks
    point = float(tcri.tl.mutual_information(adata, covariate="cov_0", n_samples=0))
    summary = tcri.tl.mutual_information(adata, covariate="cov_0", n_samples=50, random_state=0)

    assert isinstance(summary, dict)
    assert set(summary) >= {"mean", "sd", "hdi_low", "hdi_high"}
    assert summary["hdi_low"] <= summary["mean"] <= summary["hdi_high"], (
        "the posterior mean lies outside its own HDI"
    )
    assert summary["sd"] > 0, "zero spread across draws; the draw path is not sampling"
    assert np.isfinite(point)


def test_random_state_makes_draws_reproducible(blocks):
    """Same seed, same summary; different seed, different summary."""
    _model, adata, _truth = blocks
    kw = dict(covariate="cov_0", n_samples=40)
    a = tcri.tl.mutual_information(adata, random_state=0, **kw)
    b = tcri.tl.mutual_information(adata, random_state=0, **kw)
    c = tcri.tl.mutual_information(adata, random_state=1, **kw)
    assert a["mean"] == b["mean"], "same random_state gave a different answer"
    assert a["mean"] != c["mean"], "random_state does not reach the draw"


# ── accuracy, at the strength the fixture can actually support ───────────────

def test_model_ranks_the_blocks_as_the_oracle_does(blocks):
    """Ordinal, not absolute.

    The metrics read the model's ``p_ct``; the oracle reads the observed crosstab. They differ
    (issue #59), so an absolute-value assertion here would pin the fit rather than the metric.
    What must hold is that a block the oracle says is more clone→phenotype coupled scores
    higher — otherwise the metric is not tracking the structure it claims to measure.
    """
    _model, adata, truth = blocks

    grouped = tcri.tl.mutual_information(adata, covariate="cov_0", groupby="patient",
                                         normalize_mode="average")
    model_order = grouped.sort_values("MI")["patient"].tolist()
    oracle_order = sorted(truth, key=lambda p: truth[p]["nmi_average"])
    assert model_order == oracle_order, (
        f"model ranks blocks {model_order} but the oracle ranks them {oracle_order}"
    )


# ── covariate is required wherever the result is reduced ─────────────────────

@pytest.mark.parametrize("metric", ["mutual_information", "clonotypic_entropy",
                                    "phenotypic_entropy"])
def test_scalar_metrics_require_an_explicit_covariate(blocks, metric):
    """``covariate=None`` used to stack every covariate level into one table, treating each
    (covariate, clone) pair as a distinct clone.

    That inflates H(c). Measured on a 10-clone / 2-covariate fixture:

        C=20 P=4  (many clones)   min  +0.0%   average +13.8%
        C=6  P=8  (few clones)    min +12.4%   average +15.4%

    ``min`` divides by min(H(c), H(phi)). In the realistic regime for a repertoire — clones
    vastly outnumbering phenotypes — it selects H(phi), which row-splitting does not touch, so
    the default really is unaffected. The C=6/P=8 row is kept only to show that this exemption
    is a property of the ratio rather than of the metric; an earlier version of this docstring
    stated it as universal. ``average`` moves in every regime, including the realistic one.

    The three reducing metrics also disagreed about what ``covariate=None`` meant: one
    collapsed to a per-phenotype index, one kept a (covariate, clonotype) index, one stacked.
    Choosing a unification is a question about the estimand, not the code, so covariate is now
    required wherever a reduction happens rather than being silently resolved one of three ways.
    """
    _model, adata, _truth = blocks
    with pytest.raises(ValueError, match="covariate is required"):
        getattr(tcri.tl, metric)(adata)

    with pytest.raises(ValueError, match="covariate is required"):
        getattr(tcri.tl, metric)(adata, covariate=None, groupby="patient")


def test_joint_distribution_still_returns_every_covariate(blocks):
    """``joint_distribution`` is deliberately exempt: it LABELS the blocks with a covariate
    index level instead of collapsing them, so no ambiguity arises, and it stays the way to
    get every covariate level in one object."""
    _model, adata, _truth = blocks

    jd = tcri.tl.joint_distribution(adata, covariate=None)
    assert jd.index.nlevels == 2 and jd.index.names[0] == "covariate"
    assert set(jd.index.get_level_values("covariate")) == set(
        adata.uns["tcri_covariate_categories"]
    )

    one = tcri.tl.joint_distribution(adata, covariate="cov_0")
    assert one.index.nlevels == 1
    assert len(one) < len(jd), "restricting to one covariate did not reduce the table"


def test_the_guard_reaches_the_plotting_layer(blocks):
    """``pl.*`` forwards ``covariate`` straight through, so the same ambiguity would otherwise
    reappear one layer up with no guard at all."""
    _model, adata, _truth = blocks
    with pytest.raises(ValueError, match="covariate is required"):
        tcri.pl.mutual_information(adata, return_df=True)


def test_precomputed_joint_path_is_unaffected(blocks):
    """A precomputed joint has already resolved the covariate question — the table exists.
    The guard must not fire there, or ``mutual_information(jd)`` breaks for every caller that
    builds its own joint."""
    _model, adata, _truth = blocks
    jd = tcri.tl.joint_distribution(adata, covariate="cov_0")
    value = tcri.tl.mutual_information(jd)
    assert np.isfinite(float(value))


# ── defects the audit surfaced, each with the number that proves it ──────────

def test_precomputed_joint_path_cannot_bypass_the_covariate_guard(blocks):
    """D1: the guard lives in ``joint_draws``; the precomputed-joint fast path never calls it.

    ``mutual_information(joint_distribution(adata, covariate=None))`` therefore reached the old
    stacked computation one call away from the guard — measured 0.1240077 stacked against
    0.1452629 for the same table marginalised. A guard that can be walked around by composing
    two public functions is not a guard.
    """
    _model, adata, _truth = blocks
    stacked = tcri.tl.joint_distribution(adata, covariate=None)

    for metric in (tcri.tl.mutual_information, tcri.tl.clonotypic_entropy,
                   tcri.tl.phenotypic_entropy):
        with pytest.raises(ValueError, match="covariate.*index level"):
            metric(stacked)

    # the documented escape hatches must both work
    assert np.isfinite(float(tcri.tl.mutual_information(
        stacked.groupby(level="clonotype").sum())))
    assert np.isfinite(float(tcri.tl.mutual_information(
        tcri.tl.joint_distribution(adata, covariate="cov_0"))))


def test_groupby_honours_the_callers_clone_restriction(blocks):
    """D3: ``clones=`` was accepted and discarded whenever ``groupby`` was set.

    The per-group closure rebound the name to that group's clone list, shadowing the caller's,
    so ``groupby='patient', clones=[two clones]`` returned a frame ``.equals()`` the
    unrestricted call. The restriction is now intersected with each group's clones.
    """
    _model, adata, _truth = blocks
    subset = _clones_of(adata, "P0")[:2]

    unrestricted = tcri.tl.mutual_information(adata, covariate="cov_0", groupby="patient")
    restricted = tcri.tl.mutual_information(adata, covariate="cov_0", groupby="patient",
                                            clones=subset)

    assert not unrestricted.equals(restricted), "clones= is still discarded under groupby"
    assert set(restricted["patient"]) == {"P0"}, (
        "restricting to P0's clones should leave only P0; groups with no surviving clone are "
        "dropped rather than reported as NaN"
    )
    direct = float(tcri.tl.mutual_information(adata, covariate="cov_0", clones=subset))
    assert float(restricted["MI"].iloc[0]) == pytest.approx(direct, rel=1e-12)


def test_groupby_warns_when_cells_have_no_group_label(blocks):
    """D9: ``.dropna()`` on the groupby column silently excluded unlabelled cells.

    Measured 20% of cells dropped with no warning. Excluding a fifth of the data from every
    reported number is not a default anyone opted into.
    """
    _model, adata, _truth = blocks
    site = pd.Series(adata.obs["patient"].astype(str).values, index=adata.obs_names)
    site[::5] = None
    adata = adata.copy()
    adata.obs["site"] = site.astype("category")

    with pytest.warns(UserWarning, match="no group label"):
        tcri.tl.mutual_information(adata, covariate="cov_0", groupby="site")


def test_a_single_draw_reports_no_spread(blocks):
    """D11: ``n_samples=1`` returned ``sd=0.0`` and a zero-width HDI.

    One draw carries no information about spread. Reporting ``[0.289341, 0.289341]`` states a
    certainty that was never measured; NaN says what is actually known.
    """
    _model, adata, _truth = blocks
    one = tcri.tl.mutual_information(adata, covariate="cov_0", n_samples=1, random_state=0)
    assert np.isfinite(one["mean"])
    assert np.isnan(one["sd"]) and np.isnan(one["hdi_low"]) and np.isnan(one["hdi_high"])

    many = tcri.tl.mutual_information(adata, covariate="cov_0", n_samples=20, random_state=0)
    assert many["sd"] > 0 and many["hdi_high"] > many["hdi_low"], (
        "the n=1 special case must not have disabled real summaries"
    )


# ── flux: the two sides must come from one shared draw (issue #65) ───────────

@pytest.mark.parametrize("random_state", [None, 0, 7])
@pytest.mark.parametrize("distance_metric", ["kl", "l1", "jsd"])
def test_self_flux_is_exactly_zero(blocks, random_state, distance_metric):
    """The flux of a covariate against ITSELF is the distance of a distribution from itself,
    which is exactly 0 for every metric here.

    It was not. The two sides came from two independent ``joint_draws`` calls, and at the
    DEFAULT ``random_state=None`` those were independent samples — so self-flux read 0.209180
    at n_samples=16. The number being reported was the sampling noise floor, and it grew with
    the noise. The docstring already claimed the sides were drawn "coherently (same seed)";
    that was true only when a seed was passed.

    ``random_state=None`` is parametrised deliberately — it is the default, and it is the case
    that was broken. A test that only checked a seeded call would have passed throughout.
    """
    _model, adata, _truth = blocks
    out = tcri.tl.phenotypic_flux(adata, cov_from="cov_0", cov_to="cov_0", n_samples=16,
                                  distance_metric=distance_metric, random_state=random_state)
    values = out["mean"].to_numpy(float) if hasattr(out, "columns") else np.asarray(out, float)
    worst = float(np.nanmax(values))
    assert worst == pytest.approx(0.0, abs=1e-12), (
        f"flux of cov_0 against itself is {worst:.6f} under {distance_metric} at "
        f"random_state={random_state}; the two sides are not sharing a draw"
    )


def test_real_flux_keeps_posterior_spread(blocks):
    """Coupling the draws must not flatten the posterior. Two different covariate levels are
    different latent variables, so their flux retains genuine spread — this guards against
    'fixing' self-flux by collapsing the draw path altogether."""
    _model, adata, _truth = blocks
    out = tcri.tl.phenotypic_flux(adata, cov_from="cov_0", cov_to="cov_1", n_samples=16,
                                  random_state=0)
    assert (out["sd"].to_numpy(float) > 0).any(), "no spread across draws; the draw path is dead"
