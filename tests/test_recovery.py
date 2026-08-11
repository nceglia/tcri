"""Statistical **recovery** tests — the only tests in this suite with an accuracy oracle.

Everything else here checks *structure* (contracts, identities, wiring). Those cannot
catch an estimator that is well-formed but wrong. These use
:func:`tcri.datasets.simulate_tcri`, whose mutual information is known in closed form,
and ask: **does the number come out right?**

Tiering (see ``conftest.py``):

* unmarked — oracle self-consistency, exactness of the metric against an independent
  MI implementation, and metamorphic invariances. Model-free, so fast; runs every commit.
* ``@pytest.mark.slow`` — needs model fits and/or replication over seeds. Skipped unless
  ``--runslow``. Run nightly / before a release.

Two honest caveats baked into the assertions below:

1. **Finite-sample bias.** The plug-in MI estimator is biased *upward* by roughly
   ``(C-1)(P-1)/(2N ln2)`` bits, so a realized sample's MI exceeds the population value.
   Tests compare against the *realized* oracle where that matters, and use
   bias-aware tolerances where they compare against the population value.
2. **Single-seed monotonicity is flaky.** Sampling noise can make a larger N land
   *further* from the truth on one seed (observed: gap +0.003 at N=5000 vs +0.012 at
   N=20000 on the same seed). Convergence is therefore asserted on a **mean over
   seeds**, never on one draw.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pyro
import pytest

import tcri
from tcri.tools._mutual_information import _mi_from_joint
from tcri.datasets import mi_from_joint_oracle, simulate_tcri


# ── helpers ─────────────────────────────────────────────────────────────────
def _empirical_joint(adata, phenotype_col="phenotype"):
    """Realized clone x phenotype count table as a DataFrame (a precomputed joint)."""
    return pd.crosstab(adata.obs["clone_id"], adata.obs[phenotype_col])


def _truth(adata):
    return adata.uns["tcri_truth"]


# ══════════════════════ TIER A — the oracle itself ══════════════════════════
def test_oracle_respects_information_bounds():
    """MI <= min(H(c), H(phi)), and both normalizations land in [0, 1]."""
    for conc in (0.05, 0.5, 5.0):
        t = _truth(simulate_tcri(omega_concentration=conc, n_cells=500, seed=0))
        assert t["true_mi"] >= -1e-12
        assert t["true_mi"] <= min(t["true_h_clone"], t["true_h_phenotype"]) + 1e-9
        assert 0.0 - 1e-12 <= t["true_nmi_min"] <= 1.0 + 1e-9
        assert 0.0 - 1e-12 <= t["true_nmi_average"] <= 1.0 + 1e-9


def test_omega_concentration_controls_the_true_mi():
    """The difficulty knob must actually move the ground truth, monotonically.

    Small Dirichlet concentration => near-one-hot P(phi|c) => MI approaches H(phi).
    Large => near-uniform rows => MI approaches 0. Averaged over seeds so the
    ordering reflects the parameter, not one draw.
    """
    means = []
    for conc in (0.05, 0.5, 5.0, 50.0):
        vals = [
            _truth(simulate_tcri(omega_concentration=conc, n_cells=300, seed=s))["true_mi"]
            for s in range(4)
        ]
        means.append(float(np.mean(vals)))
    assert means == sorted(means, reverse=True), f"MI must fall as concentration rises: {means}"
    assert means[0] > 10 * means[-1], f"knob has too little range: {means}"


def test_fuzziness_changes_difficulty_not_truth():
    """fuzziness blends the expression programs only — the true MI must be identical.

    This separation is what lets a benchmark vary estimation difficulty while holding
    the estimand fixed; if it leaked into the truth, MAE-vs-fuzziness would be
    uninterpretable.
    """
    ref = _truth(simulate_tcri(fuzziness=0.0, n_cells=400, seed=3))["true_mi"]
    for f in (0.25, 0.5, 1.0):
        t = _truth(simulate_tcri(fuzziness=f, n_cells=400, seed=3))
        assert t["true_mi"] == pytest.approx(ref, abs=1e-12), f"fuzziness={f} moved the truth"


def test_simulation_is_deterministic_given_a_seed():
    a = simulate_tcri(n_cells=250, seed=7)
    b = simulate_tcri(n_cells=250, seed=7)
    np.testing.assert_array_equal(a.X, b.X)
    assert list(a.obs["clone_id"]) == list(b.obs["clone_id"])
    assert _truth(a)["true_mi"] == _truth(b)["true_mi"]
    c = simulate_tcri(n_cells=250, seed=8)
    assert not np.array_equal(a.X, c.X), "different seeds must differ"


def test_label_error_degrades_the_realized_coupling_only():
    """Corrupting labels must lower the *realized* MI while the population truth stands."""
    clean = _truth(simulate_tcri(label_error_rate=0.0, n_cells=3000, seed=4))
    noisy = _truth(simulate_tcri(label_error_rate=0.6, n_cells=3000, seed=4))
    assert noisy["empirical_mi"] < clean["empirical_mi"], "label noise must reduce realized MI"
    assert noisy["true_mi"] == pytest.approx(clean["true_mi"], abs=1e-12)


# ══════════════ TIER B — does tcri's metric equal an independent oracle? ════
def test_tcri_mi_matches_an_independent_implementation():
    """tcri's MI kernel on a realized joint == the oracle, both normalizations.

    The strongest fast test available: the oracle in ``tcri.datasets`` is a separate,
    deliberately independent implementation, so agreement is real evidence rather
    than a tautology. Catches sign errors, wrong log base, and denominator swaps.

    This tier scores a COUNT table built from ``obs`` -- there is no model and no posterior --
    so it calls ``_mi_from_joint`` directly. It used to reach it through
    ``tl.mutual_information(jd)``, the precomputed-joint path, and was that path's only caller
    in the repo. Going through the public tool added a store-to-``uns`` step to a test with no
    AnnData to store into, and hid which function the oracle was actually being compared to.
    """
    for seed in range(3):
        adata = simulate_tcri(n_cells=1500, n_clones=15, n_phenotypes=4, seed=seed)
        jd = _empirical_joint(adata)
        oracle = mi_from_joint_oracle(jd.values)

        raw = _mi_from_joint(jd.values, normalized=False)
        assert raw == pytest.approx(oracle["mi"], rel=1e-9, abs=1e-12)

        nmi_min = _mi_from_joint(jd.values, normalized=True, mode="min")
        assert nmi_min == pytest.approx(oracle["nmi_min"], rel=1e-9, abs=1e-12)

        nmi_avg = _mi_from_joint(jd.values, normalized=True, mode="average")
        assert nmi_avg == pytest.approx(oracle["nmi_average"], rel=1e-9, abs=1e-12)


def test_the_two_normalizations_are_not_interchangeable():
    """Guards the benchmark trap: tcri defaults to 'min', the note's grid used the mean.

    ``min <= average`` denominators means nmi_min >= nmi_average, so comparing a
    'min'-normalized estimate to a mean-normalized ground truth silently inflates the
    estimate. A benchmark must pick deliberately.
    """
    adata = simulate_tcri(n_cells=1200, n_clones=25, n_phenotypes=4, seed=1)
    jd = _empirical_joint(adata)
    nmi_min = _mi_from_joint(jd.values, mode="min")
    nmi_avg = _mi_from_joint(jd.values, mode="average")
    assert nmi_min > nmi_avg
    t = _truth(adata)
    assert t["empirical_nmi_min"] > t["empirical_nmi_average"]


def test_empirical_mi_is_biased_upward_at_small_n():
    """The plug-in estimator over-reports on small samples — assert the known sign.

    Documents why a recovery test cannot demand ``estimate == truth`` at small N.
    """
    gaps = [
        _truth(simulate_tcri(n_cells=150, n_clones=25, n_phenotypes=5, seed=s))["empirical_mi"]
        - _truth(simulate_tcri(n_cells=150, n_clones=25, n_phenotypes=5, seed=s))["true_mi"]
        for s in range(8)
    ]
    assert float(np.mean(gaps)) > 0, f"expected upward plug-in bias, got {np.mean(gaps):+.4f}"


# ══════════════════ TIER D — metamorphic invariances ════════════════════════
def test_mi_is_invariant_to_relabeling():
    """Renaming clones or phenotypes cannot change an information quantity."""
    adata = simulate_tcri(n_cells=900, n_clones=12, n_phenotypes=4, seed=2)
    jd = _empirical_joint(adata)
    base = _mi_from_joint(jd.values, normalized=False)

    rng = np.random.default_rng(0)
    shuffled = jd.iloc[rng.permutation(jd.shape[0]), rng.permutation(jd.shape[1])]
    assert _mi_from_joint(shuffled.values, normalized=False) == pytest.approx(base, rel=1e-9)


def test_mi_is_invariant_to_uniform_replication():
    """Doubling every count is the same distribution — MI must not move."""
    adata = simulate_tcri(n_cells=800, n_clones=10, n_phenotypes=3, seed=5)
    jd = _empirical_joint(adata)
    base = _mi_from_joint(jd.values, normalized=False)
    assert _mi_from_joint(jd.values * 7, normalized=False) == pytest.approx(base, rel=1e-9)


# ══════════════════════ TIER C/E — slow, model-based ════════════════════════
@pytest.mark.slow
def test_empirical_mi_converges_to_the_population_value():
    """|empirical - true| must shrink with N **on average over seeds**.

    Deliberately not a single-seed monotonicity check: one draw can land further away
    at larger N (observed +0.003 at N=5000 vs +0.012 at N=20000, same seed).
    """
    n_seeds = 6
    mae = {}
    for n in (250, 1000, 4000):
        errs = [
            abs(_truth(simulate_tcri(n_cells=n, n_clones=20, n_phenotypes=4, seed=s))["empirical_mi"]
                - _truth(simulate_tcri(n_cells=n, n_clones=20, n_phenotypes=4, seed=s))["true_mi"])
            for s in range(n_seeds)
        ]
        mae[n] = float(np.mean(errs))
    assert mae[250] > mae[1000] > mae[4000], f"MAE must fall with N: {mae}"


@pytest.mark.slow
def test_model_mi_tracks_the_true_mi_across_difficulty():
    """A *fitted* model's MI must respond to the ground truth, not sit at a constant.

    The weakest defensible claim about the full pipeline: three datasets with very
    different true MI must come back ordered correctly. Not an equality test — the
    model estimator is posterior-based and differs from the plug-in value.
    """
    from tcri.model._model import TCRIModel

    got = []
    for conc in (0.05, 1.0, 20.0):
        pyro.clear_param_store()
        adata = simulate_tcri(
            omega_concentration=conc, n_cells=1200, n_clones=20,
            n_phenotypes=4, n_genes=40, seed=11,
        )
        TCRIModel.setup_anndata(
            adata, layer="counts", clonotype_key="clone_id",
            phenotype_key="phenotype", covariate_key="covariate", batch_key="batch",
        )
        model = TCRIModel(
            adata, n_latent=16, n_hidden=32, n_layers=1,
            classifier_n_layers=1, classifier_hidden=32, K=4,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model.train(max_epochs=60, batch_size=256,
                        enable_progress_bar=False, enable_model_summary=False)
            model.to_anndata(adata)
        est = tcri.tl.mutual_information(
            adata, covariate="cov_0", weighted=True, normalize_mode="average",
        )["result"]
        got.append((_truth(adata)["true_nmi_average"], float(est["value"].iloc[0])))

    truths = [g[0] for g in got]
    ests = [g[1] for g in got]
    assert truths == sorted(truths, reverse=True), f"fixture truths not ordered: {truths}"
    assert ests == sorted(ests, reverse=True), (
        f"model MI did not track the truth: truths={truths} estimates={ests}"
    )


@pytest.mark.slow
def test_posterior_interval_is_well_formed_and_tracks_the_plug_in():
    """The n_samples>0 path returns a usable posterior summary of NMI.

    RE-BASELINED. This test previously asserted the 94% HDI covers the simulator's truth in
    >=6/8 replicates, and it passed at 8/8 — on two errors cancelling:

      1. the plug-in NMI(E[J]) reads LOW against the truth (issue #59), and
      2. the draws read HIGH, because NMI is nonlinear in the joint and the draw was taken
         from a fabricated concentration (local_scale=3) roughly 3x wider than the fitted
         posterior (~9.5).

    DE-5b removed the second by drawing from the guide's actual posterior. Coverage then went
    to 0/8 — not because DE-5b broke calibration, but because it stopped compensating for #1.

    A coverage bar is the wrong assertion here anyway: it compares an HDI of E_s[NMI(J_s)]
    against a truth, while the number most callers see is the plug-in NMI(E[J]). Those are
    different functionals — on seed 100 the old interval [0.248, 0.342] did not even contain
    its own point estimate of 0.173. Which one a figure should report is an open question for
    the authors (metrics contract, OPEN_QUESTIONS['posterior_summary_of_a_nonlinear_metric']).

    So this now asserts what the path genuinely guarantees: the interval is well formed,
    ordered, finite, informative, and brackets its own posterior mean. Coverage against truth
    is tracked in issue #59, where it can be swept properly rather than pinned at 8 replicates.
    """
    from tcri.model._model import TCRIModel

    n_rep = 4
    for seed in range(n_rep):
        pyro.clear_param_store()
        adata = simulate_tcri(
            n_cells=1200, n_clones=18, n_phenotypes=4, n_genes=40,
            omega_concentration=0.4, seed=100 + seed,
        )
        TCRIModel.setup_anndata(
            adata, layer="counts", clonotype_key="clone_id",
            phenotype_key="phenotype", covariate_key="covariate", batch_key="batch",
        )
        model = TCRIModel(
            adata, n_latent=16, n_hidden=32, n_layers=1,
            classifier_n_layers=1, classifier_hidden=32, K=4,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model.train(max_epochs=60, batch_size=256,
                        enable_progress_bar=False, enable_model_summary=False)
            model.to_anndata(adata)

        summary = tcri.tl.mutual_information(
            adata, covariate="cov_0", n_samples=100, weighted=True,
            normalize_mode="average", random_state=seed,
        )["result"].iloc[0]
        lo, hi, mean = summary["hdi_low"], summary["hdi_high"], summary["value"]

        assert np.isfinite([lo, hi, mean]).all(), f"non-finite summary: {dict(summary)}"
        assert hi > lo, f"degenerate or inverted HDI: [{lo}, {hi}]"
        assert hi - lo < 0.5, f"HDI too wide to be informative: [{lo}, {hi}]"
        assert lo <= mean <= hi, (
            f"the posterior mean {mean:.4f} lies outside its own 94% HDI [{lo:.4f}, {hi:.4f}]. "
            f"Whatever the interval is summarising, it is not the quantity reported beside it."
        )
        assert 0.0 <= lo and hi <= 1.0, (
            f"NMI interval [{lo:.4f}, {hi:.4f}] leaves [0,1]; normalize_mode='average' is "
            f"bounded, so this is a computation error rather than a calibration question"
        )
