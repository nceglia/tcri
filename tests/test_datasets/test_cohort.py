"""``simulate_cohort`` — the shape most analyses have, with clones paired across conditions.

The property that earns this function its place is the pairing. Simulating each condition
independently and giving the runs matching clone names produces ids that line up over
generative structure that does not, so a paired metric finds nothing and correctly reports
nothing. These tests pin that the pairing is real.
"""
import numpy as np
import pandas as pd
import pytest

from tcri.datasets import mi_from_joint_oracle, simulate_cohort


@pytest.fixture(scope="module")
def cohort():
    return simulate_cohort(n_patients=6, n_cells_per_sample=140, n_clones=12, seed=0)


def test_the_shape_is_patients_x_conditions_x_arms(cohort):
    obs = cohort.obs
    assert {"clone_id", "phenotype", "condition", "patient", "response"} <= set(obs.columns)
    assert set(obs["response"].unique()) == {"R", "NR"}
    assert list(obs["condition"].cat.categories) == ["pre", "post"]

    counts = obs.groupby(["patient", "condition"], observed=True).size()
    assert counts.nunique() == 1, "every (patient, condition) should be the same size"
    assert obs["patient"].nunique() == 6

    # single-valued leftovers from the per-patient sims must not survive: they would point a
    # reader at the wrong column when reaching for setup_anndata(covariate_key=...)
    assert not ({"covariate", "batch"} & set(obs.columns))


def test_clones_pair_across_conditions_within_a_patient(cohort):
    """The whole point. Without this, `phenotypic_flux` and every `delta_*` measure nothing.

    Measured by MASS, not by count. With power-law clone sizes some rare clones simply are
    not drawn at both conditions — that is what a heavy tail does to a finite sample, and it
    is the behaviour the intersection rule exists to handle. What has to hold is that the
    clones carrying the repertoire are paired: on this fixture the 8 shared clones of 12 in
    P04 cover 97.9% of its post cells.
    """
    obs = cohort.obs
    for patient, g in obs.groupby("patient", observed=True):
        at = {c: set(x["clone_id"]) for c, x in g.groupby("condition", observed=True)}
        shared = at["pre"] & at["post"]
        assert shared, f"{patient}: no clones shared at all"
        for condition, cells in g.groupby("condition", observed=True):
            covered = cells["clone_id"].isin(shared).mean()
            assert covered > 0.9, (
                f"{patient}/{condition}: shared clones cover only {covered:.1%} of cells"
            )


def test_clones_never_span_two_patients(cohort):
    """`groupby` restricts by clone id, so a clone in two patients would let one patient's
    estimate absorb another's cells. `tcri` raises on it; the generator must not produce it."""
    owner = {}
    for patient, g in cohort.obs.groupby("patient", observed=True):
        for clone in g["clone_id"].unique():
            assert owner.setdefault(clone, patient) == patient, f"{clone} spans patients"


def test_only_the_concentration_moves(cohort):
    """Responders' clones commit; non-responders' do not — and nothing was relabelled, so
    each cell's phenotype still matches the expression it was generated with."""
    truth = cohort.uns["tcri_truth"]["per_arm"]
    r_change = float(truth.loc["R", "post"] - truth.loc["R", "pre"])
    nr_change = float(truth.loc["NR", "post"] - truth.loc["NR", "pre"])
    assert r_change > nr_change, f"R moved {r_change:+.3f}, NR moved {nr_change:+.3f}"
    assert r_change > 0.1, "the responder arm barely concentrated; the fixture is inert"

    # `true_phenotype` is carried through from the source sims and still agrees, which is what
    # "nothing is relabelled" means concretely
    obs = cohort.obs
    if "true_phenotype" in obs.columns:
        agree = (obs["phenotype"].astype(str) == obs["true_phenotype"].astype(str)).mean()
        assert agree > 0.99


def test_the_oracle_is_per_patient(cohort):
    """Pooling an arm's patients first mixes clones across patients and inflates the NMI, so
    a pooled oracle compared against a per-patient estimate is a unit mismatch, not a
    benchmark. `per_sample` must therefore be one row per (patient, condition)."""
    per_sample = cohort.uns["tcri_truth"]["per_sample"]
    assert len(per_sample) == 6 * 2
    assert {"patient", "response", "condition", "empirical_nmi_min"} <= set(per_sample.columns)

    row = per_sample.iloc[0]
    g = cohort.obs.query("patient == @row.patient and condition == @row.condition")
    crosstab = pd.crosstab(g["clone_id"], g["phenotype"]).to_numpy(dtype=float)
    assert row["empirical_nmi_min"] == pytest.approx(
        mi_from_joint_oracle(crosstab)["nmi_min"], rel=1e-9)

    # and it is NOT the pooled-per-arm value, which is the mistake it exists to prevent
    arm = cohort.obs.query("response == @row.response and condition == @row.condition")
    pooled = mi_from_joint_oracle(
        pd.crosstab(arm["clone_id"], arm["phenotype"]).to_numpy(dtype=float))["nmi_min"]
    assert not np.isclose(row["empirical_nmi_min"], pooled, atol=1e-3)


def test_reproducible_and_responsive_to_its_knobs(cohort):
    a = simulate_cohort(n_patients=4, n_cells_per_sample=80, n_clones=8, seed=1)
    b = simulate_cohort(n_patients=4, n_cells_per_sample=80, n_clones=8, seed=1)
    pd.testing.assert_frame_equal(a.obs, b.obs)
    assert not np.allclose(
        a.uns["tcri_truth"]["per_arm"].to_numpy(),
        simulate_cohort(n_patients=4, n_cells_per_sample=80, n_clones=8,
                        seed=2).uns["tcri_truth"]["per_arm"].to_numpy())

    # enrichment=1.0 means the second condition is an unbiased draw too, so the arms stop
    # separating -- the knob is what creates the effect, not the labelling
    flat = simulate_cohort(n_patients=4, n_cells_per_sample=140, n_clones=10,
                           responder_enrichment=1.0, nonresponder_enrichment=1.0, seed=0)
    per_arm = flat.uns["tcri_truth"]["per_arm"]
    assert abs(float(per_arm.loc["R", "post"] - per_arm.loc["R", "pre"])) < 0.1


@pytest.mark.parametrize("kwargs,match", [
    (dict(conditions=("only_one",)), "at least two"),
    (dict(conditions=("a", "a")), "unique"),
    (dict(clone_size_distribution="zipfish"), "clone_size_distribution"),
    (dict(n_patients=1), "n_patients"),
    (dict(responder_enrichment=0.5), "responder_enrichment"),
])
def test_rejects_incoherent_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        simulate_cohort(n_cells_per_sample=40, n_clones=5, **kwargs)


def test_clone_sizes_are_heavy_tailed(cohort):
    """Real repertoires are a few large expanded clones over a long tail of singletons.

    ``simulate_tcri``'s own ``pi`` is a symmetric Dirichlet — not heavy-tailed — so a cohort
    that wants realistic clone sizes has to impose them. Checked as the log-log slope of size
    against rank, which IS ``-alpha`` for a Zipf law, plus a Gini for concentration.
    """
    def _slope(frame):
        # `clone_id` is categorical over the WHOLE cohort, so value_counts carries a zero for
        # every other patient's clones -- log(0) poisons the fit
        sizes = frame["clone_id"].value_counts()
        sizes = np.sort(sizes[sizes > 0].to_numpy())[::-1]
        ranks = np.arange(1, len(sizes) + 1)
        return np.polyfit(np.log(ranks), np.log(sizes), 1)[0], sizes

    obs = cohort.obs.query("condition == 'pre' and patient == 'P01'")
    slope, sizes = _slope(obs)
    assert slope < -0.8, f"log-log slope {slope:+.2f} is not a heavy tail"

    flat = simulate_cohort(n_patients=2, n_clones=30, n_cells_per_sample=600,
                           clone_size_distribution="uniform", seed=0)
    f_slope, f_sizes = _slope(flat.obs.query("condition == 'pre' and patient == 'P01'"))
    assert f_slope > slope, "'uniform' is no flatter than 'powerlaw'"
    assert sizes.max() / sizes.sum() > 3 * (f_sizes.max() / f_sizes.sum())


def test_conditions_generalize_past_two():
    """The enrichment ramps across an ordered series, so a timeseries works, not just pre/post."""
    a = simulate_cohort(n_patients=4, conditions=("t0", "t1", "t2", "t3"),
                        n_clones=14, n_cells_per_sample=220, seed=0)
    assert list(a.obs["condition"].cat.categories) == ["t0", "t1", "t2", "t3"]

    per_arm = a.uns["tcri_truth"]["per_arm"]
    assert list(per_arm.columns) == ["t0", "t1", "t2", "t3"], "conditions lost their order"
    r = per_arm.loc["R"].to_numpy()
    assert r[-1] > r[0], "the responder arm did not concentrate across the series"


def test_clone_counts_can_vary_across_patients():
    """Real cohorts are ragged, and a metric normalized by log2(C) is not comparable across
    patients with different C — which is what `n_clones_ref` exists to pin."""
    ragged = simulate_cohort(n_patients=6, n_clones=(8, 26), n_cells_per_sample=150, seed=0)
    counts = ragged.uns["tcri_truth"]["per_sample"].groupby("patient")["n_clones"].max()
    assert counts.nunique() > 1, "every patient got the same clone count"
    assert counts.between(8, 26).all()

    fixed = simulate_cohort(n_patients=4, n_clones=11, n_cells_per_sample=150, seed=0)
    assert fixed.uns["tcri_truth"]["per_sample"]["n_clones"].max() == 11


def test_the_cohort_can_be_written_to_h5ad(tmp_path):
    """A tuple anywhere in `uns` makes the object unwritable, and it surfaces far from the
    cause — as an `IORegistryError` out of `ut.save_tcri_session`, long after generation."""
    import anndata as ad

    a = simulate_cohort(n_patients=4, conditions=("pre", "post"), n_clones=(6, 10),
                        n_cells_per_sample=60, seed=0)
    settings = a.uns["tcri_truth"]["settings"]
    assert not any(isinstance(v, tuple) for v in settings.values()), (
        f"tuples in settings: {[k for k, v in settings.items() if isinstance(v, tuple)]}"
    )

    path = tmp_path / "cohort.h5ad"
    a.write_h5ad(path)
    back = ad.read_h5ad(path)
    assert back.n_obs == a.n_obs
    assert list(back.uns["tcri_truth"]["settings"]["conditions"]) == ["pre", "post"]
