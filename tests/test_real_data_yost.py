"""End-to-end checks on REAL repertoire data (Yost et al., top 50 clones).

Every other test in this suite runs on `simulate_tcri`, which draws ω from a Dirichlet and
generates expression from the model's own assumptions. That is circular: it can only ask whether
tcri inverts data produced by tcri's generative story. It has already proved measurably blind to
a real defect — re-measuring `reconstruction_loss_scale` (deviation [E]), the synthetic read a
perfectly calibrated posterior-predictive library ratio of 1.00 at *every* setting while real
1000-gene, 87%-dropout data read 1.40 vs 0.99.

So this file exists to check the things the synthetic structurally cannot, on data the model did
not generate. It is **not** CI: the dataset lives outside the repo and the fits take minutes.

    MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_real_data_yost.py -q --runslow

Dataset (7682 cells × 1000 genes, counts layer):
    trb        clone id      — 50 clones, verified disjoint across patients
    cluster    phenotype     — 6 CD8 states
    treatment  covariate     — pre / post
    patient    groupby       — 10 patients
    response   splitby       — R / NR, one value per patient

Point `TCRI_YOST_H5AD` at the file to override the default location.
"""
from __future__ import annotations

import os
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

pytestmark = pytest.mark.slow

DEFAULT = pathlib.Path.home() / "Data" / "tcri" / "yost_tcri_v2_top50.h5ad"
PATH = pathlib.Path(os.environ.get("TCRI_YOST_H5AD", DEFAULT))

CLONE, PHENO, COV_KEY, GROUP, SPLIT = "trb", "cluster", "treatment", "patient", "response"
#: `covariate=` selects a VALUE of the covariate column, not the column name. The column is
#: named at setup_anndata(covariate_key=...); the metric then picks one of its categories.
COV = "pre"


@pytest.fixture(scope="module")
def yost():
    if not PATH.exists():
        pytest.skip(f"real dataset not present at {PATH} (set TCRI_YOST_H5AD)")
    import anndata as ad

    return ad.read_h5ad(PATH)


@pytest.fixture(scope="module")
def fitted(yost):
    import pyro

    from tcri.model._model import TCRIModel

    a = yost.copy()
    pyro.clear_param_store()
    TCRIModel.setup_anndata(a, layer="counts", clonotype_key=CLONE, phenotype_key=PHENO,
                            covariate_key=COV_KEY, batch_key=GROUP)
    m = TCRIModel(a, n_latent=16, n_hidden=32, n_layers=1, classifier_n_layers=1,
                  classifier_hidden=32, K=a.obs[PHENO].nunique(), seed=0)
    m.train(max_epochs=120, batch_size=512, accelerator="cpu",
            enable_progress_bar=False, enable_model_summary=False)
    m.to_anndata(a)
    return m, a


def test_dataset_shape_is_what_the_pipeline_assumes(yost):
    """Guard the assumptions the rest of this file rests on, so a swapped file fails loudly
    rather than producing quietly meaningless numbers."""
    for col in (CLONE, PHENO, COV_KEY, GROUP, SPLIT):
        assert col in yost.obs, f"missing obs column {col!r}"
        assert yost.obs[col].isna().sum() == 0, f"{col} has nulls"
    assert "counts" in yost.layers

    # a clone must not span patients: the metric groupby restricts by clone id, so a shared
    # clone would let one patient's estimate absorb another's cells
    spans = yost.obs.groupby(CLONE, observed=True)[GROUP].nunique()
    assert int((spans > 1).sum()) == 0, (
        f"{int((spans > 1).sum())} clones appear in more than one patient; use a "
        f"patient-scoped clone id (trb_unique) instead of {CLONE!r}"
    )
    # response must be a property of the patient, or splitby is ill-defined
    assert yost.obs.groupby(GROUP, observed=True)[SPLIT].nunique().max() == 1
    # and COV must be an actual category of the covariate column
    assert COV in set(yost.obs[COV_KEY].astype(str)), (
        f"COV={COV!r} is not a value of {COV_KEY!r} "
        f"({sorted(set(yost.obs[COV_KEY].astype(str)))}) — `covariate=` takes a VALUE, "
        f"not the column name"
    )


def test_metrics_run_and_are_in_range(fitted):
    """The full metric surface on real data: shapes, ranges, no NaNs where there is mass."""
    import tcri

    _m, a = fitted
    mi = tcri.tl.mutual_information(a, covariate=COV, n_samples=0, weighted=True,
                                    normalize_mode="average")
    assert np.isfinite(mi) and 0.0 <= float(mi) <= 1.0, f"NMI out of range: {mi}"

    ce = tcri.tl.clonotypic_entropy(a, covariate=COV, n_samples=0)
    pe = tcri.tl.phenotypic_entropy(a, covariate=COV, n_samples=0)
    for name, s in (("clonotypic", ce), ("phenotypic", pe)):
        v = np.asarray(s, dtype=float)
        v = v[np.isfinite(v)]
        assert v.size, f"{name} entropy is all NaN"
        assert (v >= -1e-9).all() and (v <= 1.0 + 1e-9).all(), f"{name} entropy out of [0,1]"


def test_grouped_comparison_runs_over_patients(fitted):
    """`groupby=patient` + `splitby=response` is the real use case, and it exercises
    `_validate_group_clones` on genuinely ragged clone sizes."""
    import tcri

    _m, a = fitted
    df = tcri.tl.mutual_information(a, covariate=COV, groupby=GROUP, splitby=SPLIT,
                                    n_samples=0, weighted=True, normalize_mode="average")
    assert isinstance(df, pd.DataFrame) and len(df) == a.obs[GROUP].nunique()
    assert set(df[SPLIT].unique()) <= {"R", "NR"}
    assert "MI" in df.columns, f"expected an 'MI' column, got {list(df.columns)}"
    assert df["MI"].notna().any()


def test_p_ct_stays_a_probability_table_on_real_data(fitted):
    """`p_ct` is the clone×phenotype table every metric reads. Check it is well-formed and not
    degenerate on real dropout-heavy expression.

    This test previously asserted that `p_ct` tracks the observed crosstab, as evidence for
    DE-18's observed-phenotype likelihood. **DE-18 is withdrawn** — z^ϕ is latent and the
    hierarchical branch is a prior that does not see the data directly, so `p_ct` is under no
    obligation to stay near the crosstab and an L1 bound against it asserts the wrong thing.

    What is still worth checking on real data is that the table is a valid distribution and
    retains clone-to-clone structure rather than collapsing to one shared row.
    """
    import tcri

    _m, a = fitted
    jd = tcri.tl.joint_distribution(a, covariate=None, n_samples=0)

    # each (covariate, clone) row is a distribution over phenotypes. Check normalisation
    # BEFORE collapsing: summing over covariates gives a row total of 1 per covariate the
    # clone appears in, which is 2.0 for a clone seen both pre and post.
    raw = jd.to_numpy(dtype=float)
    assert np.isfinite(raw).all(), "p_ct contains NaN or inf on real data"
    assert (raw >= -1e-9).all(), "p_ct has negative mass"
    assert np.allclose(raw.sum(axis=1), 1.0, atol=1e-6), (
        f"p_ct rows are not normalised: sums span "
        f"{raw.sum(axis=1).min():.6f}–{raw.sum(axis=1).max():.6f}"
    )

    collapsed = jd.groupby(level=-1).sum() if jd.index.nlevels > 1 else jd
    v = collapsed.div(collapsed.sum(1).clip(lower=1e-12), axis=0).to_numpy(dtype=float)

    # clone-to-clone structure must survive. If every row relaxed to the same archetype the
    # table would carry no clonotype information and MI would be ~0 by construction.
    spread = float(np.abs(v - v.mean(axis=0, keepdims=True)).sum(axis=1).mean())
    assert spread > 1e-3, (
        f"every clone's phenotype distribution is effectively identical (mean L1 to the "
        f"column mean is {spread:.2e}); p_ct carries no clonotype-specific signal"
    )


def test_posterior_concentration_is_not_pinned_to_beta(fitted):
    """DE-5 on real, heavy-tailed clone sizes.

    Eq 6 specifies λ'_m ∈ ℝ^P_{>0} — a free variational parameter. The implementation had
    `conc = β · (normalized row)`, pinning every group's concentration TOTAL to β regardless of
    its cell count. This is what DE-5 fixes, and it is what this test checks: the totals must be
    free to differ across groups.
    """
    import pyro

    _m, a = fitted
    raw = pyro.get_param_store()["q_p_ct_raw"].detach().cpu().numpy()
    totals = raw.sum(axis=1)

    assert totals.max() / max(totals.min(), 1e-12) > 1.05, (
        f"concentration totals are effectively constant across groups "
        f"({totals.min():.3f}–{totals.max():.3f}); λ'_m is still pinned (DE-5)"
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Concentration does NOT track clone size, and after DE-18's withdrawal we do not "
        "expect it to. The hierarchical branch ω_c -> ϕ_m -> z^ϕ has no direct data term by "
        "design, so the only ϕ-bearing ELBO term is −KL(q(ϕ_m) ‖ Dir(β·ω)), whose optimum over "
        "a free λ'_m is λ'_m = β·ω — a total of β for every row, independent of how many cells "
        "the group has. Measured r = -0.038 on Yost top-50. DE-5 correctly frees the parameter; "
        "whether the posterior should also CONCENTRATE with data, and through what coupling, is "
        "an open model question for the forthcoming supplemental note. Kept as a live "
        "measurement rather than deleted, so the answer is recorded when it arrives."
    ),
)
def test_posterior_concentration_tracks_clone_size(fitted):
    import pyro

    _m, a = fitted
    totals = pyro.get_param_store()["q_p_ct_raw"].detach().cpu().numpy().sum(axis=1)
    sizes = np.bincount(_m.module.ct_array.cpu().numpy(), minlength=len(totals))[: len(totals)]
    keep = sizes > 0
    assert keep.sum() > 2 and totals[keep].std() > 0, "not enough spread to correlate"
    r = float(np.corrcoef(totals[keep], sizes[keep])[0, 1])
    assert r > 0.0, (
        f"posterior concentration correlates {r:+.3f} with clone-covariate group size; it "
        f"should INCREASE with more data, not decrease"
    )
