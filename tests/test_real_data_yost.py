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


def test_p_ct_tracks_the_observed_crosstab(fitted):
    """DE-18 on real expression.

    Before the observed-phenotype likelihood, `p_ct` had no data term: it was initialised at the
    observed crosstab and relaxed toward the archetype prior, so its L1 to that crosstab grew
    without bound (0.284 → 0.515 on synthetic). The synthetic cannot tell us whether that fix
    holds on real dropout-heavy expression; this can.

    The bound is deliberately loose. Real `p_ct` should NOT equal the crosstab — shrinkage across
    clones is the model's contribution — but it must stay in the same neighbourhood rather than
    departing for the prior.
    """
    import tcri

    _m, a = fitted
    tab = pd.crosstab(a.obs[CLONE], a.obs[PHENO]).astype(float)
    tab = tab.div(tab.sum(1).clip(lower=1e-12), axis=0)

    jd = tcri.tl.joint_distribution(a, covariate=None, n_samples=0)
    jd = jd.groupby(level=-1).sum() if jd.index.nlevels > 1 else jd
    jd = jd.div(jd.sum(1).clip(lower=1e-12), axis=0)

    idx = tab.index.intersection(jd.index)
    cols = [c for c in tab.columns if c in jd.columns]
    assert len(idx) and len(cols), "no overlap between the crosstab and the learned table"
    l1 = float(np.abs(tab.loc[idx, cols].values - jd.loc[idx, cols].values).sum(1).mean())

    assert l1 < 1.0, (
        f"mean per-clone L1 between p_ct and the observed crosstab is {l1:.3f} out of a possible "
        f"2.0. p_ct has drifted away from the data it is meant to describe (DE-18)."
    )


def test_posterior_concentration_responds_to_clone_size(fitted):
    """DE-5 on real, heavy-tailed clone sizes.

    Real repertoires span singletons to thousands of cells; this subset's largest clone has 901
    cells. With the concentration pinned to β every group got the same posterior width no matter
    how much data supported it, which is invisible under uniform synthetic clone sizes and
    glaring here.
    """
    import pyro

    _m, a = fitted
    raw = pyro.get_param_store()["q_p_ct_raw"].detach().cpu().numpy()
    totals = raw.sum(axis=1)

    assert totals.max() / max(totals.min(), 1e-12) > 1.05, (
        f"concentration totals are effectively constant across groups "
        f"({totals.min():.3f}–{totals.max():.3f}); the posterior width does not depend on how "
        f"many cells a group has (DE-5)"
    )

    sizes = np.bincount(_m.module.ct_array.cpu().numpy(), minlength=len(totals))[: len(totals)]
    keep = sizes > 0
    if keep.sum() > 2 and totals[keep].std() > 0:
        r = float(np.corrcoef(totals[keep], sizes[keep])[0, 1])
        assert r > 0.0, (
            f"posterior concentration correlates {r:+.3f} with clone-covariate group size; it "
            f"should INCREASE with more data, not decrease"
        )
