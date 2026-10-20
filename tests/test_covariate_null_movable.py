"""Which groups the covariate null can move, and what a metric does with one it cannot.

The covariate null shuffles each cell's covariate level within (clonotype, batch). A group whose
clones sit at one level in every batch is left exactly as the fit has it, so its adjusted value
would be 0 by construction. Whether a group can move is decided from ``obs``, never from a drawn
shuffle: a group none of whose counted clones has a (clonotype, batch) holding a level the call
reads and another level has no reference. Its value is kept, every reference column of its rows is
NaN, a warning names it and it leaves the tests. A covariate null that could move no cell anywhere
is refused before anything is fitted.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import tcri
from tcri._compute._tables import movable_clones
from tcri._state import keys as K
from tcri.datasets import simulate_tcri
from tcri.model._model import TCRIModel

KNOBS = dict(n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1, classifier_hidden=16,
             K=3, seed=0)
TRAIN = dict(max_epochs=4, batch_size=128, n_epochs_kl_warmup=2, accelerator="cpu",
             enable_progress_bar=False, enable_model_summary=False)

#: Patients sequenced as one library per covariate level: the covariate null cannot move them.
SPLIT_LIBRARIES = ("P0", "P1")


def _cohort(split=SPLIT_LIBRARIES):
    """Four patients; those in ``split`` have one library per (patient, level), the rest one
    library per patient, so only the rest hold a clone at two levels within a batch."""
    parts = []
    for i, patient in enumerate(("P0", "P1", "P2", "P3")):
        block = simulate_tcri(n_clones=6, n_phenotypes=3, n_genes=20, n_cells=160,
                              n_covariates=2, omega_concentration=0.5, seed=i)
        block.obs["clone_id"] = block.obs["clone_id"].astype(str) + "@" + patient
        block.obs["patient"] = patient
        block.obs["arm"] = "a" if i % 2 == 0 else "b"
        block.obs["library"] = ((patient + "_" + block.obs["covariate"].astype(str))
                                if patient in split else patient)
        block.obs_names = [f"{patient}_{j}" for j in range(block.n_obs)]
        parts.append(block)
    adata = ad.concat(parts, join="outer")
    for col in ("clone_id", "phenotype", "covariate", "patient", "arm", "library"):
        adata.obs[col] = adata.obs[col].astype("category")
    adata.layers["counts"] = adata.X.copy()
    return adata


def _fit(adata, name):
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="covariate",
                            batch_key="library", replicate="patient")
    model = TCRIModel(adata, name=name, **KNOBS)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(**TRAIN)
        model.to_anndata(adata)
    return model


@pytest.fixture(scope="module")
def split_libraries():
    """A fit whose patients P0 and P1 the covariate null cannot move, with two covariate nulls."""
    adata = _cohort()
    model = _fit(adata, "movable")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tcri.null.covariate(model, adata, n_nulls=2)
    return model, adata


def test_movable_clones_is_decided_from_obs():
    """A clonotype is movable at a level when one of its (clonotype, batch) groups holds a cell at
    that level and a cell at another; a group holding one level, or no cell at the level asked
    about, does not count."""
    obs = pd.DataFrame({
        "clone": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "batch": ["x", "x", "x", "y", "x", "x", "x", "x"],
        "level": ["pre", "post", "pre", "post", "pre", "pre", "mid", "post"],
    }, index=[f"c{i}" for i in range(8)]).astype("category")
    adata = ad.AnnData(np.zeros((8, 1), dtype="float32"), obs=obs)
    adata.uns[K.METADATA] = {K.Config.CLONE_COL: "clone", K.Config.BATCH_COL: "batch",
                             K.Config.COVARIATE_COL: "level"}
    adata.uns[K.COVARIATE_CATEGORIES] = ["mid", "post", "pre"]
    assert movable_clones(adata) == {"a", "d"}
    assert movable_clones(adata, ("pre",)) == {"a"}
    assert movable_clones(adata, ("post",)) == {"a", "d"}
    assert movable_clones(adata, ("mid", "pre")) == {"a", "d"}


@pytest.mark.parametrize("mode", ["change", "covariate"])
def test_an_unmovable_patient_has_no_reference(mode, split_libraries):
    """P0 and P1 keep their values; every reference column of their replicate and item rows is
    NaN; a warning names them at the levels read; the tests count the other two patients."""
    _, adata = split_libraries
    where = (dict(cov_from="cov_0", cov_to="cov_1") if mode == "change"
             else dict(covariate="cov_0", null_model="covariate"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = tcri.tl.phenotypic_entropy(adata, **where, splitby="arm", n_samples=3,
                                         random_state=0, inplace=False)
    named = [str(w.message) for w in caught if "cannot move any cell" in str(w.message)]
    assert len(named) == 1 and "'P0', 'P1'" in named[0], named
    assert all(w.filename == __file__ for w in caught
               if "cannot move any cell" in str(w.message))

    reference = ("null_value", "adjusted", "adjusted_sd", "adjusted_hdi_low", "adjusted_hdi_high",
                 "null_value_from", "null_value_to", "adjusted_from", "adjusted_to")
    for level in ("replicate", "item"):
        for frame_name in ("draws", "summary"):
            frame = res[level][frame_name]
            gone = frame["patient"].isin(SPLIT_LIBRARIES)
            columns = [c for c in reference if c in frame.columns]
            assert columns, (level, frame_name)
            assert frame.loc[gone, columns].isna().all().all(), (level, frame_name)
            assert np.isfinite(frame.loc[~gone, "adjusted"].to_numpy(dtype=float)).any()
            assert np.isfinite(frame.loc[gone, "value"].to_numpy(dtype=float)).all(), (
                "an unmovable patient's own value was blanked with its reference")
    stats = res["stats"]
    assert set(stats["quantity"]) == {"adjusted"}
    split = stats.loc[stats["comparison"] == "split"].iloc[0]
    assert split["n_a"] + split["n_b"] == 2, "an unmovable patient was counted by a test"


def test_a_phenotype_null_is_not_blanked(split_libraries):
    """The rule is the covariate null's: a reference that shuffles phenotypes moves every patient,
    so nothing is blanked and nothing is named."""
    model, adata = split_libraries
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tcri.null.phenotype(model, adata, n_nulls=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = tcri.tl.phenotypic_entropy(adata, covariate="cov_0", inplace=False)
    assert not [w for w in caught if "cannot move any cell" in str(w.message)]
    assert np.isfinite(res["replicate"]["summary"]["adjusted"].to_numpy(dtype=float)).all()


def test_a_covariate_null_that_can_move_nothing_is_refused_before_fitting():
    """With one library per (patient, level) for every patient, no shuffle within (clonotype,
    batch) moves a cell: building the covariate null raises, and ``null.all`` raises before it
    fits any reference."""
    adata = _cohort(split=("P0", "P1", "P2", "P3"))
    model = _fit(adata, "immovable")
    with pytest.raises(ValueError, match="cannot move any cell"):
        tcri.null.covariate(model, adata, n_nulls=1)
    with pytest.raises(ValueError, match="'library'"):
        tcri.null.all(model, adata, n_nulls=1)
    assert not [f for f in K.fits(adata) if f.startswith("null.")], (
        "a reference was fitted before the covariate null was refused")


def test_a_group_with_nothing_to_move_is_not_named(split_libraries):
    """A group left no counted clone by ``clones=`` has no value to adjust: the covariate-null
    warning does not name it, only the groups whose clones it cannot move."""
    _, adata = split_libraries
    kept = [c for c in adata.uns[K.CLONOTYPE_CATEGORIES] if str(c).endswith(("@P0", "@P2"))]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tcri.tl.phenotypic_entropy(adata, cov_from="cov_0", cov_to="cov_1", clones=kept,
                                   inplace=False)
    named = [str(w.message) for w in caught if "cannot move any cell" in str(w.message)]
    assert len(named) == 1 and "'P0'" in named[0], named
    assert not any(p in named[0] for p in ("'P1'", "'P2'", "'P3'")), named


def test_unmovable_groups_stay_out_of_the_tests_on_the_value(split_libraries):
    """When no group the call counts can be moved, the tests fall back to the value, and the
    unmovable groups stay out of them there too: their rows are present with ``n`` 0."""
    _, adata = split_libraries
    kept = [c for c in adata.uns[K.CLONOTYPE_CATEGORIES] if str(c).endswith(SPLIT_LIBRARIES)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = tcri.tl.phenotypic_entropy(adata, cov_from="cov_0", cov_to="cov_1", clones=kept,
                                         inplace=False)
    stats = res["stats"]
    assert set(stats["quantity"]) == {"value"}
    assert (stats["n_a"] == 0).all() and (stats["n_b"] == 0).all(), stats
    assert stats["p"].isna().all()


def test_a_change_warns_once_per_call(split_libraries):
    """The change warnings describe the call's groups; its runs on each fit of the reference read
    the same groups and do not repeat them."""
    _, adata = split_libraries
    kept = [c for c in adata.uns[K.CLONOTYPE_CATEGORIES] if not str(c).endswith("@P3")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tcri.tl.phenotypic_entropy(adata, cov_from="cov_0", cov_to="cov_1", clones=kept,
                                   inplace=False)
    counted = [w for w in caught if "have no counted clone" in str(w.message)]
    assert len(counted) == 1, [str(w.message) for w in counted]


def test_a_numeric_covariate_level_is_matched_as_the_engine_matches_it():
    """A covariate registered as numbers is found by an equal number, the engine's rule: 0 finds
    the level 0.0 for the persistent clones, the movable rule and the joint alike, so the change
    equals the one called with the registered values."""
    adata = _cohort(split=())
    adata.obs["covariate"] = adata.obs["covariate"].map({"cov_0": 0.0, "cov_1": 7.0}).astype(
        "category")
    model = _fit(adata, "numeric_levels")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tcri.null.covariate(model, adata, n_nulls=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loose = tcri.tl.phenotypic_flux(adata, cov_from=0, cov_to=7, inplace=False)
    assert not [w for w in caught if "counted clone" in str(w.message)
                or "cannot move" in str(w.message)], [str(w.message) for w in caught]
    exact = tcri.tl.phenotypic_flux(adata, cov_from=0.0, cov_to=7.0, inplace=False)
    assert len(loose["item"]["summary"]) == len(exact["item"]["summary"]) > 0
    np.testing.assert_allclose(loose["replicate"]["summary"]["adjusted"].to_numpy(dtype=float),
                               exact["replicate"]["summary"]["adjusted"].to_numpy(dtype=float))


def test_flux_names_its_own_call_form():
    """Flux takes only the pair; its error names the pair and not ``covariate=``."""
    adata = _cohort()
    _fit(adata, "flux_pair")
    with pytest.raises(ValueError, match="cov_from= and cov_to=") as err:
        tcri.tl.phenotypic_flux(adata, cov_from="cov_0", cov_to=None, null_model=None)
    assert "covariate=" not in str(err.value)
