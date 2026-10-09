"""Pools are not clones: every tool leaves them out by default.

``pp.pool_rare_clones`` pools each group's rare clones into one ``pooled@{group}`` label and
lists the pool labels in the derivation record. A pool is a mixture of many rare clones, so at
``exclude_pools=True`` every tool counts the registered clones minus the pools, which is the
explicit ``clones=[every non-pool clone]`` call. Leaving out the pools keeps every other row
where the tool puts it, while ``clones=`` orders rows by its list, so the two are compared row
for row where the orders agree and after sorting where they do not. The record, not the look of
a label, decides what a pool is: a column no pooling step wrote has nothing to exclude.

The fixture is a small simulated cohort pooled with ``pp.pool_rare_clones`` and registered on
the pooled column, so every patient has a pool. Fits are tiny; nothing here asserts accuracy.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

import tcri
from tcri._compute._repertoire import _derivation_steps, _pool_labels
from tcri._compute._tables import _resolve_clones
from tcri._state import keys as K
from tcri.datasets import simulate_cohort
from tcri.model._model import TCRIModel
from tcri.null._rebuild import rebuild
from tcri.tools._mutual_information import _mi_from_joint
from tcri.utils._utils import load_tcri_session, save_tcri_session

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

PRE, POST, SPLIT = "pre", "post", "disease_status"
CLONES = "clone_id_pooled"
KNOBS = dict(n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1, classifier_hidden=16,
             K=4, seed=0)
TRAIN = dict(max_epochs=10, batch_size=128, n_epochs_kl_warmup=3, accelerator="cpu",
             enable_progress_bar=False, enable_model_summary=False)

#: Every tool that takes ``clones=``, with the arguments it is called with here.
TOOLS = {
    "joint_distribution": (tcri.tl.joint_distribution, {}),
    "mutual_information": (tcri.tl.mutual_information, {"covariate": POST, "splitby": SPLIT}),
    "clonotypic_entropy": (tcri.tl.clonotypic_entropy, {"covariate": POST, "splitby": SPLIT}),
    "phenotypic_entropy": (tcri.tl.phenotypic_entropy, {"covariate": POST, "splitby": SPLIT}),
    "phenotypic_flux": (tcri.tl.phenotypic_flux,
                        {"cov_from": PRE, "cov_to": POST, "splitby": SPLIT}),
    "delta_clonotypic_entropy": (tcri.tl.delta_clonotypic_entropy,
                                 {"cov_from": PRE, "cov_to": POST, "splitby": SPLIT}),
    "delta_phenotypic_entropy": (tcri.tl.delta_phenotypic_entropy,
                                 {"cov_from": PRE, "cov_to": POST, "splitby": SPLIT}),
    "joint_distribution_ppc": (tcri.diag.joint_distribution_ppc, {}),
    "permutation_null": (tcri.diag.permutation_null,
                         {"covariate": POST, "groupby": "patient", "n_perm": 20,
                          "random_state": 0}),
}
STORED = {name for name in TOOLS if hasattr(tcri.tl, name)}

#: Tools whose rows ``clones=`` orders by its list where leaving out the pools keeps the tool's
#: own order: the joint stacked over covariates, and the per-clone table within each patient.
REORDERED = {"joint_distribution", "phenotypic_entropy"}

#: The label columns of a long frame, which fix its row order once sorted.
LABELS = ["covariate", "cov_from", "cov_to", "patient", "disease_status", "clonotype",
          "phenotype", "gene", "draw"]


@pytest.fixture(scope="module")
def pooled():
    """A pooled cohort, registered on the pooled column and fitted with the phenotype and
    condition nulls. Returns ``(model, adata)``."""
    from .conftest import _remember_fixture_params

    adata = simulate_cohort(n_patients=4, n_clones=(10, 16), n_cells_per_sample=60, seed=0)
    adata.layers["counts"] = adata.X.copy()
    tcri.pp.pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key=CLONES,
                            phenotype_key="phenotype", covariate_key="condition",
                            batch_key="patient", replicate="patient")
    model = TCRIModel(adata, name="pooled", **KNOBS)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(**TRAIN)
        model.to_anndata(adata)
        tcri.null.all(model, adata, kinds=("phenotype", "condition"))
    _remember_fixture_params("pooled")
    return model, adata


def _kept(adata):
    """Every non-pool clone, read off the labels rather than the derivation record."""
    return [c for c in adata.uns[K.CLONOTYPE_CATEGORIES] if not str(c).startswith("pooled@")]


def _run(name, model, adata, **kw):
    """Call one tool on ``adata``; a stored result is not written unless ``inplace=True``."""
    if name == "gene_importance":
        kw.setdefault("inplace", False)
        return tcri.perturb.gene_importance(model, adata, genes=list(adata.var_names[:3]),
                                            **kw)
    fn, fixed = TOOLS[name]
    if name in STORED:
        kw.setdefault("inplace", False)
    return fn(adata, **fixed, **kw)


def _frames(result):
    return {"frame": result} if isinstance(result, pd.DataFrame) else dict(result)


def _assert_same(got, want):
    got, want = _frames(got), _frames(want)
    assert got.keys() == want.keys()
    for slot, frame in want.items():
        if frame is None:
            assert got[slot] is None, slot
        else:
            pd.testing.assert_frame_equal(got[slot], frame, obj=slot)


def _same(got, want):
    try:
        _assert_same(got, want)
    except AssertionError:
        return False
    return True


def _sorted(result):
    """Every frame of a result in a fixed row order: by index for a labelled index, by the label
    columns for a long frame."""
    out = {}
    for slot, frame in _frames(result).items():
        if frame is None or not isinstance(frame.index, pd.RangeIndex):
            out[slot] = frame if frame is None else frame.sort_index()
        else:
            labels = [c for c in LABELS if c in frame.columns]
            out[slot] = frame.sort_values(labels, kind="stable").reset_index(drop=True)
    return out


def _drop_pools(frame):
    """``frame`` without its pool rows, every other row where it was."""
    ids = (frame.index.get_level_values("clonotype") if "clonotype" in frame.index.names
           else frame["clonotype"])
    keep = ~pd.Index(ids.astype(str)).str.startswith("pooled@")
    out = frame[np.asarray(keep)]
    return out.reset_index(drop=True) if isinstance(frame.index, pd.RangeIndex) else out


def _clone_ids(result):
    """The clone ids a result has rows for, or ``None`` when it has no clone axis."""
    for frame in _frames(result).values():
        if frame is None:
            continue
        if "clonotype" in frame.columns:
            return set(frame["clonotype"])
        if "clonotype" in frame.index.names:
            return set(frame.index.get_level_values("clonotype"))
    return None


def _manual_importance(model, adata, gene, mask):
    """``I`` and the per-phenotype shift from two ``knockout`` frames, over ``mask`` cells."""
    base = tcri.perturb.knockout(model, adata, genes=[]).loc[mask]
    pert = tcri.perturb.knockout(model, adata, genes=[gene]).loc[mask]
    shift = base.mean(axis=0) - pert.mean(axis=0)
    return float(shift.abs().sum()), shift


def test_the_fixture_has_a_pool_in_every_patient(pooled):
    _, adata = pooled
    pools = _pool_labels(adata, CLONES)
    assert pools == [f"pooled@{p}" for p in sorted(adata.obs["patient"].unique())]
    assert _resolve_clones(adata, None, True) == _kept(adata)


# ── exclude_pools is the explicit clones= call ───────────────────────────────

@pytest.mark.parametrize("name", list(TOOLS))
def test_exclude_pools_equals_explicit_clones(pooled, name):
    """``exclude_pools=True`` is ``clones=[every non-pool clone]``: every slot, the reference
    columns and the contrast included, row for row, or after sorting where ``clones=`` orders
    the rows by its list. Counting the pools moves the numbers, so the equality is not vacuous:
    a result with a clone axis gains the pool rows."""
    model, adata = pooled
    by_default = _run(name, model, adata)
    explicit = _run(name, model, adata, clones=_kept(adata), exclude_pools=False)
    if name in REORDERED:
        _assert_same(_sorted(by_default), _sorted(explicit))
    else:
        _assert_same(by_default, explicit)

    with_pools = _run(name, model, adata, exclude_pools=False)
    ids = _clone_ids(with_pools)
    if ids is None:
        assert not _same(by_default, with_pools)
    else:
        assert {c for c in ids if c.startswith("pooled@")}
        assert not {c for c in _clone_ids(by_default) if c.startswith("pooled@")}


def test_exclude_pools_equals_explicit_clones_in_gene_importance(pooled):
    """``gene_importance`` has no ``clones=``: its explicit form is the importance over the
    cells of every non-pool clone, from two ``knockout`` frames. The reference, the rebuilt
    phenotype null, leaves out the same cells."""
    model, adata = pooled
    gene = adata.var_names[1]
    kw = dict(genes=[gene], inplace=False)
    res = tcri.perturb.gene_importance(model, adata, **kw)
    with_pools = tcri.perturb.gene_importance(model, adata, exclude_pools=False, **kw)
    null = rebuild(model, adata, "null.phenotype")
    kept = adata.obs[CLONES].isin(_kept(adata)).to_numpy()
    for patient in adata.obs["patient"].unique():
        in_patient = (adata.obs["patient"] == patient).to_numpy()
        for got, mask in ((res, in_patient & kept), (with_pools, in_patient)):
            row = got["result"].query("patient == @patient")
            want_i, want_shift = _manual_importance(model, adata, gene, mask)
            assert float(row["value"].iloc[0]) == pytest.approx(want_i, abs=1e-5)
            want_null, _ = _manual_importance(null, adata, gene, mask)
            assert float(row["null_value"].iloc[0]) == pytest.approx(want_null, abs=1e-5)
            shift = (got["shift"].query("patient == @patient")
                     .set_index("phenotype")["shift"].reindex(want_shift.index))
            np.testing.assert_allclose(shift.to_numpy(), want_shift.to_numpy(), atol=1e-5)
    assert not np.allclose(res["result"]["value"], with_pools["result"]["value"])


def test_leaving_out_pools_keeps_the_row_order(pooled):
    """With no ``clones=``, leaving out the pools drops their rows and keeps every other row
    where the tool puts it with the pools counted: the joint stacked over covariates, and each
    patient's per-clone table. The values of the rows that stay do not depend on the pools."""
    _, adata = pooled
    for by_default, with_pools in (
        (tcri.tl.joint_distribution(adata, inplace=False)["result"],
         tcri.tl.joint_distribution(adata, exclude_pools=False, inplace=False)["result"]),
        (tcri.tl.phenotypic_entropy(adata, covariate=POST, null_model=None,
                                    inplace=False)["table"],
         tcri.tl.phenotypic_entropy(adata, covariate=POST, exclude_pools=False,
                                    null_model=None, inplace=False)["table"]),
    ):
        assert len(by_default) < len(with_pools)
        pd.testing.assert_frame_equal(by_default, _drop_pools(with_pools))


def test_the_reference_counts_the_clones_the_value_counts(pooled):
    """The reference is the caller's own call on the null, so it forwards ``exclude_pools``:
    with pools counted, ``null_value`` is the null's own value with pools counted."""
    _, adata = pooled
    for exclude_pools in (True, False):
        kw = dict(covariate=POST, exclude_pools=exclude_pools, inplace=False)
        with_ref = tcri.tl.mutual_information(adata, **kw)["result"]
        of_null = tcri.tl.mutual_information(adata, fit="null.phenotype", null_model=None,
                                             **kw)["result"]
        merged = with_ref.merge(of_null[["patient", "value"]], on="patient",
                                suffixes=("", "_own"))
        assert len(merged) == len(with_ref)
        np.testing.assert_allclose(merged["null_value"], merged["value_own"], atol=1e-12)


@pytest.mark.parametrize("name", sorted(STORED) + ["gene_importance"])
def test_exclude_pools_is_recorded_in_params(pooled, name):
    model, adata = pooled
    a = adata.copy()
    _run(name, model, a, inplace=True)
    assert tcri.get.params(a, name)["exclude_pools"] is True


# ── nothing to exclude ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(TOOLS) + ["gene_importance"])
def test_exclude_pools_without_a_pooling_step_changes_nothing(pooled, name):
    """The same labels with the derivation record removed, as a column pooled by hand or one
    from an object that predates the record: nothing is excluded, and the result is the one
    that counts the pools."""
    model, adata = pooled
    by_hand = adata.copy()
    del by_hand.uns[K.CLONOTYPE_DERIVATIONS]
    assert _resolve_clones(by_hand, None, True) is None
    _assert_same(_run(name, model, by_hand), _run(name, model, by_hand, exclude_pools=False))
    _assert_same(_run(name, model, by_hand), _run(name, model, adata, exclude_pools=False))


def test_without_tcri_metadata_the_model_names_the_registered_column(pooled):
    """An object ``to_anndata`` has not written records no registered column, so reading the
    object alone leaves nothing out, and nothing raises. ``gene_importance`` holds the model and
    reads the column the model registered, so it leaves out the same cells as on the written
    object."""
    model, adata = pooled
    bare = adata.copy()
    del bare.uns[K.METADATA], bare.uns[K.CLONOTYPE_CATEGORIES]
    assert _resolve_clones(bare, None, True) is None
    assert _resolve_clones(bare, None, True, column=CLONES) == _kept(adata)

    kw = dict(genes=[adata.var_names[1]], groupby="patient", null_model=None, inplace=False)
    by_default = tcri.perturb.gene_importance(model, bare, **kw)
    _assert_same(by_default, tcri.perturb.gene_importance(model, adata, **kw))
    assert not _same(by_default,
                     tcri.perturb.gene_importance(model, bare, exclude_pools=False, **kw))


# ── permutation_null restricts like the metrics ──────────────────────────────

def test_permutation_null_clones_restricts_like_the_metrics(pooled):
    """Per patient, the null's empirical joint holds the clones the metric holds at the same
    covariate, and ``observed`` is the empirical MI of exactly their cells. An explicit
    ``clones=`` restricts to exactly its clones."""
    _, adata = pooled
    obs = adata.obs
    kw = dict(covariate=POST, groupby="patient", n_perm=20, random_state=0)
    kept = _kept(adata)
    by_default = tcri.diag.permutation_null(adata, **kw)
    per_clone = tcri.tl.phenotypic_entropy(adata, covariate=POST, null_model=None,
                                           inplace=False)["result"]

    def _observed(null, patient):
        return float(null.loc[null["patient"] == patient, "observed"].iloc[0])

    def _empirical_mi(cells):
        joint = pd.crosstab(cells[CLONES].astype(str), cells["phenotype"].astype(str))
        return _mi_from_joint(joint.to_numpy(dtype=float), normalized=True, mode="min")

    for patient, rows in per_clone.groupby("patient", observed=True):
        at = obs[(obs["condition"] == POST) & (obs["patient"] == patient)]
        cells = at[at[CLONES].isin(kept)]
        assert set(rows["clonotype"]) == set(cells[CLONES])
        assert _observed(by_default, patient) == pytest.approx(_empirical_mi(cells), abs=1e-12)

    some = kept[::2]
    subset = tcri.diag.permutation_null(adata, clones=some, **kw)
    for patient in subset["patient"]:
        at = obs[(obs["condition"] == POST) & (obs["patient"] == patient)]
        cells = at[at[CLONES].isin(some)]
        assert _observed(subset, patient) == pytest.approx(_empirical_mi(cells), abs=1e-12)


# ── the record survives a saved session ──────────────────────────────────────

def test_pool_record_round_trips_through_a_saved_session(pooled, tmp_path):
    """The derivation record, the pools it lists and the clones they resolve to survive
    ``save_tcri_session`` and ``load_tcri_session``, so a reloaded object leaves out the same
    pools and reproduces the stored results and their ``exclude_pools``."""
    model, adata = pooled
    a = adata.copy()
    names = ("mutual_information", "phenotypic_flux")
    stored = {name: _run(name, model, a, inplace=True) for name in names}
    save_tcri_session(model, a, str(tmp_path / "session"))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, loaded = load_tcri_session(str(tmp_path / "session"))

    assert _derivation_steps(loaded) == _derivation_steps(adata)
    assert loaded.uns[K.METADATA][K.CLONE_COL] == CLONES
    assert _pool_labels(loaded, CLONES) == _pool_labels(adata, CLONES)
    assert _resolve_clones(loaded, None, True) == _kept(adata)
    for name, want in stored.items():
        assert bool(tcri.get.params(loaded, name)["exclude_pools"]) is True
        _assert_same(_run(name, model, loaded), want)
