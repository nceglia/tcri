"""The item rows of a change equal the recorded item rows on a fixed substrate.

A change of phenotypic entropy has one item row per persistent clone and draw, and a change of
clonotypic entropy one per phenotype and draw, recomputed on the persistent clones. This module
compares those rows, at every draw, with the rows in ``tests/snapshots/change_items.csv``: the
labels exactly, and ``value``, ``value_from`` and ``value_to`` to a relative 1e-12, the room
floating point leaves between platforms.

The substrate is fixed rather than trained. :func:`build_substrate` fits a small model only so the
object carries every key a fit writes, then writes over everything the engine reads that training
sets (``p_ct``, the guide concentration, the Dirichlet scale, the gate and the per-cell logits)
with arrays from a seeded numpy ``Generator``. Every engine output is then a function of those
arrays and the registration alone, so the comparison does not move with the optimizer. The cases
with ``n_samples > 0`` also depend on torch's seeded Dirichlet sampler, which a torch release may
change, so they run only under the torch the rows were recorded with.
"""
from __future__ import annotations

import contextlib
import io
import logging
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import tcri
from tcri._state import keys as K
from tcri.model._model import TCRIModel

SNAPSHOT = Path(__file__).parent.parent / "snapshots" / "change_items.csv"

LEVELS = ("pre", "post")
PHENOTYPES = ("phen_0", "phen_1", "phen_2", "phen_3")
#: Per patient: clones at both levels, clones at ``pre`` only, clones at ``post`` only.
CLONES = {"P0": (5, 2, 1), "P1": (4, 1, 2), "P2": (3, 2, 2)}

METRICS = {"phenotypic_entropy": "clonotype", "clonotypic_entropy": "phenotype"}
#: The draw settings compared. Every case groups by the registered replicate, ``patient``: by
#: default, or named.
CASES = {
    "plug_in": dict(n_samples=0),
    "draws": dict(n_samples=4, random_state=5),
    "weighted_draws": dict(n_samples=3, random_state=2, weighted=True),
    "named_groupby": dict(n_samples=4, random_state=5, groupby="patient"),
}
#: The torch the recorded rows were drawn with; the cases with draws need its Dirichlet sampler.
RECORDED_TORCH = "2.13.0"
LABELS = ["metric", "case", "patient", "cov_from", "cov_to", "item", "draw"]
VALUES = ["value", "value_from", "value_to"]


def build_substrate(seed: int = 0) -> ad.AnnData:
    """Three patients, each with clones at both levels and at one level only, on a fixed
    substrate.

    The cells, their counts and their phenotype labels come from ``default_rng(seed)``; after a
    two-epoch fit, ``p_ct``, the guide concentration, the per-cell logits, the Dirichlet scale and
    the gate are replaced with values from ``default_rng(seed + 1)``.
    """
    rng = np.random.default_rng(seed)
    cells = []
    for patient, (both, pre_only, post_only) in CLONES.items():
        clones = [f"c{i}@{patient}" for i in range(both + pre_only + post_only)]
        at = {"pre": clones[:both + pre_only],
              "post": clones[:both] + clones[both + pre_only:]}
        for level in LEVELS:
            for clone in at[level]:
                cells += [(clone, level, patient)] * int(rng.integers(2, 8))
    obs = pd.DataFrame(cells, columns=["clone_id", "timepoint", "patient"],
                       index=[f"cell_{i}" for i in range(len(cells))])
    obs["phenotype"] = [PHENOTYPES[i % len(PHENOTYPES)] for i in range(len(obs))]
    for col in obs.columns:
        obs[col] = obs[col].astype("category")
    obs["timepoint"] = obs["timepoint"].cat.reorder_categories(list(LEVELS))
    adata = ad.AnnData(X=rng.poisson(2.0, size=(len(obs), 15)).astype(np.float32), obs=obs)
    adata.layers["counts"] = adata.X.copy()

    logging.disable(logging.INFO)
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                                    phenotype_key="phenotype", covariate_key="timepoint",
                                    batch_key="patient", replicate="patient")
            model = TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
                              classifier_hidden=16, K=len(PHENOTYPES), seed=0,
                              name="change_items")
            model.train(max_epochs=2, batch_size=128, n_epochs_kl_warmup=1, accelerator="cpu",
                        enable_progress_bar=False, enable_model_summary=False)
            model.to_anndata(adata)
    finally:
        logging.disable(logging.NOTSET)

    sub = np.random.default_rng(seed + 1)
    n_ct, n_phen = np.asarray(adata.uns[K.P_CT]).shape
    adata.uns[K.P_CT] = sub.dirichlet(np.ones(n_phen), size=n_ct)
    adata.uns[K.CONC_CT] = sub.uniform(2.0, 30.0, size=(n_ct, n_phen))
    adata.obsm[K.X_LOGITS] = sub.normal(0.0, 1.5, size=(adata.n_obs, n_phen)).astype(np.float32)
    adata.uns[K.LOCAL_SCALE] = 10.0
    adata.uns[K.GATE_PROB] = 0.5
    return adata


def canonical(frame: pd.DataFrame, *, metric: str, case: str) -> pd.DataFrame:
    """The item draws of one call as ``LABELS + VALUES``, labels as strings, values as float64,
    sorted on the labels."""
    out = frame.rename(columns={METRICS[metric]: "item"}).assign(metric=metric, case=case)
    out = out[LABELS + VALUES].copy()
    for col in LABELS:
        out[col] = out[col].astype(int) if col == "draw" else out[col].astype(str)
    for col in VALUES:
        out[col] = out[col].astype(np.float64)
    return out.sort_values(LABELS, kind="stable").reset_index(drop=True)


def read_snapshot(path: Path = SNAPSHOT) -> pd.DataFrame:
    """The recorded rows, read back bit for bit (``float_precision="round_trip"``)."""
    frame = pd.read_csv(path, dtype={c: str for c in LABELS if c != "draw"},
                        float_precision="round_trip", keep_default_na=False,
                        na_values={c: [""] for c in VALUES})
    frame["draw"] = frame["draw"].astype(int)
    for col in VALUES:
        frame[col] = frame[col].astype(np.float64)
    return frame.sort_values(LABELS, kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def substrate():
    return build_substrate()


def _item_draws(adata, metric, case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = getattr(tcri.tl, metric)(adata, cov_from="pre", cov_to="post", null_model=None,
                                       inplace=False, **CASES[case])
    return canonical(res["item"]["draws"], metric=metric, case=case)


def test_the_substrate_has_clones_at_one_level_only(substrate):
    """Each patient has persistent clones and clones at one level only, so the item rows are a
    strict subset of the clones the replicate counts."""
    obs = substrate.obs
    for patient in CLONES:
        mine = obs.loc[obs["patient"] == patient]
        at = {lv: set(mine.loc[mine["timepoint"] == lv, "clone_id"].astype(str))
              for lv in LEVELS}
        assert at["pre"] & at["post"]
        assert at["pre"] - at["post"] and at["post"] - at["pre"]


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("metric", list(METRICS))
def test_change_item_rows_equal_the_recorded_rows(metric, case, substrate):
    """Every item row of the change from ``pre`` to ``post``, at every draw, equals its recorded
    row: the same rows and labels, and the same ``value``, ``value_from`` and ``value_to``."""
    import torch

    if CASES[case].get("n_samples") and torch.__version__.split("+")[0] != RECORDED_TORCH:
        pytest.skip(f"the draws were recorded with torch {RECORDED_TORCH}'s Dirichlet sampler")
    recorded = read_snapshot()
    expected = recorded.loc[(recorded["metric"] == metric) & (recorded["case"] == case)]
    expected = expected.reset_index(drop=True)
    assert len(expected), f"the snapshot has no rows for {metric} {case}"
    got = _item_draws(substrate, metric, case)
    pd.testing.assert_frame_equal(got[LABELS], expected[LABELS], check_exact=True)
    pd.testing.assert_frame_equal(got[VALUES], expected[VALUES], check_exact=False, rtol=1e-12,
                                  atol=1e-14)
