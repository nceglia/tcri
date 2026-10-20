"""Changes between two covariate levels: ``cov_from`` and ``cov_to`` on every metric.

What these pin, in order of how much they would cost to get wrong:

1. a change is ``cov_to`` − ``cov_from`` taken WITHIN a draw, so its interval is the interval of
   the difference: HDIs do not subtract, and the interval of a change cannot be rebuilt from the
   intervals of its endpoints;
2. both levels come from ONE shared sample, so a level against itself changes by exactly 0,
   whatever ``random_state`` is;
3. each side of a replicate's change is that level's single-covariate value, counting every
   clone the replicate has there, while the item rows hold only the persistent clones, those at
   both levels within the replicate;
4. the covariate rows of ``stats`` are a signed-rank test across replicates, and every number in
   them, ``n`` included, describes the replicates the test used, never clones or draws.

Mutual information has no items, so its change has a replicate level only. Flux is already a
distance between the two levels, so it has no endpoints.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import logging
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu, wilcoxon

import tcri
import tcri.tools._entropy as entropy_module
import tcri.tools._mutual_information as mi_module
from tcri._compute import _tables
from tcri._compute._tables import clones_at, joint_draws
from tcri._state import keys as K
from tcri._stats import stars
from tcri.datasets import simulate_tcri
from tcri.model._model import TCRIModel
from tcri.tools._entropy import _clonotypic_one, _phenotypic_one
from .test_metrics import _assert_same_result

A, B = "cov_0", "cov_1"
DIFFERENCES = ["mutual_information", "clonotypic_entropy", "phenotypic_entropy"]
METRICS = DIFFERENCES + ["phenotypic_flux"]
ITEM = {"clonotypic_entropy": "phenotype", "phenotypic_entropy": "clonotype",
        "phenotypic_flux": "clonotype"}
#: The numbers of a covariate row that summarize the replicates its test used, beside ``n``.
ROW_SUMMARY = ("mean_a", "mean_b", "sd_a", "sd_b", "ci_low_a", "ci_high_a", "ci_low_b",
               "ci_high_b", "delta")


def _levels(metric):
    return ("replicate",) if metric == "mutual_information" else ("replicate", "item")


def _run(metric, adata, **kwargs):
    """The full result of one call (``inplace=False``), its warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(tcri.tl, metric)(adata, inplace=False, **kwargs)


def _recorded(metric, adata, **kwargs):
    """``(result, warnings)`` of one call, every warning recorded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = getattr(tcri.tl, metric)(adata, inplace=False, **kwargs)
    return result, caught


def _covariate_rows(stats):
    return stats.loc[stats["comparison"] == "covariate"]


def _phenotypes(adata):
    return sorted(str(p) for p in adata.uns[K.PHENOTYPE_CATEGORIES])


# ── fixtures ─────────────────────────────────────────────────────────────────

def _blocks(specs):
    """One AnnData from ``simulate_tcri`` blocks, one per patient, clone ids patient-scoped."""
    parts = []
    for patient, kwargs in specs:
        block = simulate_tcri(n_phenotypes=4, n_genes=25, **kwargs)
        block.obs["clone_id"] = block.obs["clone_id"].astype(str) + "@" + patient
        block.obs["patient"] = patient
        block.obs_names = [f"{patient}_{j}" for j in range(block.n_obs)]
        parts.append(block)
    adata = ad.concat(parts, join="outer", label=None)
    for col in ("clone_id", "phenotype", "covariate", "patient"):
        adata.obs[col] = adata.obs[col].astype("category")
    adata.layers["counts"] = adata.X.copy()
    return adata


def _fit(adata, name, *, covariate_key="covariate", max_epochs=10, n_nulls=0):
    """Register, train and write ``adata``'s substrate, under the model name ``name``.

    ``n_nulls`` covariate nulls are fitted beside it. The model name scopes the fixture's
    parameter-store keys, so it leaves every other fixture's fit alone.
    """
    logging.disable(logging.INFO)
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                                    phenotype_key="phenotype", covariate_key=covariate_key,
                                    batch_key="patient", replicate="patient")
            n_phenotypes = len(adata.obs["phenotype"].cat.categories)
            model = TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
                              classifier_hidden=16, K=n_phenotypes, seed=0, name=name)
            model.train(max_epochs=max_epochs, batch_size=128, n_epochs_kl_warmup=4,
                        accelerator="cpu", enable_progress_bar=False,
                        enable_model_summary=False)
            model.to_anndata(adata)
            if n_nulls:
                tcri.null.covariate(model, adata, n_nulls=n_nulls, enable_progress_bar=False,
                                    enable_model_summary=False)
    finally:
        logging.disable(logging.NOTSET)
    return model


def _clones_by_level(adata, patient):
    obs = adata.obs.loc[adata.obs["patient"] == patient]
    return {lv: set(obs.loc[obs["covariate"].astype(str) == lv, "clone_id"].astype(str))
            for lv in (A, B)}


@pytest.fixture(scope="module")
def ragged():
    """One fitted patient, P0, some of whose clones are at one level only.

    Built rather than carved: the package refuses a sliced AnnData, whose full-space ``uns``
    registration arrays misalign against the subset. Thinning the cells instead makes clones at
    one level only arise the way they do in real data.
    """
    adata = _blocks([("P0", dict(n_clones=30, n_cells=260, n_covariates=2,
                                 omega_concentration=0.5, seed=3))])
    _fit(adata, "changes_ragged")
    at = _clones_by_level(adata, "P0")
    assert at[A] - at[B] and at[B] - at[A], "the fixture has no clone at one level only"
    return adata, at


@pytest.fixture(scope="module")
def uneven():
    """Three fitted patients; P2 has cells at ``cov_0`` only."""
    adata = _blocks([
        ("P0", dict(n_clones=8, n_cells=150, n_covariates=2, omega_concentration=0.3, seed=4)),
        ("P1", dict(n_clones=8, n_cells=150, n_covariates=2, omega_concentration=0.6, seed=6)),
        ("P2", dict(n_clones=8, n_cells=120, n_covariates=1, omega_concentration=0.3, seed=5)),
    ])
    _fit(adata, "changes_uneven")
    assert not _clones_by_level(adata, "P2")[B]
    return adata


@pytest.fixture(scope="module")
def contraction():
    """One patient: eight clones at ``pre``, four of them still at ``post``, every clone with one
    phenotype profile, in the fit and in its covariate null.

    The fitted substrate is written over with that profile (``p_ct``) and with logits of 0, so at
    ``n_samples=0`` the engine returns the profile itself as every clone's row. Fitting first
    gives the object every key a fit and its null carry; the covariate null keeps each clone's
    cells at each level, so it has the fit's clones at both.
    """
    rng = np.random.default_rng(0)
    clones = [f"c{i}@P0" for i in range(8)]
    cells = [(c, "pre") for c in clones for _ in range(6)]
    cells += [(c, "post") for c in clones[:4] for _ in range(6)]
    obs = pd.DataFrame(cells, columns=["clone_id", "timepoint"],
                       index=[f"cell_{i}" for i in range(len(cells))])
    obs["phenotype"] = [f"phen_{i % 3}" for i in range(len(obs))]
    obs["patient"] = "P0"
    for col in obs.columns:
        obs[col] = obs[col].astype("category")
    obs["timepoint"] = obs["timepoint"].cat.reorder_categories(["pre", "post"])
    adata = ad.AnnData(X=rng.poisson(2.0, size=(len(obs), 15)).astype(np.float32), obs=obs)
    adata.layers["counts"] = adata.X.copy()
    _fit(adata, "changes_contraction", covariate_key="timepoint", max_epochs=2, n_nulls=1)

    profile = np.array([0.5, 0.3, 0.2])
    for fit in (None, *K.members(adata, "null.covariate")):
        n_ct = np.asarray(adata.uns[K.fit_key(K.P_CT, fit)]).shape[0]
        adata.uns[K.fit_key(K.P_CT, fit)] = np.tile(profile, (n_ct, 1))
        logits = adata.obsm[K.fit_key(K.X_LOGITS, fit)]
        adata.obsm[K.fit_key(K.X_LOGITS, fit)] = np.zeros_like(np.asarray(logits))
    return adata


#: patient -> (its clones at ``pre``, its clones at ``post``), the ``post`` clones its first ones
STEADY = {"P0": (4, 4), "P1": (8, 4), "P2": (4, 1), "P3": (16, 2)}


@pytest.fixture(scope="module")
def steady():
    """Four patients, every clone with one phenotype profile, six cells per clone and level,
    each with the clone counts of ``STEADY``.

    The fitted substrate is written over as in :func:`contraction`, so at ``n_samples=0`` every
    clone's row is the profile and a level's clonotypic entropy in bits is log2 of its clone
    count: from ``pre`` to ``post`` P0 changes by exactly 0, its two sides being one computation
    on four clones, and P1, P2 and P3 by −1, −2 and −3.
    """
    rng = np.random.default_rng(0)
    cells = []
    for patient, (n_pre, n_post) in STEADY.items():
        clones = [f"c{i}@{patient}" for i in range(max(n_pre, n_post))]
        cells += [(c, "pre", patient) for c in clones[:n_pre] for _ in range(6)]
        cells += [(c, "post", patient) for c in clones[:n_post] for _ in range(6)]
    obs = pd.DataFrame(cells, columns=["clone_id", "timepoint", "patient"],
                       index=[f"cell_{i}" for i in range(len(cells))])
    obs["phenotype"] = [f"phen_{i % 3}" for i in range(len(obs))]
    for col in obs.columns:
        obs[col] = obs[col].astype("category")
    obs["timepoint"] = obs["timepoint"].cat.reorder_categories(["pre", "post"])
    adata = ad.AnnData(X=rng.poisson(2.0, size=(len(obs), 15)).astype(np.float32), obs=obs)
    adata.layers["counts"] = adata.X.copy()
    _fit(adata, "changes_steady", covariate_key="timepoint", max_epochs=2)

    n_ct = np.asarray(adata.uns[K.P_CT]).shape[0]
    adata.uns[K.P_CT] = np.tile(np.array([0.5, 0.3, 0.2]), (n_ct, 1))
    adata.obsm[K.X_LOGITS] = np.zeros_like(np.asarray(adata.obsm[K.X_LOGITS]))
    return adata


# ── the estimand ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("metric", DIFFERENCES)
def test_a_change_is_to_minus_from_within_a_draw(metric, cohort):
    """Positive means it increased, and the subtraction happens per draw, on the value and on the
    adjusted value, at both levels.

    The per-draw part is what a caller cannot do from two stored results: the distribution of a
    difference is not recoverable from two marginal summaries. The summaries reduce a change and
    its endpoints under one mask, so the identity holds in them as on every draw.
    """
    _, adata = cohort
    res = _run(metric, adata, cov_from=A, cov_to=B, n_samples=20, random_state=0)

    for level in _levels(metric):
        draws, summary = res[level]["draws"], res[level]["summary"]
        for name, frame in (("draws", draws), ("summary", summary)):
            assert {"cov_from", "cov_to", "value", "value_from", "value_to"} <= set(frame.columns)
            assert "covariate" not in frame.columns, "a change is labeled by its two levels"
            assert set(frame["cov_from"]) == {A} and set(frame["cov_to"]) == {B}
            for col in ("value", "adjusted"):
                change = frame[col].to_numpy(dtype=float)
                ends = (frame[f"{col}_to"].to_numpy(dtype=float)
                        - frame[f"{col}_from"].to_numpy(dtype=float))
                np.testing.assert_allclose(change, ends, rtol=0, atol=1e-12,
                                           err_msg=f"{level} {name}: {col} is not to - from")
        assert draws["draw"].nunique() == 20, "the endpoints were reduced before the subtraction"

        # the summary's value is the mean of the per-draw changes, and its HDI is the change's
        # own. The draws hold one block per null with the same value in each.
        key = ["patient"] + ([ITEM[metric]] if level == "item" else [])
        own = draws.drop_duplicates(subset=key + ["draw"])
        agg = own.groupby(key, observed=True)["value"].agg(["mean", "median"]).reset_index()
        got = summary.merge(agg, on=key, validate="one_to_one")
        assert len(got) == len(summary)
        np.testing.assert_allclose(got["value"], got["mean"], rtol=0, atol=1e-12)

        # The HDI brackets the MEDIAN, not necessarily the mean: an HDI is the narrowest
        # interval holding 94% of the mass, so one outlying draw can pull the mean outside it.
        # Any interval holding more than half the mass contains the median.
        fin = got.dropna(subset=["hdi_low", "hdi_high", "median"])
        assert len(fin)
        assert ((fin["hdi_low"] <= fin["median"]) & (fin["median"] <= fin["hdi_high"])).all()

    if metric == "mutual_information":
        frame = res["replicate"]["summary"]
        assert {"denom_from", "denom_to"} <= set(frame.columns) and "denom" not in frame.columns


#: (metric, settings) whose replicate change must be two single-covariate calls' difference.
SIDES = [
    ("mutual_information", {}),
    ("mutual_information", {"normalize_mode": "average"}),
    ("mutual_information", {"normalized": False}),
    ("clonotypic_entropy", {}),
    ("clonotypic_entropy", {"normalized": False}),
    ("clonotypic_entropy", {"n_clones_ref": 40}),
    ("clonotypic_entropy", {"weighted": True}),
    ("phenotypic_entropy", {}),
    ("phenotypic_entropy", {"weighted": True}),
]


@pytest.mark.parametrize("metric, settings", SIDES,
                         ids=[f"{m}-{'-'.join(f'{k}={v}' for k, v in s.items()) or 'default'}"
                              for m, s in SIDES])
def test_each_side_of_a_replicate_change_is_the_single_covariate_value(metric, settings,
                                                                       ragged):
    """Every per-replicate change except flux's is the per-draw difference of two
    single-covariate calls with the same integer seed, normalized clonotypic entropy included:
    each side counts every clone the replicate has at that level and is normalized there.

    On a fixture with clones at one level only, so a side computed on the persistent clones
    alone would differ.
    """
    adata, _ = ragged
    kwargs = dict(n_samples=5, random_state=13, null_model=None, **settings)
    change = _run(metric, adata, cov_from=A, cov_to=B, **kwargs)["replicate"]["draws"]
    assert len(change) == 5

    for side, level in (("from", A), ("to", B)):
        single = _run(metric, adata, covariate=level, **kwargs)["replicate"]["draws"]
        cols = ["value"] + (["denom"] if metric == "mutual_information" else [])
        single = single[["patient", "draw", *cols]].rename(columns={c: f"{c}_single"
                                                                    for c in cols})
        both = change.merge(single, on=["patient", "draw"], validate="one_to_one")
        assert len(both) == len(change)
        for col in cols:
            np.testing.assert_array_equal(both[f"{col}_{side}"], both[f"{col}_single"],
                                          err_msg=f"{col}_{side} is not the call at {level}")
    np.testing.assert_array_equal(change["value"], change["value_to"] - change["value_from"])


@pytest.mark.parametrize("random_state", [None, 0, "generator"],
                         ids=["None", "integer", "numpy-generator"])
@pytest.mark.parametrize("metric", METRICS)
def test_a_level_against_itself_changes_by_exactly_zero(metric, random_state, cohort):
    """Both levels come from one shared sample, so a level against itself changes by exactly 0
    on every draw, at both levels, in the value and in the adjusted value.

    Independent draws would report the sampling noise floor instead, which grows with
    ``n_samples``. A change of exactly 0 leaves the signed-rank test, so its covariate row has
    no test and summarizes no replicate: ``n`` is 0 and the arms and ``delta`` are NaN.

    One draw serves both sides here, so this cannot see two levels drawn from two seeds:
    :func:`test_one_change_call_hands_every_engine_call_one_integer_seed` does.
    """
    _, adata = cohort
    rs = np.random.default_rng(7) if random_state == "generator" else random_state
    res = _run(metric, adata, cov_from=A, cov_to=A, n_samples=12, random_state=rs)

    for level in _levels(metric):
        for slot in ("draws", "summary"):
            frame = res[level][slot]
            for col in ("value", "adjusted"):
                v = frame[col].to_numpy(dtype=float)
                assert np.isfinite(v).all(), f"{level} {slot}: a non-finite {col}"
                assert np.abs(v).max() == 0.0, (
                    f"{level} {slot}: a level against itself changes {col} by "
                    f"{np.abs(v).max():.3g}; the two sides are not one draw")
            if metric != "phenotypic_flux":
                np.testing.assert_array_equal(frame["value_from"], frame["value_to"])

    rows = _covariate_rows(res["stats"])
    assert len(rows) == 1
    assert (rows["n_a"] == 0).all() and (rows["n_b"] == 0).all()
    assert rows["p"].isna().all() and rows["stat"].isna().all() and (rows["stars"] == "").all()
    for col in ROW_SUMMARY:
        assert rows[col].isna().all(), f"{col} is set on a row that used no replicate"


@pytest.mark.parametrize("null_model", [None, "auto"], ids=["no-reference", "reference"])
@pytest.mark.parametrize("random_state", [None, "generator"], ids=["None", "numpy-generator"])
@pytest.mark.parametrize("metric", METRICS)
def test_one_change_call_hands_every_engine_call_one_integer_seed(metric, random_state,
                                                                  null_model, cohort,
                                                                  monkeypatch):
    """Between two different levels, every engine call of one change call, both levels of every
    replicate, receives the same integer seed, so both levels come from one shared sample.

    ``None`` and a numpy ``Generator`` are the stateful cases: handed to the engine as they are,
    each call would draw its own seed and the two levels would be independent. A level against
    itself cannot show this, since its two sides are one draw. With a reference, each of its
    fits is a call of its own, with its own seed (``_reference.null_seed``), and the same holds
    within each fit.
    """
    _, adata = cohort
    calls = []

    def spy(adata, covariate, **kwargs):
        calls.append((kwargs.get("fit"), covariate, kwargs["random_state"]))
        return joint_draws(adata, covariate, **kwargs)

    for module in (_tables, entropy_module, mi_module):
        monkeypatch.setattr(module, "joint_draws", spy)
    rs = np.random.default_rng(7) if random_state == "generator" else random_state
    _run(metric, adata, cov_from=A, cov_to=B, n_samples=3, random_state=rs,
         null_model=null_model)

    by_fit = {}
    for fit, level, seed in calls:
        by_fit.setdefault(fit, []).append((level, seed))
    expected_fits = {None} | (set() if null_model is None
                              else set(K.members(adata, "null.covariate")))
    assert set(by_fit) == expected_fits
    n_replicates = adata.obs["patient"].nunique()
    for fit, seen in by_fit.items():
        assert sorted(level for level, _ in seen) == [A] * n_replicates + [B] * n_replicates, (
            f"fit {fit!r}: not one engine call per level and replicate")
        seeds = {seed for _, seed in seen}
        assert len(seeds) == 1, f"fit {fit!r}: {len(seeds)} seeds in one change call"
        seed = seeds.pop()
        assert isinstance(seed, int) and not isinstance(seed, bool), (
            f"fit {fit!r}: the engine received {type(seed).__name__}, not an integer")


# ── the items: the persistent clones ─────────────────────────────────────────

@pytest.mark.parametrize("metric", ["phenotypic_entropy", "phenotypic_flux",
                                    "clonotypic_entropy"])
def test_item_rows_hold_exactly_the_persistent_clones(metric, ragged):
    """A clone at one level only has no item row: per clone, the items are the persistent
    clones; per phenotype, every phenotype has its row, computed on the persistent clones
    (:func:`test_item_changes_are_the_kernels_on_the_persistent_clones`). The call does not warn
    about the clones it leaves out of the items.
    """
    adata, at = ragged
    res, caught = _recorded(metric, adata, cov_from=A, cov_to=B, null_model=None)
    assert not [str(w.message) for w in caught if str(w.message).startswith(f"{metric}:")], (
        "a replicate with a clone at both levels needs no warning")

    for slot in ("draws", "summary"):
        items = res["item"][slot]
        if metric == "clonotypic_entropy":
            assert sorted(set(items["phenotype"].astype(str))) == _phenotypes(adata)
            assert len(items) == len(_phenotypes(adata)) * (items["draw"].nunique()
                                                            if slot == "draws" else 1)
        else:
            assert set(items["clonotype"].astype(str)) == at[A] & at[B]
        if metric == "phenotypic_flux":
            assert not {"value_from", "value_to"} & set(items.columns), (
                "flux is a distance between the levels and has no endpoints")


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("metric", ["phenotypic_entropy", "clonotypic_entropy"])
def test_item_changes_are_the_kernels_on_the_persistent_clones(metric, weighted, ragged):
    """The item rows of a change are the item kernels run on the persistent clones' rows of each
    level's joint, from one shared draw: one value per persistent clone, or each phenotype
    recomputed on the persistent clones so both sides share one clone set and one normalizer.

    The expected values come from the kernels and ``joint_draws`` directly, with the same seed,
    and are compared exactly, at ``weighted=False`` and ``weighted=True``.
    """
    adata, _ = ragged
    draw_kw = dict(n_samples=4, random_state=5, weighted=weighted)
    items = _run(metric, adata, cov_from=A, cov_to=B, null_model=None,
                 **draw_kw)["item"]["draws"]

    to_clones = set(clones_at(adata, B))
    persistent = [c for c in clones_at(adata, A) if c in to_clones]
    if metric == "phenotypic_entropy":
        def kernel(J, cols):
            return _phenotypic_one(persistent, J, cols, normalized=True)
    else:
        def kernel(J, cols):
            return _clonotypic_one(J, cols, normalized=True)

    expected = {}
    for side, level in (("from", A), ("to", B)):
        draws, cols = joint_draws(adata, level, temperature=1.0, clones=None, **draw_kw)
        rows = []
        for ids, J in draws:
            index = {c: i for i, c in enumerate(ids)}
            rows.append(kernel(np.asarray(J)[[index[c] for c in persistent]], cols))
        expected[side] = rows

    item_col = ITEM[metric]
    assert items["draw"].nunique() == 4
    for draw, chunk in items.groupby("draw"):
        labels = chunk[item_col].astype(str).tolist()
        h_from = np.array([expected["from"][draw][k] for k in labels])
        h_to = np.array([expected["to"][draw][k] for k in labels])
        assert sorted(labels) == sorted(expected["from"][draw])
        np.testing.assert_array_equal(chunk["value_from"].to_numpy(dtype=float), h_from)
        np.testing.assert_array_equal(chunk["value_to"].to_numpy(dtype=float), h_to)
        np.testing.assert_array_equal(chunk["value"].to_numpy(dtype=float), h_to - h_from)


def test_the_persistent_clones_set_the_item_normalizer_not_the_replicate_one(ragged):
    """Per phenotype, a change recomputes H[P(c|φ)] on the persistent clones, which is not the
    difference of two single-covariate item values: each of those is normalized by its own
    level's clone count, so a contracting repertoire would report a change it did not have. Per
    replicate the opposite holds: the change IS the difference of the two single-covariate
    values.

    ``null_model=None``: both are statements about the value, and a reference subtracted from
    each side would not change which computation each one is.
    """
    adata, _ = ragged
    change = _run("clonotypic_entropy", adata, cov_from=A, cov_to=B, null_model=None)
    single = {lv: _run("clonotypic_entropy", adata, covariate=lv, null_model=None)
              for lv in (A, B)}

    def items(res):
        return res["item"]["summary"].set_index("phenotype")["value"]

    naive = items(single[B]) - items(single[A])
    assert not np.allclose(items(change).reindex(naive.index), naive, equal_nan=True), (
        "the persistent clones made no difference; the fixture cannot discriminate")

    rep = change["replicate"]["summary"]["value"].item()
    difference = (single[B]["replicate"]["summary"]["value"].item()
                  - single[A]["replicate"]["summary"]["value"].item())
    assert rep == pytest.approx(difference, abs=1e-12)


@pytest.mark.parametrize("side", ["from", "to"])
@pytest.mark.parametrize("metric", DIFFERENCES)
def test_a_clone_at_one_level_moves_that_levels_replicate_value(metric, side, ragged):
    """Each side of a replicate's change counts every clone the replicate has at that level, so
    removing a clone that is at one level only (with ``clones=``) moves that side and the change,
    leaves the other side exactly as it was, and leaves the items, the persistent clones,
    unchanged."""
    adata, at = ragged
    level, other = (A, B) if side == "from" else (B, A)
    one_sided = sorted(at[level] - at[other])[0]
    every = sorted(at[A] | at[B])
    kwargs = dict(cov_from=A, cov_to=B, null_model=None)
    full = _run(metric, adata, clones=every, **kwargs)
    without = _run(metric, adata, clones=[c for c in every if c != one_sided], **kwargs)

    f = full["replicate"]["summary"].iloc[0]
    w = without["replicate"]["summary"].iloc[0]
    kept = "value_to" if side == "from" else "value_from"
    assert f[f"value_{side}"] != w[f"value_{side}"], (
        f"removing {one_sided}, at {level} only, did not move value_{side}")
    assert f["value"] != w["value"]
    assert f[kept] == w[kept], f"removing a clone absent from {other} moved {kept}"
    if metric != "mutual_information":
        pd.testing.assert_frame_equal(full["item"]["summary"], without["item"]["summary"])


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_a_clone_at_one_level_does_not_move_the_flux(weighted, ragged):
    """Flux is defined on the clones at both levels: a clone at one level only is in neither its
    items nor its replicate value."""
    adata, at = ragged
    every = sorted(at[A] | at[B])
    persistent = sorted(at[A] & at[B])
    kwargs = dict(cov_from=A, cov_to=B, null_model=None, weighted=weighted)
    full = _run("phenotypic_flux", adata, clones=every, **kwargs)
    shared = _run("phenotypic_flux", adata, clones=persistent, **kwargs)
    for level in ("replicate", "item"):
        pd.testing.assert_frame_equal(full[level]["summary"], shared[level]["summary"],
                                      obj=level)


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_per_replicate_flux_is_the_weighted_mean_of_the_clone_flux(weighted, ragged):
    """Per draw, a replicate's flux is its persistent clones' flux weighted by each clone's share
    of the ``cov_from`` joint, renormalized over them: its cell count there at ``weighted=True``,
    and an equal share at ``weighted=False``, where it is the plain mean of the finite items."""
    adata, _ = ragged
    res = _run("phenotypic_flux", adata, cov_from=A, cov_to=B, n_samples=3, random_state=4,
               weighted=weighted, null_model=None)
    items, rep = res["item"]["draws"], res["replicate"]["draws"]
    at_from = adata.obs.loc[adata.obs["covariate"].astype(str) == A]
    counts = at_from["clone_id"].astype(str).value_counts()

    assert len(rep) == 3
    for (patient, draw), chunk in items.groupby(["patient", "draw"], observed=True):
        d = chunk["value"].to_numpy(dtype=float)
        w = (counts.reindex(chunk["clonotype"].astype(str)).to_numpy(dtype=float) if weighted
             else np.ones(len(d)))
        keep = np.isfinite(d)
        assert keep.any()
        expected = float(np.sum(w[keep] * d[keep]) / np.sum(w[keep]))
        got = rep.loc[(rep["patient"] == patient) & (rep["draw"] == draw), "value"].item()
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-15)


def test_a_contraction_without_redistribution_is_no_normalized_change(contraction):
    """Eight clones with one profile, four of them left at ``post``: the repertoire contracted
    and redistributed nothing. Normalized per level, each side is 1, so the change is 0, in the
    value and in the adjusted value; in bits the same contraction is log2(4) − log2(8) = −1,
    which is what the per-level normalizer takes out. The covariate null has the fit's clones at
    each level, so the adjusted change is 0 in bits too.
    """
    adata = contraction
    kwargs = dict(cov_from="pre", cov_to="post")
    res = _run("clonotypic_entropy", adata, **kwargs)
    rep = res["replicate"]["summary"].iloc[0]
    assert rep["value_from"] == pytest.approx(1.0, abs=1e-12)
    assert rep["value_to"] == pytest.approx(1.0, abs=1e-12)
    assert rep["value"] == pytest.approx(0.0, abs=1e-12)
    assert rep["adjusted"] == pytest.approx(0.0, abs=1e-12)
    assert res["stats"]["quantity"].tolist() == ["adjusted"]

    bits = _run("clonotypic_entropy", adata, normalized=False, **kwargs)
    rep = bits["replicate"]["summary"].iloc[0]
    assert rep["value_from"] == pytest.approx(3.0, abs=1e-12)
    assert rep["value_to"] == pytest.approx(2.0, abs=1e-12)
    assert rep["value"] == pytest.approx(-1.0, abs=1e-12)
    assert rep["adjusted"] == pytest.approx(0.0, abs=1e-12)

    # per phenotype, both sides are the four persistent clones, uniform
    items = res["item"]["summary"]
    assert len(items) == 3
    np.testing.assert_allclose(items["value"], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(items["value_from"], 1.0, rtol=0, atol=1e-12)


# ── what a replicate's inputs leave out ──────────────────────────────────────

@pytest.mark.parametrize("metric", METRICS)
def test_a_replicate_with_no_cells_at_a_level_is_counted_and_nan(metric, uneven):
    """A replicate with no cells at a level keeps its replicate row, NaN, and leaves the test's
    ``n`` (on the three differences: flux at ``null_model=None`` has no test); the call says how
    many such replicates there are, from the caller's line. Its item rows: one NaN row per
    phenotype for clonotypic entropy, none per clone.

    Its draws keep the level it has, its single-covariate value; its summary reduces the change
    and both endpoints under one mask, so a draw with a NaN side enters none of them."""
    res, caught = _recorded(metric, uneven, cov_from=A, cov_to=B, null_model=None)
    mine = [w for w in caught if str(w.message).startswith(f"{metric}:")]
    assert [str(w.message) for w in mine] == [
        f"{metric}: 1 of 3 patient groups have no counted clone at '{A}' or at '{B}'; their "
        f"values are NaN and they leave the tests."
    ]
    assert all(w.filename == __file__ for w in mine), "the warning names a line in the package"

    for slot in ("draws", "summary"):
        rep = res["replicate"][slot]
        assert set(rep["patient"].astype(str)) == {"P0", "P1", "P2"}
        p2 = rep["patient"].astype(str) == "P2"
        assert rep.loc[p2, "value"].isna().all()
        assert np.isfinite(rep.loc[~p2, "value"].to_numpy(dtype=float)).all()
        if metric != "phenotypic_flux":
            assert rep.loc[p2, "value_to"].isna().all()
            p2_from = rep.loc[p2, "value_from"].to_numpy(dtype=float)
            if slot == "draws":
                assert np.isfinite(p2_from).all(), "P2's cov_0 value is gone from its draws"
            else:
                assert np.isnan(p2_from).all(), "the summary reduced value_from on its own mask"

    if metric != "mutual_information":
        items = res["item"]["summary"]
        p2 = items.loc[items["patient"].astype(str) == "P2"]
        if metric == "clonotypic_entropy":
            assert sorted(p2["phenotype"].astype(str)) == _phenotypes(uneven)
            assert p2["value"].isna().all()
        else:
            assert p2.empty

    row = _covariate_rows(res["stats"]).iloc[0]
    if metric == "phenotypic_flux":
        # a flux row without a usable reference is pinned by its missing test alone
        assert np.isnan(row["p"]) and row["stars"] == ""
    else:
        assert row["n_a"] == row["n_b"] == 2


@pytest.mark.parametrize("metric", ["mutual_information", "clonotypic_entropy"])
def test_one_clone_at_a_level_raises_naming_the_replicate_and_that_level(metric, ragged):
    """A normalized replicate value whose normalizer is 0 raises, naming the metric, the
    replicate and the level where the replicate has one clone, whichever side of the change it
    is on; the same call in bits returns, with 0 at that level."""
    adata, at = ragged
    shared = sorted(at[A] & at[B])[0]
    for lonely, crowded in ((B, A), (A, B)):
        kept = [shared, sorted(at[crowded] - at[lonely])[0]]
        for cov_from, cov_to in ((A, B), (B, A)):
            with pytest.raises(ValueError,
                               match=rf"^{metric}: patient='P0' at covariate '{lonely}'"):
                _run(metric, adata, cov_from=cov_from, cov_to=cov_to, clones=kept,
                     null_model=None)
        bits = _run(metric, adata, cov_from=A, cov_to=B, clones=kept, null_model=None,
                    normalized=False)["replicate"]["summary"].iloc[0]
        assert bits["value_to" if lonely == B else "value_from"] == pytest.approx(0.0, abs=1e-12)


def test_one_persistent_clone_gets_item_rows_not_an_error(ragged):
    """Item values keep their kernels: with two clones at each level and one of them at both, the
    replicate values are defined and the items are computed on the one persistent clone, a
    clonotypic entropy of 0 per phenotype rather than an error."""
    adata, at = ragged
    shared = sorted(at[A] & at[B])[0]
    kept = [shared, sorted(at[A] - at[B])[0], sorted(at[B] - at[A])[0]]
    kwargs = dict(cov_from=A, cov_to=B, clones=kept, null_model=None)

    ce = _run("clonotypic_entropy", adata, **kwargs)
    assert np.isfinite(ce["replicate"]["summary"]["value"].to_numpy(dtype=float)).all()
    items = ce["item"]["summary"]
    assert sorted(items["phenotype"].astype(str)) == _phenotypes(adata)
    np.testing.assert_array_equal(items[["value", "value_from", "value_to"]].to_numpy(dtype=float),
                                  0.0)

    pe = _run("phenotypic_entropy", adata, **kwargs)
    assert pe["item"]["summary"]["clonotype"].astype(str).tolist() == [shared]


# ── the tests: covariate rows and split rows ─────────────────────────────────

@pytest.mark.parametrize("splitby", [None, "disease_status"], ids=["no-split", "split"])
@pytest.mark.parametrize("metric", DIFFERENCES)
def test_the_covariate_rows_are_the_signed_rank_test_of_the_replicate_changes(metric, splitby,
                                                                              cohort):
    """A change's covariate rows come first: one per split level with ``splitby`` (did each split
    change), one over every replicate without it. Each is the Wilcoxon signed-rank test of the
    replicates' adjusted changes, equal to scipy on a frame built by hand from the replicate
    summary, and every number in it describes the replicates that test used: the arms are the
    adjusted endpoints, ``delta = mean_b − mean_a``, and ``n`` counts replicates."""
    _, adata = cohort
    res = _run(metric, adata, cov_from=A, cov_to=B, splitby=splitby)
    stats, rep = res["stats"], res["replicate"]["summary"]

    rows = _covariate_rows(stats)
    assert stats["comparison"].tolist()[:len(rows)] == ["covariate"] * len(rows), (
        "the covariate rows come before the split rows")
    levels = [None] if splitby is None else sorted(rep[splitby].unique().tolist(), key=str)
    assert len(rows) == len(levels)
    assert len(stats) == len(rows) + (0 if splitby is None else 1)

    for (_, row), level in zip(rows.iterrows(), levels):
        assert row["quantity"] == "adjusted"
        assert (row["level_a"], row["level_b"]) == (A, B)
        assert row["replicate_unit"] == "patient"
        sub = rep if level is None else rep.loc[rep[splitby] == level]
        if splitby is not None:
            assert row[splitby] == level
        a = sub["adjusted_from"].to_numpy(dtype=float)
        b = sub["adjusted_to"].to_numpy(dtype=float)
        used = np.isfinite(a) & np.isfinite(b) & (b - a != 0)
        assert used.sum() == len(sub) == (6 if splitby is None else 3)
        assert row["n_a"] == row["n_b"] == used.sum()

        W, p = wilcoxon(b[used] - a[used])
        assert row["stat"] == pytest.approx(W) and row["p"] == pytest.approx(p)
        assert row["stars"] == stars(p)
        assert row["mean_a"] == pytest.approx(a[used].mean())
        assert row["mean_b"] == pytest.approx(b[used].mean())
        assert row["sd_a"] == pytest.approx(a[used].std(ddof=1))
        assert row["sd_b"] == pytest.approx(b[used].std(ddof=1))
        assert row["delta"] == pytest.approx(row["mean_b"] - row["mean_a"], abs=1e-12)


def test_a_covariate_row_summarizes_only_the_replicates_its_test_used(steady):
    """A replicate whose change is exactly 0 leaves the signed-rank test and, with it, every
    number of the covariate row: each arm's mean, sd and interval, ``delta`` and ``n`` are over
    the replicates with a non-zero change, so the row describes one set of replicates.

    On :func:`steady`, whose P0 changes by exactly 0 beside three non-zero changes, at
    ``null_model=None`` and in bits, where every endpoint is log2 of a clone count.
    """
    res = _run("clonotypic_entropy", steady, cov_from="pre", cov_to="post", normalized=False,
               null_model=None)
    rep = res["replicate"]["summary"].set_index("patient").loc[list(STEADY)]
    assert rep.loc["P0", "value"] == 0.0
    assert (rep["value"].drop("P0") != 0).all()
    np.testing.assert_allclose(rep["value_from"], [np.log2(n) for n, _ in STEADY.values()],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(rep["value_to"], [np.log2(n) for _, n in STEADY.values()],
                               rtol=0, atol=1e-12)

    used = rep.drop(index="P0")
    a = used["value_from"].to_numpy(dtype=float)
    b = used["value_to"].to_numpy(dtype=float)
    for name, every, kept in (("mean_a", rep["value_from"].mean(), a.mean()),
                              ("mean_b", rep["value_to"].mean(), b.mean()),
                              ("delta", rep["value"].mean(), (b - a).mean())):
        assert not np.isclose(every, kept), f"P0 does not move {name}; the fixture is inert"

    row = _covariate_rows(res["stats"]).iloc[0]
    assert row["quantity"] == "value"
    assert row["n_a"] == row["n_b"] == 3
    for suffix, arm in (("a", a), ("b", b)):
        lo, hi = np.percentile(arm, [2.5, 97.5])
        assert row[f"mean_{suffix}"] == pytest.approx(arm.mean())
        assert row[f"sd_{suffix}"] == pytest.approx(arm.std(ddof=1))
        assert row[f"ci_low_{suffix}"] == pytest.approx(lo)
        assert row[f"ci_high_{suffix}"] == pytest.approx(hi)
    assert row["delta"] == pytest.approx(b.mean() - a.mean())
    W, p = wilcoxon(b - a)
    assert row["stat"] == pytest.approx(W) and row["p"] == pytest.approx(p)


@pytest.mark.parametrize("metric", METRICS)
def test_the_split_rows_of_a_change_compare_the_replicate_changes(metric, cohort):
    """With ``splitby``, the split row contrasts the per-replicate changes between the splits: a
    Mann–Whitney test over patients, equal to scipy's, ``n`` the 3 patients per arm, however many
    items the change has."""
    _, adata = cohort
    res = _run(metric, adata, cov_from=A, cov_to=B, groupby="patient",
               splitby="disease_status")
    split = res["stats"].loc[res["stats"]["comparison"] == "split"]
    assert len(split) == 1
    row = split.iloc[0]
    assert row["disease_status"] == "control vs disease"
    assert row["replicate_unit"] == "patient" and row["quantity"] == "adjusted"
    assert row["n_a"] == 3 and row["n_b"] == 3

    rep = res["replicate"]["summary"]
    arm = {lv: rep.loc[rep["disease_status"] == lv, "adjusted"].to_numpy(dtype=float)
           for lv in ("control", "disease")}
    U, p = mannwhitneyu(arm["control"], arm["disease"], alternative="two-sided")
    assert row["stat"] == pytest.approx(U) and row["p"] == pytest.approx(p)
    assert row["delta"] == pytest.approx(arm["disease"].mean() - arm["control"].mean())
    if metric != "mutual_information":
        assert len(res["item"]["summary"]) > 6, "fixture too small to discriminate"


@pytest.mark.parametrize("splitby", [None, "disease_status"], ids=["no-split", "split"])
def test_the_flux_covariate_rows_test_the_adjusted_flux_against_zero(splitby, cohort):
    """Flux is never negative, so its covariate row tests the adjusted flux against 0: no arms,
    ``delta`` the mean adjusted flux over the replicates the test used, and the signed-rank test
    of those values against 0. Without a usable reference there is no test."""
    _, adata = cohort
    res = _run("phenotypic_flux", adata, cov_from=A, cov_to=B, splitby=splitby)
    rep = res["replicate"]["summary"]
    rows = _covariate_rows(res["stats"])
    levels = [None] if splitby is None else ["control", "disease"]
    assert len(rows) == len(levels)
    for (_, row), level in zip(rows.iterrows(), levels):
        assert row["quantity"] == "adjusted"
        for col in ("mean_a", "mean_b", "sd_a", "sd_b", "ci_low_a", "ci_high_a", "ci_low_b",
                    "ci_high_b"):
            assert np.isnan(row[col]), f"flux's covariate row has no arm, but {col} is set"
        sub = rep if level is None else rep.loc[rep["disease_status"] == level]
        adjusted = sub["adjusted"].to_numpy(dtype=float)
        used = np.isfinite(adjusted) & (adjusted != 0)
        assert row["n_a"] == row["n_b"] == used.sum() == len(sub)
        assert row["delta"] == pytest.approx(adjusted[used].mean())
        W, p = wilcoxon(adjusted[used])
        assert row["stat"] == pytest.approx(W) and row["p"] == pytest.approx(p)

    bare = _covariate_rows(_run("phenotypic_flux", adata, cov_from=A, cov_to=B,
                                splitby=splitby, null_model=None)["stats"])
    assert len(bare) == len(levels)
    assert bare["p"].isna().all() and (bare["stars"] == "").all()


@pytest.mark.parametrize("metric", METRICS)
def test_with_no_replicate_a_change_is_one_row_with_its_covariate_row(metric, cohort):
    """No replicate registered and no ``groupby``: the whole object is one replicate, whose
    change has its row, and its covariate row is present with ``n`` of 1 (0 when the change is
    exactly 0, and then its arms and ``delta`` are NaN) and no test."""
    _, adata = cohort
    bare = adata.copy()
    bare.uns[K.METADATA] = {**bare.uns[K.METADATA], "replicate": None}
    for cov_to, n in ((B, 1), (A, 0)):
        res = _run(metric, bare, cov_from=A, cov_to=cov_to, n_samples=3, random_state=0)
        summary = res["replicate"]["summary"]
        assert len(summary) == 1 and "patient" not in summary.columns
        assert np.isfinite(summary[["value", "null_value", "adjusted"]]
                           .to_numpy(dtype=float)).all()
        stats = res["stats"]
        assert stats["comparison"].tolist() == ["covariate"]
        row = stats.iloc[0]
        assert row["n_a"] == row["n_b"] == n
        assert np.isnan(row["p"]) and row["stars"] == ""
        if n == 0:
            for col in ROW_SUMMARY:
                assert np.isnan(row[col]), f"{col} is set on a row that used no replicate"


# ── the stored result ────────────────────────────────────────────────────────

@pytest.mark.parametrize("metric", METRICS)
def test_every_comparison_of_a_change_is_answered(metric, cohort):
    """The change rows of the comparison table, read through ``tcri.get`` from one stored call:
    the change per item (none for mutual information), per replicate, whether it is nonzero
    across replicates (a covariate row per split level) and whether it differs between splits
    (the split row). Each frame names its item and its two levels."""
    _, adata = cohort
    adata = adata.copy()
    assert getattr(tcri.tl, metric)(adata, cov_from=A, cov_to=B, groupby="patient",
                                    splitby="disease_status") is None

    replicates = tcri.get.replicates(adata, metric)
    assert sorted(replicates["patient"].astype(str)) == [f"P{i}" for i in range(6)]
    assert {"cov_from", "cov_to"} <= set(replicates.columns)
    assert "covariate" not in replicates.columns
    assert len(tcri.get.draws(adata, metric)) == len(replicates) * tcri.get.draws(
        adata, metric)["null"].nunique()
    if metric == "mutual_information":
        with pytest.raises(KeyError, match="no item level"):
            tcri.get.items(adata, metric)
    else:
        items = tcri.get.items(adata, metric)
        assert ITEM[metric] in items.columns and len(items) > len(replicates)
        assert {"cov_from", "cov_to"} <= set(tcri.get.draws(adata, metric, items=True).columns)
    stats = tcri.get.stats(adata, metric)
    assert stats["comparison"].tolist() == ["covariate", "covariate", "split"]
    assert stats["disease_status"].tolist() == ["control", "disease", "control vs disease"]


@pytest.mark.parametrize("metric", METRICS)
def test_stored_and_returned_agree(metric, cohort):
    """The store-once invariant on a change: the call stores its result and returns ``None``,
    ``inplace=False`` returns the result it stores, and the provenance records both levels."""
    _, adata = cohort
    adata = adata.copy()
    assert getattr(tcri.tl, metric)(adata, cov_from=A, cov_to=B, groupby="patient") is None
    returned = _run(metric, adata, cov_from=A, cov_to=B, groupby="patient")
    _assert_same_result(tcri.get.result(adata, metric), returned, metric)
    params = tcri.get.params(adata, metric)
    assert params["cov_from"] == A and params["cov_to"] == B
    if metric != "phenotypic_flux":
        assert params["covariate"] is None


def test_a_change_with_labels_that_are_not_strings_round_trips_through_h5ad(cohort, tmp_path):
    """A boolean split and integer covariate levels: a change's ``stats`` holds covariate rows
    and split rows in one frame, so its labels are stored as text, and the result writes to
    ``.h5ad`` and reads back the same."""
    from tests.test_covariate_null_movable import _cohort, _fit

    _, adata = cohort
    adata = adata.copy()
    adata.obs["responder"] = (adata.obs["disease_status"].astype(str) == "disease").to_numpy()
    tcri.tl.mutual_information(adata, cov_from=A, cov_to=B, splitby="responder",
                               null_model=None)

    numeric = _cohort(split=())
    numeric.obs["covariate"] = numeric.obs["covariate"].map({"cov_0": 0, "cov_1": 1}).astype(
        "category")
    _fit(numeric, "integer_levels")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tcri.tl.mutual_information(numeric, cov_from=0, cov_to=1, splitby="arm",
                                   null_model=None)

    for obj, name in ((adata, "responder"), (numeric, "integer levels")):
        stats = tcri.get.stats(obj, "mutual_information")
        assert set(stats["comparison"]) == {"covariate", "split"}, name
        path = tmp_path / f"{name}.h5ad"
        obj.write_h5ad(path)
        pd.testing.assert_frame_equal(tcri.get.stats(ad.read_h5ad(path), "mutual_information"),
                                      stats, obj=name)


@pytest.mark.parametrize("replicate", [True, False], ids=["replicate", "no-replicate"])
def test_a_stored_change_round_trips_through_h5ad(replicate, cohort, tmp_path):
    """Every metric's stored change, written to ``.h5ad`` and read back, gives the same
    replicate summary, items, ``stats`` and params through ``tcri.get``, with the fixture's
    covariate null as the reference and an integer ``random_state``.

    With a registered replicate, by patient and with ``splitby``, so ``stats`` holds covariate
    rows per split and the split row; without one, the whole object is one replicate and its
    covariate row has no replicate unit.
    """
    _, adata = cohort
    adata = adata.copy()
    kwargs = dict(cov_from=A, cov_to=B, n_samples=3, random_state=0)
    if replicate:
        kwargs["splitby"] = "disease_status"
    else:
        adata.uns[K.METADATA] = {**adata.uns[K.METADATA], "replicate": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for metric in METRICS:
            assert getattr(tcri.tl, metric)(adata, **kwargs) is None
    path = tmp_path / "changes.h5ad"
    adata.write_h5ad(path)
    back = ad.read_h5ad(path)

    for metric in METRICS:
        stats = tcri.get.stats(adata, metric)
        assert set(stats["quantity"]) == {"adjusted"}, "the fixture's reference was not used"
        assert (stats["comparison"].tolist() == ["covariate", "covariate", "split"] if replicate
                else stats["replicate_unit"].isna().all())
        views = [tcri.get.replicates, tcri.get.stats]
        if metric != "mutual_information":
            views.append(tcri.get.items)
        for view in views:
            pd.testing.assert_frame_equal(view(back, metric), view(adata, metric),
                                          obj=f"{metric} {view.__name__}")
        if metric == "mutual_information":
            with pytest.raises(KeyError, match="no item level"):
                tcri.get.items(back, metric)
        assert tcri.get.params(back, metric) == tcri.get.params(adata, metric), metric


@pytest.mark.parametrize("metric", METRICS)
def test_no_direction_probability_column(metric, cohort):
    """``p_gt`` is deliberately absent from a change. It reads as a frequentist p-value, its
    resolution caps at 1/n_samples, and per item it invites filtering on the survivors; the HDI
    answers direction already, and the graded version is one line off the draws."""
    _, adata = cohort
    res = _run(metric, adata, cov_from=A, cov_to=B, n_samples=10, random_state=0)
    for level in _levels(metric):
        for slot in ("draws", "summary"):
            assert not ({"p_gt", "p_lt", "prob_increase"} & set(res[level][slot].columns))

    # one graded value per row of the summary, not one value for every row. Flux is a distance
    # and never negative, so its share of positive draws carries no direction and is not
    # compared across rows.
    level = "replicate" if metric == "mutual_information" else "item"
    key = ["patient"] + ([ITEM[metric]] if level == "item" else [])
    graded = (res[level]["draws"].groupby(key, observed=True)["value"]
              .apply(lambda v: (v > 0).mean()))
    rows = res[level]["summary"].set_index(key).index
    assert len(graded) == len(rows) and sorted(graded.index) == sorted(rows)
    if metric != "phenotypic_flux":
        assert graded.nunique() > 1, "one graded value for every row"


def test_a_change_is_a_call_of_the_metric_itself():
    """A change is the metric called with ``cov_from`` and ``cov_to``, on every metric, mutual
    information included; no metric has a separate function, accessor or stored key for it."""
    for namespace in (tcri.tl, tcri.pl, tcri.get):
        assert not [n for n in dir(namespace) if n.startswith("delta_")], namespace.__name__
    assert not [n for n in tcri.tl.__all__ if n.startswith("delta_")]
    assert not [n for n in tcri.get._RESULTS if n.startswith("delta")]
    assert not [n for n in dir(K) if n.startswith("DELTA_")]
    for metric in METRICS:
        params = inspect.signature(getattr(tcri.tl, metric)).parameters
        assert {"cov_from", "cov_to"} <= set(params), metric
