"""``tcri.null``: a reference is the same model, fitted the same way, on permuted labels.

Everything that makes a null a *reference* rather than noise is in two places. The strata decide
which cells a label may move between, so that the quantity being scored is destroyed and nothing
else is. And the fit itself is the parent's -- same knobs, same seed, same split, same minibatch
order, same training arguments -- so the only difference between a null and its parent is which
cell carries which label.

The load-bearing invariant, asserted here for all three kinds: the clone x covariate index is the
PARENT'S. Same ``n_ct``, same order, same ``ct_to_c``, same ``ct_to_cov``. That is what makes the
``null_value`` join total with no NaN rows. What moves per cell differs by kind, and asserting
movement for all three is the mistake an earlier revision of the design made -- under the
phenotype null ``ct_array`` is the parent's cell for cell, because neither the clone nor the
covariate has been touched.

Rows of plan §3.4 that assert reader surface (``fit=``, ``null_model=``) land with that surface.

Fits are deliberately tiny; nothing here asserts accuracy.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pyro
import pytest
import torch

import tcri
from tcri._compute._tables import clones_at, fit_clone_labels
from tcri._state import keys as K
from tcri.datasets import simulate_tcri
from tcri.model._model import TCRIModel, _owns_param
from tcri.null import _permute
from tcri.null._rebuild import rebuild

TRAIN = dict(max_epochs=3, batch_size=64, n_steps_kl_warmup=4, accelerator="cpu",
             enable_progress_bar=False, enable_model_summary=False)
KNOBS = dict(n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
             classifier_hidden=16, K=3, seed=0)


def _adata(seed=0, n_covariates=2):
    a = simulate_tcri(n_clones=8, n_phenotypes=3, n_genes=20, n_cells=200,
                      n_covariates=n_covariates, omega_concentration=0.4, seed=seed)
    a.layers["counts"] = a.X.copy()
    a.obs["site"] = pd.Categorical(["s0", "s1"] * (a.n_obs // 2))
    return a


def _setup(a):
    TCRIModel.setup_anndata(a, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="covariate",
                            batch_key="batch")


def _parent(a, name):
    _setup(a)
    model = TCRIModel(a, name=name, **KNOBS)
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(**TRAIN)
        model.to_anndata(a)
    return model


@pytest.fixture(scope="module")
def fitted():
    """A parent and its three nulls, fitted once. Returns ``(model, adata, nulls)``."""
    from .conftest import _remember_fixture_params

    a = _adata()
    model = _parent(a, "nullparent")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nulls = tcri.null.all(model, a)
    _remember_fixture_params("nullparent")
    return model, a, nulls


# ── the permutation ──────────────────────────────────────────────────────────

def test_permutation_preserves_the_strata(fitted):
    """Per stratum, the multiset of permuted codes equals the original.

    This is what "within" means, and it is the only thing standing between a reference and a
    number computed on a differently shaped dataset. Also asserted per kind: clone SIZES survive
    the clonotype shuffle, and each clone keeps its own cells under the condition shuffle.
    """
    _, adata, _ = fitted
    meta = adata.uns[K.METADATA]
    for kind in ("phenotype", "clonotype", "condition"):
        fit = f"null.{kind}"
        perm = np.asarray(adata.uns[K.fit_key(K.PERMUTATION, fit)])
        strata = list(adata.uns[K.fit_key(K.FIT_SETTINGS, fit)]["strata"])
        assert sorted(perm.tolist()) == list(range(adata.n_obs)), f"{kind}: not a permutation"

        col = {"phenotype": meta[K.Config.PHENOTYPE_COL],
               "clonotype": meta[K.Config.CLONE_COL],
               "condition": meta[K.Config.COVARIATE_COL]}[kind]
        codes = adata.obs[col].astype("category").cat.codes.to_numpy()
        moved = codes[perm]
        keys = pd.MultiIndex.from_arrays([adata.obs[c].astype(str).to_numpy() for c in strata])
        for _, idx in pd.Series(np.arange(adata.n_obs), index=keys).groupby(
                level=list(range(len(strata))), observed=True):
            i = idx.to_numpy()
            assert sorted(codes[i].tolist()) == sorted(moved[i].tolist()), (
                f"{kind}: the permutation crossed a stratum boundary")
        assert not np.array_equal(codes, moved), f"{kind}: the permutation is the identity"

    # clone sizes survive the clonotype shuffle
    clone = adata.obs[meta[K.Config.CLONE_COL]].astype("category").cat.codes.to_numpy()
    p = np.asarray(adata.uns[K.fit_key(K.PERMUTATION, "null.clonotype")])
    assert sorted(np.bincount(clone).tolist()) == sorted(np.bincount(clone[p]).tolist())

    # each clone keeps its own cells under the condition shuffle
    p = np.asarray(adata.uns[K.fit_key(K.PERMUTATION, "null.condition")])
    assert np.array_equal(clone, clone[p]), "the condition null moved cells between clones"


def test_the_ct_index_is_the_parents(fitted):
    """``n_ct``, the ct pair list, ``ct_to_c``, ``ct_to_cov`` and every per-ct count are the
    parent's, under all three kinds. ``ct_array`` is the parent's cell for cell under the
    PHENOTYPE null only.

    The asymmetry is the point. A phenotype permutation touches neither the clone nor the
    covariate, so nothing about the ct index or a cell's place in it can move; the clonotype and
    condition nulls move most cells. Asserting movement for all three would be green on a null
    that permuted nothing at all.
    """
    _, adata, _ = fitted
    parent_ct = np.asarray(adata.uns[K.CT_ARRAY])
    n_ct = int(adata.uns[K.FIT_SETTINGS]["n_ct"])
    for kind in ("phenotype", "clonotype", "condition"):
        fit = f"null.{kind}"
        assert int(adata.uns[K.fit_key(K.FIT_SETTINGS, fit)]["n_ct"]) == n_ct, kind
        for key in (K.CT_TO_C, K.CT_TO_COV):
            np.testing.assert_array_equal(np.asarray(adata.uns[K.fit_key(key, fit)]),
                                          np.asarray(adata.uns[key]),
                                          err_msg=f"{kind}: {key} is not the parent's")
        ct = np.asarray(adata.uns[K.fit_key(K.CT_ARRAY, fit)])
        np.testing.assert_array_equal(np.bincount(ct, minlength=n_ct),
                                      np.bincount(parent_ct, minlength=n_ct),
                                      err_msg=f"{kind}: per-ct cell counts moved")
        same = int((ct == parent_ct).sum())
        if kind == "phenotype":
            assert same == adata.n_obs, "the phenotype null moved a cell's ct row"
        else:
            assert same < adata.n_obs, f"{kind}: no cell changed ct row"


def test_the_clone_map_is_a_no_op_on_the_main_fit(fitted):
    """§4.2a's substrate-derived helpers agree with the obs reading they replace.

    ``metric_table`` and the delta endpoints now build their clone lists from the fit rather
    than from ``obs``. On the main fit the two must be the same thing; if they ever are not,
    every metric in the package silently moved.
    """
    _, adata, _ = fitted
    meta = adata.uns[K.METADATA]
    obs_clone = adata.obs[meta[K.Config.CLONE_COL]]
    labels = fit_clone_labels(adata, None)
    assert labels.index.equals(adata.obs_names)
    assert (labels.astype(str) == obs_clone.astype(str)).all(), "the clone map is not a no-op"

    for level in adata.uns[K.COVARIATE_CATEGORIES]:
        at_obs = set(adata.obs.loc[
            adata.obs[meta[K.Config.COVARIATE_COL]].astype(str) == str(level),
            meta[K.Config.CLONE_COL]].dropna().unique())
        assert set(clones_at(adata, level)) == at_obs, f"clones_at disagrees at {level!r}"

    # ...and under the clonotype null it reads the PERMUTED map, which is the whole purpose.
    assert not (fit_clone_labels(adata, "null.clonotype").astype(str)
                == obs_clone.astype(str)).all()


def test_within_is_validated_not_extended(fitted):
    """A ``within`` that drops a required column raises; a refining one is accepted and recorded.

    Silently unioning the missing column back in would make the recorded strata differ from the
    strata the caller believes they asked for -- and on a covariate-sparse frame a ``within``
    that drops the covariate takes the shared-clone set from 6 to 36, so the join would be onto
    a row set six times larger with no error anywhere.
    """
    model, adata, _ = fitted
    covariate_col = adata.uns[K.METADATA][K.Config.COVARIATE_COL]

    with pytest.raises(ValueError, match=covariate_col):
        _permute.resolve_strata(adata, "phenotype", ["batch"])
    with pytest.raises(ValueError, match="not in adata.obs"):
        _permute.resolve_strata(adata, "clonotype", [covariate_col, "no_such_column"])
    with pytest.raises(ValueError, match="no `within`"):
        _permute.resolve_strata(adata, "condition", ["batch"])

    refined = ["batch", covariate_col, "site"]
    assert _permute.resolve_strata(adata, "phenotype", refined) == refined

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tcri.null.phenotype(model, adata, within=refined, key_added="site")
    settings = adata.uns[K.fit_key(K.FIT_SETTINGS, "null.phenotype.site")]
    assert list(settings["strata"]) == refined
    assert settings["seed"] != adata.uns[K.fit_key(K.FIT_SETTINGS, "null.phenotype")]["seed"], (
        "a second null of the same kind reused the first's permutation stream")


# ── the null is a fit of the parent's model ──────────────────────────────────

def test_null_is_the_parent_model_on_permuted_labels(fitted):
    """Every constructor argument except ``name`` and ``permutation`` is the parent's, and so is
    the train/validation split. ``seed=`` moves the permutation and nothing else."""
    model, adata, nulls = fitted
    from tcri.null._nulls import _init_params

    assert _init_params(model) == _init_params(nulls["phenotype"])
    np.testing.assert_array_equal(np.sort(model.train_indices),
                                  np.sort(nulls["phenotype"].train_indices))

    a, b = _adata(), _adata()
    _setup(a)
    _setup(b)
    perm = np.asarray(adata.uns[K.fit_key(K.PERMUTATION, "null.phenotype")])
    fresh_parent = TCRIModel(a, name="fresh.parent", **KNOBS)
    fresh_null = TCRIModel(b, name="fresh.null", permutation=("phenotype", perm), **KNOBS)
    for (kp, vp), (kn, vn) in zip(fresh_parent.module.named_parameters(),
                                  fresh_null.module.named_parameters()):
        assert kp == kn and torch.equal(vp, vn), f"initial weights differ at {kp}"


def test_fitting_a_null_does_not_move_the_parent():
    """Every parent store leaf and module tensor is bitwise unchanged by its null's fit.

    The gradient-leak test. A null shares the parent's AnnData and is constructed from the
    parent's arguments, so a shared tensor or a second registration under the parent's namespace
    would be easy to introduce and invisible in every other assertion here.
    """
    a = _adata()
    model = _parent(a, "leakcheck")
    store = pyro.get_param_store()
    before = {k: v.detach().clone() for k, v in store.named_parameters()
              if _owns_param("leakcheck", k)}
    assert before, "the parent registered no parameters"
    state = {k: v.detach().clone() for k, v in model.module.state_dict().items()}
    p_ct = np.asarray(a.uns[K.P_CT]).copy()

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tcri.null.phenotype(model, a)

    assert not [w for w in caught if "already holds" in str(w.message)], (
        "the parent warned about a collision with its own null")
    for k, v in store.named_parameters():
        if _owns_param("leakcheck", k):
            assert torch.equal(v.detach(), before[k]), f"the null's fit moved {k}"
    for k, v in model.module.state_dict().items():
        assert torch.equal(v, state[k]), f"the null's fit moved module.{k}"
    np.testing.assert_array_equal(np.asarray(a.uns[K.P_CT]), p_ct)


def test_null_writes_its_substrate_and_provenance(fitted):
    """The prefixed substrate, the permutation vector and the settings are all in the object,
    the fit is listed, and the bare kind resolves to it."""
    model, adata, _ = fitted
    fit = "null.phenotype"
    assert fit in K.fits(adata)
    assert K.resolve_fit(adata, "phenotype") == fit

    for key in (K.P_CT, K.CONC_CT, K.LOCAL_SCALE, K.GATE_PROB, K.CLASSIFIER_TEMPERATURE,
                K.CT_TO_COV, K.CT_TO_C, K.CT_ARRAY, K.COV_ARRAY, K.PERMUTATION, K.BUFFERS,
                K.FIT_SETTINGS):
        assert K.fit_key(key, fit) in adata.uns, f"{key} missing for {fit}"
    for key in (K.X_TCRI, K.X_LOGITS, K.X_LOGPOSTERIOR, K.X_PROBABILITIES):
        assert K.fit_key(key, fit) in adata.obsm, f"{key} missing for {fit}"
    assert K.fit_key(K.PHENOTYPE, fit) in adata.obs

    settings = adata.uns[K.fit_key(K.FIT_SETTINGS, fit)]
    assert settings["kind"] == "phenotype" and settings["parent"] == model.name
    assert settings["namespace"] == f"{model.name}.{fit}"
    assert settings["n_obs"] == adata.n_obs
    assert int(settings["n_strata"]) == len(settings["stratum_sizes"])
    assert settings["train"]["max_epochs"] == TRAIN["max_epochs"]
    # the joinability record carries the categories AS THAT MODEL SAW THEM -- the shared keys
    # cannot do this job, because comparing them between two fits compares an object with itself
    assert list(settings[K.PHENOTYPE_CATEGORIES]) == list(adata.uns[K.PHENOTYPE_CATEGORIES])

    # and the main fit is untouched by all of it
    assert adata.uns[K.METADATA][K.Config.CLONE_COL] == "clone_id"


def test_a_fit_name_is_claimed_by_one_parent(fitted):
    """A second parent writing ``null.phenotype`` into the same object raises and names the way
    out; with ``key_added`` both coexist and neither's substrate moved."""
    _, adata, _ = fitted
    other = TCRIModel(adata, name="otherparent", **KNOBS)
    with contextlib.redirect_stdout(io.StringIO()):
        other.train(**TRAIN)

    held = np.asarray(adata.uns[K.fit_key(K.P_CT, "null.phenotype")]).copy()
    with pytest.raises(ValueError, match="key_added"):
        tcri.null.phenotype(other, adata)
    np.testing.assert_array_equal(np.asarray(adata.uns[K.fit_key(K.P_CT, "null.phenotype")]), held)

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tcri.null.phenotype(other, adata, key_added="b")
    assert "null.phenotype.b" in K.fits(adata)
    np.testing.assert_array_equal(np.asarray(adata.uns[K.fit_key(K.P_CT, "null.phenotype")]), held)


def test_all_fits_every_applicable_kind():
    """``all`` matches the single functions under the same seed, and skips ``condition`` loudly
    when there is only one condition to permute between."""
    a = _adata()
    model = _parent(a, "allkinds")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = tcri.null.all(model, a)
    assert sorted(out) == ["clonotype", "condition", "phenotype"]
    for kind in out:
        assert f"null.{kind}" in K.fits(a)

    single = _adata(n_covariates=1)
    single_model = _parent(single, "onecondition")
    with contextlib.redirect_stdout(io.StringIO()), pytest.warns(UserWarning, match="condition"):
        out = tcri.null.all(single_model, single)
    assert sorted(out) == ["clonotype", "phenotype"]
    assert "null.condition" not in K.fits(single)

    with pytest.raises(ValueError, match="condition null takes no"):
        tcri.null.all(model, a, within=["batch", "covariate"])
    with pytest.raises(ValueError, match="not one of"):
        tcri.null.all(model, a, kinds=("phenotype", "nonsense"))


# ── recovery ─────────────────────────────────────────────────────────────────

def test_null_is_recoverable_from_the_parent_session(fitted, tmp_path):
    """Save the PARENT, clear the store, load, rebuild: the null predicts what it predicted.

    A null has no session of its own and needs none. Its parameters travel in the parent's store
    file because the store is whole-process; its buffers and permutation travel in the AnnData.
    BatchNorm running statistics are included, which is what the buffer round-trip is for.
    """
    from tcri.utils import load_tcri_session, save_tcri_session

    model, adata, nulls = fitted
    expected = nulls["clonotype"].predict(adata).to_numpy()

    with contextlib.redirect_stdout(io.StringIO()):
        save_tcri_session(model, adata, str(tmp_path / "run"))
    pyro.clear_param_store()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded, loaded_adata = load_tcri_session(str(tmp_path / "run"), map_location="cpu")
        back = rebuild(loaded, loaded_adata, "clonotype")

    assert loaded._train_kwargs["max_epochs"] == TRAIN["max_epochs"], (
        "the parent's training arguments did not survive the session")
    np.testing.assert_allclose(back.predict(loaded_adata).to_numpy(), expected, atol=1e-6)


def test_rebuild_names_the_missing_namespace(fitted):
    """With the store cleared, ``rebuild`` raises and names both fixes rather than handing back
    a randomly initialised model that answers every question without complaint."""
    model, adata, _ = fitted
    pyro.clear_param_store()
    with pytest.raises(RuntimeError, match="load_tcri_session"):
        rebuild(model, adata, "phenotype")
    with pytest.raises(KeyError, match="no fit named"):
        rebuild(model, adata, "not_a_fit")


# ── a null is a fit, so everything that takes a fit takes a null (PR C) ──────

def test_a_null_is_a_fit_for_every_reader(fitted):
    """The evaluation table, row by row. A null whose row does not look like this is a wrong
    null, and there is nowhere else in the package where that is checked.

    Measured 2026-09-11 on this fixture at 30 epochs, seeds 0/1/2 (majority-class rate of the
    true labels in brackets):

    | quantity | parent | phenotype null | clonotype null | condition null |
    |---|---|---|---|---|
    | head accuracy [0.405] seed 0 | 0.630 | 0.545 | 0.460 | 0.630 |
    | head accuracy [0.535] seed 1 | 0.820 | 0.535 | 0.545 | 0.820 |
    | head accuracy [0.480] seed 2 | 0.865 | 0.480 | 0.480 | 0.865 |
    | mutual information seed 0 | 0.2075 | 0.0075 | 0.0130 | 0.2069 |
    | mutual information seed 1 | 0.4744 | 0.0223 | 0.0331 | 0.4749 |
    | mutual information seed 2 | 0.4522 | 0.0386 | 0.0076 | 0.4520 |

    Two things in that table are worth stating because they are NOT what one would predict.

    A label null's head does not fall to chance (1/P); it falls to the MAJORITY-CLASS RATE. The
    permutation preserves each stratum's label multiset, so the marginal survives it and a head
    with nothing else to learn predicts the marginal. On seeds 1 and 2 the accuracy equals that
    rate to three decimals. Asserting "at chance" would be asserting something false.

    And the CLONOTYPE null's head falls too, rather than staying at its parent's. The hard label
    is the argmax of the GATED posterior, which mixes the head with ``log p_ct``, and a
    clonotype permutation destroys exactly that prior; the noisy-label readout also ties the
    head to the hierarchy. So "the head is intact" is true of ``f_cls`` and not of the call.

    What IS clean is the mutual information row and the condition null's column: the condition
    null reproduces its parent's accuracy exactly on all three seeds, because neither the
    phenotype nor the clone of any cell moved.
    """
    import tcri

    model, adata, nulls = fitted
    cov = list(adata.uns[K.COVARIATE_CATEGORIES])[0]
    pheno_col = adata.uns[K.METADATA][K.Config.PHENOTYPE_COL]
    truth = adata.obs[pheno_col].astype(str).to_numpy()
    majority = float(adata.obs[pheno_col].value_counts(normalize=True).max())

    def accuracy(fit):
        pred = adata.obs[K.fit_key(K.PHENOTYPE, fit)].astype(str).to_numpy()
        return float((pred == truth).mean())

    # 1) expression is untouched by every permutation, so reconstruction cannot move
    for kind, null in nulls.items():
        with contextlib.redirect_stdout(io.StringIO()):
            ppc = tcri.diag.reconstruction_ppc(null, adata, n_sims=2, random_state=0)
        assert ppc is not None and len(ppc), kind

    # 2) the substrate readers take fit=, and read that fit's own probabilities
    for kind in nulls:
        cal = tcri.diag.phenotype_calibration(adata, fit=f"null.{kind}")
        assert cal is not None and len(cal), kind
    assert not np.allclose(
        adata.obsm[K.fit_key(K.X_PROBABILITIES, "null.phenotype")],
        adata.obsm[K.X_PROBABILITIES]), "calibration read the main fit's probabilities"

    # 3) the head: both label nulls collapse toward the marginal, the condition null does not
    parent_accuracy = accuracy(None)
    for kind in ("phenotype", "clonotype"):
        assert accuracy(f"null.{kind}") < parent_accuracy, kind
        assert accuracy(f"null.{kind}") <= majority + 0.15, (
            f"the {kind} null still reads the true labels: {accuracy(f'null.{kind}'):.3f} "
            f"against a majority-class rate of {majority:.3f}")
    assert accuracy("null.condition") == pytest.approx(parent_accuracy, abs=0.05), (
        "the condition null moved the phenotype call, and it moves neither phenotype nor clone")

    # 4) mutual information: only the two label nulls destroy the coupling it measures
    mi = {k: float(tcri.tl.mutual_information(adata, covariate=cov, fit=f"null.{k}",
                                              null_model=None,
                                              inplace=False)["result"]["value"].iloc[0])
          for k in nulls}
    parent_mi = float(tcri.tl.mutual_information(adata, covariate=cov, null_model=None,
                                                 inplace=False)["result"]["value"].iloc[0])
    assert mi["phenotype"] < 0.25 * parent_mi and mi["clonotype"] < 0.25 * parent_mi, mi
    assert mi["condition"] == pytest.approx(parent_mi, rel=0.1), (
        f"the condition null destroyed clone-phenotype coupling, which is not its axis: {mi}")


def test_a_metric_never_trains(fitted, monkeypatch):
    """With the substrate absent the metric raises, and no fit runs.

    `train` is monkeypatched to fail, so a metric that quietly built its own null would error
    here with a different message rather than passing. The raise is the whole opt-in: fitting is
    a decision with a cost and it is the caller's.
    """
    import tcri
    from tcri.model._model import TCRIModel

    _, adata, _ = fitted
    adata = adata.copy()
    cov = list(adata.uns[K.COVARIATE_CATEGORIES])[0]
    for key in list(adata.uns):
        if "null.phenotype" in key:
            del adata.uns[key]
    adata.uns[K.METADATA] = {**adata.uns[K.METADATA],
                             K.FITS: [f for f in K.fits(adata) if f != "null.phenotype"]}

    def _explode(*a, **k):
        raise AssertionError("a metric trained a model")

    monkeypatch.setattr(TCRIModel, "train", _explode)
    with pytest.raises(KeyError, match=r"tcri\.null\.all"):
        tcri.tl.mutual_information(adata, covariate=cov, inplace=False)


def test_a_groupby_finer_than_the_strata_raises():
    """The silently wrong group metric, made loud.

    On a fixture where a (batch, covariate) stratum spans two replicates, the clonotype null's
    shuffle moves clones between those replicates while the group mask still comes from obs. A
    group's clone list then selects the other group's null rows -- measured on one patient with
    two replicates, 20 of 20 clones span both and about half of R1's selected rows are R2's
    cells. The parent's obs passes every disjointness check, so nothing else catches it.

    The strata alone cannot close this: they are fixed at fit time while `groupby` is chosen at
    metric time. Only the substrate-derived clone map is evaluated against the groupby actually
    used.
    """
    import tcri

    a = _adata()
    # two replicates inside one (batch, covariate) stratum, and clone ids that span them
    a.obs["replicate"] = pd.Categorical(["R1", "R2"] * (a.n_obs // 2))
    _setup(a)
    model = _parent(a, "finegroupby")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tcri.null.clonotype(model, a)

    spans = a.obs.groupby([a.uns[K.METADATA][K.Config.BATCH_COL],
                           a.uns[K.METADATA][K.Config.COVARIATE_COL]],
                          observed=True)["replicate"].nunique().max()
    assert spans > 1, "the fixture cannot reach the defect"

    cov = list(a.uns[K.COVARIATE_CATEGORIES])[0]
    with pytest.raises(ValueError, match="spans groups"):
        tcri.tl.mutual_information(a, covariate=cov, groupby="replicate",
                                    fit="null.clonotype", null_model=None, inplace=False)
    # the message names the fit, its strata and the refit that fixes it
    try:
        tcri.tl.mutual_information(a, covariate=cov, groupby="replicate",
                                    fit="null.clonotype", null_model=None, inplace=False)
    except ValueError as exc:
        assert "null.clonotype" in str(exc) and "within=" in str(exc), str(exc)
