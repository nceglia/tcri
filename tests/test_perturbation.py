"""``tcri.perturb`` -- the identities that pin the perturbation.

The perturbation is a query on a fitted model, so it has no reference joint and no golden
value (``METRICS_CONTRACT.md``, "The perturbation"). What can be pinned is structure:
``knockout`` of nothing is ``predict()``; ``knockout`` of a gene is ``predict()`` on the
zeroed matrix; an unexpressed gene has importance exactly zero; importance is bounded and is
the L1 norm of a shift that sums to zero; groups reduce cells by row and bind the prior by
global id; draws come from the guide's Dirichlet and are seeded; the contrast is per gene over
groups. The fixtures fit for a few epochs -- the numbers here are identities, not accuracies.

Each test says what it catches. The mutations run against this file before it was trusted, and
what failed: flipping the sign of ``shift`` (the sign test alone); binding the prior by loader
position instead of global id (the cell-order test alone); dropping the draw path (the draws
test alone); zeroing the neighbouring column in either kernel path (the knockout-vs-predict or
the unexpressed-gene test, plus every hand comparison, because the two paths then disagree).

Every model call here goes through the ``cohort`` fixture. That used to be forced: the Pyro
parameter store is process-global and each session fixture cleared it when built, so a model
method on ``trained_model`` after ``cohort`` existed read the wrong ``q_p_ct_raw`` (P=4 against
a P=3 head). Since 0.12 the fixtures are namespaced and coexist, so this is now one fixture for
consistency rather than a constraint. The no-replicate case is a copy of the cohort with the
registered replicate removed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import mannwhitneyu

import tcri
from tcri._state import keys as K


def _zeroed(adata, j):
    """A copy with gene column ``j`` set to zero in ``X`` and in the registered layer."""
    a = adata.copy()
    a.X[:, j] = 0
    if "counts" in a.layers:
        a.layers["counts"][:, j] = 0
    return a


def _without_replicate(adata):
    """A copy whose registry names no replicate, so ``groupby=None`` means every cell."""
    a = adata.copy()
    a.uns[K.METADATA] = {**a.uns[K.METADATA], K.Config.REPLICATE: None}
    return a


def _manual_importance(model, adata, gene, *, mask=None):
    """``I`` and the per-phenotype shift from two ``knockout`` frames, over ``mask`` cells."""
    base = tcri.perturb.knockout(model, adata, genes=[])
    pert = tcri.perturb.knockout(model, adata, genes=[gene])
    if mask is not None:
        base, pert = base.loc[mask], pert.loc[mask]
    shift = base.mean(axis=0) - pert.mean(axis=0)
    return float(shift.abs().sum()), shift


# ── knockout is predict() on the intervened matrix ───────────────────────────

def test_knockout_of_nothing_is_predict(cohort):
    """The two per-cell rules cannot drift: silencing nothing must be ``predict()`` exactly."""
    model, adata = cohort
    k0 = tcri.perturb.knockout(model, adata, genes=[])
    p = model.predict(adata)
    assert list(k0.index) == list(p.index) and list(k0.columns) == list(p.columns)
    np.testing.assert_allclose(k0.to_numpy(), p.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(k0.sum(axis=1).to_numpy(), 1.0, atol=1e-5)


def test_knockout_is_predict_on_the_zeroed_matrix(cohort):
    """Catches zeroing the wrong column: ``knockout([j])`` must equal ``predict()`` on a copy
    whose column ``j`` is zero, for one gene and for a two-gene program."""
    model, adata = cohort
    j = 3
    kg = tcri.perturb.knockout(model, adata, genes=[adata.var_names[j]])
    pz = model.predict(_zeroed(adata, j))
    np.testing.assert_allclose(kg.to_numpy(), pz.to_numpy(), atol=1e-6)

    program = [adata.var_names[1], adata.var_names[7]]
    kp = tcri.perturb.knockout(model, adata, genes=program)
    pp = model.predict(_zeroed(_zeroed(adata, 1), 7))
    np.testing.assert_allclose(kp.to_numpy(), pp.to_numpy(), atol=1e-6)


def test_gate_off_is_the_head_alone(cohort, monkeypatch):
    """``use_gate=False`` is ``softmax(f_cls(μ_i))``: invariant to the gate and the prior."""
    model, adata = cohort
    module = model.module
    z = torch.as_tensor(model.get_latent_representation(adata), dtype=torch.float32)
    device = next(module.parameters()).device
    with torch.no_grad():
        want = torch.softmax(module.classifier(z.to(device)), dim=-1).cpu().numpy()
    got = tcri.perturb.knockout(model, adata, genes=[], use_gate=False)
    np.testing.assert_allclose(got.to_numpy(), want, atol=1e-5)

    head = tcri.perturb.gene_importance(model, adata, genes=[0, 1], use_gate=False,
                                        inplace=False)["result"]
    gated = tcri.perturb.gene_importance(model, adata, genes=[0, 1], inplace=False)["result"]
    assert not np.allclose(head["value"], gated["value"]), "the gate changed nothing"

    original = module.gate_prob
    monkeypatch.setattr(module, "gate_prob", 0.9 if original != 0.9 else 0.1)
    moved = tcri.perturb.gene_importance(model, adata, genes=[0, 1], use_gate=False,
                                         inplace=False)["result"]
    np.testing.assert_allclose(moved["value"], head["value"], atol=1e-7)


# ── the importance identities ────────────────────────────────────────────────

def test_unexpressed_gene_has_exactly_zero_importance(cohort):
    """Catches any baseline/perturbed asymmetry: zeroing a column that is already zero changes
    nothing, so the importance is 0.0 with no floor to arrange."""
    model, adata = cohort
    a = _zeroed(adata, 0)
    res = tcri.perturb.gene_importance(model, a, genes=[0], inplace=False)
    assert (res["result"]["value"].abs() < 1e-12).all(), res["result"]
    assert (res["shift"]["shift"].abs() < 1e-12).all()


def test_importance_is_bounded_and_is_the_l1_of_a_zero_sum_shift(cohort):
    """Catches a non-simplex probability leaking through, and ties ``result`` to ``shift``:
    ``value`` must equal ``Σ_p |shift|`` row for row at ``n_samples=0``."""
    model, adata = cohort
    res = tcri.perturb.gene_importance(model, adata, inplace=False)
    result, shift = res["result"], res["shift"]
    assert ((result["value"] >= 0) & (result["value"] <= 2)).all()

    by = shift.groupby(["gene", "patient"], observed=True, sort=False)
    assert (by["shift"].sum().abs() < 1e-5).all(), "shift does not sum to zero"
    assert (by["baseline"].sum() - 1.0).abs().max() < 1e-5
    assert (by["perturbed"].sum() - 1.0).abs().max() < 1e-5

    l1 = by["shift"].apply(lambda s: float(np.abs(s).sum())).rename("l1").reset_index()
    merged = result.merge(l1, on=["gene", "patient"])
    assert len(merged) == len(result)
    np.testing.assert_allclose(merged["value"], merged["l1"], atol=1e-9)


def test_groups_reduce_cells_and_shift_carries_its_sign(cohort):
    """Catches a wrong mask, the clone machinery creeping in, and a sign flip of ``shift``:
    the per-patient importance and shift must equal what two ``knockout`` frames give by hand,
    ``baseline − perturbed``."""
    model, adata = cohort
    gene = adata.var_names[5]
    res = tcri.perturb.gene_importance(model, adata, genes=[gene], inplace=False)
    for patient in adata.obs["patient"].cat.categories:
        mask = (adata.obs["patient"] == patient).to_numpy()
        want_i, want_shift = _manual_importance(model, adata, gene, mask=mask)
        row = res["result"].query("patient == @patient")
        assert len(row) == 1
        assert float(row["value"].iloc[0]) == pytest.approx(want_i, abs=1e-5)
        got = (res["shift"].query("patient == @patient")
               .set_index("phenotype")["shift"].reindex(want_shift.index))
        np.testing.assert_allclose(got.to_numpy(), want_shift.to_numpy(), atol=1e-5)


def test_covariate_restricts_the_cells(cohort):
    """``covariate`` selects the cells, not the prior: the prior stays each cell's own."""
    model, adata = cohort
    gene = adata.var_names[2]
    cov = list(adata.uns[K.COVARIATE_CATEGORIES])[0]
    res = tcri.perturb.gene_importance(model, adata, genes=[gene], covariate=cov,
                                       inplace=False)
    assert (res["result"]["covariate"] == cov).all()
    for patient in adata.obs["patient"].cat.categories:
        mask = ((adata.obs["patient"] == patient) & (adata.obs["covariate"] == cov)).to_numpy()
        want_i, _ = _manual_importance(model, adata, gene, mask=mask)
        got = float(res["result"].query("patient == @patient")["value"].iloc[0])
        assert got == pytest.approx(want_i, abs=1e-5)
    with pytest.raises(ValueError, match="not found among"):
        tcri.perturb.gene_importance(model, adata, genes=[gene], covariate="no_such_level",
                                     inplace=False)


def test_cell_order_invariant(cohort):
    """Catches loader-position indexing: a reversed view must give the same result. Labels
    follow rows, the prior follows the cell's global id."""
    model, adata = cohort
    rev = adata[::-1].copy()
    genes = list(adata.var_names[:4])
    a = tcri.perturb.gene_importance(model, adata, genes=genes, inplace=False)["result"]
    b = tcri.perturb.gene_importance(model, rev, genes=genes, inplace=False)["result"]
    a = a.sort_values(["gene", "patient"]).reset_index(drop=True)
    b = b.sort_values(["gene", "patient"]).reset_index(drop=True)
    assert list(a["gene"]) == list(b["gene"]) and list(a["patient"]) == list(b["patient"])
    np.testing.assert_allclose(a["value"], b["value"], atol=1e-6)

    ka = tcri.perturb.knockout(model, adata, genes=genes[:1])
    kb = tcri.perturb.knockout(model, rev, genes=genes[:1]).reindex(ka.index)
    np.testing.assert_allclose(ka.to_numpy(), kb.to_numpy(), atol=1e-6)


def test_batch_size_invariant(cohort):
    """Catches accumulation errors across batches."""
    model, adata = cohort
    genes = list(adata.var_names[:3])
    small = tcri.perturb.gene_importance(model, adata, genes=genes, batch_size=64,
                                         inplace=False)["result"]
    large = tcri.perturb.gene_importance(model, adata, genes=genes, batch_size=4096,
                                         inplace=False)["result"]
    np.testing.assert_allclose(small["value"], large["value"], atol=1e-5)


# ── draws ────────────────────────────────────────────────────────────────────

def test_draws_are_dirichlet_and_seeded(cohort, monkeypatch):
    """``n_samples=0`` is deterministic; ``>0`` is seeded, differs across seeds, has a spread,
    and collapses onto the plug-in when the guide's concentration is made enormous -- which is
    what shows the draws come from ``get_conc_ct`` and not from anywhere else."""
    model, adata = cohort
    genes = list(adata.var_names[:3])
    kw = dict(genes=genes, inplace=False)

    a = tcri.perturb.gene_importance(model, adata, **kw)["result"]
    b = tcri.perturb.gene_importance(model, adata, **kw)["result"]
    np.testing.assert_array_equal(a["value"], b["value"])
    assert a["sd"].isna().all(), "no draws, so no spread to report"

    s0 = tcri.perturb.gene_importance(model, adata, n_samples=8, random_state=0, **kw)
    s0b = tcri.perturb.gene_importance(model, adata, n_samples=8, random_state=0, **kw)
    s1 = tcri.perturb.gene_importance(model, adata, n_samples=8, random_state=1, **kw)
    assert s0["table"]["draw"].nunique() == 8
    np.testing.assert_array_equal(s0["result"]["value"], s0b["result"]["value"])
    assert not np.array_equal(s0["result"]["value"], s1["result"]["value"])
    assert np.isfinite(s0["result"]["sd"]).all() and (s0["result"]["sd"] > 0).any()
    assert (s0["result"]["hdi_low"] <= s0["result"]["value"] + 1e-12).all()
    assert (s0["result"]["hdi_high"] >= s0["result"]["value"] - 1e-12).all()

    original = model.module.get_conc_ct
    monkeypatch.setattr(model.module, "get_conc_ct", lambda: original() * 1e8)
    tight = tcri.perturb.gene_importance(model, adata, n_samples=8, random_state=0, **kw)
    np.testing.assert_allclose(tight["result"]["value"], a["value"], atol=5e-4)


def test_use_gate_false_ignores_n_samples_with_a_warning(cohort):
    """The head alone has no posterior to draw from; a zero-width interval would state a
    certainty never measured."""
    model, adata = cohort
    with pytest.warns(UserWarning, match="n_samples has no effect"):
        res = tcri.perturb.gene_importance(model, adata, genes=[0], use_gate=False,
                                           n_samples=4, inplace=False)
    assert res["table"]["draw"].nunique() == 1
    assert res["result"]["sd"].isna().all()


# ── groups, contrasts, storage ───────────────────────────────────────────────

def test_splitby_contrast_is_per_gene_over_groups(cohort):
    """Catches pseudoreplication: ``stats`` has one row per gene, its ``n`` is the number of
    patients per arm, and ``stat``/``p`` are a Mann-Whitney over the per-patient values."""
    model, adata = cohort
    res = tcri.perturb.gene_importance(model, adata, splitby="disease_status", inplace=False)
    stats, result = res["stats"], res["result"]
    assert stats is not None and len(stats) == adata.n_vars
    assert (stats["replicate_unit"] == "patient").all()
    assert (stats["n_a"] == 3).all() and (stats["n_b"] == 3).all()

    gene = adata.var_names[4]
    row = stats.query("gene == @gene").iloc[0]
    per = result.query("gene == @gene")
    va = per.loc[per["disease_status"] == row["level_a"], "value"].to_numpy()
    vb = per.loc[per["disease_status"] == row["level_b"], "value"].to_numpy()
    U, p = mannwhitneyu(va, vb, alternative="two-sided")
    assert float(row["stat"]) == pytest.approx(float(U))
    assert float(row["p"]) == pytest.approx(float(p))

    assert tcri.perturb.gene_importance(model, adata, genes=[0], inplace=False)["stats"] is None


def test_no_replicate_registered_gives_one_group(cohort):
    """Without a registered replicate and without ``groupby`` the reduction is over every
    cell: one row per gene, no group column, ``groupby`` recorded as ``None``."""
    model, adata = cohort
    adata = _without_replicate(adata)
    res = tcri.perturb.gene_importance(model, adata, genes=[0, 1, 2])
    assert list(res["result"]["gene"]) == [adata.var_names[i] for i in (0, 1, 2)]
    assert "patient" not in res["result"].columns
    assert tcri.get.params(adata, "gene_importance")["groupby"] is None
    want, _ = _manual_importance(model, adata, adata.var_names[1])
    assert float(res["result"]["value"].iloc[1]) == pytest.approx(want, abs=1e-5)


def test_missing_group_labels_are_excluded_with_a_warning(cohort):
    """A group column with gaps (not the batch column, which scvi itself refuses)."""
    model, adata = cohort
    a = adata.copy()
    a.obs["site"] = a.obs["patient"].astype(object)
    a.obs.loc[a.obs["site"] == "P5", "site"] = np.nan
    with pytest.warns(UserWarning, match="have no group label"):
        res = tcri.perturb.gene_importance(model, a, genes=[0], groupby="site", inplace=False)
    assert set(res["result"]["site"]) == {"P0", "P1", "P2", "P3", "P4"}
    want, _ = _manual_importance(model, a, a.var_names[0], mask=(a.obs["site"] == "P0").to_numpy())
    assert float(res["result"].query("site == 'P0'")["value"].iloc[0]) == pytest.approx(want, abs=1e-5)


def test_result_is_cached_readable_and_h5ad_safe(cohort, tmp_path):
    """The payload round-trips through ``tcri.get`` and a real ``.h5ad`` write; the
    provenance records the settings and never the model."""
    model, adata = cohort
    adata = adata.copy()
    res = tcri.perturb.gene_importance(model, adata, genes=[0, 1], n_samples=3,
                                       random_state=0, splitby="disease_status")
    back = tcri.get.result(adata, "gene_importance")
    for slot in ("table", "result", "stats", "shift"):
        pd.testing.assert_frame_equal(back[slot], res[slot])
    pd.testing.assert_frame_equal(tcri.get.gene_importance(adata, which="shift"), res["shift"])

    params = tcri.get.params(adata, "gene_importance")
    assert "model" not in params and "adata" not in params
    assert params["use_gate"] is True and params["n_samples"] == 3 and params["n_draws"] == 3
    assert params["groupby"] == "patient" and params["gate_prob"] == model.module.gate_prob

    adata.write_h5ad(tmp_path / "x.h5ad")
    import anndata as ad
    reread = tcri.get.result(ad.read_h5ad(tmp_path / "x.h5ad"), "gene_importance")
    pd.testing.assert_frame_equal(reread["result"], res["result"])

    tcri.perturb.gene_importance(model, adata, genes=[0], key_added="gi_one")
    assert "gi_one" in adata.uns and len(tcri.get.result(adata, "gene_importance",
                                                         key="gi_one")["result"]) == 6
    before = set(adata.uns)
    tcri.perturb.gene_importance(model, adata, genes=[0], inplace=False, key_added="never")
    assert "never" not in adata.uns and set(adata.uns) == before


def test_the_error_names_the_perturb_call(cohort):
    """Plotting or reading before computing is the easy mistake; the message is the fix."""
    _, adata = cohort
    a = adata.copy()
    a.uns.pop(K.GENE_IMPORTANCE, None)
    with pytest.raises(KeyError, match=r"tcri\.perturb\.gene_importance\(model, adata"):
        tcri.get.result(a, "gene_importance")


def test_genes_accept_names_and_positions_in_order(cohort):
    model, adata = cohort
    res = tcri.perturb.gene_importance(model, adata, genes=[3, "gene_1", 3], inplace=False)
    assert list(dict.fromkeys(res["result"]["gene"])) == ["gene_3", "gene_1"]
    with pytest.raises(KeyError, match="no_such_gene"):
        tcri.perturb.knockout(model, adata, genes=["no_such_gene"])
    with pytest.raises(IndexError, match="out of range"):
        tcri.perturb.knockout(model, adata, genes=[adata.n_vars])


def test_knockout_key_added_writes_obsm(cohort):
    model, adata = cohort
    a = adata.copy()
    frame = tcri.perturb.knockout(model, a, genes=[0], key_added="X_ko_gene0")
    assert a.obsm["X_ko_gene0"].shape == (a.n_obs, frame.shape[1])
    assert a.obsm["X_ko_gene0"].dtype == np.float32
    np.testing.assert_allclose(a.obsm["X_ko_gene0"], frame.to_numpy(), atol=1e-6)
    assert "X_ko_gene0" not in adata.obsm, "the fixture object must not be written to"
