"""``tcri.perturb`` -- the identities that pin the perturbation.

The perturbation is a query on a fitted model, so it has no reference joint and no golden
value (``METRICS_CONTRACT.md``, "The perturbation"). What can be pinned is structure:
``knockout`` of nothing is ``predict()``; ``knockout`` of a gene is ``predict()`` on the
zeroed matrix; an unexpressed gene has importance exactly zero; importance is bounded and is
the L1 norm of a shift that sums to zero, per draw, on the fit and on every null run; the stored
adjusted value is the fit's value minus its paired null's, draw ``d`` paired with null ``d mod
n_nulls``; groups reduce cells by row and bind the prior by global id; draws come from the guide's
Dirichlet and are seeded; the contrast is per gene over groups. The fixtures fit for a few epochs
-- the numbers here are identities, not accuracies.

Each test names the mutation it catches: a flipped sign on ``shift``, a prior bound by loader
position instead of global id, a dropped draw path, a zeroed neighboring column in either
kernel path.

Every model call goes through the ``cohort`` fixture, so every comparison is against one fit.
The no-replicate case is a copy of that cohort with the registered replicate removed.
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


def _summary(res, level="replicate"):
    """A level's summary of a full result (``inplace=False``)."""
    return res[level]["summary"]


def _gi(model, adata, **kw):
    """The full result of a ``gene_importance`` call, stored nowhere."""
    return tcri.perturb.gene_importance(model, adata, inplace=False, **kw)


def _null_names(adata):
    """The fits of the phenotype reference, in the order ``null_model="auto"`` pairs them."""
    return list(K.resolve_group(adata, "phenotype")[1])


@pytest.fixture(scope="module")
def nulls(cohort):
    """The phenotype reference's fits, rebuilt as models, in pairing order."""
    from tcri.null._rebuild import rebuild

    model, adata = cohort
    return [rebuild(model, adata, name) for name in _null_names(adata)]


def _manual_importance(model, adata, gene, *, mask=None):
    """``I`` and the per-phenotype shift from two ``knockout`` frames, over ``mask`` cells."""
    base = tcri.perturb.knockout(model, adata, genes=[])
    pert = tcri.perturb.knockout(model, adata, genes=[gene])
    if mask is not None:
        base, pert = base.loc[mask], pert.loc[mask]
    shift = base.mean(axis=0) - pert.mean(axis=0)
    return float(shift.abs().sum()), shift


def _manual_adjusted(model, nulls, adata, gene, *, mask=None):
    """Per null, in pairing order, the adjusted importance and adjusted shift at
    ``n_samples=0``: the model's minus that null's, each from ``knockout`` frames."""
    i_fit, s_fit = _manual_importance(model, adata, gene, mask=mask)
    out = []
    for null in nulls:
        i_null, s_null = _manual_importance(null, adata, gene, mask=mask)
        out.append((i_fit - i_null, s_fit - s_null))
    return out


def _paired(res, level, **labels):
    """The stored per-pair adjusted values of one row, indexed by ``null``, at one draw."""
    draws = res[level]["draws"]
    for col, value in labels.items():
        draws = draws[draws[col] == value]
    return draws.set_index("null")


class _Spy:
    """Records each call's arguments and return value, and returns it unchanged."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, *args, **kwargs):
        out = self.fn(*args, **kwargs)
        self.calls.append((args, kwargs, out))
        return out


def _spy(monkeypatch, name):
    from tcri.perturbation import _importance

    spy = _Spy(getattr(_importance, name))
    monkeypatch.setattr(_importance, name, spy)
    return spy


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

    head = _summary(_gi(model, adata, genes=[0, 1], use_gate=False))
    gated = _summary(_gi(model, adata, genes=[0, 1]))
    assert not np.allclose(head["adjusted"], gated["adjusted"]), "the gate changed nothing"

    original = module.gate_prob
    monkeypatch.setattr(module, "gate_prob", 0.9 if original != 0.9 else 0.1)
    moved = _summary(_gi(model, adata, genes=[0, 1], use_gate=False))
    np.testing.assert_allclose(moved["adjusted"], head["adjusted"], atol=1e-7)


# ── the importance identities ────────────────────────────────────────────────

def test_unexpressed_gene_has_exactly_zero_importance(cohort):
    """Catches any baseline/perturbed asymmetry: zeroing a column that is already zero changes
    nothing, on the fit and on every null, so the adjusted importance is 0.0 with no floor to
    arrange."""
    model, adata = cohort
    res = _gi(model, _zeroed(adata, 0), genes=[0])
    for level in ("replicate", "item"):
        assert (res[level]["draws"]["adjusted"].abs() < 1e-12).all(), level


def test_importance_is_bounded_and_is_the_l1_of_a_zero_sum_shift(cohort, monkeypatch):
    """Catches a non-simplex probability leaking through, and ties the two levels, on the fit's
    run and on every null run: per draw the importance is in [0, 2] and is ``Σ_p |shift|`` over
    that gene and group's item rows, and the shift sums to zero over phenotypes."""
    model, adata = cohort
    frames = _spy(monkeypatch, "importance_frames")
    _gi(model, adata, n_samples=4, random_state=0)
    assert len(frames.calls) == 1 + len(_null_names(adata))
    keys = ["gene", "patient", "draw"]
    for _, _, (rep, item) in frames.calls:
        assert ((rep["value"] >= 0) & (rep["value"] <= 2)).all()
        by = item.groupby(keys, observed=True, sort=False)["value"]
        assert (by.sum().abs() < 1e-5).all(), "shift does not sum to zero"
        l1 = by.apply(lambda s: float(np.abs(s).sum())).rename("l1").reset_index()
        merged = rep.merge(l1, on=keys, validate="one_to_one")
        assert len(merged) == len(rep)
        np.testing.assert_allclose(merged["value"], merged["l1"], atol=1e-12)


def test_the_shift_draws_reduce_to_the_draw_averaged_shift(cohort, monkeypatch):
    """Catches item rows labeled with the wrong gene, group or phenotype: averaged over draws,
    the fit's per-draw shift is ``mean_draw(baseline) − mean_draw(perturbed)`` from the sums
    accumulated in the same call."""
    model, adata = cohort
    accumulate = _spy(monkeypatch, "_accumulate")
    frames = _spy(monkeypatch, "importance_frames")
    genes = list(adata.var_names[:4])
    _gi(model, adata, genes=genes, n_samples=4, random_state=2)
    base, pert, counts = (t.numpy() for t in accumulate.calls[0][2])
    denom = np.clip(counts, 1.0, None)[None, :, None]
    want = (base / denom).mean(axis=0)[None] - (pert / denom[None]).mean(axis=1)  # [G, groups, P]

    phenotypes = list(adata.uns[K.PHENOTYPE_CATEGORIES])
    patients = list(adata.obs["patient"].dropna().unique())
    item = frames.calls[0][2][1]
    got = item.groupby(["gene", "patient", "phenotype"], observed=True)["value"].mean()
    for g, gene in enumerate(genes):
        for k, patient in enumerate(patients):
            row = got.loc[(gene, patient)].reindex(phenotypes).to_numpy()
            np.testing.assert_allclose(row, want[g, k], atol=1e-12)


def test_the_adjusted_value_is_the_fit_minus_its_paired_null(cohort, monkeypatch):
    """Per pair, on both levels: ``adjusted`` is the fit's value at draw ``d`` minus the value of
    null ``i = d mod n_nulls`` at its own draw ``(d − i) / n_nulls``. The adjusted shift sums to
    zero over phenotypes, and its absolute values do not sum to the adjusted importance: the L1
    of a difference is not the difference of the L1s."""
    model, adata = cohort
    frames = _spy(monkeypatch, "importance_frames")
    res = _gi(model, adata, genes=list(adata.var_names[:3]), n_samples=4, random_state=0)
    n_nulls = len(_null_names(adata))
    for level, slot, labels in (("replicate", 0, ["gene", "patient"]),
                                ("item", 1, ["gene", "patient", "phenotype"])):
        fit = frames.calls[0][2][slot]
        runs = []
        for i in range(n_nulls):
            run = frames.calls[1 + i][2][slot].copy()
            runs.append(run.assign(null=i, draw=i + run["draw"] * n_nulls))
        want = (fit.merge(pd.concat(runs), on=labels + ["draw"], suffixes=("", "_null"),
                          validate="one_to_one")
                .assign(want=lambda f: f["value"] - f["value_null"]))
        stored = res[level]["draws"].merge(want[labels + ["null", "draw", "want"]],
                                           on=labels + ["null", "draw"], validate="one_to_one")
        assert len(stored) == len(res[level]["draws"]) == len(fit)
        np.testing.assert_allclose(stored["adjusted"], stored["want"], atol=1e-12)

    item, rep = res["item"]["draws"], res["replicate"]["draws"]
    by = item.groupby(["gene", "patient", "null", "draw"], observed=True)["adjusted"]
    assert (by.sum().abs() < 1e-5).all()
    l1 = by.apply(lambda s: float(np.abs(s).sum())).rename("l1").reset_index()
    merged = rep.merge(l1, on=["gene", "patient", "null", "draw"], validate="one_to_one")
    assert not np.allclose(merged["adjusted"], merged["l1"])


# ── the pairing ──────────────────────────────────────────────────────────────

def test_rotation_pairs_each_draw_with_one_null(cohort, monkeypatch):
    """At ``n_samples > 0`` fit draw ``d`` is paired with null ``d mod n_nulls`` only, and each
    null draws only its share: ``n_samples / n_nulls`` for a multiple, the first nulls one more
    otherwise."""
    model, adata = cohort
    n_nulls = len(_null_names(adata))
    for n_samples in (2 * n_nulls, 2 * n_nulls + 1):
        prior = _spy(monkeypatch, "prior_draws")
        res = _gi(model, adata, genes=[0, 1], n_samples=n_samples, random_state=1)
        shares = [len(range(i, n_samples, n_nulls)) for i in range(n_nulls)]
        assert [kw["n_samples"] for _, kw, _ in prior.calls] == [n_samples, *shares]
        for level in ("replicate", "item"):
            draws = res[level]["draws"]
            assert (draws["null"] == draws["draw"] % n_nulls).all(), level
            for i in range(n_nulls):
                taken = set(draws.loc[draws["null"] == i, "draw"])
                assert taken == set(range(i, n_samples, n_nulls)), (level, i)
        monkeypatch.undo()


def test_the_plug_in_is_paired_with_every_null(cohort):
    """At ``n_samples=0`` the one fit draw is paired with each null, so a row's adjusted value is
    the mean over the nulls and ``adjusted_sd`` the spread between them, with no HDI."""
    model, adata = cohort
    n_nulls = len(_null_names(adata))
    res = _gi(model, adata, genes=[0, 1])
    for level in ("replicate", "item"):
        draws, summary = res[level]["draws"], res[level]["summary"]
        assert set(draws["null"]) == set(range(n_nulls)) and set(draws["draw"]) == {0}
        labels = [c for c in summary.columns if not c.startswith("adjusted")]
        by = draws.groupby(labels, observed=True, sort=False)["adjusted"]
        want = pd.DataFrame({"mean": by.mean(), "sd": by.std(ddof=1)}).reset_index()
        merged = summary.merge(want, on=labels, validate="one_to_one")
        np.testing.assert_allclose(merged["adjusted"], merged["mean"], atol=1e-12)
        np.testing.assert_allclose(merged["adjusted_sd"], merged["sd"], atol=1e-12)
        assert summary["adjusted_hdi_low"].isna().all() and summary["adjusted_hdi_high"].isna().all()


def test_fewer_draws_than_nulls_raises_before_anything_is_computed(cohort, monkeypatch):
    model, adata = cohort
    n_nulls = len(_null_names(adata))
    assert n_nulls > 1, "the fixture needs two nulls for this check"
    accumulate = _spy(monkeypatch, "_accumulate")
    with pytest.raises(ValueError, match=f"fewer than the reference's {n_nulls} fits"):
        _gi(model, adata, genes=[0], n_samples=n_nulls - 1, random_state=0)
    assert not accumulate.calls


def test_no_reference_raises(cohort):
    """The adjusted value is the only one stored, so a call with nothing to adjust against
    raises rather than storing a bare importance."""
    model, adata = cohort
    with pytest.raises(ValueError, match="needs a reference"):
        _gi(model, adata, genes=[0], null_model=None)
    with pytest.raises(ValueError, match="needs a reference"):
        tcri.perturb.gene_importance(model, adata.copy(), genes=[0], null_model=None)


@pytest.mark.parametrize("n_samples", [0, 2, 4])
def test_only_the_adjusted_value_is_stored(cohort, n_samples):
    """Each draws frame has genes × replicates (× phenotypes) × max(n_samples, n_nulls) rows,
    one per pair, and carries the labels, the pair and ``adjusted``; each summary carries
    ``adjusted`` and its spread. The bare value and the null's value are not stored."""
    model, adata = cohort
    n_nulls = len(_null_names(adata))
    n_pheno = len(adata.uns[K.PHENOTYPE_CATEGORIES])
    pairs = max(n_samples, n_nulls)
    res = _gi(model, adata, genes=[0, 1, 2], n_samples=n_samples, random_state=0)
    for level, per, labels in (("replicate", 1, {"gene", "patient"}),
                               ("item", n_pheno, {"gene", "patient", "phenotype"})):
        draws, summary = res[level]["draws"], res[level]["summary"]
        assert len(draws) == 3 * 6 * per * pairs, level
        assert len(summary) == 3 * 6 * per, level
        assert set(draws.columns) == labels | {"null", "draw", "adjusted"}, level
        assert set(summary.columns) == labels | {"adjusted", "adjusted_sd", "adjusted_hdi_low",
                                                 "adjusted_hdi_high"}, level


def test_nothing_is_stored_for_the_null_runs(cohort):
    model, adata = cohort
    a = adata.copy()
    for key in [k for k in a.uns if "gene_importance" in k or k.startswith("gi_")]:
        del a.uns[key]
    before = set(a.uns)
    tcri.perturb.gene_importance(model, a, genes=[0], n_samples=2, random_state=0)
    tcri.perturb.gene_importance(model, a, genes=[0], key_added="gi_keyed")
    assert set(a.uns) - before == {K.GENE_IMPORTANCE, "gi_keyed"}


def test_a_group_with_no_cell_at_the_covariate_has_nan_rows(cohort):
    """The rows are fixed by the genes, the groups, the phenotypes and the pairs: a group with
    no cell at ``covariate`` keeps NaN rows on both levels, while the other groups' values are
    those of the same cells grouped by patient."""
    model, adata = cohort
    cov = list(adata.uns[K.COVARIATE_CATEGORIES])[0]
    # P2's cells at `cov` form a group of their own, so the group P2 has none at `cov`
    a = adata.copy()
    a.obs["site"] = a.obs["patient"].astype(object)
    a.obs.loc[((a.obs["patient"] == "P2") & (a.obs["covariate"] == cov)).to_numpy(),
              "site"] = "P2 at cov"
    kw = dict(genes=[0, 1], covariate=cov, n_samples=2, random_state=0)
    res = _gi(model, a, groupby="site", **kw)
    full = _gi(model, adata, groupby="patient", **kw)
    n_pheno = len(adata.uns[K.PHENOTYPE_CATEGORIES])
    pairs = max(2, len(_null_names(adata)))
    for level, per in (("replicate", 1), ("item", n_pheno)):
        draws, summary = res[level]["draws"], res[level]["summary"]
        assert len(summary) == 2 * 7 * per and len(draws) == pairs * len(summary)
        empty = summary["site"] == "P2"
        assert empty.sum() == 2 * per and summary.loc[empty, "adjusted"].isna().all()
        assert draws.loc[draws["site"] == "P2", "adjusted"].isna().all()
        assert np.isfinite(summary.loc[~empty, "adjusted"]).all()
        labels = [c for c in ("gene", "phenotype") if c in summary.columns]
        got = (summary.loc[~empty].replace({"site": {"P2 at cov": "P2"}})
               .rename(columns={"site": "patient"}))
        merged = got.merge(full[level]["summary"], on=labels + ["patient"],
                           suffixes=("", "_full"), validate="one_to_one")
        assert len(merged) == len(got)
        np.testing.assert_allclose(merged["adjusted"], merged["adjusted_full"], atol=1e-6)


def test_groups_reduce_cells_and_shift_carries_its_sign(cohort, nulls):
    """Catches a wrong mask, the clone machinery creeping in, and a sign flip of the shift:
    per patient and null, the adjusted importance and shift equal what ``knockout`` frames of
    the model and of that null give by hand, ``baseline − perturbed``."""
    model, adata = cohort
    gene = adata.var_names[5]
    res = _gi(model, adata, genes=[gene])
    for patient in adata.obs["patient"].cat.categories:
        mask = (adata.obs["patient"] == patient).to_numpy()
        rep = _paired(res, "replicate", patient=patient)
        item = _paired(res, "item", patient=patient)
        for i, (want_i, want_shift) in enumerate(_manual_adjusted(model, nulls, adata, gene,
                                                                  mask=mask)):
            assert float(rep.loc[i, "adjusted"]) == pytest.approx(want_i, abs=1e-5)
            got = item.loc[i].set_index("phenotype")["adjusted"].reindex(want_shift.index)
            np.testing.assert_allclose(got.to_numpy(), want_shift.to_numpy(), atol=1e-5)


def test_covariate_restricts_the_cells(cohort, nulls):
    """``covariate`` selects the cells, not the prior: the prior stays each cell's own."""
    model, adata = cohort
    gene = adata.var_names[2]
    cov = list(adata.uns[K.COVARIATE_CATEGORIES])[0]
    res = _gi(model, adata, genes=[gene], covariate=cov)
    assert (_summary(res)["covariate"] == cov).all()
    assert (_summary(res, "item")["covariate"] == cov).all()
    for patient in adata.obs["patient"].cat.categories:
        mask = ((adata.obs["patient"] == patient) & (adata.obs["covariate"] == cov)).to_numpy()
        rep = _paired(res, "replicate", patient=patient)
        for i, (want_i, _) in enumerate(_manual_adjusted(model, nulls, adata, gene, mask=mask)):
            assert float(rep.loc[i, "adjusted"]) == pytest.approx(want_i, abs=1e-5)
    with pytest.raises(ValueError, match="not found among"):
        _gi(model, adata, genes=[gene], covariate="no_such_level")


def test_cell_order_invariant(cohort):
    """Catches loader-position indexing: a reversed view must give the same result. Labels
    follow rows, the prior follows the cell's global id."""
    model, adata = cohort
    rev = adata[::-1].copy()
    genes = list(adata.var_names[:4])
    a = _summary(_gi(model, adata, genes=genes))
    b = _summary(_gi(model, rev, genes=genes))
    a = a.sort_values(["gene", "patient"]).reset_index(drop=True)
    b = b.sort_values(["gene", "patient"]).reset_index(drop=True)
    assert list(a["gene"]) == list(b["gene"]) and list(a["patient"]) == list(b["patient"])
    np.testing.assert_allclose(a["adjusted"], b["adjusted"], atol=1e-6)

    ka = tcri.perturb.knockout(model, adata, genes=genes[:1])
    kb = tcri.perturb.knockout(model, rev, genes=genes[:1]).reindex(ka.index)
    np.testing.assert_allclose(ka.to_numpy(), kb.to_numpy(), atol=1e-6)


def test_batch_size_invariant(cohort):
    """Catches accumulation errors across batches."""
    model, adata = cohort
    genes = list(adata.var_names[:3])
    small = _summary(_gi(model, adata, genes=genes, batch_size=64))
    large = _summary(_gi(model, adata, genes=genes, batch_size=4096))
    np.testing.assert_allclose(small["adjusted"], large["adjusted"], atol=1e-5)


# ── draws ────────────────────────────────────────────────────────────────────

def test_draws_are_dirichlet_and_seeded(cohort, monkeypatch):
    """``n_samples=0`` is deterministic; ``>0`` is seeded, differs across seeds, has a spread,
    and collapses onto the plug-in when the guide's concentration is made enormous on the fit
    and on its nulls -- which is what shows the draws come from ``get_conc_ct`` and not from
    anywhere else."""
    model, adata = cohort
    genes = list(adata.var_names[:3])
    kw = dict(genes=genes)

    a = _summary(_gi(model, adata, **kw))
    b = _summary(_gi(model, adata, **kw))
    np.testing.assert_array_equal(a["adjusted"], b["adjusted"])
    assert a["adjusted_hdi_low"].isna().all(), "no draws, so no interval to report"

    s0 = _gi(model, adata, n_samples=8, random_state=0, **kw)
    s0b = _gi(model, adata, n_samples=8, random_state=0, **kw)
    s1 = _gi(model, adata, n_samples=8, random_state=1, **kw)
    for level in ("replicate", "item"):
        assert s0[level]["draws"]["draw"].nunique() == 8
        r0, r0b, r1 = (_summary(s, level) for s in (s0, s0b, s1))
        np.testing.assert_array_equal(r0["adjusted"], r0b["adjusted"])
        assert not np.array_equal(r0["adjusted"], r1["adjusted"])
        assert np.isfinite(r0["adjusted_sd"]).all() and (r0["adjusted_sd"] > 0).any()
        assert (r0["adjusted_hdi_low"] <= r0["adjusted"] + 1e-12).all()
        assert (r0["adjusted_hdi_high"] >= r0["adjusted"] - 1e-12).all()

    module = type(model.module)
    original = module.get_conc_ct
    monkeypatch.setattr(module, "get_conc_ct", lambda self: original(self) * 1e8)
    tight = _gi(model, adata, n_samples=8, random_state=0, **kw)
    np.testing.assert_allclose(_summary(tight)["adjusted"], a["adjusted"], atol=5e-4)


def test_use_gate_false_ignores_n_samples_with_a_warning(cohort):
    """The head alone has no posterior to draw from; a zero-width interval would state a
    certainty never measured. With nothing drawn, the one draw is paired with every null."""
    model, adata = cohort
    n_nulls = len(_null_names(adata))
    with pytest.warns(UserWarning, match="n_samples has no effect"):
        res = _gi(model, adata, genes=[0], use_gate=False, n_samples=2 * n_nulls)
    draws = res["replicate"]["draws"]
    assert draws["draw"].nunique() == 1 and set(draws["null"]) == set(range(n_nulls))
    assert _summary(res)["adjusted_hdi_low"].isna().all()


# ── groups, contrasts, storage ───────────────────────────────────────────────

def test_splitby_contrast_is_per_gene_over_groups(cohort):
    """Catches pseudoreplication: ``stats`` has one row per gene, its ``n`` is the number of
    patients per arm, and ``stat``/``p`` are a Mann-Whitney over the per-patient adjusted
    values of that gene, the one quantity stored."""
    model, adata = cohort
    res = _gi(model, adata, splitby="disease_status")
    stats, result = res["stats"], _summary(res)
    assert stats is not None
    assert set(stats["quantity"]) == {"adjusted"}
    assert len(stats) == adata.n_vars
    assert stats["gene"].is_unique
    assert (stats["replicate_unit"] == "patient").all()
    assert (stats["n_a"] == 3).all() and (stats["n_b"] == 3).all()

    for gene in adata.var_names:
        row = stats.query("gene == @gene").iloc[0]
        per = result.query("gene == @gene")
        va = per.loc[per["disease_status"] == row["level_a"], "adjusted"].to_numpy()
        vb = per.loc[per["disease_status"] == row["level_b"], "adjusted"].to_numpy()
        U, p = mannwhitneyu(va, vb, alternative="two-sided")
        assert float(row["stat"]) == pytest.approx(float(U))
        assert float(row["p"]) == pytest.approx(float(p))

    assert _gi(model, adata, genes=[0])["stats"] is None


def test_no_replicate_registered_gives_one_group(cohort, nulls):
    """Without a registered replicate and without ``groupby`` the reduction is over every
    cell: one row per gene, no group column, ``groupby`` recorded as ``None``."""
    model, adata = cohort
    adata = _without_replicate(adata)
    tcri.perturb.gene_importance(model, adata, genes=[0, 1, 2])
    result = tcri.get.replicates(adata, "gene_importance")
    assert list(result["gene"]) == [adata.var_names[i] for i in (0, 1, 2)]
    assert "patient" not in result.columns
    assert "patient" not in tcri.get.items(adata, "gene_importance").columns
    assert tcri.get.params(adata, "gene_importance")["groupby"] is None
    want = np.mean([i for i, _ in _manual_adjusted(model, nulls, adata, adata.var_names[1])])
    assert float(result["adjusted"].iloc[1]) == pytest.approx(want, abs=1e-5)


def test_missing_group_labels_are_excluded_with_a_warning(cohort, nulls):
    """A group column with gaps (not the batch column, which scvi itself refuses)."""
    model, adata = cohort
    a = adata.copy()
    a.obs["site"] = a.obs["patient"].astype(object)
    a.obs.loc[a.obs["site"] == "P5", "site"] = np.nan
    with pytest.warns(UserWarning, match="have no group label"):
        res = _gi(model, a, genes=[0], groupby="site")
    assert set(_summary(res)["site"]) == {"P0", "P1", "P2", "P3", "P4"}
    assert set(_summary(res, "item")["site"]) == {"P0", "P1", "P2", "P3", "P4"}
    mask = (a.obs["site"] == "P0").to_numpy()
    want = np.mean([i for i, _ in _manual_adjusted(model, nulls, a, a.var_names[0], mask=mask)])
    got = float(_summary(res).query("site == 'P0'")["adjusted"].iloc[0])
    assert got == pytest.approx(want, abs=1e-5)


def test_result_is_cached_readable_and_h5ad_safe(cohort, tmp_path):
    """The payload round-trips through ``tcri.get`` and a real ``.h5ad`` write, and
    ``inplace=False`` returns the stored result; the provenance records the settings and never
    the model."""
    model, adata = cohort
    adata = adata.copy()
    kw = dict(genes=[0, 1], n_samples=4, random_state=0, splitby="disease_status")
    assert tcri.perturb.gene_importance(model, adata, **kw) is None
    res = _gi(model, adata.copy(), **kw)
    back = tcri.get.result(adata, "gene_importance")
    for level in ("replicate", "item"):
        for frame in ("draws", "summary"):
            pd.testing.assert_frame_equal(back[level][frame], res[level][frame])
    pd.testing.assert_frame_equal(back["stats"], res["stats"])
    pd.testing.assert_frame_equal(tcri.get.items(adata, "gene_importance"), res["item"]["summary"])
    pd.testing.assert_frame_equal(tcri.get.draws(adata, "gene_importance", items=True),
                                  res["item"]["draws"])

    params = tcri.get.params(adata, "gene_importance")
    assert "model" not in params and "adata" not in params
    assert params["use_gate"] is True and params["n_samples"] == 4 and params["n_draws"] == 4
    assert params["groupby"] == "patient" and params["gate_prob"] == model.module.gate_prob

    adata.write_h5ad(tmp_path / "x.h5ad")
    import anndata as ad
    reread = ad.read_h5ad(tmp_path / "x.h5ad")
    pd.testing.assert_frame_equal(tcri.get.replicates(reread, "gene_importance"),
                                  res["replicate"]["summary"])
    pd.testing.assert_frame_equal(tcri.get.items(reread, "gene_importance"),
                                  res["item"]["summary"])

    tcri.perturb.gene_importance(model, adata, genes=[0], key_added="gi_one")
    assert "gi_one" in adata.uns and len(tcri.get.replicates(adata, "gene_importance",
                                                             key="gi_one")) == 6
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
    assert list(dict.fromkeys(_summary(res)["gene"])) == ["gene_3", "gene_1"]
    assert list(dict.fromkeys(_summary(res, "item")["gene"])) == ["gene_3", "gene_1"]
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
