"""``diag.joint_distribution_ppc``: the sampled-posterior check of a fit's labels.

The calibration tests build the substrate by hand, so the posterior is known exactly: a clone's
labels drawn from its distribution, and the stored posterior the conjugate one given those
labels. Under that model each draw's ``p`` is uniform across clones, which is the property the
check's readout rests on.
"""
import inspect

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import tcri
from tcri._state import keys as K

#: Asymptotic one-sample Kolmogorov-Smirnov critical constant at the 1% level.
KS_1PCT = 1.628


def _substrate(n_cells, conc, labels, *, logits=None, gate=float("nan"), eps=0.0):
    """One covariate, one ct row per clone, ``conc`` the stored posterior concentration."""
    n_clones, P = conc.shape
    clone = np.repeat(np.arange(n_clones), n_cells)
    phen = [f"ph{i}" for i in range(P)]
    clones = [f"c{i}" for i in range(n_clones)]
    obs = pd.DataFrame({"clone": pd.Categorical(np.array(clones)[clone], categories=clones),
                        "phenotype": pd.Categorical(np.array(phen)[labels], categories=phen),
                        "cov": pd.Categorical(["t0"] * len(clone))},
                       index=[f"cell{i}" for i in range(len(clone))])
    a = ad.AnnData(X=np.zeros((len(clone), 1), dtype="float32"), obs=obs)
    a.uns[K.METADATA] = {"clone_col": "clone", "phenotype_col": "phenotype",
                         "covariate_col": "cov", "batch_col": "cov"}
    a.uns[K.PHENOTYPE_CATEGORIES] = phen
    a.uns[K.CLONOTYPE_CATEGORIES] = clones
    a.uns[K.COVARIATE_CATEGORIES] = ["t0"]
    a.uns[K.P_CT] = conc / conc.sum(1, keepdims=True)
    a.uns[K.CONC_CT] = conc
    a.uns[K.LOCAL_SCALE] = 1.0
    a.uns[K.CT_TO_C] = np.arange(n_clones)
    a.uns[K.CT_TO_COV] = np.zeros(n_clones, dtype=int)
    a.uns[K.CT_ARRAY] = clone
    a.uns[K.COV_ARRAY] = np.zeros(len(clone), dtype=int)
    a.uns[K.GATE_PROB] = gate
    a.uns[K.LABEL_ERROR_RATE] = eps
    a.obsm[K.X_LOGITS] = np.zeros((len(clone), P)) if logits is None else logits
    return a


def _exact(n_clones=400, P=4, alpha=2.0, seed=0, sizes=(1, 1, 1, 2, 3, 5, 8, 13, 30)):
    """Labels from each clone's distribution, and the exact conjugate posterior stored."""
    rng = np.random.default_rng(seed)
    n_cells = rng.choice(sizes, n_clones)
    phi = rng.dirichlet(np.full(P, alpha), n_clones)
    labels = np.concatenate([rng.choice(P, k, p=phi[c]) for c, k in enumerate(n_cells)])
    counts = np.zeros((n_clones, P))
    np.add.at(counts, (np.repeat(np.arange(n_clones), n_cells), labels), 1.0)
    return _substrate(n_cells, alpha + counts, labels)


def _share_rejected(cal):
    """Share of (size_bin, draw) rows whose KS distance exceeds the 1% critical value."""
    return float((cal["ks"] > KS_1PCT / np.sqrt(cal["n_clones"])).mean())


def test_p_is_uniform_when_the_data_come_from_the_model():
    """Each draw's ``p`` is uniform across clones in every size bin, single-cell clones
    included: the randomized tie break is what keeps a 1-cell clone's discrete ``p`` uniform.
    About 1% of (bin, draw) rows reject at the 1% level by chance; a broken tie rule or
    likelihood rejects in every draw of a bin."""
    res = tcri.diag.joint_distribution_ppc(_exact(), n_samples=20, n_sims=100,
                                           random_state=0, plot=False)
    cal = res["calibration"]
    assert cal.query("size_bin == '1'")["n_clones"].iloc[0] > 100
    assert _share_rejected(cal) <= 0.05, cal.groupby("size_bin")["ks"].median()
    mean_p = res["draws"].groupby("n_cells")["p"].mean()
    assert ((mean_p - 0.5).abs() < 0.06).all(), mean_p


def test_a_misfit_is_visible():
    """The same labels against posteriors that belong to other clones: ``p`` piles near 0, so
    the readout rejects in nearly every row. Without this, the uniformity test could pass on a
    check that cannot see anything."""
    a = _exact()
    rows = np.random.default_rng(1).permutation(a.uns[K.CONC_CT].shape[0])
    a.uns[K.CONC_CT] = a.uns[K.CONC_CT][rows]
    a.uns[K.P_CT] = a.uns[K.P_CT][rows]
    res = tcri.diag.joint_distribution_ppc(a, n_samples=10, n_sims=100, random_state=0,
                                           plot=False)
    allbin = res["calibration"].query("size_bin == 'all'")
    assert (allbin["ks"] > KS_1PCT / np.sqrt(allbin["n_clones"])).all()
    assert res["draws"]["p"].mean() < 0.4


def _cell_label_probs(logits, phi_of_cell, gate, eps):
    """softmax(g * l + (1 - g) * log phi) @ C, or softmax(l + log phi) @ C without a gate, in
    numpy: the model's ``phenotype_label`` likelihood, written independently of the package."""
    ell = (logits + np.log(phi_of_cell) if gate is None
           else gate * logits + (1 - gate) * np.log(phi_of_cell))
    q = np.exp(ell - ell.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    P = q.shape[1]
    C = np.full((P, P), eps / (P - 1))
    np.fill_diagonal(C, 1 - eps)
    return q @ C


def _per_cell_model(eps, gate, seed=0, n_clones=300, P=3):
    """Labels drawn cell by cell from the label likelihood at a known phi, with a posterior
    concentrated on that phi. ``gate=None`` is the additive combine."""
    rng = np.random.default_rng(seed)
    n_cells = rng.choice((1, 2, 4, 8, 16), n_clones)
    phi = rng.dirichlet(np.full(P, 1.5), n_clones)
    clone = np.repeat(np.arange(n_clones), n_cells)
    logits = rng.normal(scale=1.5, size=(len(clone), P))
    probs = _cell_label_probs(logits, phi[clone], gate, eps)
    labels = np.array([rng.choice(P, p=row) for row in probs])
    return _substrate(n_cells, 1e6 * phi, labels, logits=logits,
                      gate=float("nan") if gate is None else gate, eps=eps)


@pytest.mark.parametrize("gate", [0.5, None], ids=["gated", "additive"])
def test_the_replicate_is_the_per_cell_likelihood_with_label_noise(gate):
    """With informative logits and ε = 0.4, labels drawn from the model's own per-cell
    likelihood give uniform ``p``, gated and additive. Storing ε = 0 makes the replicates too
    clean, and the check sees it: the observed labels sit farther from the fit than the
    replicates do, and the "all" bin rejects uniformity in every draw."""
    a = _per_cell_model(eps=0.4, gate=gate)
    right = tcri.diag.joint_distribution_ppc(a, n_samples=5, n_sims=200, random_state=0,
                                             plot=False)
    assert _share_rejected(right["calibration"]) <= 0.1
    assert abs(right["draws"]["p"].mean() - 0.5) < 0.04

    a.uns[K.LABEL_ERROR_RATE] = 0.0
    wrong = tcri.diag.joint_distribution_ppc(a, n_samples=5, n_sims=200, random_state=0,
                                             plot=False)
    allbin = wrong["calibration"].query("size_bin == 'all'")
    assert (allbin["ks"] > KS_1PCT / np.sqrt(allbin["n_clones"])).all()
    assert wrong["draws"]["p"].mean() < right["draws"]["p"].mean() - 0.05


@pytest.mark.parametrize("gate", [0.5, None], ids=["gated", "additive"])
def test_mean_replicate_frequencies_are_the_models_expected_frequencies(gate):
    """With ε > 0 and nonzero logits, the replicate labels' mean frequencies per clone are the
    clone's expected label frequencies under the model, computed here in numpy from the
    likelihood: the per-cell probabilities and the sampler are both checked."""
    import torch

    from tcri._compute._joint import _cell_probs
    from tcri.diagnostics._ppc import _confusion, _replicate_counts

    rng = np.random.default_rng(0)
    P, eps, n_rep = 4, 0.3, 4000
    n_cells = np.array([1, 3, 7, 20])
    clone = np.repeat(np.arange(len(n_cells)), n_cells)
    phi = rng.dirichlet(np.ones(P), len(n_cells))
    logits = rng.normal(scale=2.0, size=(len(clone), P))
    expected = np.stack([_cell_label_probs(logits, phi[clone], gate, eps)[clone == k].mean(0)
                         for k in range(len(n_cells))])

    bases = torch.as_tensor(phi, dtype=torch.float64).unsqueeze(0)
    ell = torch.as_tensor(logits, dtype=torch.float64).unsqueeze(0)
    qC = (_cell_probs(bases, torch.as_tensor(clone), ell, gate)[0]
          @ torch.as_tensor(_confusion(eps, P)))
    gen = torch.Generator().manual_seed(0)
    counts = _replicate_counts(qC.cumsum(-1), torch.as_tensor(clone), len(n_cells), n_rep, gen)
    freq = (counts / torch.as_tensor(n_cells, dtype=torch.float64)[None, :, None]).mean(0).numpy()

    se = np.sqrt(expected * (1 - expected) / (n_cells[:, None] * n_rep))
    assert (np.abs(freq - expected) < 5 * se + 1e-12).all(), np.abs(freq - expected) / se


def test_same_seed_same_frames_and_the_draws_are_tls(trained_model, monkeypatch):
    """A seed reproduces every frame, and the posterior draws are the ones
    ``tl.joint_distribution`` makes for that seed: both go through ``_posterior_bases``."""
    from tcri._compute import _joint

    _, adata = trained_model
    seen = []
    real = _joint._posterior_bases

    def record(*args, **kwargs):
        out = real(*args, **kwargs)
        seen.append(out[0].clone())
        return out

    monkeypatch.setattr(_joint, "_posterior_bases", record)
    kw = dict(n_samples=4, n_sims=10, random_state=3, plot=False)
    one = tcri.diag.joint_distribution_ppc(adata, **kw)
    two = tcri.diag.joint_distribution_ppc(adata, **kw)
    for name in ("draws", "summary", "calibration"):
        pd.testing.assert_frame_equal(one[name], two[name], obj=name)
    tcri.tl.joint_distribution(adata, n_samples=4, random_state=3, inplace=False)
    assert len(seen) == 3 and np.array_equal(seen[0].numpy(), seen[2].numpy())

    other = tcri.diag.joint_distribution_ppc(adata, **{**kw, "random_state": 4})
    assert not np.allclose(one["draws"]["p"], other["draws"]["p"])


def test_frames_and_their_columns(trained_model):
    _, adata = trained_model
    res = tcri.diag.joint_distribution_ppc(adata, n_samples=3, n_sims=5, random_state=0,
                                           plot=False)
    assert list(res) == ["draws", "summary", "calibration"]
    assert list(res["draws"].columns) == ["covariate", "clonotype", "draw", "n_cells",
                                          "distance_observed", "distance_replicated", "p"]
    assert list(res["summary"].columns) == ["covariate", "clonotype", "n_cells",
                                            "distance_observed", "distance_replicated"]
    assert list(res["calibration"].columns) == ["covariate", "size_bin", "draw", "n_clones",
                                                "ks"]
    d = res["draws"]
    assert len(d) == len(res["summary"]) * 3
    assert ((d["p"] > 0) & (d["p"] < 1)).all()
    one = tcri.diag.joint_distribution_ppc(adata, covariate="T2", n_samples=3, n_sims=5,
                                           random_state=0, plot=False)
    assert set(one["draws"]["covariate"]) == {"T2"}


@pytest.mark.parametrize("arg", ["n_samples", "n_sims"])
@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_counts_must_be_integers_of_at_least_one(trained_model, arg, bad):
    _, adata = trained_model
    with pytest.raises(ValueError, match=f"{arg} must be an integer >= 1"):
        tcri.diag.joint_distribution_ppc(adata, plot=False, **{arg: bad})


def test_temperature_is_gone():
    assert "temperature" not in inspect.signature(tcri.diag.joint_distribution_ppc).parameters


def test_a_fit_without_its_label_error_rate_raises(trained_model):
    _, adata = trained_model
    a = adata.copy()
    del a.uns[K.LABEL_ERROR_RATE]
    with pytest.raises(RuntimeError, match="to_anndata"):
        tcri.diag.joint_distribution_ppc(a, n_samples=1, n_sims=1, plot=False)
    a.uns[K.LABEL_ERROR_RATE] = float("nan")
    with pytest.raises(ValueError, match="label-free"):
        tcri.diag.joint_distribution_ppc(a, n_samples=1, n_sims=1, plot=False)


def test_a_phenotype_null_is_checked_against_its_permuted_labels(trained_model):
    from tcri.diagnostics._ppc import _fit_labels

    _, adata = trained_model
    phenos = [str(p) for p in adata.uns[K.PHENOTYPE_CATEGORIES]]
    registered = _fit_labels(adata, None, phenos)
    perm = np.asarray(adata.uns[K.fit_key(K.PERMUTATION, "null.phenotype[0]")])
    np.testing.assert_array_equal(_fit_labels(adata, "null.phenotype[0]", phenos),
                                  registered[perm])
    # the other axes leave the labels as registered
    np.testing.assert_array_equal(_fit_labels(adata, "null.clonotype[0]", phenos), registered)


def test_a_clonotype_null_is_checked_on_its_own_groups(trained_model):
    """A clone's ``n_cells`` is its size in the null's own grouping, read through the null's
    ct rows, not the registered clone's size."""
    from tcri._compute._tables import fit_clone_labels

    _, adata = trained_model
    fit = "null.clonotype[0]"
    res = tcri.diag.joint_distribution_ppc(adata, fit=fit, n_samples=2, n_sims=5,
                                           random_state=0, plot=False)
    cov_cats = list(adata.uns[K.COVARIATE_CATEGORIES])
    cov = np.asarray(cov_cats, dtype=object)[np.asarray(adata.uns[K.fit_key(K.COV_ARRAY, fit)])]
    sizes = (pd.DataFrame({"covariate": cov,
                           "clonotype": fit_clone_labels(adata, fit).astype(str).to_numpy()})
             .value_counts().rename("n").reset_index())
    merged = res["summary"].merge(sizes, on=["covariate", "clonotype"])
    assert len(merged) == len(res["summary"])
    np.testing.assert_array_equal(merged["n_cells"], merged["n"])


@pytest.mark.parametrize("p", [(0.3, 0.3, 0.999, 0.999, 0.999), "random"])
def test_the_figure_curve_leaves_the_band_exactly_when_ks_exceeds_it(p):
    """Each draw's step curve in the QQ panels is ``p``'s empirical quantile function, whose
    largest distance from the diagonal is the KS distance ``calibration`` reports. So a curve
    leaving the band is the same event as ``ks`` exceeding the band's critical value. The fixed
    case has KS 0.599 above the 95% value 0.563 at five clones; a linearly interpolated
    quantile curve of it stays within 0.499."""
    from tcri.diagnostics._plot import _quantile_steps
    from tcri.diagnostics._ppc import _ks_uniform

    samples = ([np.asarray(p)] if p != "random"
               else [np.random.default_rng(s).uniform(size=n) for s, n in
                     enumerate((1, 2, 5, 17, 60, 300))])
    for values in samples:
        u, y = _quantile_steps(values)
        assert u[0] == 0 and u[-1] == 1
        assert np.max(np.abs(y - u)) == pytest.approx(_ks_uniform(values), abs=1e-12)
    if p != "random":
        from scipy.stats import kstwo

        assert _ks_uniform(p) > kstwo.ppf(0.95, 5)
