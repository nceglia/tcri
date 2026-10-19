"""``tcri.diag`` posterior-predictive checks. Each computes and returns a dict of named
DataFrames, one per level of detail, and with ``plot=True`` draws its figure from those values;
nothing is cached. ``model`` is required only
for ``reconstruction_ppc`` (the live ZINB decoder); the rest are adata-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._state import keys as K

__all__ = ["joint_distribution_ppc", "phenotype_calibration", "reconstruction_ppc", "permutation_null"]


#: Elements per temporary in the replicate draw (``replicates x cells x P``): ~64 MB at float64.
_BUDGET = 8_000_000


def _count_arg(name, value):
    """``value`` as an int, raising unless it is an integer >= 1."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return int(value)


def _check_save(plot, save):
    """``save`` writes the figure, so it needs one to write."""
    if save is not None and not plot:
        raise ValueError("save= writes the figure, and plot=False draws none; pass plot=True "
                         "or drop save=.")


def _size_bins(n_cells):
    """Doubling clone-size bins: a label per clone (``"1"``, ``"2-3"``, ``"4-7"``, ...) and the
    labels present, smallest first."""
    n_cells = np.asarray(n_cells, dtype=np.int64)
    exp = np.floor(np.log2(np.maximum(n_cells, 1))).astype(np.int64)

    def label(e):
        lo, hi = 2 ** int(e), 2 ** (int(e) + 1) - 1
        return str(lo) if lo == hi else f"{lo}-{hi}"

    return np.array([label(e) for e in exp], dtype=object), [label(e) for e in np.unique(exp)]


def _ks_uniform(p):
    """One-sample Kolmogorov-Smirnov distance of ``p`` from Uniform(0, 1)."""
    p = np.sort(np.asarray(p, dtype=float))
    n = len(p)
    i = np.arange(1, n + 1)
    return float(max((i / n - p).max(), (p - (i - 1) / n).max()))


def _label_error_rate(adata, fit):
    """The fit's ε, floored at 1e-6 as the model floors it."""
    key = K.fit_key(K.LABEL_ERROR_RATE, fit)
    if key not in adata.uns:
        raise RuntimeError(
            f"uns[{key!r}] missing: the check draws replicate labels from the fit's label "
            f"likelihood, and its error rate is written by to_anndata. Run "
            f"model.to_anndata(adata{', fit=' + repr(fit) if fit else ''})."
        )
    eps = float(adata.uns[key])
    if np.isnan(eps):
        raise ValueError(
            f"fit {fit!r} is label-free (label_error_rate=None): it has no label likelihood, so "
            f"there is nothing for joint_distribution_ppc to check."
        )
    return max(eps, 1e-6)


def _confusion(eps, P):
    """The label-noise matrix: ``1 - eps`` on the diagonal, ``eps / (P - 1)`` off it."""
    if P == 1:
        return np.ones((1, 1))
    C = np.full((P, P), eps / (P - 1))
    np.fill_diagonal(C, 1.0 - eps)
    return C


def _replicate_counts(cdf, clone_of_cell, n_clones, n_rep, gen):
    """``n_rep`` replicate label sets, one label per cell drawn from its row of ``cdf`` (the
    cumulative label probabilities, ``[n_cells, P]``), counted per clone: ``[n_rep, n_clones, P]``.
    """
    import torch

    P = cdf.shape[-1]
    u = torch.rand((n_rep, cdf.shape[0], 1), generator=gen, dtype=torch.float64)
    codes = (cdf.unsqueeze(0) < u).sum(-1).clamp_max(P - 1)                 # [n_rep, n_cells]
    flat = clone_of_cell.unsqueeze(0) * P + codes
    counts = torch.zeros((n_rep, n_clones * P), dtype=torch.float64)
    counts.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float64))
    return counts.view(n_rep, n_clones, P)


def _fit_labels(adata, fit, phenos):
    """Each cell's phenotype code as the fit was trained on it.

    The registered labels, in ``PHENOTYPE_CATEGORIES`` order. A phenotype null was trained on
    them permuted, so its stored permutation is applied the way the model applied it; any other
    fit sees them as they are.
    """
    from ..model._model import PHENOTYPE_AXIS, _apply_permutation

    col = adata.uns[K.METADATA]["phenotype_col"]
    codes = pd.Categorical(adata.obs[col].astype(str), categories=phenos).codes.astype(np.int64)
    if (codes < 0).any():
        raise ValueError(
            f"{int((codes < 0).sum())} cells have a phenotype in adata.obs[{col!r}] that is not "
            f"among the registered phenotypes."
        )
    if fit is None:
        return codes
    perm = adata.uns.get(K.fit_key(K.PERMUTATION, fit))
    if perm is None:
        return codes
    settings = adata.uns.get(K.fit_key(K.FIT_SETTINGS, fit)) or {}
    axis = settings.get("axis")
    axis = str(settings.get("kind")) if axis is None else str(axis)
    return _apply_permutation(codes, PHENOTYPE_AXIS, (axis, perm))


def joint_distribution_ppc(adata, *, covariate=None, distance_metric="l1", n_samples=100,
                           n_sims=100, clones=None, exclude_pools=True,
                           random_state=None, fit=None, plot=True, save=None):
    """Posterior predictive check of the clone x phenotype fit: does the fit reproduce the
    phenotype labels it was trained on, clone by clone?

    For each of ``n_samples`` posterior draws, every cell's phenotype probabilities are rebuilt
    from the fit (its classifier logits gated with its clone's drawn distribution, then passed
    through the label-noise matrix), and ``n_sims`` simulated label sets are drawn from
    them, cell by cell. Each clone's observed label frequencies and each replicate's are compared
    to the clone's expected frequencies under that draw with ``distance_metric``. The clone's
    ``p`` at that draw is where its observed distance falls among its replicate distances, with
    ties broken at random::

        p = (#{D_rep > D_obs} + U * (#{D_rep = D_obs} + 1)) / (n_sims + 1)

    This is the sampled-posterior p: replicate and discrepancy use one posterior draw, and
    ``p`` is never averaged over draws. When the data come from the model, each draw's ``p`` is
    uniform across clones, so the readout is that distribution, overall and by clone size,
    rather than a flag per clone. A QQ curve on the diagonal means the fit reproduces its data.
    A curve below it means the observed labels sit farther from the fit than its replicates do.
    A curve above it means they sit closer than its replicates do, as when ``label_error_rate``
    is higher than the labels' real noise. The posterior is the fit's variational one and the classifier is a point estimate, so
    uniformity is approximate even for a well-specified fit.

    Each fit is checked against its own training data: a phenotype null against its permuted
    labels, a clonotype or covariate null against its own groups. A null that fits its own data
    passes like the main fit does.

    Cost scales with cells x ``n_samples`` x ``n_sims`` x phenotypes. Nothing is cached;
    every call recomputes.

    Parameters
    ----------
    adata : AnnData
        An object ``model.to_anndata`` wrote, carrying the fit's substrate and logits.
    covariate : str | None
        One covariate value; ``None`` checks every covariate.
    distance_metric : str | callable
        ``"l1"`` (default), ``"kl"``/``"dkl"``, or ``f(p, q) -> float``.
    n_samples : int
        Posterior draws, each a full set of per-clone ``p``. Integer >= 1.
    n_sims : int
        Simulated label sets per draw. ``p`` moves in steps of ``1 / (n_sims + 1)``.
        Integer >= 1.
    clones : list | None
        The fit's clones to check; ``None`` checks every one.
    exclude_pools : bool
        ``True`` leaves out the ``pooled@{group}`` labels ``tcri.pp.pool_rare_clones`` writes,
        as in ``tl.joint_distribution``. ``False`` checks each pool as one clone, which the fit
        modeled it as.
    random_state : int | numpy.Generator | torch.Generator | None
        Seeds the posterior draws, which then equal ``tl.joint_distribution(n_samples=...,
        random_state=...)``'s, and, through a second stream from the same seed, the replicates
        and tie breaks. ``None`` is unseeded.
    fit : str | None
        Which fit to check; ``None`` is the main one.
    plot : bool
        Draw the QQ figure, a row per covariate and a column per clone-size bin plus one for all
        clones: each draw's quantile function of ``p`` as a faint step curve over the exact 95%
        Kolmogorov-Smirnov range for the panel's clone count. A draw's curve leaves the range
        exactly when its KS distance exceeds the 95% critical value. A size bin of fewer than 5
        clones is merged into its smaller neighbor in the figure; ``calibration`` keeps every
        bin. The panel count depends on the data, so the check always draws its own figure.
    save : str | path | None
        Where to save the figure. Needs ``plot=True``.

    Returns
    -------
    dict
        ``"draws"``: one row per (covariate, clonotype, draw) with ``n_cells``,
        ``distance_observed``, ``distance_replicated`` (mean over the draw's replicates) and
        ``p``. ``"summary"``: one row per (covariate, clonotype) with ``n_cells`` and both
        distances averaged over draws. ``"calibration"``: one row per (covariate, size_bin,
        draw) with ``n_clones`` and ``ks``, the Kolmogorov-Smirnov distance of that draw's ``p``
        from uniform; ``size_bin`` is the clone's cell count in the covariate in doubling bins,
        plus ``"all"``.

    Raises
    ------
    ValueError
        ``n_samples`` or ``n_sims`` below 1, an unknown ``covariate``, a label-free fit,
        which has no label likelihood to check, or ``save`` with ``plot=False``.
    RuntimeError
        The fit's logits or label error rate are missing: rerun ``model.to_anndata``.
    """
    import torch

    from .._compute._distance import phenotype_distance_rows
    from .._compute._joint import _cell_probs, _gate, _posterior_bases, _seed_int
    from .._compute._tables import _clone_codes, _resolve_clones
    from ..tools._joint import _substrate

    _check_save(plot, save)
    S = _count_arg("n_samples", n_samples)
    R = _count_arg("n_sims", n_sims)
    dist_rows = phenotype_distance_rows(distance_metric)
    fit = K.resolve_fit(adata, fit)
    phenos = [str(x) for x in adata.uns[K.PHENOTYPE_CATEGORIES]]
    clone_cats = [str(x) for x in adata.uns[K.CLONOTYPE_CATEGORIES]]
    cov_cats = list(adata.uns[K.COVARIATE_CATEGORIES])
    if covariate is not None and covariate not in cov_cats:
        raise ValueError(f"covariate {covariate!r} not found among {cov_cats}")
    covs = [cov_cats.index(covariate)] if covariate is not None else range(len(cov_cats))
    P = len(phenos)

    sub = _substrate(adata, fit, use_logits=True, n_samples=S, caller="joint_distribution_ppc")
    C = torch.as_tensor(_confusion(_label_error_rate(adata, fit), P), dtype=torch.float64)
    y = _fit_labels(adata, fit, phenos)

    # One integer seed for both streams, so a numpy generator is drawn from once. The
    # posterior draws take it as tl does; the replicates take a stream spawned from it, so they
    # never reuse the draws' random numbers.
    seed = _seed_int(random_state)
    dev = torch.device("cpu")
    bases, _ = _posterior_bases(sub["p_ct"], local_scale=sub["local_scale"],
                                conc_ct=sub["conc_ct"], n_samples=S, random_state=seed, dev=dev)
    gen = torch.Generator(device=dev)
    if seed is None:
        gen.seed()
    else:
        gen.manual_seed(int(np.random.SeedSequence(seed).spawn(2)[1].generate_state(1)[0]))

    # The fit's own groups: a clonotype or covariate null's cells sit in its permuted rows.
    ct_arr = np.asarray(sub["ct_array"]).astype(np.int64)
    cov_arr = np.asarray(sub["cov_array"]).astype(np.int64)
    cell_clone = np.asarray(_clone_codes(adata, fit)).astype(np.int64)[ct_arr]
    counted = _resolve_clones(adata, clones, exclude_pools)
    if counted is None:
        order = list(range(len(clone_cats)))
    else:
        index = {c: i for i, c in enumerate(clone_cats)}
        order = list(dict.fromkeys(index[str(c)] for c in counted if str(c) in index))
    rank = np.full(len(clone_cats), -1, dtype=np.int64)
    rank[order] = np.arange(len(order))
    cell_rank = rank[cell_clone]
    logits = torch.as_tensor(np.asarray(sub["logits"]), dtype=torch.float64)
    gate = _gate(sub["gate_prob"])

    frames = []
    for m in covs:
        cells = np.nonzero((cov_arr == m) & (cell_rank >= 0))[0]
        if not len(cells):
            continue
        present = np.unique(cell_rank[cells])
        k_of_cell = np.searchsorted(present, cell_rank[cells])
        n_k = np.bincount(k_of_cell, minlength=len(present))
        n_clones, n = len(present), len(cells)
        f_obs = np.zeros((n_clones, P))
        np.add.at(f_obs, (k_of_cell, y[cells]), 1.0)
        f_obs /= n_k[:, None]

        ell = logits[cells].unsqueeze(0)
        ct_cells = torch.as_tensor(ct_arr[cells])
        k_t = torch.as_tensor(k_of_cell)
        n_k_t = torch.as_tensor(n_k, dtype=torch.float64)
        chunk = max(1, min(R, _BUDGET // max(1, n * P)))
        d_obs = np.empty((S, n_clones))
        d_rep = np.empty((S, n_clones))
        p = np.empty((S, n_clones))
        for s in range(S):
            qC = _cell_probs(bases[s:s + 1], ct_cells, ell, gate)[0] @ C        # [n, P]
            r = (torch.zeros((n_clones, P), dtype=torch.float64).index_add_(0, k_t, qC)
                 / n_k_t[:, None]).numpy()
            cdf = qC.cumsum(-1)
            gt = np.zeros(n_clones)
            eq = np.zeros(n_clones)
            total = np.zeros(n_clones)
            for r0 in range(0, R, chunk):
                rc = min(chunk, R - r0)
                counts = _replicate_counts(cdf, k_t, n_clones, rc, gen)
                f_rep = (counts / n_k_t[None, :, None]).numpy()
                # observed and replicates in ONE call: a replicate equal to the observed labels
                # then gets a bit-identical distance, so ties are exact rather than rounding
                d = dist_rows(np.concatenate([f_obs[None], f_rep]), r[None])
                gt += (d[1:] > d[0]).sum(0)
                eq += (d[1:] == d[0]).sum(0)
                total += d[1:].sum(0)
            tie = torch.rand((n_clones,), generator=gen, dtype=torch.float64).numpy()
            d_obs[s], d_rep[s] = d[0], total / R
            p[s] = (gt + tie * (eq + 1.0)) / (R + 1.0)

        names = np.asarray(clone_cats, dtype=object)[np.asarray(order)[present]]
        frames.append(pd.DataFrame({
            "covariate": [cov_cats[m]] * (n_clones * S),
            "clonotype": np.repeat(names, S),
            "draw": np.tile(np.arange(S), n_clones),
            "n_cells": np.repeat(n_k, S),
            "distance_observed": d_obs.T.ravel(),
            "distance_replicated": d_rep.T.ravel(),
            "p": p.T.ravel(),
        }))

    columns = ["covariate", "clonotype", "draw", "n_cells", "distance_observed",
               "distance_replicated", "p"]
    draws = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    summary = (draws.groupby(["covariate", "clonotype"], sort=False)
               .agg(n_cells=("n_cells", "first"),
                    distance_observed=("distance_observed", "mean"),
                    distance_replicated=("distance_replicated", "mean"))
               .reset_index())
    result = {"draws": draws, "summary": summary, "calibration": _calibration(draws)}
    if plot:
        from ._plot import ppc_figure

        ppc_figure(result, save=save)
    return result


def _calibration(draws):
    """One row per (covariate, size_bin, draw): ``n_clones`` and ``ks``."""
    rows = []
    for m, d in draws.groupby("covariate", sort=False):
        per_clone = d.groupby("clonotype", sort=False)["n_cells"].first()
        labels, bins = _size_bins(per_clone.to_numpy())
        bin_of = dict(zip(per_clone.index, labels))
        pmat = d.pivot(index="draw", columns="clonotype", values="p")[per_clone.index]
        for b in [*bins, "all"]:
            keep = [c for c in per_clone.index if b == "all" or bin_of[c] == b]
            for draw, values in zip(pmat.index, pmat[keep].to_numpy()):
                rows.append({"covariate": m, "size_bin": b, "draw": int(draw),
                             "n_clones": len(keep), "ks": _ks_uniform(values)})
    return pd.DataFrame(rows, columns=["covariate", "size_bin", "draw", "n_clones", "ks"])


def phenotype_calibration(adata, *, n_bins=10, fit=None, plot=True, ax=None, save=None):
    """Reliability of predict() probabilities: bin cells by predicted max-prob and compare the
    mean predicted prob to the empirical accuracy per bin. adata-only.

    ``fit`` selects whose probabilities are read; the TRUE labels are always the registered
    ones, so a phenotype null scores at chance here and a clonotype null scores like its
    parent. That contrast is how a null is evaluated (see :mod:`tcri.null`).

    ``plot=True`` draws the reliability diagram: mean predicted probability against observed
    accuracy per non-empty bin, marker area by the bin's cell count, the diagonal as reference,
    and the ECE in the title. ``ax`` is the one Axes to draw on (``None`` makes the figure);
    ``save`` is a path to write it to and needs ``plot=True``. The return is the same either
    way.

    Returns a dict: ``"bins"``, one row per bin with ``bin``, ``mean_pred``, ``emp_freq`` and
    ``count`` (NaN means for an empty bin); ``"summary"``, one row with ``ece``, the expected
    calibration error (bin accuracy gaps weighted by bin size), and ``n_cells``.
    """
    _check_save(plot, save)
    fit = K.resolve_fit(adata, fit)
    probs = np.asarray(adata.obsm[K.fit_key(K.X_PROBABILITIES, fit)], dtype=float)
    phenos = list(adata.uns[K.PHENOTYPE_CATEGORIES])
    conf = probs.max(axis=1)
    pred = np.asarray(phenos)[probs.argmax(axis=1)]
    true = adata.obs[adata.uns[K.METADATA]["phenotype_col"]].astype(str).to_numpy()
    correct = (pred == true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    N = max(len(conf), 1)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf >= lo) & (conf <= hi if b == n_bins - 1 else conf < hi)
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": b, "mean_pred": np.nan, "emp_freq": np.nan, "count": 0})
            continue
        mp, acc = float(conf[m].mean()), float(correct[m].mean())
        rows.append({"bin": b, "mean_pred": mp, "emp_freq": acc, "count": n})
        ece += (n / N) * abs(acc - mp)
    result = {"bins": pd.DataFrame(rows),
              "summary": pd.DataFrame([{"ece": float(ece), "n_cells": int(len(conf))}])}
    if plot:
        from ._plot import calibration_figure

        calibration_figure(result, ax=ax, save=save)
    return result


def reconstruction_ppc(model, adata=None, *, n_sims=100, random_state=None, plot=True,
                       ax=None, save=None):
    """ZINB reconstruction PPC: simulate counts from the fitted decoder and compare library
    size / dropout / mean / variance vs observed. ``model`` REQUIRED (live decoder).

    ``n_sims`` simulated count matrices are drawn and each statistic
    is averaged over them, which is what makes this a posterior-predictive check rather than a
    single-draw comparison. ``random_state`` seeds them, inside a fork of the torch random state,
    so the caller's global state is the same afterwards; ``None`` draws unseeded from it.

    The encoder/decoder run once per batch and only the sampling is repeated, so cost is
    roughly linear in ``n_sims`` on the cheap part. Statistics accumulate per simulation
    rather than materializing every one, so memory does not scale with
    ``n_sims * n_cells * n_genes``.

    ``plot=True`` draws simulated / observed for each statistic on a log axis centered on 1, one
    dot per replicate and the mean as a bar; a statistic observed as 0 has no ratio, and its
    column says so. ``ax`` is the one Axes to draw on (``None`` makes the figure); ``save`` is a
    path to write it to and needs ``plot=True``. The return is the same either way.

    Returns a dict: ``"summary"``, one row per statistic with ``observed``, ``simulated`` (the
    mean over replicates) and ``discrepancy`` (their absolute difference).
    """
    import torch

    _check_save(plot, save)
    n_sims = _count_arg("n_sims", n_sims)

    module = model.module
    module.eval()
    adata = model._validate_anndata(adata)
    device = next(module.parameters()).device
    if random_state is None:
        return _reconstruction_ppc(model, module, adata, device, n_sims, plot, ax, save)
    # Seeded inside a fork, so the caller's global torch random state is the same afterwards.
    # The CPU state is always forked; a CUDA device's is forked too. device_type is always a
    # real type: torch 2.4 builds a device from it, so None raises there.
    cuda = device.type == "cuda"
    with torch.random.fork_rng(devices=[device.index or 0] if cuda else [], device_type="cuda"):
        torch.manual_seed(int(random_state))
        return _reconstruction_ppc(model, module, adata, device, n_sims, plot, ax, save)


def _reconstruction_ppc(model, module, adata, device, n_sims, plot, ax, save):
    """The body of :func:`reconstruction_ppc`, run with the torch random state seeded."""
    import torch
    import pyro.distributions as dist
    from scvi import REGISTRY_KEYS

    loader = model._make_data_loader(adata=adata, batch_size=256)

    obs_lib, obs_all = [], []
    sim_lib = [[] for _ in range(n_sims)]          # per draw: library size per cell
    sim_zero = np.zeros(n_sims)                    # per draw: count of zero entries
    sim_sum = np.zeros(n_sims)                     # per draw: sum of counts
    sim_sq = np.zeros(n_sims)                      # per draw: sum of squared counts
    n_entries = 0

    with torch.no_grad():
        for tensors in loader:
            x = tensors[REGISTRY_KEYS.X_KEY].to(device)
            b = tensors[REGISTRY_KEYS.BATCH_KEY].long().to(device)
            log_lib = torch.log(x.sum(1, keepdim=True) + 1e-6)
            z_loc, _, _ = module.encoder(x, b)
            _px_scale, _px_r, px_rate, px_dropout = module.decoder("gene", z_loc, log_lib, b)
            gate = torch.sigmoid(px_dropout).clamp(1e-3, 1 - 1e-3)
            # mirror the module's generative distribution exactly (same eps + [-10,10] clamp,
            # tcri/model/_module.py) so the PPC samples from what the model actually defines
            nb_logits = (px_rate + module.eps).log() - (module.px_r.exp() + module.eps).log()
            nb_logits = torch.clamp(nb_logits, min=-10.0, max=10.0)
            total = module.px_r.exp().clamp(max=1e4)
            xd = dist.ZeroInflatedNegativeBinomial(gate=gate, total_count=total,
                                                   logits=nb_logits, validate_args=False)

            obs_lib.append(x.sum(1).cpu().numpy())
            obs_all.append(x.cpu().numpy())
            n_entries += x.numel()

            for s in range(n_sims):
                sim = xd.sample()
                sim_lib[s].append(sim.sum(1).cpu().numpy())
                sim_zero[s] += float((sim == 0).sum())
                sim_sum[s] += float(sim.sum())
                sim_sq[s] += float((sim ** 2).sum())

    ol = np.concatenate(obs_lib)
    oa = np.concatenate(obs_all)
    lib_per_draw = [np.concatenate(d) for d in sim_lib]

    sim_mean = sim_sum / n_entries
    # E[x^2] - (E[x])^2, per draw, then averaged -- not the variance of the pooled draws,
    # which would fold between-draw spread into a within-draw statistic.
    sim_var = sim_sq / n_entries - sim_mean ** 2

    stats = {
        "mean_library_size": (float(ol.mean()),
                              float(np.mean([d.mean() for d in lib_per_draw]))),
        "median_library_size": (float(np.median(ol)),
                                float(np.mean([np.median(d) for d in lib_per_draw]))),
        "dropout_fraction": (float((oa == 0).mean()), float((sim_zero / n_entries).mean())),
        "mean_expression": (float(oa.mean()), float(sim_mean.mean())),
        "var_expression": (float(oa.var()), float(sim_var.mean())),
    }
    df = pd.DataFrame([
        {"statistic": k, "observed": o, "simulated": s, "discrepancy": abs(o - s)}
        for k, (o, s) in stats.items()
    ])
    result = {"summary": df}
    if plot:
        from ._plot import reconstruction_figure

        per_sim = {
            "mean_library_size": [d.mean() for d in lib_per_draw],
            "median_library_size": [np.median(d) for d in lib_per_draw],
            "dropout_fraction": sim_zero / n_entries,
            "mean_expression": sim_mean,
            "var_expression": sim_var,
        }
        reconstruction_figure(df, per_sim, ax=ax, save=save)
    return result


def permutation_null(adata, *, metric="mutual_information", covariate=None, groupby=None,
                     clones=None, exclude_pools=True, normalize_mode="min", n_perm=1000,
                     random_state=None, plot=True, ax=None, save=None):
    """Permutation null for a clone↔phenotype metric: permute phenotype labels within each
    covariate, recompute the metric on the **empirical** clone×phenotype joint to form a null;
    report observed, null mean/sd, z, p. adata-only, model-free.

    ``normalize_mode`` must match the mode used for the statistic this is a null FOR. A null
    computed on the other scale is not a weaker null; it is not a null for that statistic at
    all.

    ``groupby`` gives a per-group null, which is what a per-group statistic needs: a
    per-patient MI is read against a per-patient null. Cells are restricted to the group and
    phenotypes are permuted WITHIN each (covariate, group) stratum, so the null conditions on
    exactly what the reported statistic conditions on.

    The same clone-disjointness check the metric applies is applied here, so the null raises in
    exactly the cases the metric raises rather than quietly answering a different question.

    ``clones`` and ``exclude_pools`` restrict the empirical joint the way they restrict a
    metric's: only the cells of the counted clones enter it, and the phenotypes are permuted
    among them. ``clones`` names the clones to count. ``exclude_pools`` leaves out the
    ``pooled@{group}`` labels ``tcri.pp.pool_rare_clones`` writes: a pool is a mixture of many
    rare clones, not a clone, so by default it is left out and the null counts exactly the
    clones ``pool_rare_clones`` kept. ``False`` counts each pool as one clone.

    Note this is deliberately model-free: it uses the EMPIRICAL clone x phenotype crosstab,
    not the model's ``p_ct``, so it shares no Dirichlet draw stack with ``tl.*``.

    ``plot=True`` draws one panel per row of ``summary``: a histogram of the permuted values with
    the observed value marked. ``ax`` is one Axes per row, or ``None`` to make the figure;
    ``save`` is a path to write it to and needs ``plot=True``. The return is the same either way.

    Returns a dict: ``"summary"``, one row per covariate (and group, with ``groupby``) with
    ``observed``, ``null_mean``, ``null_sd``, ``z`` and ``p``.
    """
    from .._compute._tables import _resolve_clones
    from ..tools._mutual_information import _mi_from_joint

    _check_save(plot, save)
    if metric != "mutual_information":
        raise ValueError("permutation_null currently supports metric='mutual_information'.")
    rng = np.random.default_rng(random_state)
    meta = adata.uns[K.METADATA]
    clone_col, pheno_col, cov_col = meta["clone_col"], meta["phenotype_col"], meta["covariate_col"]
    phenos = list(adata.uns[K.PHENOTYPE_CATEGORIES])
    pheno_index = {p: i for i, p in enumerate(phenos)}
    covs = [covariate] if covariate is not None else list(adata.uns[K.COVARIATE_CATEGORIES])
    counted = _resolve_clones(adata, clones, exclude_pools)
    in_counted = (np.ones(adata.n_obs, dtype=bool) if counted is None
                  else adata.obs[clone_col].isin(counted).to_numpy())

    n_phenos = len(phenos)

    if normalize_mode not in ("min", "average"):
        raise ValueError(
            f"normalize_mode must be 'min' or 'average', got {normalize_mode!r}"
        )

    def _empirical_mi(clones_codes, pheno_codes, n_clones):
        # bincount on the flattened (clone, phenotype) key rather than np.add.at —
        # identical counts, and faster: np.add.at takes the unbuffered ufunc path.
        flat = np.bincount(
            clones_codes * n_phenos + pheno_codes, minlength=n_clones * n_phenos
        )
        J = flat.astype(float).reshape(n_clones, n_phenos)
        return _mi_from_joint(J, normalized=True, mode=normalize_mode)

    if groupby is not None:
        if groupby not in adata.obs.columns:
            raise ValueError(f"groupby={groupby!r} is not a column of adata.obs")
        from .._compute._tables import _validate_group_clones
        # `obs[clone_col]`, not `fit_clone_labels`: `permutation_null` is model-free by
        # design -- it reads only the three shared keys.
        _validate_group_clones(adata.obs[clone_col], adata.obs[groupby], groupby)
        groups = adata.obs[groupby].dropna().unique().tolist()
    else:
        groups = [None]

    rows, nulls = [], []
    for m in covs:
        cov_mask = (adata.obs[cov_col].astype(str) == str(m)).to_numpy() & in_counted
        for g in groups:
            # Stratum = (covariate, group). Permuting within it conditions the null on exactly
            # what the reported statistic conditions on; permuting across strata would destroy
            # structure the statistic never claimed.
            mask = cov_mask if g is None else (
                cov_mask & (adata.obs[groupby] == g).to_numpy()
            )
            clones = adata.obs.loc[mask, clone_col].astype(str).to_numpy()
            uc = {c: i for i, c in enumerate(sorted(set(clones)))}
            cc = np.array([uc[c] for c in clones])
            pc = adata.obs.loc[mask, pheno_col].astype(str).map(pheno_index).to_numpy()
            if len(cc) == 0:
                continue
            obs_mi = _empirical_mi(cc, pc, len(uc))
            null = np.array([_empirical_mi(cc, rng.permutation(pc), len(uc))
                             for _ in range(int(n_perm))])
            mu, sd = float(null.mean()), float(null.std(ddof=1) if null.size > 1 else 0.0)
            z = (obs_mi - mu) / sd if sd > 0 else np.nan
            # (1 + #{null >= obs}) / (1 + n_perm), not the raw fraction. A permutation p-value
            # estimated from a finite number of shuffles can never be 0: the observed statistic
            # is itself one realization under the null. The unfloored fraction would return
            # exactly 0.0 whenever no shuffle beat the observation, claiming infinite evidence
            # from a finite sample. Phipson & Smyth (2010).
            p = float((1.0 + np.count_nonzero(null >= obs_mi)) / (1.0 + int(n_perm)))
            row = {"covariate": m, "observed": obs_mi, "null_mean": mu, "null_sd": sd,
                   "z": z, "p": p}
            if g is not None:
                row[groupby] = g
            rows.append(row)
            nulls.append(null)
    df = pd.DataFrame(rows)
    if plot:
        from ._plot import permutation_null_figure

        permutation_null_figure(df, nulls, groupby=groupby, ax=ax, save=save)
    return {"summary": df}
