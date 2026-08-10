"""``tcri.diag`` posterior-predictive checks (§9.1). All return DataFrames. ``model`` is
required only for ``reconstruction_ppc`` (the live ZINB decoder); the rest are adata-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _keys as K

__all__ = ["joint_distribution_ppc", "phenotype_calibration", "reconstruction_ppc", "permutation_null"]


def joint_distribution_ppc(adata, *, covariate=None, distance_metric="l1", temperature=1.0,
                           clones=None, random_state=None):
    """Model vs empirical per-clone phenotype frequencies (the fixed
    ``compare_joint_distribution``). Per-clone distance + per-covariate aggregate. adata-only."""
    from .._distance import phenotype_distance
    from ..tools import joint_distribution

    dist_fn = phenotype_distance(distance_metric)
    meta = adata.uns[K.METADATA]
    clone_col, pheno_col, cov_col = meta["clone_col"], meta["phenotype_col"], meta["covariate_col"]
    phenos = list(adata.uns[K.PHENOTYPE_CATEGORIES])
    covs = [covariate] if covariate is not None else list(adata.uns[K.COVARIATE_CATEGORIES])

    rows = []
    for m in covs:
        Jm = joint_distribution(adata, covariate=m, use_logits=True, n_samples=0,
                                temperature=temperature, clones=clones)
        cmask = adata.obs[cov_col].astype(str) == str(m)
        sub = adata.obs.loc[cmask, [clone_col, pheno_col]]
        for c in Jm.index:
            emp = (sub.loc[sub[clone_col] == c, pheno_col].value_counts()
                   .reindex(phenos).fillna(0.0).to_numpy(dtype=float))
            if emp.sum() <= 0:
                continue
            emp = emp / emp.sum()
            model_p = np.asarray(Jm.loc[c].reindex(phenos).to_numpy(), dtype=float)
            rows.append({"covariate": m, "clonotype": c,
                         "distance": float(dist_fn(emp, model_p))})
    df = pd.DataFrame(rows)
    if len(df):
        agg = df.groupby("covariate")["distance"].mean().rename("mean_distance").reset_index()
        df.attrs["per_covariate"] = agg
    return df


def phenotype_calibration(adata, *, n_bins=10):
    """Reliability of predict() probabilities: bin by predicted max-prob, compare mean
    predicted prob to empirical accuracy per bin; scalar ECE in ``df.attrs['ECE']``. adata-only."""
    probs = np.asarray(adata.obsm[K.X_PROBABILITIES], dtype=float)
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
    df = pd.DataFrame(rows)
    df.attrs["ECE"] = float(ece)
    return df


def reconstruction_ppc(model, adata=None, *, n_sims=100, random_state=0):
    """ZINB reconstruction PPC: simulate counts from the fitted decoder and compare library
    size / dropout / mean / variance vs observed. ``model`` REQUIRED (live decoder).

    ``n_sims`` posterior-predictive replicates are drawn per cell and each statistic is
    averaged over them, which is what makes this a posterior-predictive check rather than a
    single-draw comparison. It previously did nothing: the body drew exactly one replicate,
    so ``n_sims=1`` and ``n_sims=1000`` returned bit-identical frames in identical wall-clock
    while the knob was declared in the frozen contract.

    The encoder/decoder run once per batch and only the sampling is repeated, so cost is
    roughly linear in ``n_sims`` on the cheap part. Statistics accumulate per draw rather than
    materialising every replicate, so memory does not scale with ``n_sims * n_cells * n_genes``.
    """
    import torch
    import pyro.distributions as dist
    from scvi import REGISTRY_KEYS

    n_sims = int(n_sims)
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1, got {n_sims}")

    module = model.module
    module.eval()
    adata = model._validate_anndata(adata)
    device = next(module.parameters()).device
    torch.manual_seed(int(random_state))
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
    return pd.DataFrame([
        {"statistic": k, "observed": o, "simulated": s, "discrepancy": abs(o - s)}
        for k, (o, s) in stats.items()
    ])


def permutation_null(adata, *, metric="mutual_information", covariate=None,
                     normalize_mode="min", n_perm=1000, random_state=None):
    """Permutation null for a clone↔phenotype metric: permute phenotype labels within each
    covariate, recompute the metric on the **empirical** clone×phenotype joint to form a null;
    report observed, null mean/sd, z, p. adata-only, model-free.

    ``normalize_mode`` must match the mode used for the statistic this is a null FOR. It was
    hardcoded to ``"min"``, so a caller working in ``"average"`` compared their number against
    a null on a different scale -- measured on a 12-clone fixture, the same data gives 0.378
    under ``"min"`` and a materially different value under ``"average"``. A null on the wrong
    scale is not a weaker null; it is not a null for that statistic at all.

    ``groupby`` was accepted and never read -- passing it returned a bit-identical frame.
    Removed rather than implemented: strata here are covariate levels, which is what the
    permutation is conditioned on.

    Note this is deliberately model-free: it uses the EMPIRICAL clone x phenotype crosstab,
    not the model's ``p_ct``, so it shares no Dirichlet draw stack with ``tl.*``.
    """
    from ..tools._mutual_information import _mi_from_joint

    if metric != "mutual_information":
        raise ValueError("permutation_null currently supports metric='mutual_information'.")
    rng = np.random.default_rng(random_state)
    meta = adata.uns[K.METADATA]
    clone_col, pheno_col, cov_col = meta["clone_col"], meta["phenotype_col"], meta["covariate_col"]
    phenos = list(adata.uns[K.PHENOTYPE_CATEGORIES])
    pheno_index = {p: i for i, p in enumerate(phenos)}
    covs = [covariate] if covariate is not None else list(adata.uns[K.COVARIATE_CATEGORIES])

    n_phenos = len(phenos)

    if normalize_mode not in ("min", "average"):
        raise ValueError(
            f"normalize_mode must be 'min' or 'average', got {normalize_mode!r}"
        )

    def _empirical_mi(clones_codes, pheno_codes, n_clones):
        # bincount on the flattened (clone, phenotype) key rather than np.add.at —
        # identical counts, ~3.7x faster (np.add.at is the unbuffered ufunc path).
        flat = np.bincount(
            clones_codes * n_phenos + pheno_codes, minlength=n_clones * n_phenos
        )
        J = flat.astype(float).reshape(n_clones, n_phenos)
        return _mi_from_joint(J, normalized=True, mode=normalize_mode)

    rows = []
    for m in covs:
        cmask = (adata.obs[cov_col].astype(str) == str(m)).to_numpy()
        clones = adata.obs.loc[cmask, clone_col].astype(str).to_numpy()
        uc = {c: i for i, c in enumerate(sorted(set(clones)))}
        cc = np.array([uc[c] for c in clones])
        pc = adata.obs.loc[cmask, pheno_col].astype(str).map(pheno_index).to_numpy()
        if len(cc) == 0:
            continue
        obs_mi = _empirical_mi(cc, pc, len(uc))
        null = np.array([_empirical_mi(cc, rng.permutation(pc), len(uc)) for _ in range(int(n_perm))])
        mu, sd = float(null.mean()), float(null.std(ddof=1) if null.size > 1 else 0.0)
        z = (obs_mi - mu) / sd if sd > 0 else np.nan
        # (1 + #{null >= obs}) / (1 + n_perm), not the raw fraction. A permutation p-value
        # estimated from a finite number of shuffles can never be 0: the observed statistic is
        # itself one realisation under the null. The unfloored version returned exactly 0.0
        # whenever no shuffle beat the observation, which claims infinite evidence from a
        # finite sample. Phipson & Smyth (2010), "Permutation p-values should never be zero".
        p = float((1.0 + np.count_nonzero(null >= obs_mi)) / (1.0 + int(n_perm)))
        rows.append({"covariate": m, "observed": obs_mi, "null_mean": mu, "null_sd": sd,
                     "z": z, "p": p})
    return pd.DataFrame(rows)
