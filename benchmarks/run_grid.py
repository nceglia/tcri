#!/usr/bin/env python3
"""Synthetic benchmark grid — MAE of the NMI estimate against a known oracle.

Reproduces the design of Supplementary Note 1's "Benchmarks" section: sweep the
difficulty (``fuzziness``), the sample size (``N``), and the number of phenotypes
supplied at inference (``K``), and compare TCRi's normalized MI against the closed-form
truth from :func:`tcri.datasets.simulate_tcri` — plus a GMM/KMeans baseline.

Two things this harness is careful about, both of which are easy to get wrong:

* **Normalization.** The note's eq 6 uses the MEAN denominator; tcri defaults to
  ``min``. Comparing a ``min``-normalized estimate against a mean-normalized truth
  silently inflates the estimate, so ``--normalize-mode`` is explicit and defaults to
  ``average`` (i.e. eq 6).
* **Which oracle.** ``true_*`` is the population value; ``empirical_*`` is what a
  perfect estimator returns on the realized finite sample. MAE is reported against
  both, because at small N they differ by the plug-in bias.

Usage::

    python benchmarks/run_grid.py --preset reduced --device cuda --out results.csv
    python benchmarks/run_grid.py --preset full    --device cuda --profile
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# the published benchmark's axes (Supplementary Note 1, Benchmarks)
PUBLISHED = dict(fuzziness=[round(0.1*i,1) for i in range(10)],
                 n_cells=[250,500,1000,2000,5000],
                 k_infer=[8,10,12], temperature=[0.1,0.5,1.0],
                 seeds=list(range(10)))

PRESETS = {
    # shake-out grid: proves the pipeline and gives a real MAE-vs-fuzziness curve
    "smoke":   dict(fuzziness=[0.0, 0.9], n_cells=[1000], k_infer=[5], seeds=[0]),
    "reduced": dict(fuzziness=[0.0, 0.3, 0.6, 0.9], n_cells=[500, 2000],
                    k_infer=[5], seeds=[0, 1, 2]),
    # the note's grid (K supplied at inference varies around the true 5)
    "full":    dict(fuzziness=[round(0.1 * i, 1) for i in range(10)],
                    n_cells=[250, 500, 1000, 2000, 5000],
                    k_infer=[4, 5, 6], seeds=list(range(10))),
    # reproduce the published figures: needs --fit-params
    "published":       PUBLISHED,
    "published_quick": dict(fuzziness=[0.1], n_cells=[250,1000,5000],
                            k_infer=[10], temperature=[0.1,0.5,1.0],
                            seeds=list(range(3))),
}


def _baseline_nmi(adata, k, seed, method="kmeans"):
    """Cluster expression, then compute NMI from the (clone, cluster) table.

    The comparison point from the note: an estimator that ignores the hierarchical
    model and just clusters cells, then measures clone/cluster coupling.
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    from tcri.datasets import mi_from_joint_oracle

    X = np.asarray(adata.layers["counts"], dtype=float)
    X = X / np.clip(X.sum(1, keepdims=True), 1e-9, None) * 1e4
    X = StandardScaler().fit_transform(X)
    if method == "gmm":
        labels = GaussianMixture(n_components=k, random_state=seed,
                                 covariance_type="diag").fit_predict(X)
    else:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    tab = pd.crosstab(adata.obs["clone_id"], pd.Series(labels, index=adata.obs_names))
    return mi_from_joint_oracle(tab.values)


def run_cell(fuzz, n_cells, k_infer, seed, *, device, n_samples, epochs,
             normalize_mode, baseline, temperature=1.0, fit_params=None, profile=False,
             local_scale=None):
    """One grid point -> a dict of results."""
    import pyro

    import tcri
    from tcri.datasets import simulate_tcri
    from tcri.model._model import TCRIModel

    t_all = time.time()
    if fit_params is not None:
        from tcri.datasets import simulate_from_fit_params
        adata = simulate_from_fit_params(
            fit_params, n_cells=n_cells, temperature=temperature,
            fuzziness=fuzz, seed=seed,
        )
    else:
        adata = simulate_tcri(
            n_clones=40, n_phenotypes=5, n_genes=200, n_cells=n_cells,
            omega_concentration=0.4, fuzziness=fuzz, seed=seed,
        )
    truth = adata.uns["tcri_truth"]

    pyro.clear_param_store()
    TCRIModel.setup_anndata(
        adata, layer="counts", clonotype_key="clone_id", phenotype_key="phenotype",
        covariate_key="covariate", batch_key="batch",
    )
    # local_scale sets the TOTAL Dirichlet concentration on p_ct, so per-entry
    # concentration is local_scale/P. Below 1 the draws are corner-seeking, which is the
    # proposed source of the upward NMI bias. Note it moves BOTH sides at once: the guide's
    # posterior and, via uns, the metric's draw -- so a change here is not attributable to
    # one or the other without a follow-up that pins the metric side separately.
    # DE-19: the cell's seed reached the simulator and the metric draw but never the model, so
    # two runs of the same cell differed by ~1.8e-3 NMI -- larger than several of the effects
    # this grid is meant to measure.
    mk = dict(n_latent=32, n_hidden=64, n_layers=2,
              classifier_n_layers=1, classifier_hidden=64, K=k_infer, seed=seed)
    if local_scale is not None:
        mk["local_scale"] = float(local_scale)
    model = TCRIModel(adata, **mk)

    acc = "gpu" if device == "cuda" else "cpu"
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=epochs, batch_size=1024, accelerator=acc,
                    enable_progress_bar=False, enable_model_summary=False)
        model.to_anndata(adata)
    t_train = time.time() - t0

    # DE-8: record what ACTUALLY happened, not what was requested. `max_epochs` is a ceiling --
    # early stopping is on by default (`_model.py` sets early_stopping=True), so a run can stop
    # well short of it. max_epochs=4000 trained 2964 epochs and max_epochs=8000 trained 2774;
    # nothing recorded the difference, and the two near-identical results were read as an
    # estimator bias floor for a full day. One column would have made it obvious.
    hist = getattr(model, "history", {}) or {}
    _tr = hist.get("elbo_train")
    epochs_actual = int(len(_tr)) if _tr is not None else -1
    stopped_early = bool(epochs_actual >= 0 and epochs_actual < epochs)

    t0 = time.time()
    est = tcri.tl.mutual_information(
        adata, covariate="cov_0", n_samples=n_samples, weighted=True,
        normalize_mode=normalize_mode, device=device, random_state=seed,
    )
    t_metric = time.time() - t0
    est_mean = float(est["mean"]) if isinstance(est, dict) else float(est)

    key = "nmi_average" if normalize_mode == "average" else "nmi_min"
    true_v, emp_v = truth[f"true_{key}"], truth[f"empirical_{key}"]

    row = dict(
        fuzziness=fuzz, n_cells=n_cells, k_infer=k_infer, seed=seed, device=device,
        temperature=temperature, epochs=epochs,
        epochs_actual=epochs_actual, stopped_early=stopped_early,
        local_scale=(local_scale if local_scale is not None else float("nan")),
        true_nmi=true_v, empirical_nmi=emp_v, tcri_nmi=est_mean,
        ae_vs_true=abs(est_mean - true_v), ae_vs_empirical=abs(est_mean - emp_v),
        t_train=t_train, t_metric=t_metric, t_total=time.time() - t_all,
    )
    if isinstance(est, dict):
        row.update(hdi_low=est["hdi_low"], hdi_high=est["hdi_high"],
                   covers_empirical=bool(est["hdi_low"] <= emp_v <= est["hdi_high"]))
    if baseline:
        b = _baseline_nmi(adata, k_infer, seed, method=baseline)
        row[f"{baseline}_nmi"] = b[key]
        row[f"ae_{baseline}_vs_true"] = abs(b[key] - true_v)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default="reduced")
    ap.add_argument("--device", default=None, help="None|cpu|cuda (metrics engine)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--normalize-mode", choices=["average", "min"], default="average",
                    help="'average' == the note's eq 6 (default); 'min' == tcri's default")
    ap.add_argument("--baseline", choices=["kmeans", "gmm", "none"], default="kmeans")
    ap.add_argument("--out", default="benchmark_results.csv")
    ap.add_argument("--profile", action="store_true", help="torch profiler on one cell")
    ap.add_argument("--fit-params", default=None,
                    help="path to a fitted params.pkl -> reproduce the published benchmark")
    ap.add_argument("--k-infer", type=int, default=None,
                    help="override the preset's k_infer. Needed when sweeping FIXTURES: the "
                         "presets pin k_infer=10, so running a K=8 or K=12 fit unchanged would "
                         "conflate 'different omega' with 'wrong K at inference'.")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the preset's temperature sweep with a single value")
    ap.add_argument("--n-cells", type=int, default=None,
                    help="override the preset's n_cells sweep with a single value")
    ap.add_argument("--local-scale", type=float, default=None,
                    help="total Dirichlet concentration on p_ct (per-entry = local_scale/P). "
                         "Below 1 per entry the posterior draws are corner-seeking.")
    args = ap.parse_args()

    grid = PRESETS[args.preset]
    if args.k_infer is not None:
        grid = dict(grid, k_infer=[args.k_infer])
    if args.n_cells is not None:
        grid = dict(grid, n_cells=[args.n_cells])
    if args.temperature is not None:
        grid = dict(grid, temperature=[args.temperature])
    temps = grid.get("temperature", [1.0])
    if args.fit_params is None and temps != [1.0]:
        ap.error("--fit-params is required for a preset that sweeps temperature "
                 "(the synthetic omega cannot reproduce the published anchors)")
    combos = list(itertools.product(grid["fuzziness"], grid["n_cells"],
                                    grid["k_infer"], temps, grid["seeds"]))
    baseline = None if args.baseline == "none" else args.baseline
    print(f"preset={args.preset}  cells={len(combos)}  device={args.device}  "
          f"normalize_mode={args.normalize_mode}  baseline={baseline}", flush=True)

    rows = []
    t0 = time.time()
    for i, (f, n, k, T, s) in enumerate(combos, 1):
        r = run_cell(f, n, k, s, device=args.device, n_samples=args.n_samples,
                     epochs=args.epochs, normalize_mode=args.normalize_mode,
                     baseline=baseline, temperature=T, fit_params=args.fit_params,
                     local_scale=args.local_scale)
        rows.append(r)
        print(f"[{i:>4}/{len(combos)}] f={f} N={n} K={k} T={T} s={s} | "
              f"tcri={r['tcri_nmi']:.4f} true={r['true_nmi']:.4f} "
              f"AE={r['ae_vs_true']:.4f} | "
              f"ep={r['epochs_actual']}/{args.epochs}{'*' if r['stopped_early'] else ''} "
              f"| {r['t_total']:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows, {time.time()-t0:.0f}s total)")

    print("\n=== MAE vs fuzziness (mean over N, K, seeds) ===")
    cols = ["ae_vs_true", "ae_vs_empirical"] + (
        [f"ae_{baseline}_vs_true"] if baseline else [])
    print(df.groupby("fuzziness")[cols].mean().round(4).to_string())
    if "covers_empirical" in df:
        print(f"\nHDI coverage of the realized value: "
              f"{df['covers_empirical'].mean():.1%} ({int(df['covers_empirical'].sum())}/{len(df)})")

    if args.profile:
        _profile_one(args)


def _profile_one(args):
    """torch-profiler breakdown of a single training run + device-sync count."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    import pyro

    from tcri.datasets import simulate_tcri
    from tcri.model._model import TCRIModel

    print("\n=== profile: one training run ===", flush=True)
    adata = simulate_tcri(n_clones=40, n_phenotypes=5, n_genes=200, n_cells=4000, seed=0)
    pyro.clear_param_store()
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="covariate",
                            batch_key="batch")
    model = TCRIModel(adata, n_latent=32, n_hidden=64, n_layers=2,
                      classifier_n_layers=1, classifier_hidden=64, K=5)
    acts = [ProfilerActivity.CPU]
    if args.device == "cuda" and torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    acc = "gpu" if args.device == "cuda" else "cpu"
    with profile(activities=acts, record_shapes=False) as prof:
        with contextlib.redirect_stdout(io.StringIO()):
            model.train(max_epochs=10, batch_size=1024, accelerator=acc,
                        enable_progress_bar=False, enable_model_summary=False)
    sort_key = "cuda_time_total" if ProfilerActivity.CUDA in acts else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=18))


if __name__ == "__main__":
    main()
