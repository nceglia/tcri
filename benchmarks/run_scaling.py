#!/usr/bin/env python3
"""Where does tcri get slow, and how much does a GPU help?

Scales a synthetic cohort up and times each stage separately, because "it got slow" is not
actionable -- training, the posterior draws, and the per-metric reductions live on different
hardware paths and scale with different things:

* **training** is minibatched SGD through scvi/pyro -- GPU work, scales with cells x epochs;
* **to_anndata** runs the encoder and classifier over every cell once -- GPU work, scales
  with cells;
* **joint_distribution / metrics** draw the Dirichlet posterior and reduce -- this is the
  ``device=`` seam, and it scales with (clones x covariates x draws), NOT with cells;
* **plotting** is pure matplotlib on an already-reduced frame and should be flat.

Splitting them matters because the last two do not benefit from a GPU the way the first two
do, so a single wall-clock number hides which knob to turn.

Usage::

    python benchmarks/run_scaling.py --preset reduced --device cpu  --out cpu.csv
    python benchmarks/run_scaling.py --preset reduced --device cuda --out gpu.csv
    python benchmarks/run_scaling.py --plot-only cpu.csv gpu.csv --plot scaling.png
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import logging
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

#: (label, n_patients, n_clones, cells per (patient, condition))
PRESETS = {
    "smoke":   [("xs", 4, 10, 150)],
    "reduced": [("xs", 4, 12, 200), ("s", 8, 20, 400), ("m", 16, 30, 600)],
    "full":    [("xs", 4, 12, 200), ("s", 8, 20, 400), ("m", 16, 30, 600),
                ("l", 24, 50, 1000), ("xl", 40, 80, 1500)],
}
STAGES = ["simulate", "setup", "train", "to_anndata", "joint", "metrics", "deltas", "plot"]


class Timer:
    """Wall-clock per stage, with the CUDA queue drained so the number means something."""

    def __init__(self, device):
        self.device = device
        self.times = {}

    def _sync(self):
        if self.device == "cuda":
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    @contextlib.contextmanager
    def __call__(self, stage):
        self._sync()
        t0 = time.perf_counter()
        yield
        self._sync()
        self.times[stage] = time.perf_counter() - t0


def run_cell(label, n_patients, n_clones, n_cells, *, device, epochs, n_samples, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyro

    import tcri
    from tcri.datasets import simulate_cohort
    from tcri.model import TCRIModel

    accelerator = "gpu" if device == "cuda" else "cpu"
    t = Timer(device)

    with t("simulate"):
        adata = simulate_cohort(n_patients=n_patients, conditions=("pre", "post"),
                                n_clones=n_clones, n_cells_per_sample=n_cells,
                                clone_size_distribution="powerlaw", seed=seed)

    with t("setup"):
        pyro.clear_param_store()
        TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                                phenotype_key="phenotype", covariate_key="condition",
                                batch_key="patient", replicate="patient")
        model = TCRIModel(adata, n_latent=10, n_hidden=64, n_layers=2,
                          classifier_n_layers=1, classifier_hidden=64, K=4, seed=seed)

    with t("train"), contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=epochs, batch_size=512, accelerator=accelerator,
                    enable_progress_bar=False, enable_model_summary=False)

    with t("to_anndata"), contextlib.redirect_stdout(io.StringIO()):
        model.to_anndata(adata)

    # `device=` routes the metric engine's draw+reduce, independently of where training ran
    with t("joint"):
        tcri.tl.joint_distribution(adata, covariate="post", n_samples=n_samples,
                                   random_state=seed, device=device)

    with t("metrics"):
        for fn, kw in ((tcri.tl.mutual_information, {}),
                       (tcri.tl.clonotypic_entropy, {}),
                       (tcri.tl.phenotypic_entropy, {})):
            fn(adata, covariate="post", groupby="patient", splitby="response",
               n_samples=n_samples, random_state=seed, device=device, **kw)
        tcri.tl.phenotypic_flux(adata, cov_from="pre", cov_to="post", groupby="patient",
                                splitby="response", n_samples=n_samples,
                                random_state=seed, device=device)

    with t("deltas"):
        tcri.tl.delta_phenotypic_entropy(adata, cov_from="pre", cov_to="post",
                                         groupby="patient", splitby="response",
                                         n_samples=n_samples, random_state=seed, device=device)
        tcri.tl.delta_clonotypic_entropy(adata, cov_from="pre", cov_to="post",
                                         groupby="patient", splitby="response",
                                         n_samples=n_samples, random_state=seed, device=device)

    with t("plot"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        tcri.pl.mutual_information(adata, ax=axes[0, 0])
        tcri.pl.clonotypic_entropy(adata, ax=axes[0, 1])
        tcri.pl.phenotypic_entropy(adata, ax=axes[1, 0])
        tcri.pl.phenotypic_flux(adata, ax=axes[1, 1])
        plt.close(fig)

    peak = np.nan
    if device == "cuda":
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()

    row = {"label": label, "device": device, "seed": seed,
           "n_patients": n_patients, "n_clones": n_clones, "cells_per_sample": n_cells,
           "cells": int(adata.n_obs),
           "clones": int(adata.obs["clone_id"].nunique()),
           "epochs": epochs, "n_samples": n_samples,
           "peak_gpu_gb": peak, **t.times}
    row["total"] = sum(t.times.values())
    del adata, model
    gc.collect()
    return row


def make_plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    devices = [d for d in ("cpu", "cuda") if d in set(df["device"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (1) stacked stage breakdown, one bar group per size per device
    ax = axes[0]
    sizes = df.groupby("cells")["cells"].first().sort_values().index.tolist()
    width = 0.38
    cmap = plt.get_cmap("viridis")
    colours = {s: cmap(i / max(len(STAGES) - 1, 1)) for i, s in enumerate(STAGES)}
    for di, dev in enumerate(devices):
        d = df[df.device == dev].groupby("cells")[STAGES].mean()
        bottom = np.zeros(len(d))
        x = np.arange(len(d)) + (di - (len(devices) - 1) / 2) * width
        for stage in STAGES:
            ax.bar(x, d[stage], width, bottom=bottom, color=colours[stage],
                   edgecolor="white", linewidth=0.4,
                   label=stage if di == 0 else None)
            bottom += d[stage].to_numpy()
        for xi, tot in zip(x, bottom):
            ax.text(xi, tot, dev, ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels([f"{c/1000:.1f}k" for c in d.index])
    ax.set_xlabel("cells"); ax.set_ylabel("seconds"); ax.set_title("where the time goes")
    ax.legend(fontsize=7, ncol=2, frameon=False)

    # (2) scaling per stage
    ax = axes[1]
    for stage in STAGES:
        for dev, ls in zip(devices, ("-", "--")):
            d = df[df.device == dev].groupby("cells")[stage].mean()
            if d.sum() > 0:
                ax.plot(d.index, d.to_numpy(), ls, marker="o", ms=3,
                        color=colours[stage], label=f"{stage} ({dev})" if len(devices) > 1
                        else stage)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cells"); ax.set_ylabel("seconds"); ax.set_title("how each stage scales")
    ax.legend(fontsize=6, ncol=2, frameon=False)

    # (3) speedup, only meaningful with both devices
    ax = axes[2]
    if len(devices) == 2:
        cpu = df[df.device == "cpu"].groupby("cells")[STAGES + ["total"]].mean()
        gpu = df[df.device == "cuda"].groupby("cells")[STAGES + ["total"]].mean()
        common = cpu.index.intersection(gpu.index)
        for stage in STAGES + ["total"]:
            speed = cpu.loc[common, stage] / gpu.loc[common, stage].replace(0, np.nan)
            ax.plot(common, speed, marker="o", ms=4,
                    lw=2.2 if stage == "total" else 1.2,
                    color="black" if stage == "total" else colours.get(stage),
                    label=stage)
        ax.axhline(1.0, ls="--", lw=0.8, c="0.5")
        ax.set_xscale("log")
        ax.set_ylabel("CPU time / GPU time  (>1 = GPU wins)")
        ax.set_xlabel("cells"); ax.set_title("GPU speedup by stage")
        ax.legend(fontsize=6, ncol=2, frameon=False)
    else:
        ax.text(0.5, 0.5, f"only {devices[0]} measured\nrun both for a speedup panel",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default="reduced")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="scaling.csv")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--plot-only", nargs="*", default=None,
                    help="skip timing; build the figure from existing CSVs")
    args = ap.parse_args()

    if args.plot_only:
        df = pd.concat([pd.read_csv(f) for f in args.plot_only], ignore_index=True)
        make_plot(df, args.plot or "scaling.png")
        return

    rows = []
    for label, n_patients, n_clones, n_cells in PRESETS[args.preset]:
        for seed in range(args.seeds):
            try:
                row = run_cell(label, n_patients, n_clones, n_cells, device=args.device,
                               epochs=args.epochs, n_samples=args.n_samples, seed=seed)
            except Exception as exc:                      # OOM is a result, not a crash
                print(f"  {label:>3s} seed {seed}: FAILED {type(exc).__name__}: "
                      f"{str(exc)[:120]}", flush=True)
                rows.append({"label": label, "device": args.device, "seed": seed,
                             "n_patients": n_patients, "n_clones": n_clones,
                             "cells_per_sample": n_cells, "failed": type(exc).__name__})
                continue
            rows.append(row)
            print(f"  {label:>3s} seed {seed}: {row['cells']:>7,} cells, "
                  f"{row['clones']:>4} clones | "
                  + "  ".join(f"{s} {row[s]:.1f}s" for s in STAGES)
                  + f"  | TOTAL {row['total']:.1f}s"
                  + (f"  peak {row['peak_gpu_gb']:.1f}GB" if args.device == "cuda" else ""),
                  flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    ok = df[df.get("failed").isna()] if "failed" in df else df
    if len(ok):
        print("\nseconds per stage:")
        print(ok.groupby("cells")[STAGES + ["total"]].mean().round(2).to_string())
    if args.plot:
        make_plot(ok, args.plot)


if __name__ == "__main__":
    main()
