#!/usr/bin/env python3
"""Profile one tcri configuration on a prepared real dataset.

Writes a JSON per run so a sweep can be collected afterwards. Times each stage separately --
training, the per-cell pass, the posterior draw, each metric family -- because they sit on
different hardware paths and a single wall-clock number cannot say which knob to turn.

**`--ramp-by-epoch` rather than `--n-steps-kl-warmup`.** The KL warmup counts OPTIMIZER STEPS,
so at a fixed step count a larger batch means fewer steps per epoch and a ramp that finishes
far later in training -- at ``batch_size = n_obs`` the 2000-step default becomes a 2000-EPOCH
warmup and a normal run trains almost entirely with the prior scaled to nothing. Comparing two
batch sizes at a fixed step count therefore compares two different models. This derives the
step count from ``ceil(n_obs * (1 - val) / batch) * ramp_by_epoch`` so every configuration
completes its ramp at the same point in training, and records what actually happened.
(DE-17/DUX-2 in the training contract, open.)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import os
import platform
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)


class Stages:
    def __init__(self, device):
        self.device, self.t = device, {}

    def _sync(self):
        if self.device == "cuda":
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    @contextlib.contextmanager
    def __call__(self, name):
        self._sync()
        t0 = time.perf_counter()
        yield
        self._sync()
        self.t[name] = round(time.perf_counter() - t0, 2)
        print(f"    [{self.t[name]:>8.2f}s] {name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--ramp-by-epoch", type=float, default=0.6,
                    help="fraction of max_epochs by which the KL ramp should finish")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--n-latent", type=int, default=20)
    ap.add_argument("--n-hidden", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--clonotype-key", default="trb_unique")
    ap.add_argument("--phenotype-key", default="CellType")
    ap.add_argument("--covariate-key", default="treatment")
    ap.add_argument("--batch-key", default="patient_ID")
    ap.add_argument("--replicate", default="patient_ID")
    ap.add_argument("--validation-size", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import anndata as ad
    import numpy as np
    import pyro
    import torch

    import tcri
    from tcri.model import TCRIModel

    accelerator = "gpu" if args.device == "cuda" else "cpu"
    s = Stages(args.device)
    rec = {"tag": args.tag, "argv": vars(args), "host": platform.node(),
           "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "slurm_job": os.environ.get("SLURM_JOB_ID")}
    print(f"=== {args.tag} on {rec['host']} ({rec['gpu'] or args.device}) ===", flush=True)

    with s("read"):
        adata = ad.read_h5ad(args.data)
    rec.update(n_obs=int(adata.n_obs), n_vars=int(adata.n_vars))

    # the ramp has to finish at the same point in training for every batch size, or the
    # configurations are not comparable -- see the module docstring
    n_train = adata.n_obs * (1.0 - args.validation_size)
    steps_per_epoch = max(1, math.ceil(n_train / args.batch_size))
    n_steps_kl_warmup = max(1, int(round(steps_per_epoch * args.epochs * args.ramp_by_epoch)))
    rec.update(steps_per_epoch=steps_per_epoch, n_steps_kl_warmup=n_steps_kl_warmup)
    print(f"    batch={args.batch_size} -> {steps_per_epoch} steps/epoch, "
          f"n_steps_kl_warmup={n_steps_kl_warmup} "
          f"(ramp done by epoch {args.epochs * args.ramp_by_epoch:.0f})", flush=True)

    with s("setup"):
        pyro.clear_param_store()
        TCRIModel.setup_anndata(adata, layer="counts", clonotype_key=args.clonotype_key,
                               phenotype_key=args.phenotype_key,
                               covariate_key=args.covariate_key,
                               batch_key=args.batch_key, replicate=args.replicate)
        model = TCRIModel(adata, n_latent=args.n_latent, n_hidden=args.n_hidden,
                          n_layers=args.n_layers, classifier_n_layers=1,
                          classifier_hidden=args.n_hidden, K=args.k, seed=0)

    with s("train"), contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=args.epochs, batch_size=args.batch_size,
                    n_steps_kl_warmup=n_steps_kl_warmup, accelerator=accelerator,
                    validation_size=args.validation_size,
                    enable_progress_bar=False, enable_model_summary=False)
    rec["training_record"] = {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                  else str(v))
                              for k, v in (model.training_record_ or {}).items()}

    with s("to_anndata"), contextlib.redirect_stdout(io.StringIO()):
        model.to_anndata(adata)

    covs = list(adata.uns["tcri_covariate_categories"])
    rec["covariates"] = [str(c) for c in covs]
    cov = str(covs[0])
    dev = args.device

    with s("joint"):
        tcri.tl.joint_distribution(adata, covariate=cov, n_samples=args.n_samples,
                                   random_state=0, device=dev)
    with s("mutual_information"):
        tcri.tl.mutual_information(adata, covariate=cov, groupby=args.replicate,
                                   n_samples=args.n_samples, random_state=0, device=dev)
    with s("clonotypic_entropy"):
        tcri.tl.clonotypic_entropy(adata, covariate=cov, groupby=args.replicate,
                                   n_samples=args.n_samples, random_state=0, device=dev)
    with s("phenotypic_entropy"):
        tcri.tl.phenotypic_entropy(adata, covariate=cov, groupby=args.replicate,
                                   n_samples=args.n_samples, random_state=0, device=dev)

    # flux and the deltas need two covariate levels; skipping is a result, not a failure
    if len(covs) >= 2:
        a, b = str(covs[0]), str(covs[1])
        with s("phenotypic_flux"):
            tcri.tl.phenotypic_flux(adata, cov_from=a, cov_to=b, groupby=args.replicate,
                                    n_samples=args.n_samples, random_state=0, device=dev)
        with s("delta_phenotypic_entropy"):
            tcri.tl.delta_phenotypic_entropy(adata, cov_from=a, cov_to=b,
                                             groupby=args.replicate,
                                             n_samples=args.n_samples, random_state=0,
                                             device=dev)
        with s("delta_clonotypic_entropy"):
            tcri.tl.delta_clonotypic_entropy(adata, cov_from=a, cov_to=b,
                                             groupby=args.replicate,
                                             n_samples=args.n_samples, random_state=0,
                                             device=dev)
    else:
        rec["skipped"] = "flux/deltas need >=2 covariate levels"
        print(f"    SKIPPED flux+deltas: covariate has {len(covs)} level(s)", flush=True)

    with s("diagnostics"):
        tcri.diag.joint_distribution_ppc(adata, covariate=cov, distance_metric="l1")
        tcri.diag.phenotype_calibration(adata, n_bins=10)

    if torch.cuda.is_available() and args.device == "cuda":
        rec["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    rec["stages_seconds"] = s.t
    rec["total_seconds"] = round(sum(s.t.values()), 1)
    # a headline MI so a config that trained badly is visible, not just slow
    try:
        mi = tcri.get.result(adata, "mutual_information")["result"]["value"]
        rec["mi_mean"] = float(np.nanmean(mi))
    except Exception:
        pass

    with open(args.out, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"\nTOTAL {rec['total_seconds']}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
