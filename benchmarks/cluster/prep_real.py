#!/usr/bin/env python3
"""Prepare a real dataset for tcri once, so N training jobs can share the result.

HVG selection and singleton grouping are expensive and deterministic, so doing them inside
every training job would burn hours re-deriving the same object and make the timings
incomparable (each job would carry a different prep cost). This writes a prepared `.h5ad`
and a JSON of what it did.

Steps, in order, because the order matters:

1. ``X -> layers['counts']`` if there is no counts layer. ``seurat_v3`` HVG needs RAW counts;
   run it on normalized data and the variance model is meaningless.
2. HVG to ``n_top_genes`` via rapids_singlecell on the GPU, falling back to scanpy.
3. Subset to those genes and convert to CSR. The source is CSC, which is the wrong layout for
   the row slicing every minibatch does -- leaving it CSC makes training pay for it forever.
4. ``tcri.pp.group_singletons`` to build the clonotype column. On this dataset ``trb`` exists
   and ``trb_unique`` does not; grouping is what creates it, per patient.

Usage::

    python prep_real.py --in smith_new.h5ad --out prepped.h5ad --n-top-genes 2000 \
        --clonotype-key trb --min-clone-size 10
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-top-genes", type=int, default=2000)
    ap.add_argument("--clonotype-key", default="trb")
    ap.add_argument("--target-col", default="trb_unique")
    ap.add_argument("--groupby", default="patient_ID")
    ap.add_argument("--min-clone-size", type=int, default=10)
    ap.add_argument("--phenotype-key", default="CellType")
    ap.add_argument("--covariate-key", default="treatment")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    import tcri

    stamps, t0 = {}, time.perf_counter()

    def mark(stage):
        stamps[stage] = round(time.perf_counter() - t0, 1)
        print(f"  [{stamps[stage]:>7.1f}s] {stage}", flush=True)

    adata = ad.read_h5ad(args.src)
    mark(f"read {adata.shape}")

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
        mark("X -> layers['counts']")

    n_before = adata.n_vars
    if adata.n_vars > args.n_top_genes:
        used = None
        try:
            import rapids_singlecell as rsc
            rsc.get.anndata_to_GPU(adata)
            rsc.pp.highly_variable_genes(adata, n_top_genes=args.n_top_genes,
                                         flavor="seurat_v3", layer="counts")
            rsc.get.anndata_to_CPU(adata)
            used = "rapids_singlecell"
        except Exception as exc:
            print(f"  rapids HVG failed ({type(exc).__name__}: {str(exc)[:160]}); "
                  f"falling back to scanpy", flush=True)
            try:
                rsc.get.anndata_to_CPU(adata)
            except Exception:
                pass
            import scanpy as sc
            sc.pp.highly_variable_genes(adata, n_top_genes=args.n_top_genes,
                                        flavor="seurat_v3", layer="counts")
            used = "scanpy"
        mark(f"HVG via {used}")
        adata = adata[:, adata.var["highly_variable"]].copy()
        mark(f"subset to {adata.n_vars} genes")

    # CSC is the wrong layout for the row slicing a minibatch does; pay the conversion once
    for name, mat in [("X", adata.X)] + list(adata.layers.items()):
        if sp.issparse(mat) and not sp.isspmatrix_csr(mat):
            if name == "X":
                adata.X = mat.tocsr()
            else:
                adata.layers[name] = mat.tocsr()
    mark("-> CSR")

    if args.target_col not in adata.obs:
        tcri.pp.group_singletons(adata, clonotype_key=args.clonotype_key,
                                 groupby=args.groupby, target_col=args.target_col,
                                 min_clone_size=args.min_clone_size)
        mark(f"group_singletons(min_clone_size={args.min_clone_size})")

    obs = adata.obs
    summary = {
        "source": args.src, "out": args.out,
        "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars), "n_vars_before": int(n_before),
        "n_clonotypes": int(obs[args.target_col].nunique()),
        "n_phenotypes": int(obs[args.phenotype_key].nunique()),
        "n_covariates": int(obs[args.covariate_key].nunique()),
        "n_patients": int(obs[args.groupby].nunique()),
        "largest_clone": int(obs[args.target_col].value_counts().max()),
        "min_clone_size": args.min_clone_size,
        "stages_seconds": stamps,
    }
    print(json.dumps(summary, indent=2), flush=True)

    adata.write_h5ad(args.out, compression="gzip")
    mark("write")
    summary["stages_seconds"] = stamps
    with open(args.json or (args.out + ".prep.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
