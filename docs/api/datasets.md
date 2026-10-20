# Datasets (`tcri.datasets`)

Synthetic data generators with **known ground truth**. `simulate_tcri` produces an
`AnnData` whose clone × phenotype mutual information is known in closed form (stored in
`adata.uns["tcri_truth"]`), which is what makes statistical recovery testing — and the
[walkthrough](../tutorials/index.md) — possible.

```{eval-rst}
.. automodule:: tcri.datasets._simulate
   :members: simulate_cohort, simulate_tcri, mi_from_joint_oracle, simulate_from_fit_params, temperature_scale
   :member-order: bysource
```

## Published data

`yost2019` downloads the Yost et al. (2019) BCC and SCC T-cell dataset from GEO (GSE123813), builds
a scirpy `MuData` with `gex` and `airr` modalities and TRB amino-acid clonotypes, and caches it.
It needs the `datasets` extra: `pip install 'tcri[datasets]'`.

```python
mdata = tcri.datasets.yost2019()
adata = tcri.pp.from_mudata(mdata, clonotype_key="clone_id", covariate_key="treatment")
```

```{eval-rst}
.. automodule:: tcri.datasets._yost
   :members: yost2019
```
