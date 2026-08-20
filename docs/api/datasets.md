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
