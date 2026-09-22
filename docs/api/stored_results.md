# Stored results

Every `tcri.tl` function, and `tcri.perturb.gene_importance`, stores its result under
`adata.uns["<key>"]` and returns the same object. For a named fit the fit name follows the `tcri_`
prefix, as in `tcri_null.phenotype_mutual_information`. Read a result back with
{func}`tcri.get.result`, the frames inside it with {func}`tcri.get.table`, and the arguments it ran
with using {func}`tcri.get.params`.

Beside the payload, each entry written by tcri 0.13 or later carries `params`, the schema `version`
listed below, the `tool` that wrote it and the `tcri_version` that was installed. Entries written
earlier carry only `params` and `version`, and their columns may predate the layout below.

Reading a result whose schema version is newer than the installed tcri raises an error asking you to
upgrade. Reading an older version raises an error asking you to recompute the result.

The fields below are generated from the same snapshot the test suite pins, so they match what the
tools store. Columns marked "always" are present in every result; the others appear when the metric
was given a group, a split or a reference.

```{include} _generated/stored_results.md
```
