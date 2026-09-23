# Stored results: compute once, plot from the store

Every `tcri.tl` metric, and `tcri.perturb.gene_importance`, computes a result, writes it to
`adata.uns` and returns the same object. Every `tcri.pl` function reads that stored result and
never recomputes it. A plot therefore cannot disagree with the frame in your hand:

```python
res = tcri.tl.mutual_information(adata, covariate="pre", groupby="patient", null_model=None)

cached = tcri.get.result(adata, "mutual_information")
pd.testing.assert_frame_equal(cached["result"], res["result"])

tcri.pl.mutual_information(adata)          # draws res["result"], nothing is recomputed
```

`null_model=None` is in that call because the metrics default to `null_model="auto"`, which needs
a reference fit on the object; the reference is the last section below.

This page describes that convention — where a result lands, what is recorded beside it, how it
survives a round trip through `.h5ad`, and what stops it from changing by accident.

## Why it is a convention and not a habit

Before it, every `pl` function took `adata` and recomputed its metric internally. Two costs
followed. A plot could silently disagree with the table the user had computed, because the plot
call resolved its own arguments; and a plot could only draw what it could recompute, so
`pl.phenotypic_flux` manufactured a `groupby` out of the batch column to get a frame it could
consume.

One decorator, `@tl_result`, now realises the convention as code, so a metric cannot drift from
it:

```python
@tl_result(key=K.MUTUAL_INFORMATION, version=1, schema=schemas.MutualInformation,
           values=("value", "denom"), denominators=("denom",), default_null="phenotype")
def mutual_information(adata, *, covariate=None, groupby=None, ..., key_added=None, inplace=True):
    ...
```

## Where a result lands

The key is the tool's name, so the key says what wrote it.

| Call | Key |
|---|---|
| `tl.mutual_information(adata)` | `uns["tcri_mutual_information"]` |
| `tl.mutual_information(adata, key_added="mi_run2")` | `uns["mi_run2"]` |
| `tl.mutual_information(adata, fit="null.phenotype")` | `uns["tcri_null.phenotype_mutual_information"]` |
| `perturb.gene_importance(model, adata)` | `uns["tcri_gene_importance"]` |

`inplace=False` returns the result without storing it. A metric's payload is a dict of frames:
`table` is the unreduced substrate, one row per covariate, group and item (and draw, with
`n_samples > 0`), and `result` is the reduced per-group frame the plots consume. The metrics that
contrast groups add `stats`; `joint_distribution`, which contrasts nothing, has only the two;
`gene_importance` adds `shift`, its per-phenotype decomposition. The snapshot below is the
written record of which slots each tool has.

## What is recorded beside the payload

Four provenance keys sit in the stored blob next to the payload:

| Key | What it holds |
|---|---|
| `params` | every declared argument of the call |
| `version` | the integer schema version of this result's layout |
| `tool` | the canonical key of the tool that wrote it |
| `tcri_version` | the installed tcri that wrote it |

`params` is the part worth knowing about, because it decides what a plot draws. It records
**every argument in the signature, including defaults the caller never passed** — provenance
that records only explicit arguments cannot answer "what was this run with". Excluded are the
data argument itself, any object argument before it (the fitted model in
`gene_importance(model, adata, ...)`), and the two arguments about storage, `key_added` and
`inplace`.

Arguments the tool *resolved* are recorded as resolved, not as the caller spelled them. A
`groupby=None` that the tool resolved to the registered replicate column is recorded as that
column, so a reader six months later sees the column actually used rather than a placeholder.
The same holds for a `null_model="auto"`, which is recorded as the fit it resolved to.

`pl` reads its axes from this block: `groupby`, `splitby` and the rest come from `params`, not
from plot arguments, which is what makes a plot a view of a specific computation.

## Reading it back

`tcri.get` is the read side. Nothing else needs to know the blob format.

| Call | Returns |
|---|---|
| `get.result(adata, "mutual_information")` | exactly what `tl` returned, provenance stripped |
| `get.table(adata, "mutual_information")` | the `result` frame |
| `get.table(adata, "mutual_information", which="table")` | the unreduced substrate |
| `get.mutual_information(adata)` | the same, one accessor per tool |
| `get.params(adata, "mutual_information")` | the provenance block |
| `get.fits(adata)` | the named fits this object carries |

Every accessor but `get.fits` takes `key=` to read a result stored under another key and `fit=`
to read a named fit's result. Reading a result that was never computed raises and names the call to run:

```text
adata.uns['tcri_mutual_information'] not found. Run tcri.tl.mutual_information(adata, ...)
first — tcri.pl.* renders the stored result and never recomputes it.
```

`pl` functions take `return_df=True` to hand back the frame they would have drawn, which is the
stored `result`.

## Named fits, side by side

An `AnnData` can carry several fits: the main one, and any alternative or null model written
with `to_anndata(fit=...)`. Passing `fit=` to a metric computes it on that fit and stores it
under the fit's own key, leaving the main fit's result in place. The name goes after the
namespace prefix, so the keys still sort together: `tcri_null.phenotype_mutual_information`, or
`<key_added>_<fit>` when the destination is a `key_added` of your own.

A bare kind resolves to the fit that carries it, the way scanpy's `basis="umap"` finds `X_umap`:
`fit="phenotype"` and `fit="null.phenotype"` are one call, one key and one `params` block. An
ambiguous name raises rather than silently preferring one.

## The reference alongside

When a metric declares `null_model`, the decorator runs the whole call a second time against the
named reference fit, with every other argument forwarded verbatim, and joins the two: the result
gains `null_*` columns and, for the value columns, `excess*` columns. Forwarding verbatim is the
point — `weighted`, `normalized`, `clones`, `n_samples` and the rest each change the estimand,
and a reference computed at defaults would be a different quantity subtracted from a different
quantity.

The reference run is itself a stored result under its own fit key, so it is plottable with
`key=`, readable through `tcri.get`, and never computed twice. `pl` draws `quantity="value"`
with the reference behind it, or `quantity="excess"` against a zero rule.

## Surviving `.h5ad`

`uns` is written to HDF5, where `/` is a path separator and a tuple has no writer. A result is
therefore encoded before it is stored, and the encoding never uses a user label as a dict key:

| In the result | Stored as |
|---|---|
| `DataFrame` | columns by position, the labels as an array of values |
| `Series` | its values and its index |
| a dict with any key that is not a valid group name | parallel `keys` and `values` arrays |
| `MultiIndex` | one array per level, plus the level names |
| ndarray, str, int, float, None, list | unchanged — anndata already round-trips these |

Storing columns positionally is what lets a phenotype called `CD8/GZMK` and two columns with the
same label survive the trip. Reading inverts the encoding, so `get.result` gives back the slots
`tl` returned, with their values, column labels and index levels. Each encoded structure is
tagged (`__tcri_df__`, `__tcri_series__`, `__tcri_map__`, `__tcri_multiindex__`), so the decoder
recognises its own output and passes everything else through.

## What stops a result from changing quietly

`version` is the layout's version, declared as `@tl_result(version=N)`. It is guarded by a
snapshot test rather than by review: `tests/test_result_schemas.py` runs every tool twice — a
minimal call and a maximal one — and compares the slots, their column names and their index
levels against `tests/snapshots/tl_schemas.json`.

- A renamed, added or dropped column fails the test until the version is bumped, and the message
  says so.
- A version that goes down fails too.
- The snapshot is rewritten deliberately, with
  `pytest tests/test_result_schemas.py --update-schema-snapshot`.

Reading is versioned in the same terms, and it is the schema version that is compared, not the
package version: a result whose schema is newer than the installed tcri writes is refused rather
than half-read, and one whose schema is older raises and asks to be recomputed.

The procedure for bumping a version is in the
[contributing guide](../development/contributing.md).

## Adding a tool that stores its result

1. Declare `key_added=None` and `inplace=True` in the signature, even though the body never
   reads them, and decorate the function with `@tl_result(key=..., version=1, schema=...)`.
2. Give the payload a `TypedDict` in `tcri/_state/schemas.py`; its required keys are checked on
   every call.
3. Add the canonical key to `tcri/_state/keys.py` and the tool to `tcri.get`, which gives it an
   accessor and lets `pl` find it.
4. Record the new schema in the snapshot, so the guard covers it from the first release.
