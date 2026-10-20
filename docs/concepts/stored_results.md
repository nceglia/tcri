# Stored results: compute once, plot from the store

Every `tcri.tl` metric, and `tcri.perturb.gene_importance`, computes a result, writes it to
`adata.uns` and returns `None`, the scanpy convention. `tcri.get` is the only reader of the stored
result, and every `tcri.pl` function draws it through `tcri.get` and never recomputes it. There
is one copy of each result, so a plot cannot disagree with the frame in your hand:

```python
tcri.tl.mutual_information(adata, covariate="pre", groupby="patient", null_model=None)

mi = tcri.get.replicates(adata, "mutual_information")   # one row per patient
tcri.pl.mutual_information(adata)                         # draws mi, nothing is recomputed
```

With `inplace=False` the call stores nothing and returns the whole result instead:

```python
res = tcri.tl.mutual_information(adata, covariate="pre", groupby="patient", null_model=None,
                                 inplace=False)
res["replicate"]["summary"]                               # the same frame, and nothing stored
```

`null_model=None` is in those calls because the metrics default to `null_model="auto"`, which
needs a reference fit on the object; the reference is the last section below.

This page describes that convention — where a result lands, what is recorded beside it, how it
survives a round trip through `.h5ad`, and what stops it from changing by accident.

## Why it is a convention and not a habit

Before it, every `pl` function took `adata` and recomputed its metric internally. Two costs
followed. A plot could silently disagree with the table the user had computed, because the plot
call resolved its own arguments; and a plot could only draw what it could recompute, so
`pl.phenotypic_flux` manufactured a `groupby` out of the batch column to get a frame it could
consume.

One decorator, `@tl_result`, now realizes the convention as code, so a metric cannot drift from
it:

```python
@tl_result(key=K.MUTUAL_INFORMATION, version=4, schema=schemas.MutualInformation,
           values=("value", "denom", "value_from", "value_to", "denom_from", "denom_to"),
           denominators=("denom", "denom_from", "denom_to"),
           default_null={"covariate": "phenotype", "change": "covariate"})
def mutual_information(adata, *, covariate=None, cov_from=None, cov_to=None, groupby=None, ...,
                       fit=None):
    ...
```

The function declares the parameters it computes with. The decorator adds `key_added`, `inplace`
and, for a metric with a reference, `null_model` after them, and reads those three itself.
`values` lists every value column the metric can return, and a call skips those it does not
return; `default_null` is the reference `null_model="auto"` resolves to, here the phenotype null
at one covariate level and the covariate null on a change.

## Where a result lands

The key is the tool's name, so the key says what wrote it.

| Call | Key |
|---|---|
| `tl.mutual_information(adata)` | `uns["tcri_mutual_information"]` |
| `tl.mutual_information(adata, key_added="mi_run2")` | `uns["mi_run2"]` |
| `tl.mutual_information(adata, fit="null.phenotype[0]")` | `uns["tcri_null.phenotype[0]_mutual_information"]` |
| `perturb.gene_importance(model, adata)` | `uns["tcri_gene_importance"]` |

Mutual information, the two entropies and flux store their result by level:

```text
adata.uns["tcri_<metric>"]
├── replicate
│   ├── draws      one row per replicate and draw (and null fit)
│   └── summary    one row per replicate           ← get.replicates
├── item           (the entropies and flux; mutual information has none)
│   ├── draws      one row per replicate, item and draw (and null fit)
│   └── summary    one row per replicate and item
├── stats          one row per test; None when nothing is compared
└── params, version, tool, tcri_version
```

The replicate summary is the main result, read with `get.replicates`. Every replicate has its
replicate row, NaN when it has no cells at the covariate or no clone left by `clones=`.

A call at one covariate level labels its rows with `covariate`. A change, `cov_from` with
`cov_to`, labels them with `cov_from` and `cov_to` instead, and its `value` is `cov_to − cov_from`
with the endpoints beside it, `value_from` and `value_to` (and `denom_from`, `denom_to` for mutual
information). A reference adds a `null_*` column for each of them and the adjusted endpoints,
`adjusted_from` and `adjusted_to`. A summary reduces a change and its endpoints over one set of
draws, so `value = value_to − value_from` and `adjusted = adjusted_to − adjusted_from` hold in the
summary as they do on every draw. On a change the item rows are the persistent clones, those at
both levels within the replicate.

Flux is a distance between the two levels, so it has no endpoints. Its item rows hold one
distance per persistent clone, and its replicate rows those distances weighted by each clone's
share of the replicate's `cov_from` joint, renormalized over the persistent clones (equally at
`weighted=False`, by cells at `cov_from` with `weighted=True`); a replicate with no
persistent clone has a NaN replicate row.

The other tools, `joint_distribution` and `gene_importance`, store a dict of frames: `table` is
the unreduced substrate, one row per covariate, group and item (and draw, with `n_samples > 0`),
and `result` is the reduced per-group frame the plots consume. `joint_distribution`, which
contrasts nothing, has only the two; `gene_importance` adds `stats` and `shift`, its per-phenotype
decomposition. The snapshot below is the written record of which slots each tool has.

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
The same holds for a `null_model="auto"`, which is recorded as the reference it resolved to.

`pl` reads its axes from this block: `groupby`, `splitby` and the rest come from `params`, not
from plot arguments, which is what makes a plot a view of a specific computation.

## Reading it back

`tcri.get` is the read side, and the only one: a `tl` call returns `None`, so a stored result is
read here or not at all. Nothing else needs to know the blob format.

| Call | Returns |
|---|---|
| `get.result(adata, "phenotypic_entropy")` | the whole stored result, provenance stripped |
| `get.replicates(adata, "phenotypic_entropy")` | the replicate summary |
| `get.items(adata, "phenotypic_entropy")` | the item summary |
| `get.stats(adata, "phenotypic_entropy")` | the tests, or `None` |
| `get.draws(adata, "phenotypic_entropy")` | the replicate draws |
| `get.draws(adata, "phenotypic_entropy", items=True)` | the item draws |
| `get.params(adata, "phenotypic_entropy")` | the provenance block |
| `get.fits(adata)` | the named fits this object carries |

Every accessor but `get.fits` takes `key=` to read a result stored under another key and `fit=`
to read a named fit's result. Reading a result that was never computed raises and names the call to run:

```text
adata.uns['tcri_mutual_information'] not found. Run tcri.tl.mutual_information(adata, ...)
first — tcri.pl.* renders the stored result and never recomputes it.
```

`get.items` and `get.draws(items=True)` raise for mutual information, which has no items, and
`get.replicates`, `get.items` and `get.draws` raise for a result stored as `table` and `result`
(the joint and gene importance), which `get.result` reads.

`pl` functions take `return_df=True` to hand back the frame they would have drawn: the replicate
summary, or with `view="item"` the item summary, or the stored `result` of the other tools.

## Named fits, side by side

An `AnnData` can carry several fits: the main one, and any alternative or null model written
with `to_anndata(fit=...)`. Passing `fit=` to a metric computes it on that fit and stores it
under the fit's own key, leaving the main fit's result in place. The name goes after the
namespace prefix, so the keys still sort together: `tcri_null.phenotype[0]_mutual_information`,
or `<key_added>_<fit>` when the destination is a `key_added` of your own.

A bare kind resolves to the fit that carries it, the way scanpy's `basis="umap"` finds `X_umap`:
`fit="phenotype[0]"` and `fit="null.phenotype[0]"` are one call, one key and one `params` block.
An ambiguous name raises rather than silently preferring one, and so does a reference's name
(`fit="phenotype"`), since `fit=` selects one fit and a reference is several.

## The reference alongside

When a metric declares `null_model`, the decorator runs the whole call again on each fit of the
named reference, with every other argument forwarded verbatim and each null's draws seeded from
their own seed, and joins them draw by draw: every per-draw frame (each level's `draws`, or
`table`) gains a `null` column numbering the fits, every frame gains `null_*` columns and, for
the value columns, `adjusted*` columns, and each summary (or `result`) carries the adjusted
value's spread over all the pairs. Forwarding verbatim is
the point — `weighted`, `normalized`, `clones`, `n_samples` and the rest each change the
estimand, and a reference computed at defaults would be a different quantity subtracted from a
different quantity.

The reference run is itself a stored result under its own fit key, so it is plottable with
`key=`, readable through `tcri.get`, and never computed twice. `pl` draws the adjusted value
against a zero rule by default, or `quantity="value"` with the reference behind it.

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
`tl` computed, with their values, column labels and index levels. Each encoded structure is
tagged (`__tcri_df__`, `__tcri_series__`, `__tcri_map__`, `__tcri_multiindex__`), so the decoder
recognizes its own output and passes everything else through.

## What stops a result from changing quietly

`version` is the layout's version, declared as `@tl_result(version=N)`. It is guarded by a
snapshot test rather than by review: `tests/test_result_schemas.py` runs every tool with a
minimal call and a maximal one, and the metrics that take a change run both again with `cov_from`
and `cov_to`. It compares the slots, their column names and their index levels against
`tests/snapshots/tl_schemas.json`.

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

1. Declare only the parameters the body computes with, and decorate the function with
   `@tl_result(key=..., version=1, schema=...)`, which adds `key_added` and `inplace` (and
   `null_model`, given a `default_null`). Declaring one of those three yourself raises.
2. Give the payload a `TypedDict` in `tcri/_state/schemas.py`; its required keys are checked on
   every call.
3. Add the canonical key to `tcri/_state/keys.py` and the tool to the registry in `tcri.get`,
   which lets `tcri.get` and `pl` find it by name.
4. Record the new schema in the snapshot, so the guard covers it from the first release.
