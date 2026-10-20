"""The Yost et al. (2019) BCC/SCC T-cell dataset from GEO (GSE123813), as a MuData."""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

__all__ = ["yost2019"]

_GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE123nnn/GSE123813/suppl"

#: The GEO supplementary files and their SHA-256 hashes. pooch checks every download against
#: its hash and downloads again when a cached file no longer matches.
_FILES = {
    "GSE123813_bcc_scRNA_counts.txt.gz":
        "sha256:3f8134b3de192e4790d92e678a1591f3b385dc1e8943d9ff33e038685e33bb22",
    "GSE123813_bcc_tcell_metadata.txt.gz":
        "sha256:baf6e34744ba110f869cf348aa83525bd20fc6ca3c3b572f2255610cd16c981e",
    "GSE123813_bcc_tcr.txt.gz":
        "sha256:cb5a34ec3b73b8e1de416db1c9eb7d2200ac590e7c1f1cb5c11a36acb3b00763",
    "GSE123813_scc_scRNA_counts.txt.gz":
        "sha256:cd26a7289babd8b81b9ccb1b2e598bac096e26a92f5a891cf65e8570393fbd7a",
    "GSE123813_scc_metadata.txt.gz":
        "sha256:82533566eab224ba71f5cf1de0fa8714b20de57becbfc5b5d0317d49aa6f2837",
    "GSE123813_scc_tcr.txt.gz":
        "sha256:15eb07d9e1b59691f3af020e988327c1029dbbc773a531cb0d77bf74c80933c6",
}

#: Counts, metadata and TCR file of each cohort. The metadata file lists the cohort's T cells.
_COHORTS = {
    "bcc": ("GSE123813_bcc_scRNA_counts.txt.gz", "GSE123813_bcc_tcell_metadata.txt.gz",
            "GSE123813_bcc_tcr.txt.gz"),
    "scc": ("GSE123813_scc_scRNA_counts.txt.gz", "GSE123813_scc_metadata.txt.gz",
            "GSE123813_scc_tcr.txt.gz"),
}

#: BCC's metadata spells the Treg cluster "Tregs"; it takes SCC's spelling so both cohorts share
#: one label.
_CLUSTER_RENAMES = {"Tregs": "Treg"}

#: Lineage of each cohort's clusters, by cluster name. The cohorts differ only in "Naive". BCC's
#: "Naive" is CD8: GEO's all-cell annotation of the BCC cells (GSE123813_bcc_all_metadata.txt)
#: labels most of its cells as CD8 T cells, and CD8A or CD8B is detected in more of them than
#: CD4. SCC has a separate "CD8_naive" cluster, and its "Naive" cluster is CD4.
_LINEAGE = {
    "bcc": {"CD8_act": "CD8", "CD8_eff": "CD8", "CD8_ex": "CD8", "CD8_ex_act": "CD8",
            "CD8_mem": "CD8", "Naive": "CD8", "Tfh": "CD4", "Th17": "CD4", "Treg": "CD4"},
    "scc": {"CD8_act": "CD8", "CD8_eff": "CD8", "CD8_ex": "CD8", "CD8_ex_act": "CD8",
            "CD8_mem": "CD8", "CD8_naive": "CD8", "Naive": "CD4", "Tfh": "CD4", "Th17": "CD4",
            "Treg": "CD4"},
}

#: Version of the build recipe, part of the cached file's name. Raising it makes every
#: existing cache stale, so the next call rebuilds from the downloaded files.
_BUILD = 1

#: Genes parsed per chunk of a counts file. A chunk is held dense, so its size bounds the
#: memory a read needs; the sparse matrix is assembled from the chunks.
_CHUNK_ROWS = 500

_INSTALL_HINT = "pip install 'tcri[datasets]'"


def _require_dependencies():
    """Import and return scirpy, mudata and pooch, raising with the install command if any is missing."""
    try:
        import mudata
        import pooch
        import scirpy
    except ImportError as err:
        raise ImportError(
            f"tcri.datasets.yost2019 needs scirpy, mudata and pooch ({err.name} is missing). "
            f"Install them with: {_INSTALL_HINT}"
        ) from err
    return scirpy, mudata, pooch


def yost2019(cache_dir: str | os.PathLike | None = None):
    """Return the Yost et al. (2019) BCC and SCC T-cell dataset as a MuData.

    The data are GEO series GSE123813: single-cell RNA and TCR sequencing of T cells from basal
    cell carcinoma (BCC) and squamous cell carcinoma (SCC) samples taken before and after PD-1
    blockade (Yost et al., *Nat. Med.* 25, 1251–1259, 2019).

    The first call downloads the six supplementary files from GEO, checks each against its
    SHA-256 hash, builds the object and caches it in ``cache_dir`` as a gzip-compressed ``.h5mu``
    file. Later calls read the cached file. The downloaded files stay in ``cache_dir``.

    The cells are each cohort's T cells, the cells listed in its metadata file
    (``GSE123813_bcc_tcell_metadata.txt`` and ``GSE123813_scc_metadata.txt``). Cell ids are the
    GEO ids, which start with the cohort name.

    The ``gex`` modality:

    - ``X`` and ``layers["counts"]`` hold the raw counts as a sparse float32 matrix. The cohorts
      are joined on the union of their genes, so a gene measured in one cohort only is 0 in
      the other.
    - ``obs`` holds ``patient``, ``treatment`` (``"pre"`` or ``"post"``) and ``cluster`` from the
      metadata, ``lineage`` (``"CD4"`` or ``"CD8"``) and ``disease`` (``"bcc"`` or ``"scc"``), all
      categorical. BCC's ``Tregs`` cluster takes SCC's name, ``Treg``. ``lineage`` follows the
      cluster: ``CD8_*`` clusters are CD8, and ``Tfh``, ``Th17`` and ``Treg`` are CD4. ``Naive``
      is CD8 in BCC and CD4 in SCC, which has a separate ``CD8_naive`` cluster.
    - ``obsm["X_umap"]`` holds the published UMAP coordinates. Each cohort has its own
      embedding, so coordinates are comparable only within a cohort.

    The ``airr`` modality holds the same cells. Each chain of the GEO TCR table becomes a chain
    with its locus and its CDR3 nucleotide and amino-acid sequences (the AIRR ``junction`` and
    ``junction_aa`` fields). The table has no productivity, V or J genes, or UMI counts, so every
    chain is marked productive and those fields are empty. A cell without a TCR has no chains.
    Scirpy's ``ir.pp.index_chains`` and ``ir.tl.chain_qc`` have been run; no cells are filtered,
    so the ``chain_qc`` columns (``receptor_type``, ``receptor_subtype``, ``chain_pairing``) are
    there to filter on.

    Clonotypes are defined by the identity of the primary TRB chain's CDR3 amino-acid sequence:
    ``ir.pp.ir_dist(metric="identity", sequence="aa")`` and then
    ``ir.tl.define_clonotype_clusters(sequence="aa", metric="identity", receptor_arms="VDJ",
    dual_ir="primary_only", within_group=None, key_added="clone_id")``, which writes ``clone_id``
    to the ``airr`` modality's ``obs``. Cells whose TRB CDR3s differ in nucleotide sequence but
    translate to the same amino-acid sequence share a clonotype, and so do cells of any
    ``receptor_type``: a multichain cell joins the clonotype of its primary TRB chain. Cells
    without a TRB chain have no ``clone_id``. With no UMI counts, ``index_chains`` ranks a
    cell's chains by their nucleotide sequence, so a cell with two TRB chains takes the one with
    the lexicographically greater CDR3 nucleotide sequence as its primary chain.

    Parameters
    ----------
    cache_dir
        The folder for the downloaded files and the built object, created if missing. ``None``
        uses ``pooch.os_cache("tcri")``, tcri's folder in the operating system's cache.

    Returns
    -------
    mudata.MuData
        With ``gex`` and ``airr`` modalities holding the same cells, in the same order. Pass it
        to :func:`tcri.pp.from_mudata`, for example with ``clonotype_key="clone_id"`` and
        ``covariate_key="treatment"``.

    Raises
    ------
    ImportError
        When scirpy, mudata or pooch is missing; ``pip install 'tcri[datasets]'`` installs them.
    ValueError
        When a downloaded file does not match its SHA-256 hash, or its contents are not in the
        GEO layout the build reads.
    """
    ir, mudata, pooch = _require_dependencies()
    folder = Path(pooch.os_cache("tcri") if cache_dir is None else cache_dir).expanduser()
    cached = folder / f"yost2019-build{_BUILD}.h5mu"
    if cached.exists():
        return mudata.read_h5mu(cached)

    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        name: Path(pooch.retrieve(f"{_GEO}/{name}", known_hash=known_hash, fname=name,
                                  path=folder, progressbar=False))
        for name, known_hash in _FILES.items()
    }
    mdata = _build(paths, ir, mudata)
    partial = cached.with_suffix(".h5mu.partial")
    mdata.write_h5mu(partial, compression="gzip")
    os.replace(partial, cached)
    return mdata


def _build(paths: dict[str, Path], ir, mudata):
    """Build the MuData from the downloaded files, keyed by file name."""
    import anndata as ad

    gex, tcr = {}, {}
    for cohort, (counts, metadata, receptors) in _COHORTS.items():
        gex[cohort] = _read_gex(paths[counts], paths[metadata], cohort)
        tcr[cohort] = _read_tcr(paths[receptors])

    ids = [gex[c].obs_names for c in _COHORTS]
    if len(ids[0].intersection(ids[1])):
        for cohort in _COHORTS:
            gex[cohort].obs_names = f"{cohort}_" + gex[cohort].obs_names
            tcr[cohort].index = f"{cohort}_" + tcr[cohort].index

    joined = ad.concat(list(gex.values()), join="outer")
    for column in ("patient", "treatment", "cluster", "lineage", "disease"):
        joined.obs[column] = joined.obs[column].astype("category")
    chains = pd.concat(list(tcr.values()))
    airr = ir.io.from_airr_cells(_airr_cells(joined.obs_names, chains, ir))

    mdata = mudata.MuData({"gex": joined, "airr": airr})
    ir.pp.index_chains(mdata)
    ir.tl.chain_qc(mdata)
    ir.pp.ir_dist(mdata, metric="identity", sequence="aa")
    ir.tl.define_clonotype_clusters(mdata, sequence="aa", metric="identity", receptor_arms="VDJ",
                                    dual_ir="primary_only", within_group=None, key_added="clone_id")
    return mdata


def _read_gex(counts: Path, metadata: Path, cohort: str):
    """One cohort's AnnData: the metadata's cells, their raw counts, obs and UMAP."""
    import anndata as ad

    meta = pd.read_csv(metadata, sep="\t", index_col="cell.id")
    meta.index.name = None
    with gzip.open(counts, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    position = pd.Index(header).get_indexer(meta.index)
    if (position < 0).any():
        raise ValueError(
            f"{int((position < 0).sum())} cells of {metadata.name} are not columns of {counts.name}"
        )
    keep = np.sort(position)
    cells = pd.Index(header)[keep]

    genes, blocks = [], []
    # Each body row is a gene name followed by one value per header field, so field 0 is the
    # gene and the cell in header position i is field i + 1.
    reader = pd.read_csv(counts, sep="\t", header=None, skiprows=1, index_col=0,
                         usecols=[0, *(keep + 1)], chunksize=_CHUNK_ROWS)
    for chunk in reader:
        values = chunk.to_numpy()
        if not np.issubdtype(values.dtype, np.integer) and not np.array_equal(values, np.round(values)):
            raise ValueError(f"{counts.name} holds non-integer counts")
        genes.extend(chunk.index.astype(str))
        blocks.append(sparse.csr_matrix(values.astype(np.float32)))
    X = sparse.vstack(blocks, format="csr").T.tocsr()

    obs = meta.loc[cells, ["patient", "treatment", "cluster"]].copy()
    obs["cluster"] = obs["cluster"].replace(_CLUSTER_RENAMES)
    unknown = sorted(set(obs["cluster"]) - set(_LINEAGE[cohort]))
    if unknown:
        raise ValueError(f"{metadata.name} has clusters with no lineage: {unknown}")
    obs["lineage"] = obs["cluster"].map(_LINEAGE[cohort])
    obs["disease"] = cohort
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes)))
    adata.layers["counts"] = X.copy()
    adata.obsm["X_umap"] = meta.loc[cells, ["UMAP1", "UMAP2"]].to_numpy(dtype=float)
    return adata


def _read_tcr(path: Path) -> pd.DataFrame:
    """The TCR table: per cell, ``;``-separated ``LOCUS:CDR3`` nucleotide and amino-acid strings."""
    table = pd.read_csv(path, sep="\t", header=None, skiprows=1, index_col=0,
                        names=["cell", "cdr3s_nt", "cdr3s_aa"])
    table.index.name = None
    return table


def _airr_cells(cells: pd.Index, chains: pd.DataFrame, ir) -> list:
    """One ``AirrCell`` per cell, in order; a cell absent from ``chains`` has no chains."""
    out = []
    table = chains.reindex(cells)
    for cell_id, nt, aa in zip(cells, table["cdr3s_nt"], table["cdr3s_aa"]):
        cell = ir.io.AirrCell(cell_id)
        if isinstance(nt, str):
            nt_chains, aa_chains = nt.split(";"), aa.split(";")
            if len(nt_chains) != len(aa_chains):
                raise ValueError(f"cell {cell_id}: {len(nt_chains)} nucleotide and "
                                 f"{len(aa_chains)} amino-acid CDR3 sequences")
            for nt_chain, aa_chain in zip(nt_chains, aa_chains):
                locus, junction = nt_chain.split(":", 1)
                locus_aa, junction_aa = aa_chain.split(":", 1)
                if locus != locus_aa:
                    raise ValueError(f"cell {cell_id}: chain loci {locus} and {locus_aa} differ")
                chain = ir.io.AirrCell.empty_chain_dict()
                chain.update(locus=locus, junction=junction, junction_aa=junction_aa,
                             productive=True)
                cell.add_chain(chain)
        out.append(cell)
    return out
