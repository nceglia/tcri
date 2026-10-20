"""``yost2019`` — the GEO GSE123813 files built into a scirpy MuData that ``from_mudata`` takes.

The offline tests write small files in the GEO layouts to a temporary ``cache_dir`` and point the
hash registry at them, so pooch finds every file in place and nothing is downloaded. The fixture
is built with one gene per counts chunk, so the chunked read is exercised across chunk borders.

The live test downloads the real files and runs only when selected with ``pytest -m network``.
"""
from __future__ import annotations

import gzip
import hashlib
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import tcri
from tcri.datasets import _yost

BCC_CELLS = ["bcc.su001.pre.tcell_c1", "bcc.su001.pre.tcell_c2", "bcc.su001.post.tcell_c3",
             "bcc.su002.post.tcell_c4"]
#: In the BCC counts but not the T-cell metadata, so not in the built object.
BCC_OTHER = "bcc.su001.pre.tcell_n1"
SCC_CELLS = ["scc.su010.pre_s1", "scc.su010.post_s2", "scc.su011.pre_s3", "scc.su011.post_s4"]

BCC_COUNTS = {"G1": [0, 3, 1, 0, 9], "G2": [2, 0, 0, 1, 9], "G3": [1, 1, 0, 4, 9]}
SCC_COUNTS = {"G2": [5, 0, 1, 2], "G3": [0, 2, 0, 0], "G4": [1, 1, 7, 3]}

BCC_TCR = {
    "bcc.su001.pre.tcell_c1": ("TRA:AAA;TRB:CCC", "TRA:CAVA;TRB:CASSB1"),
    "bcc.su001.pre.tcell_c2": ("TRA:AAT;TRB:CCA", "TRA:CAVB;TRB:CASSB1"),
    "bcc.su001.post.tcell_c3": ("TRA:AAA;TRB:CCG", "TRA:CAVA;TRB:CASSB2"),
    BCC_OTHER: ("TRB:CCT", "TRB:CASSB9"),
}
SCC_TCR = {
    "scc.su010.pre_s1": ("TRB:CCC", "TRB:CASSB1"),
    "scc.su010.post_s2": ("TRA:AGA;TRB:GGA;TRB:GGT", "TRA:CAVC;TRB:CASSB4;TRB:CASSB5"),
    "scc.su011.pre_s3": ("TRA:ATA", "TRA:CAVD"),
    # Three TRA chains make it a multichain cell; its TRB matches c1's.
    "scc.su011.post_s4": ("TRA:TAA;TRA:TAC;TRA:TAG;TRB:CCC",
                          "TRA:CAVE;TRA:CAVF;TRA:CAVG;TRB:CASSB1"),
}


def _write_gz(path, text):
    with gzip.open(path, "wt") as handle:
        handle.write(text)


def _write_counts(path, cells, counts):
    lines = ["\t".join(cells)]
    lines += [f"{gene}\t" + "\t".join(map(str, values)) for gene, values in counts.items()]
    _write_gz(path, "\n".join(lines) + "\n")


def _write_metadata(path, rows):
    lines = ["cell.id\tpatient\ttreatment\tcluster\tUMAP1\tUMAP2"]
    lines += ["\t".join(map(str, row)) for row in rows]
    _write_gz(path, "\n".join(lines) + "\n")


def _write_tcr(path, tcr):
    lines = ["cdr3s_nt\tcdr3s_aa"] + [f"{cell}\t{nt}\t{aa}" for cell, (nt, aa) in tcr.items()]
    _write_gz(path, "\n".join(lines) + "\n")


def _metadata_rows(cells, clusters):
    rows = []
    for i, (cell, cluster) in enumerate(zip(cells, clusters)):
        patient = cell.split(".")[1] if "." in cell else "su000"
        treatment = "post" if ".post" in cell else "pre"
        rows.append((cell, patient, treatment, cluster, float(i), float(-i)))
    return rows


def _write_fixture(folder, bcc_cells=BCC_CELLS, scc_cells=SCC_CELLS):
    """The six GEO files for the given cell ids; returns the hash registry that matches them."""
    bcc_counts_cells = [*bcc_cells, BCC_OTHER]
    _write_counts(folder / "GSE123813_bcc_scRNA_counts.txt.gz", bcc_counts_cells, BCC_COUNTS)
    _write_counts(folder / "GSE123813_scc_scRNA_counts.txt.gz", scc_cells, SCC_COUNTS)
    # Metadata rows in reverse order: the built object follows the counts' cell order.
    _write_metadata(folder / "GSE123813_bcc_tcell_metadata.txt.gz",
                    _metadata_rows(bcc_cells, ["CD8_mem", "Tregs", "Tfh", "Naive"])[::-1])
    _write_metadata(folder / "GSE123813_scc_metadata.txt.gz",
                    _metadata_rows(scc_cells, ["CD8_ex", "CD8_naive", "Treg", "Naive"])[::-1])
    rename_bcc = dict(zip(BCC_CELLS, bcc_cells))
    rename_scc = dict(zip(SCC_CELLS, scc_cells))
    _write_tcr(folder / "GSE123813_bcc_tcr.txt.gz",
               {rename_bcc.get(c, c): v for c, v in BCC_TCR.items()})
    _write_tcr(folder / "GSE123813_scc_tcr.txt.gz",
               {rename_scc.get(c, c): v for c, v in SCC_TCR.items()})
    return {
        name: "sha256:" + hashlib.sha256((folder / name).read_bytes()).hexdigest()
        for name in _yost._FILES
    }


def _offline(monkeypatch, registry):
    """Point the loader at the fixture files and make any download attempt fail at once."""
    monkeypatch.setattr(_yost, "_FILES", registry)
    monkeypatch.setattr(_yost, "_GEO", "http://127.0.0.1:9")
    monkeypatch.setattr(_yost, "_CHUNK_ROWS", 1)


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    folder = tmp_path_factory.mktemp("yost2019")
    registry = _write_fixture(folder)
    with pytest.MonkeyPatch.context() as mp:
        _offline(mp, registry)
        mdata = tcri.datasets.yost2019(cache_dir=folder)
    return folder, registry, mdata


@pytest.fixture(scope="module")
def mdata(cache):
    return cache[2]


def _clone(mdata, cell):
    return mdata.mod["airr"].obs.loc[cell, "clone_id"]


def test_modalities_hold_the_same_t_cells_in_the_same_order(mdata):
    gex, airr = mdata.mod["gex"], mdata.mod["airr"]
    assert list(gex.obs_names) == BCC_CELLS + SCC_CELLS
    assert list(airr.obs_names) == list(gex.obs_names)
    assert BCC_OTHER not in gex.obs_names


def test_counts_are_raw_integers_joined_on_the_union_of_genes(mdata):
    gex = mdata.mod["gex"]
    assert sparse.issparse(gex.X)
    assert set(gex.var_names) == {"G1", "G2", "G3", "G4"}
    X = pd.DataFrame(gex.X.toarray(), index=gex.obs_names, columns=gex.var_names)
    assert np.array_equal(X.to_numpy(), np.round(X.to_numpy()))
    assert (X.to_numpy() >= 0).all()
    assert np.array_equal(gex.layers["counts"].toarray(), gex.X.toarray())
    for gene, values in BCC_COUNTS.items():
        assert X.loc[BCC_CELLS, gene].tolist() == values[:4]
    for gene, values in SCC_COUNTS.items():
        assert X.loc[SCC_CELLS, gene].tolist() == values
    assert (X.loc[SCC_CELLS, "G1"] == 0).all()
    assert (X.loc[BCC_CELLS, "G4"] == 0).all()


def test_obs_and_umap_come_from_the_metadata(mdata):
    gex = mdata.mod["gex"]
    obs = gex.obs
    for column in ("patient", "treatment", "cluster", "lineage", "disease"):
        assert isinstance(obs[column].dtype, pd.CategoricalDtype)
    assert set(obs["disease"].cat.categories) == {"bcc", "scc"}
    assert (obs.loc[BCC_CELLS, "disease"] == "bcc").all()
    assert (obs.loc[SCC_CELLS, "disease"] == "scc").all()
    assert obs.loc["bcc.su001.post.tcell_c3", "cluster"] == "Tfh"
    # BCC's "Tregs" takes SCC's spelling.
    assert set(obs["cluster"].cat.categories) == {"CD8_mem", "Treg", "Tfh", "Naive", "CD8_ex",
                                                  "CD8_naive"}
    assert obs.loc["bcc.su001.pre.tcell_c2", "cluster"] == "Treg"
    # Lineage follows the cluster; "Naive" is CD8 in BCC and CD4 in SCC.
    assert obs.loc[BCC_CELLS, "lineage"].tolist() == ["CD8", "CD4", "CD4", "CD8"]
    assert obs.loc[SCC_CELLS, "lineage"].tolist() == ["CD8", "CD8", "CD4", "CD4"]
    assert obs.loc["scc.su010.post_s2", "treatment"] == "post"
    assert "UMAP1" not in obs and "UMAP2" not in obs
    umap = pd.DataFrame(gex.obsm["X_umap"], index=gex.obs_names)
    assert umap.loc["bcc.su001.post.tcell_c3"].tolist() == [2.0, -2.0]
    assert umap.loc["scc.su011.pre_s3"].tolist() == [2.0, -2.0]


def test_clones_are_defined_by_trb_cdr3_alone(mdata):
    # Same TRB amino acids (different nucleotides), different TRA: one clone.
    # Same TRA, different TRB: two.
    assert _clone(mdata, "bcc.su001.pre.tcell_c1") == _clone(mdata, "bcc.su001.pre.tcell_c2")
    assert _clone(mdata, "bcc.su001.pre.tcell_c1") != _clone(mdata, "bcc.su001.post.tcell_c3")
    # Identity of the CDR3 amino-acid sequence holds across cohorts too.
    assert _clone(mdata, "scc.su010.pre_s1") == _clone(mdata, "bcc.su001.pre.tcell_c1")
    # A multichain cell joins the clone of its primary TRB.
    assert mdata.mod["airr"].obs.loc["scc.su011.post_s4", "chain_pairing"] == "multichain"
    assert _clone(mdata, "scc.su011.post_s4") == _clone(mdata, "bcc.su001.pre.tcell_c1")
    # No TCR, or no TRB: no clone.
    assert pd.isna(_clone(mdata, "bcc.su002.post.tcell_c4"))
    assert pd.isna(_clone(mdata, "scc.su011.pre_s3"))


def test_every_chain_is_productive_and_chain_qc_ran(mdata):
    import scirpy as ir

    airr = mdata.mod["airr"]
    chains = ir.get.airr(airr, ["productive", "junction_aa", "locus"])
    productive = chains.filter(like="productive").to_numpy(dtype=object).ravel()
    present = [v for v in productive if not pd.isna(v)]
    assert present and all(bool(v) for v in present)
    assert chains.loc["bcc.su001.pre.tcell_c1", "VDJ_1_junction_aa"] == "CASSB1"
    assert chains.loc["bcc.su001.pre.tcell_c1", "VJ_1_junction_aa"] == "CAVA"
    assert {"receptor_type", "receptor_subtype", "chain_pairing"} <= set(airr.obs.columns)
    assert airr.obs.loc["bcc.su002.post.tcell_c4", "chain_pairing"] == "no IR"


def test_two_trb_chains_rank_by_nucleotide_sequence(mdata):
    import scirpy as ir

    chains = ir.get.airr(mdata.mod["airr"], "junction_aa", ["VDJ_1", "VDJ_2"])
    # GGT > GGA, so CASSB5 is the primary chain.
    assert chains.loc["scc.su010.post_s2", "VDJ_1_junction_aa"] == "CASSB5"
    assert chains.loc["scc.su010.post_s2", "VDJ_2_junction_aa"] == "CASSB4"


def test_a_second_call_reads_the_cache(cache, monkeypatch):
    folder, registry, first = cache
    _offline(monkeypatch, registry)
    assert (folder / f"yost2019-build{_yost._BUILD}.h5mu").exists()

    def no_rebuild(*args, **kwargs):
        raise AssertionError("rebuilt instead of reading the cache")

    monkeypatch.setattr(_yost, "_build", no_rebuild)
    again = tcri.datasets.yost2019(cache_dir=folder)
    assert list(again.mod["gex"].obs_names) == list(first.mod["gex"].obs_names)
    assert again.mod["airr"].obs["clone_id"].equals(first.mod["airr"].obs["clone_id"])
    assert np.array_equal(again.mod["gex"].X.toarray(), first.mod["gex"].X.toarray())


def test_ids_shared_across_cohorts_get_the_cohort_prefix(tmp_path, monkeypatch):
    shared = [f"cell_{i}" for i in range(4)]
    _offline(monkeypatch, _write_fixture(tmp_path, bcc_cells=shared, scc_cells=shared))
    mdata = tcri.datasets.yost2019(cache_dir=tmp_path)
    expected = [f"bcc_{c}" for c in shared] + [f"scc_{c}" for c in shared]
    assert list(mdata.mod["gex"].obs_names) == expected
    assert list(mdata.mod["airr"].obs_names) == expected
    assert _clone(mdata, "bcc_cell_0") == _clone(mdata, "scc_cell_0")


def test_from_mudata_returns_a_model_ready_anndata(mdata):
    from tcri.model._model import TCRIModel

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adata = tcri.pp.from_mudata(mdata, clonotype_key="clone_id", covariate_key="treatment")
    assert any("no clonotype" in str(w.message) for w in caught)
    with_clone = mdata.mod["airr"].obs["clone_id"].dropna().index
    assert set(adata.obs_names) == set(with_clone)
    assert {"clone_id", "treatment", "cluster", "patient", "disease"} <= set(adata.obs.columns)
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="cluster", covariate_key="treatment",
                            batch_key="patient")


@pytest.mark.parametrize("module", ["scirpy", "mudata", "pooch"])
def test_a_missing_dependency_raises_with_the_install_command(module, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match=r"pip install 'tcri\[datasets\]'"):
        tcri.datasets.yost2019(cache_dir=tmp_path)


@pytest.mark.network
def test_live_download_and_build(tmp_path):
    """Download the GEO files, build, and print the measured sizes."""
    mdata = tcri.datasets.yost2019(cache_dir=tmp_path)
    gex, airr = mdata.mod["gex"], mdata.mod["airr"]
    assert list(airr.obs_names) == list(gex.obs_names)
    assert set(gex.obs["disease"].cat.categories) == {"bcc", "scc"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = tcri.pp.from_mudata(mdata, clonotype_key="clone_id", covariate_key="treatment")
    clones = airr.obs["clone_id"]
    print()
    print("cells per cohort:", gex.obs["disease"].value_counts().to_dict())
    print("genes:", gex.n_vars)
    print("cells with a clone_id:", int(clones.notna().sum()), "clones:", clones.nunique())
    print("cells with a clone_id per cohort:",
          gex.obs.loc[clones.notna(), "disease"].value_counts().to_dict())
    print("from_mudata shape:", adata.shape)
