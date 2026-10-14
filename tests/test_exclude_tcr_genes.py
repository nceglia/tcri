"""Leaving TCR segment genes out: which genes match, the keep list, the two return modes, gene
symbols in a var column, and the warning at setup.

A gene matches when its symbol is ``TR``, a locus letter, a segment letter and then a digit or
``P``, ignoring case. The default ``keep`` holds the invariant segments of MAIT, iNKT and
Vγ9Vδ2 cells, which mark a phenotype rather than a clone.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import tcri
from tcri.model._model import TCRIModel
from tcri._compute._repertoire import TCRIDataWarning

HUMAN_SEGMENTS = ["TRAV12-2", "TRAV14DV4", "TRAJ42", "TRBV20-1", "TRBD1", "TRBJ2-7", "TRGV4",
                  "TRGJ1", "TRGJP", "TRGJP1", "TRGJP2", "TRDV1", "TRDD3", "TRDJ1"]
MOUSE_SEGMENTS = ["Trav6-7-dv9", "Traj12", "Trbv13-2", "Trbd1", "Trbj2-7", "Trgv2", "Trgj1",
                  "Trdv4", "Trdj1"]
NOT_SEGMENTS = ["TRADD", "TRAC", "TRBC1", "TRBC2", "TRGC1", "TRGC2", "TRDC", "Trac", "Trbc2",
                "TRAF1", "TRAT1", "CD3E", "GZMB"]
DEFAULT_KEEP = ["TRAV1-2", "TRAJ33", "TRAV10", "TRAJ18", "TRGV9", "TRDV2", "Trav1", "Trav11",
                "Traj33", "Traj18", "trav1-2"]


def _adata(genes, **var):
    """One AnnData of counts over ``genes``, with any extra ``var`` columns."""
    rng = np.random.default_rng(0)
    X = rng.poisson(5.0, size=(4, len(genes))).astype("float32")
    return AnnData(X=X, var=pd.DataFrame(var, index=pd.Index(genes)))


def test_human_and_mouse_segment_genes_match_and_others_do_not():
    """Every human and mouse V, D and J gene is excluded, the TRGJP joining genes among them;
    TRADD, the constant genes and other TR-prefixed genes stay."""
    genes = HUMAN_SEGMENTS + MOUSE_SEGMENTS + NOT_SEGMENTS
    mask = tcri.pp.exclude_tcr_genes(_adata(genes), inplace=False)
    assert dict(zip(genes, mask)) == {
        **{g: False for g in HUMAN_SEGMENTS + MOUSE_SEGMENTS}, **{g: True for g in NOT_SEGMENTS}
    }


def test_default_keep_stays_in_either_case():
    """The invariant segments stay whatever their case, so the mouse Traj33 and Traj18 are kept
    through the human entries."""
    genes = DEFAULT_KEEP + ["TRBV20-1", "Trbv13-2"]
    mask = tcri.pp.exclude_tcr_genes(_adata(genes), inplace=False)
    assert mask.tolist() == [True] * len(DEFAULT_KEEP) + [False, False]


def test_custom_keep_replaces_the_default():
    """A keep passed in replaces the default, compared ignoring case; () keeps no segment."""
    genes = ["TRAV1-2", "TRBV20-1", "TRBJ2-7", "CD3E"]
    mask = tcri.pp.exclude_tcr_genes(_adata(genes), keep=("trbv20-1",), inplace=False)
    assert mask.tolist() == [False, True, False, True]
    mask = tcri.pp.exclude_tcr_genes(_adata(genes), keep=(), inplace=False)
    assert mask.tolist() == [False, False, False, True]


def test_keep_refuses_a_single_string():
    with pytest.raises(TypeError, match="not one string"):
        tcri.pp.exclude_tcr_genes(_adata(["TRBV20-1"]), keep="TRBV20-1")


def test_inplace_removes_the_segment_genes():
    """In place, the excluded genes are removed, the others keep their order and values, and
    nothing is returned."""
    genes = ["CD3E", "TRBV20-1", "TRAV1-2", "TRAJ42", "GZMB"]
    adata = _adata(genes)
    before = adata.X.copy()
    assert tcri.pp.exclude_tcr_genes(adata) is None
    assert adata.var_names.tolist() == ["CD3E", "TRAV1-2", "GZMB"]
    np.testing.assert_array_equal(adata.X, before[:, [0, 2, 4]])


def test_not_inplace_returns_the_mask_and_leaves_adata_unchanged():
    genes = ["CD3E", "TRBV20-1", "TRAV1-2", "TRAJ42", "GZMB"]
    adata = _adata(genes)
    mask = tcri.pp.exclude_tcr_genes(adata, inplace=False)
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert mask.tolist() == [True, False, True, False, True]
    assert adata.var_names.tolist() == genes
    assert adata[:, mask].var_names.tolist() == ["CD3E", "TRAV1-2", "GZMB"]


def test_inplace_on_a_view():
    """A view is made an actual object and subset, as scanpy's own gene filters do."""
    view = _adata(["CD3E", "TRBV20-1", "GZMB"])[:2]
    tcri.pp.exclude_tcr_genes(view)
    assert view.var_names.tolist() == ["CD3E", "GZMB"] and view.n_obs == 2


def test_ensembl_var_names_without_gene_symbols_raise():
    """Ensembl IDs carry no symbol to match, so the call raises rather than excluding nothing,
    and nothing is removed."""
    ids = ["ENSG00000211751", "ENSG00000198851.10", "ENSG00000211772", "CD3E"]
    adata = _adata(ids)
    with pytest.raises(ValueError, match="gene_symbols"):
        tcri.pp.exclude_tcr_genes(adata)
    assert adata.var_names.tolist() == ids


def test_gene_symbols_column_is_matched():
    """With gene_symbols, the column is matched and var_names are not."""
    ids = ["ENSG00000211751", "ENSG00000198851", "ENSG00000211772", "ENSMUSG00000076498"]
    adata = _adata(ids, symbol=pd.Categorical(["TRBC1", "CD3E", "TRBV20-1", "Trbv13-2"]))
    mask = tcri.pp.exclude_tcr_genes(adata, gene_symbols="symbol", inplace=False)
    assert mask.tolist() == [True, True, False, False]
    tcri.pp.exclude_tcr_genes(adata, gene_symbols="symbol")
    assert adata.var_names.tolist() == ["ENSG00000211751", "ENSG00000198851"]


def test_unknown_gene_symbols_column_raises():
    with pytest.raises(KeyError, match="symbol"):
        tcri.pp.exclude_tcr_genes(_adata(["CD3E"]), gene_symbols="symbol")


def test_logs_one_line_naming_the_excluded_genes(caplog):
    adata = _adata(["CD3E", "TRBV20-1", "TRAV1-2", "TRAJ42"])
    with caplog.at_level(logging.INFO, logger="tcri"):
        tcri.pp.exclude_tcr_genes(adata)
        tcri.pp.exclude_tcr_genes(adata)
    lines = [r.getMessage() for r in caplog.records if r.name.startswith("tcri")]
    assert lines == ["exclude_tcr_genes: excluded 2 of 4 genes: TRBV20-1, TRAJ42.",
                     "exclude_tcr_genes: excluded 0 of 2 genes."]


def _labeled(genes):
    """An unregistered AnnData over ``genes`` with two clones, two phenotypes and two
    conditions."""
    adata = _adata(genes)
    adata = AnnData(X=np.vstack([adata.X] * 3), var=adata.var)
    adata.obs = pd.DataFrame({"clone_id": ["c1", "c2"] * 6, "phenotype": ["A", "B"] * 6,
                              "covariate": ["pre"] * 6 + ["post"] * 6, "patient": "P1"},
                             index=[f"cell_{i}" for i in range(12)])
    adata.layers["counts"] = adata.X.copy()
    return adata


def _segment_warnings(adata):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                                phenotype_key="phenotype", covariate_key="covariate",
                                batch_key="patient")
    return [w for w in record if "segment genes" in str(w.message)]


def test_setup_warns_on_segment_genes_and_not_after_exclusion():
    """Setup names a few of the segment genes outside the default keep and points to
    exclude_tcr_genes; after it, setup does not warn."""
    adata = _labeled(["CD3E", "TRAV1-2", "TRBV20-1", "TRBJ2-7", "TRAJ42", "TRGV4", "GZMB"])
    found = _segment_warnings(adata)
    assert len(found) == 1 and issubclass(found[0].category, TCRIDataWarning)
    message = str(found[0].message)
    assert message.startswith("4 of the 7 registered genes are TCR V, D or J segment genes "
                              "(TRBV20-1, TRBJ2-7, TRAJ42, ...)")
    assert "tcri.pp.exclude_tcr_genes" in message and "TRAV1-2" not in message

    tcri.pp.exclude_tcr_genes(adata)
    assert _segment_warnings(adata) == []
