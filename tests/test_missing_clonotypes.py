"""Every registered cell has a clonotype.

A cell without one belongs to no clone. Its clone code is -1, and a clone x covariate row built
from that code reads back through the category list as index -1, the last clone. So
``setup_anndata`` refuses such a cell before registering anything, the constructor refuses a
clonotype, phenotype or covariate code of -1 that reaches it after setup, and the substrate
readers refuse a stored negative code rather than masking it. With a clonotype on every cell,
the clone x covariate index is built from the categories alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from tcri._compute._repertoire import _missing_clonotypes
from tcri._compute._tables import clones_at, fit_clone_labels
from tcri._state import keys as K
from tcri.datasets import simulate_tcri
from tcri.model._model import TCRIModel

# scvi warns about categories with fewer than three cells, which every frame here has.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

KEYS = dict(clonotype_key="clone_id", phenotype_key="phenotype", covariate_key="covariate",
            batch_key="batch")
KNOBS = dict(n_latent=4, n_hidden=8, n_layers=1, classifier_n_layers=1, classifier_hidden=8,
             K=2)
MISSING = [np.nan, None, "", "   ", "nan"]
DTYPES = ["object", "category", "string", "str"]


def _adata(clones, dtype="object", covariates=("t0", "t1", "t0", "t1", "t1", "t0")):
    names = [f"cell_{i}" for i in range(len(clones))]
    obs = pd.DataFrame(
        {
            "clone_id": pd.Series(clones, index=names, dtype=dtype),
            "phenotype": pd.Categorical(["A", "B"] * (len(clones) // 2)),
            "covariate": pd.Categorical(list(covariates)),
            "batch": pd.Categorical(["b0"] * len(clones)),
        },
        index=names,
    )
    X = np.random.default_rng(0).poisson(2.0, size=(len(clones), 5)).astype(np.float32)
    return AnnData(X=X, obs=obs)


def test_missing_clonotype_definition():
    """NaN or None, an empty or whitespace-only string and the literal ``"nan"`` are missing in
    every dtype. Anything else is a clonotype, near misses included, and an unused category marks
    no cell."""
    values = ["c1", np.nan, None, "", " \t", "nan", "NaN", " nan", "None", "0"]
    expected = [False, True, True, True, True, True, False, False, False, False]
    for dtype in DTYPES:
        mask = _missing_clonotypes(pd.Series(values, dtype=dtype))
        assert mask.tolist() == expected, f"{dtype}: {mask.tolist()}"

    unused = pd.Series(pd.Categorical(["c1", "c2"], categories=["c1", "c2", "", "nan"]))
    assert not _missing_clonotypes(unused).any()
    assert _missing_clonotypes(pd.Series([1.0, np.nan])).tolist() == [False, True]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value", MISSING, ids=repr)
def test_setup_raises_on_missing_clonotype(value, dtype):
    """setup_anndata refuses cells without a clonotype before writing anything to the object,
    and the message names the column and the number of cells."""
    clones = ["c1", "c2", "c3", "c1", "c2", "c3"]
    clones[1] = clones[4] = value
    adata = _adata(clones, dtype)
    obs_columns, uns_keys = list(adata.obs.columns), set(adata.uns)

    with pytest.raises(ValueError,
                       match=r"2 of 6 cells have no clonotype in adata\.obs\['clone_id'\]"):
        TCRIModel.setup_anndata(adata, **KEYS)
    assert list(adata.obs.columns) == obs_columns, "setup wrote to obs before raising"
    assert set(adata.uns) == uns_keys, "setup wrote to uns before raising"


@pytest.mark.parametrize("permuted", [False, True], ids=["model", "null"])
@pytest.mark.parametrize(("column", "label", "axis"), [
    ("clone_id", "clonotype", "clonotype"),
    ("phenotype", "phenotype", "phenotype"),
    ("covariate", "covariate", "condition"),
])
def test_init_raises_on_a_label_removed_after_setup(column, label, axis, permuted):
    """A clonotype, phenotype or covariate removed from ``obs`` after setup gives that cell code
    -1, which reads back as the column's last category. The constructor refuses it before
    anything is derived from the codes, for a model and for a null of that axis, which is built
    on the parent's setup without running it again."""
    adata = _adata(["c1", "c2", "c3", "c1", "c2", "c3"])
    TCRIModel.setup_anndata(adata, **KEYS)
    values = adata.obs[column].astype(object).to_numpy().copy()
    values[2] = np.nan
    adata.obs[column] = pd.Series(values, index=adata.obs_names, dtype=object)
    permutation = (axis, np.random.default_rng(0).permutation(adata.n_obs)) if permuted else None

    with pytest.raises(ValueError,
                       match=rf"1 of 6 cells have no {label} in adata\.obs\['{column}'\]"):
        TCRIModel(adata, permutation=permutation, **KNOBS)


def _substrate(ct_to_c, fit):
    """Three clones over two covariate levels, one cell in each clone x covariate row."""
    adata = AnnData(X=np.zeros((4, 1), dtype=np.float32))
    adata.uns[K.METADATA] = {K.Config.CLONE_COL: "clone_id"}
    adata.uns[K.CLONOTYPE_CATEGORIES] = ["c1", "c2", "c3"]
    adata.uns[K.COVARIATE_CATEGORIES] = ["t0", "t1"]
    adata.uns[K.fit_key(K.CT_TO_C, fit)] = np.asarray(ct_to_c)
    adata.uns[K.fit_key(K.CT_TO_COV, fit)] = np.array([0, 1, 0, 1])
    adata.uns[K.fit_key(K.CT_ARRAY, fit)] = np.arange(4)
    return adata


@pytest.mark.parametrize("fit", [None, "null.clonotype"])
def test_substrate_readers_raise_on_a_negative_clone_code(fit):
    """fit_clone_labels and clones_at read clone ids through the category list, and refuse a
    substrate that stores a negative clone code instead of masking it or reading it as the last
    clone."""
    good = _substrate([0, 0, 1, 2], fit)
    assert fit_clone_labels(good, fit).tolist() == ["c1", "c1", "c2", "c3"]
    assert clones_at(good, "t0", fit=fit) == ["c1", "c2"]
    assert clones_at(good, "t1", fit=fit) == ["c1", "c3"]

    bad = _substrate([-1, 0, 1, 2], fit)
    with pytest.raises(ValueError, match="negative clone code.*Redo the fit"):
        fit_clone_labels(bad, fit)
    for level in ("t0", "t1"):
        with pytest.raises(ValueError, match="negative clone code.*Redo the fit"):
            clones_at(bad, level, fit=fit)


def _declared_order():
    """Clone categories declared out of label order, with one unused."""
    adata = _adata(["c2", "c1", "c3", "c1", "c2", "c3"])
    adata.obs["clone_id"] = pd.Categorical(adata.obs["clone_id"],
                                           categories=["c3", "c1", "unused", "c2"])
    return adata


def _simulated():
    return simulate_tcri(n_clones=8, n_phenotypes=3, n_genes=20, n_cells=200, n_covariates=2,
                         omega_concentration=0.4, seed=0)


@pytest.mark.parametrize("make", [_declared_order, _simulated], ids=["declared-order", "simulated"])
def test_datasets_without_missing_clonotypes_are_unchanged(make):
    """With a clonotype on every cell, the rows are the sorted set of observed (clone,
    covariate) pairs, each coded by its position in the column's categories: ``n_ct``,
    ``ct_to_c``, ``ct_to_cov`` and the per-cell row all follow from the categories alone."""
    adata = make()
    TCRIModel.setup_anndata(adata, **KEYS)
    module = TCRIModel(adata, **KNOBS).module

    clones, covariates = adata.obs["clone_id"], adata.obs["covariate"]
    clone_cats = list(clones.astype("category").cat.categories)
    cov_cats = list(covariates.astype("category").cat.categories)
    cells = [(clone_cats.index(c), cov_cats.index(t)) for c, t in zip(clones, covariates)]
    rows = sorted(set(cells))

    assert module.ct_count == len(rows)
    np.testing.assert_array_equal(module.ct_to_c.numpy(), [c for c, _ in rows])
    np.testing.assert_array_equal(module.ct_to_cov.numpy(), [t for _, t in rows])
    np.testing.assert_array_equal(module.ct_array.numpy(), [rows.index(cell) for cell in cells])
