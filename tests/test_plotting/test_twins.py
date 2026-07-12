"""The four pl↔tl twins (cache renderers) + resolve_palette (PR7). Smoke: each runs and
returns a Matplotlib axes (or the tidy df via return_df)."""
import matplotlib

matplotlib.use("Agg")

import tcri
from tcri import _keys as K


def _cov(adata):
    return list(adata.uns[K.COVARIATE_CATEGORIES])[0]


def test_pl_mutual_information(trained_model):
    _, adata = trained_model
    ax = tcri.pl.mutual_information(adata, covariate=_cov(adata))
    assert ax is not None
    df = tcri.pl.mutual_information(adata, covariate=_cov(adata), return_df=True)
    assert "MI" in df.columns


def test_pl_entropy_twins(trained_model):
    _, adata = trained_model
    cov = _cov(adata)
    assert tcri.pl.clonotypic_entropy(adata, covariate=cov, groupby="patient") is not None
    assert tcri.pl.phenotypic_entropy(adata, covariate=cov, groupby="patient") is not None
    # no-groupby falls back to a bar of the Series
    assert tcri.pl.clonotypic_entropy(adata, covariate=cov) is not None


def test_pl_phenotypic_flux(trained_model):
    _, adata = trained_model
    covs = list(adata.uns[K.COVARIATE_CATEGORIES])
    ax = tcri.pl.phenotypic_flux(adata, order=covs)
    assert ax is not None
    df = tcri.pl.phenotypic_flux(adata, order=covs, return_axes=True)
    assert df is not None


def test_resolve_palette_in_place(trained_model):
    _, adata = trained_model
    mapping = tcri.pl.resolve_palette(adata, "patient")
    assert "patient_colors" in adata.uns
    assert isinstance(mapping, dict) and "patient" in mapping
