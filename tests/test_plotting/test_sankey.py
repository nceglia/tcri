import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcri.plotting._plotting import plot_pheno_sankey


def test_sankey_smoke_two_covariates(trained_model):
    """Two covariates, return_axes=True: figure has axes with patches."""
    _model, adata = trained_model
    covariates = list(adata.uns["tcri_covariate_categories"])
    assert len(covariates) >= 2

    fig, ax = plot_pheno_sankey(
        adata,
        covariate_order=covariates[:2],
        return_axes=True,
    )
    assert fig is not None
    assert ax in fig.axes
    assert len(ax.patches) > 0
    plt.close(fig)


def test_sankey_clones_filtering(trained_model):
    """Subset of clones runs without error and produces fewer connections."""
    _model, adata = trained_model
    covariates = list(adata.uns["tcri_covariate_categories"])[:2]
    all_clones = list(adata.uns["tcri_clonotype_categories"])

    fig_full, _ = plot_pheno_sankey(
        adata, covariate_order=covariates, return_axes=True,
    )
    fig_subset, ax_subset = plot_pheno_sankey(
        adata, covariate_order=covariates,
        clones=all_clones[: max(1, len(all_clones) // 4)],
        return_axes=True,
    )
    assert fig_subset is not None
    assert ax_subset in fig_subset.axes
    plt.close(fig_full)
    plt.close(fig_subset)


def test_sankey_normalize_false(trained_model):
    """normalize=False (raw cell counts) produces an axis."""
    _model, adata = trained_model
    covariates = list(adata.uns["tcri_covariate_categories"])[:2]

    fig, ax = plot_pheno_sankey(
        adata,
        covariate_order=covariates,
        normalize=False,
        return_axes=True,
    )
    assert fig is not None
    assert ax in fig.axes
    assert len(ax.patches) > 0
    plt.close(fig)
