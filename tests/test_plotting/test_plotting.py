import matplotlib.pyplot as plt

from tcri.plotting._plotting import clonality


def test_clonality_plot(mock_adata):
    """Test clonality plotting function."""
    # Test basic functionality
    ax = clonality(mock_adata, groupby="timepoint")
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test with splitby parameter
    ax = clonality(mock_adata, groupby="timepoint", splitby="patient")
    assert isinstance(ax, plt.Axes)
    plt.close()
