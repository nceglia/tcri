import pytest
import numpy as np
import matplotlib.pyplot as plt
from tcri.plotting._plotting import (
    polar_plot,
    probability_ternary,
    mutual_information,
    clonality
)

def test_polar_plot(mock_adata):
    """Test polar_plot function."""
    # Test with distribution statistic
    ax = polar_plot(
        mock_adata,
        phenotypes=['A', 'B', 'C'],
        statistic="distribution",
        splitby="timepoint"
    )
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with entropy statistic
    ax = polar_plot(
        mock_adata,
        phenotypes=['A', 'B', 'C'],
        statistic="entropy",
        splitby="timepoint"
    )
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with custom color dictionary
    color_dict = {'T1': 'red', 'T2': 'blue'}
    ax = polar_plot(
        mock_adata,
        phenotypes=['A', 'B', 'C'],
        statistic="distribution",
        splitby="timepoint",
        color_dict=color_dict
    )
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_probability_ternary(mock_adata):
    """Test probability_ternary function."""
    # Test basic functionality
    fig, ax = probability_ternary(
        mock_adata,
        phenotype_names=['A', 'B', 'C'],
        splitby="timepoint",
        conditions=['T1', 'T2']
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with top_n parameter
    fig, ax = probability_ternary(
        mock_adata,
        phenotype_names=['A', 'B', 'C'],
        splitby="timepoint",
        conditions=['T1', 'T2'],
        top_n=5
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_mutual_information_plot(mock_adata):
    """Test mutual_information plotting function."""
    # Test basic functionality
    ax = mutual_information(
        mock_adata,
        splitby="timepoint",
        temperature=1.0
    )
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with different temperature
    ax = mutual_information(
        mock_adata,
        splitby="timepoint",
        temperature=0.5
    )
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_clonality_plot(mock_adata):
    """Test clonality plotting function."""
    # Test basic functionality
    ax = clonality(
        mock_adata,
        groupby="timepoint"
    )
    assert isinstance(ax, plt.Axes)
    plt.close()
    
    # Test with splitby parameter
    ax = clonality(
        mock_adata,
        groupby="timepoint",
        splitby="patient"
    )
    assert isinstance(ax, plt.Axes)
    plt.close() 