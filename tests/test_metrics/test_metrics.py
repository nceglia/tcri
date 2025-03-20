import pytest
import numpy as np
from tcri.metrics._metrics import (
    clonotypic_entropy,
    phenotypic_entropy,
    mutual_information,
    clonality
)

def test_clonotypic_entropy(mock_adata, mock_joint_distribution):
    """Test clonotypic_entropy function."""
    # Test with mock data
    entropy_val = clonotypic_entropy(
        mock_adata,
        covariate="T1",
        phenotype="A",
        temperature=1.0
    )
    
    # Check that entropy is between 0 and 1 (normalized)
    assert 0 <= entropy_val <= 1
    
    # Test with different temperature
    entropy_val_temp = clonotypic_entropy(
        mock_adata,
        covariate="T1",
        phenotype="A",
        temperature=0.5
    )
    assert entropy_val_temp != entropy_val  # Should be different with different temperature

def test_phenotypic_entropy(mock_adata, mock_joint_distribution):
    """Test phenotypic_entropy function."""
    # Test with mock data
    entropy_val = phenotypic_entropy(
        mock_adata,
        covariate="T1",
        clonotype="clone_0",
        temperature=1.0
    )
    
    # Check that entropy is between 0 and 1 (normalized)
    assert 0 <= entropy_val <= 1
    
    # Test with different temperature
    entropy_val_temp = phenotypic_entropy(
        mock_adata,
        covariate="T1",
        clonotype="clone_0",
        temperature=0.5
    )
    assert entropy_val_temp != entropy_val  # Should be different with different temperature

def test_mutual_information(mock_adata, mock_joint_distribution):
    """Test mutual_information function."""
    # Test with mock data
    mi_val = mutual_information(
        mock_adata,
        covariate="T1",
        temperature=1.0
    )
    
    # Check that MI is non-negative
    assert mi_val >= 0
    
    # Test with different temperature
    mi_val_temp = mutual_information(
        mock_adata,
        covariate="T1",
        temperature=0.5
    )
    assert mi_val_temp != mi_val  # Should be different with different temperature
    
    # Test with weighted=True
    mi_val_weighted = mutual_information(
        mock_adata,
        covariate="T1",
        temperature=1.0,
        weighted=True
    )
    assert mi_val_weighted != mi_val  # Should be different when weighted

def test_clonality(mock_adata):
    """Test clonality function."""
    # Test with mock data
    clonality_dict = clonality(mock_adata)
    
    # Check that clonality values are between 0 and 1
    for val in clonality_dict.values():
        assert 0 <= val <= 1
    
    # Check that we have clonality values for all phenotypes
    assert set(clonality_dict.keys()) == set(mock_adata.uns["tcri_phenotype_categories"]) 