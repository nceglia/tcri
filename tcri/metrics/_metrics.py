# Standard library imports
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc

# Third-party imports
from scipy.stats import entropy
from scipy.spatial import distance
import gseapy as gp

# Local imports
from ..preprocessing._preprocessing import joint_distribution

warnings.filterwarnings('ignore')

def dkl(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return entropy(p, q) 

def clonotypic_entropy(adata, covariate, phenotype, temperature=1.0):
    """Calculate clonotypic entropy for a given covariate and phenotype."""
    jd = joint_distribution(adata, covariate, temperature=temperature)
    if phenotype not in jd.columns:
        return 0.0
    res = jd[phenotype].to_numpy()
    res = res[res > 0]  # Remove zeros
    if len(res) == 0:
        return 0.0
    return entropy(res, base=2) / np.log2(len(res))

def phenotypic_entropy(adata, covariate, clonotype, temperature=1.0):
    """Calculate phenotypic entropy for a given covariate and clonotype."""
    jd = joint_distribution(adata, covariate, temperature=temperature)
    if clonotype not in jd.index:
        return 0.0
    res = jd.loc[clonotype].to_numpy()
    res = res[res > 0]  # Remove zeros
    if len(res) == 0:
        return 0.0
    return entropy(res, base=2) / np.log2(len(res))

def phenotypic_entropies(adata, covariate, base=2, normalized=True, temperature=1., n_samples=0):
    tcr_sequences = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    unique_tcrs = np.unique(tcr_sequences)
    tcr_to_entropy_dict = {
        tcr: phenotypic_entropy(
            adata, 
            covariate, 
            tcr, 
            base=base, 
            normalized=normalized, 
            temperature=temperature,
            n_samples=n_samples
        ) 
        for tcr in unique_tcrs
    }
    return tcr_to_entropy_dict

def clonotypic_entropies(adata, covariate, normalized=True, base=2, temperature=1., decimals=5, n_samples=0):
    unique_phenotypes = adata.uns["tcri_phenotype_categories"]
    phenotype_entropies = dict()
    for phenotype in unique_phenotypes:
        cent = clonotypic_entropy(adata, covariate, phenotype, temperature=temperature)
        phenotype_entropies[phenotype] = np.round(cent,decimals=decimals)
    return phenotype_entropies

def clonality(adata):
    phenotypes = adata.obs[adata.uns["tcri_phenotype_key"]].tolist()
    unique_phenotypes = np.unique(phenotypes)
    entropys = dict()
    df = adata.obs.copy()
    for phenotype in unique_phenotypes:
        pheno_df = df[df[adata.uns['tcri_phenotype_key']]==phenotype]
        clonotypes = pheno_df[adata.uns["tcri_clone_key"]].tolist()
        unique_clonotypes = np.unique(clonotypes)
        nums = []
        for tcr in unique_clonotypes:
            tcrs = pheno_df[pheno_df[adata.uns["tcri_clone_key"]]==tcr]
            nums.append(len(tcrs))
        clonality = 1 - entropy(np.array(nums),base=2) / np.log2(len(nums))
        entropys[phenotype] = np.nan_to_num(clonality)
    return entropys

def clone_fraction(adata, groupby):
    frequencies = dict()
    for group in set(adata.obs[groupby]):
        frequencies[group] = dict()
        sdata = adata[adata.obs[groupby] == group]
        total_cells = len(sdata.obs.index.tolist())
        clones = sdata.obs[sdata.uns["tcri_clone_key"]].tolist()
        for c in set(clones):
            frequencies[group][c] = clones.count(c) / total_cells
    return frequencies

def flux(adata, from_this, to_that, clones=None, temperature=1., distance_metric="l1", n_samples=0):
    distances = dict()
    jd_this = joint_distribution(adata,from_this,temperature=temperature,n_samples=n_samples,clones=clones)
    jd_that = joint_distribution(adata,to_that,temperature=temperature,n_samples=n_samples,clones=clones)
    common_indices = jd_this.index.intersection(jd_that.index)
    if distance_metric == "l1":
        distances = (jd_this.loc[common_indices] - jd_that.loc[common_indices]).abs().sum(axis=1)
    else:
        distances = pd.Series(
            {idx: dkl(jd_this.loc[idx], jd_that.loc[idx]) for idx in common_indices}
        )
    return distances

def mutual_information(adata, covariate, temperature=1.0, weighted=False):
    """Calculate mutual information between clonotypes and phenotypes for a given covariate."""
    jd = joint_distribution(adata, covariate, temperature=temperature)
    mi = 0.0
    total_weight = 0.0
    
    for clonotype in jd.index:
        p = jd.loc[clonotype].to_numpy()
        p = p[p > 0]  # Remove zeros
        if len(p) == 0:
            continue
            
        weight = 1.0
        if weighted:
            weight = np.sum(p)
            total_weight += weight
            
        mi += weight * entropy(p, base=2)
    
    if weighted and total_weight > 0:
        mi = mi / total_weight
        
    return mi / np.log2(len(jd.columns))