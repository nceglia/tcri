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

def clonotypic_entropy(adata, covariate, phenotype, base=2, normalized=True, temperature=1., clones=None, n_samples=0):
    logf = lambda x : np.log(x) / np.log(base)
    jd = joint_distribution(adata,covariate,temperature=temperature, n_samples=n_samples,clones=clones).T
    res = jd.loc[phenotype].to_numpy()
    cent = entropy(res,base=base)
    if normalized:
        cent = cent / logf(len(res))
    return cent

def phenotypic_entropy(adata, covariate, clonotype, base=2, normalized=True, temperature=1., n_samples=0, clones=None):
    logf = lambda x : np.log(x) / np.log(base)
    jd = joint_distribution(adata,covariate, temperature=temperature, n_samples=n_samples, clones=clones).T
    res = jd.loc[clonotype].to_numpy()
    pent = entropy(res,base=base)
    if normalized:
        pent = pent / logf(len(res))
    return pent

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


def mutual_information(
    adata, 
    covariate,
    temperature=1,
    n_samples=0,
    clones=None,
    weighted=False
):
    """
    Compute mutual information from the joint distribution returned by joint_distribution.
    If weighted=True, large clones contribute more to MI.
    """
    # Retrieve the distribution as a DataFrame
    pxy_df = joint_distribution(
        adata=adata,
        covariate_label=covariate,
        temperature=temperature,
        n_samples=n_samples,
        clones=clones,
        weighted=weighted
    )
    # Convert to numpy
    pxy = pxy_df.to_numpy()
    # Ensure total sums to 1
    total = pxy.sum()
    if total < 1e-15:
        return 0.0
    pxy /= total

    # Marginals
    px = pxy.sum(axis=1, keepdims=True)  # sum over phenotypes
    py = pxy.sum(axis=0, keepdims=True)  # sum over clonotypes

    px_py = px @ py
    # Avoid zeros
    eps = 1e-15
    mask = (pxy > eps)
    mi = np.sum(pxy[mask] * np.log2((pxy[mask] + eps) / (px_py[mask] + eps)))
    return mi