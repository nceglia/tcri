from scipy.stats import entropy
import numpy as np
import operator
from scipy.spatial import distance
import pandas as pd
import warnings
import gseapy as gp
import math
import pandas as pd
import torch
from scipy.stats import entropy
import numpy
import torch.nn.functional as F
import scanpy as sc

from ..preprocessing._preprocessing import joint_distribution


warnings.filterwarnings('ignore')


def dkl(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return entropy(p, q) 

def get_largest_clonotypes(adata, n=20):
    df = adata.obs[[adata.uns["tcri_clone_key"],"clone_size"]]
    return df.sort_values("clone_size",ascending=False)[adata.uns['tcri_clone_key']].unique().tolist()[:n]

def clonotypic_entropy(adata, covariate, phenotype, base=2, normalized=True, temperature=1., n_samples=0):
    logf = lambda x : np.log(x) / np.log(base)
    jd = joint_distribution(adata,covariate,temperature=temperature, n_samples=n_samples).T
    res = jd.loc[phenotype].to_numpy()
    cent = entropy(res,base=base)
    if normalized:
        cent = cent / logf(len(res))
    return cent

def phenotypic_entropy(adata, covariate, clonotype, base=2, normalized=True, temperature=1., n_samples=0):
    logf = lambda x : np.log(x) / np.log(base)
    jd = joint_distribution(adata,covariate, temperature=temperature, n_samples=n_samples)
    res = jd.loc[clonotype].to_numpy()
    pent = entropy(res,base=base)
    if normalized:
        pent = pent / logf(len(res))
    return pent

def phenotypic_entropies(adata, covariate, base=2, normalized=True, temperature=1., n_samples=0):
    tcr_sequences = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    unique_tcrs = np.unique(tcr_sequences)
    jd = joint_distribution(adata,covariate, temperature=temperature).to_numpy().T
    clonotype_entropies = np.zeros(jd.shape[1])
    max_entropy = np.log2(jd.shape[0])
    for i, clonotype_distribution in enumerate(jd.T):
        normalized_distribution = clonotype_distribution / np.sum(clonotype_distribution)
        epsilon = np.finfo(float).eps
        if normalized:
            clonotype_entropies[i] = -np.sum(normalized_distribution * np.log2(normalized_distribution + epsilon)) / max_entropy
        else:
            clonotype_entropies[i] = -np.sum(normalized_distribution * np.log2(normalized_distribution + epsilon))
    tcr_to_entropy_dict = dict(zip(unique_tcrs, clonotype_entropies))
    return tcr_to_entropy_dict

def clonotypic_entropies(adata, covariate, normalized=True, base=2, temperature=1., decimals=5, n_samples=0):
    unique_phenotypes = adata.uns["tcri_phenotype_categories"]
    phenotype_entropies = dict()
    for phenotype in unique_phenotypes:
        cent = clonotypic_entropy(adata, covariate, phenotype, base=base, normalized=normalized, temperature=temperature)
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
        clonality = 1 - entropy(numpy.array(nums),base=2) / numpy.log2(len(nums))
        entropys[phenotype] = numpy.nan_to_num(clonality)
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
    jd_this = joint_distribution(adata,from_this,temperature=temperature,n_samples=n_samples)
    jd_that = joint_distribution(adata,to_that,temperature=temperature,n_samples=n_samples)
    common_indices = jd_this.index.intersection(jd_that.index)
    if distance_metric == "l1":
        distances = (jd_this.loc[common_indices] - jd_that.loc[common_indices]).abs().sum(axis=1)
    else:
        distances = pd.Series(
            {idx: dkl(jd_this.loc[idx], jd_that.loc[idx]) for idx in common_indices}
        )
    return distances

def mutual_information(adata, covariate,temperature=1, n_samples=0, clones=None):
    pxy = joint_distribution(adata,covariate,temperature=temperature, n_samples=n_samples, clones=clones).to_numpy()
    pxy = pxy / pxy.sum()
    px = np.sum(pxy, axis=1)
    px = px / px.sum()
    py = np.sum(pxy, axis=0)
    py = py / py.sum()
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log2((pxy[nzs] / px_py[nzs])))
    return mi