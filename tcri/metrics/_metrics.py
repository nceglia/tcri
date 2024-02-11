from scipy.stats import entropy
import numpy as np
import operator
from scipy.spatial import distance
import pandas as pd
import warnings
import gseapy as gp
import math
import pandas as pd

from scipy.stats import entropy
import numpy

from ..preprocessing._preprocessing import clone_size, joint_distribution

warnings.filterwarnings('ignore')

def clonotypic_entropy(adata, phenotype, method="probabilistic", base=2, normalized=False):
    logf = lambda x : np.log(x) / np.log(base)
    joint_distribution(adata, method=method)
    jd = adata.uns['joint_distribution']
    res = jd.loc[phenotype].to_numpy()
    cent = entropy(res,base=base)
    if normalized:
        cent = cent / logf(len(res))
    return cent

def phenotypic_entropy(adata, clonotype, base=2, normalized=False, method="probabilistic"):
    logf = lambda x : np.log(x) / np.log(base)
    joint_distribution(adata, method=method)
    jd = adata.uns['joint_distribution'].T
    res = jd.loc[clonotype].to_numpy()
    pent = entropy(res,base=base)
    if normalized:
        pent = pent / logf(len(res))
    return pent

def get_largest_clonotypes(adata, n=20):
    df = adata.obs[[adata.uns["tcri_clone_key"],"clone_size"]]
    return df.sort_values("clone_size",ascending=False)[adata.uns['tcri_clone_key']].unique().tolist()[:n]

def probability_distribution(adata, method="probabilistic"):
    joint_distribution(adata,method=method)
    if method == "probabilistic":
        joint_distribution(adata) 
        total = adata.uns["joint_distribution"].to_numpy().sum()
        pdist = adata.uns["joint_distribution"].sum(axis=1) / total
    else:
        raise NotImplementedError("In progress!")
    return pdist

def phenotypic_entropies(adata, method="probabilistic", normalized=True, decimals=5):
    joint_distribution(adata, method=method)
    tcr_sequences = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    unique_tcrs = np.unique(tcr_sequences)
    jd = adata.uns["joint_distribution"].to_numpy() 
    jd = jd / np.sum(jd)
    clonotype_entropies = np.zeros(jd.shape[1])
    max_entropy = np.log2(jd.shape[0])
    for i, clonotype_distribution in enumerate(jd.T):
        normalized_distribution = clonotype_distribution / np.sum(clonotype_distribution)
        epsilon = np.finfo(float).eps
        if normalized:
            clonotype_entropies[i] = -np.sum(normalized_distribution * np.log2(normalized_distribution + epsilon)) / max_entropy
        else:
            clonotype_entropies[i] = -np.sum(normalized_distribution * np.log2(normalized_distribution + epsilon))
    clonotype_entropies = [abs(round(x,decimals)) for x in clonotype_entropies]
    tcr_to_entropy_dict = dict(zip(unique_tcrs, clonotype_entropies))
    return tcr_to_entropy_dict

def clonotypic_entropies(adata, method="probabilistic", normalized=True, base=2, decimals=5):
    unique_phenotypes = adata.uns["tcri_unique_phenotypes"]
    phenotype_entropies = dict()
    for phenotype in unique_phenotypes:
        cent = clonotypic_entropy(adata, phenotype, base=base, method=method, normalized=normalized)
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

def marginal_phenotypic(adata, clones=None, probability=False, method="probabilistic"):
    joint_distribution(adata,method=method)
    if clones == None:
        clones = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    dist = adata.uns["joint_distribution"][clones].to_numpy()
    dist = dist.sum(axis=1)
    if probability:
        dist /= dist.sum()
    return np.nan_to_num(dist).T

def flux(adata, key, from_this, to_that, clones=None, method="probabilistic", distance_metric="l1"):
    this = adata[adata.obs[key] == from_this]
    that = adata[adata.obs[key] == to_that]
    this_clones = set(this.obs[this.uns["tcri_clone_key"]])
    that_clones = set(that.obs[this.uns["tcri_clone_key"]])
    clones = list(this_clones.intersection(that_clones))
    distances = dict()
    joint_distribution(this, method=method)
    joint_distribution(that, method=method)
    for clone in clones:
        this_distribution = marginal_phenotypic(this,clones=[clone], probability=True)
        that_distribution = marginal_phenotypic(that,clones=[clone], probability=True)
        if distance_metric == "l1":
            dist = distance.cityblock(this_distribution, that_distribution)
        elif distance_metric == "dkl":
            dist = entropy(this_distribution, that_distribution, base=2, axis=0)
        distances[clone] = dist
    return distances

def mutual_information(adata, method="probabilistic"):
    joint_distribution(adata,method=method)
    pxy = adata.uns['joint_distribution'].to_numpy()
    pxy = pxy / pxy.sum()
    px = np.sum(pxy, axis=1)
    px = px / px.sum()
    py = np.sum(pxy, axis=0)
    py = py / py.sum()
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log2((pxy[nzs] / px_py[nzs])))
    return mi

