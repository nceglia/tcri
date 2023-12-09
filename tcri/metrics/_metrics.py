from scipy.stats import entropy
import numpy as np
import operator
from scipy.spatial import distance
import pandas as pd
import warnings
import gseapy as gp
import pandas as pd

from scipy.stats import entropy
import numpy

from ..preprocessing._preprocessing import clone_size, joint_distribution

warnings.filterwarnings('ignore')

def phenotypic_entropy(adata, method="probabilistic"):
    joint_distribution(adata,method=method)
    jd = adata.uns["joint_distribution"].to_numpy() 
    jd = jd / np.sum(jd)
    marginal_pheno_probs = np.sum(jd, axis=1)  # Sum over phenotypes
    epsilon = np.finfo(float).eps  # Smallest positive float number
    pheno_entropy = -np.sum(marginal_pheno_probs * np.log(marginal_pheno_probs + epsilon))
    return pheno_entropy

def clonotypic_entropy(adata, method="probabilistic"):
    joint_distribution(adata, method=method)
    jd = adata.uns["joint_distribution"].to_numpy().T
    jd = jd / np.sum(jd)
    marginal_tcr_probs = np.sum(jd, axis=1)  # Sum over phenotypes
    epsilon = np.finfo(float).eps  # Smallest positive float number
    tcr_entropy = -np.sum(marginal_tcr_probs * np.log(marginal_tcr_probs + epsilon))
    return tcr_entropy

def phenotypic_entropies(adata, method="probabilistic", normalized=True):
    joint_distribution(adata, method=method)
    tcr_sequences = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    unique_tcrs = np.unique(tcr_sequences)
    jd = adata.uns["joint_distribution"].to_numpy() 
    jd = jd / np.sum(jd)
    clonotype_entropies = np.zeros(jd.shape[1])
    max_entropy = np.log(jd.shape[0])
    for i, clonotype_distribution in enumerate(jd):
        normalized_distribution = clonotype_distribution / np.sum(clonotype_distribution)
        epsilon = np.finfo(float).eps
        if normalized:
            clonotype_entropies[i] = -np.sum(normalized_distribution * np.log(normalized_distribution + epsilon)) / max_entropy
        else:
            clonotype_entropies[i] = -np.sum(normalized_distribution * np.log(normalized_distribution + epsilon))
    tcr_to_entropy_dict = dict(zip(unique_tcrs, clonotype_entropies))
    return tcr_to_entropy_dict

def clonotypic_entropies(adata, method="probabilistic", normalized=True):
    joint_distribution(adata, method=method)
    unique_phenotypes = adata.uns["tcri_unique_phenotypes"]
    jd = adata.uns["joint_distribution"].to_numpy()
    phenotype_entropies = np.zeros(jd.shape[0])
    max_entropy = np.log(jd.shape[1])
    for i, phenotype_distribution in enumerate(jd):
        normalized_distribution = phenotype_distribution / np.sum(phenotype_distribution)
        epsilon = np.finfo(float).eps
        if normalized:
            phenotype_entropies[i] = -np.sum(normalized_distribution * np.log(normalized_distribution + epsilon)) / max_entropy
        else:
            phenotype_entropies[i] = -np.sum(normalized_distribution * np.log(normalized_distribution + epsilon))
        phenotype_entropies[i] = np.round(phenotype_entropies[i],decimals=5)
    phenotype_to_entropy_dict = dict(zip(unique_phenotypes, phenotype_entropies))
    return phenotype_to_entropy_dict

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


def marginal_phenotypic(adata, clones=None, probability=False):
    if clones == None:
        clones = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    dist = adata.uns["joint_distribution"][clones].to_numpy()
    dist = dist.sum(axis=1)
    if probability:
        dist /= dist.sum()
    return np.nan_to_num(dist).T

def marginal_phenotypic(adata, clones=None,method="probabilistic"):
    joint_distribution(adata,method=method)
    if clones == None:
        clones = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    dist = adata.uns["joint_distribution"][clones].to_numpy()
    dist = dist.sum(axis=1)
    dist /= dist.sum()
    return np.nan_to_num(dist).T

def flux(adata, key, from_this, to_that, clones=None, method="probabilistic", distance_metric="l1"):
    this = adata[adata.obs[key] == from_this]
    that = adata[adata.obs[key] == to_that]
    this_clones = set(this.obs[this.uns["tcri_clone_key"]])
    that_clones = set(that.obs[this.uns["tcri_clone_key"]])
    clones = list(this_clones.intersection(that_clones))
    distances = dict()
    for clone in clones:
        this_distribution = marginal_phenotypic(this,clones=[clone])
        that_distribution = marginal_phenotypic(that,clones=[clone])
        if distance_metric == "l1":
            dist = distance.cityblock(this_distribution, that_distribution)
        elif distance_metric == "dkl":
            dist = entropy(this_distribution, that_distribution, base=2, axis=0)
        distances[clone] = dist
    return distances 

def mutual_information(adata):
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
