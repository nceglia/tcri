from scipy.stats import entropy
import numpy as np
import operator
from scipy.spatial import distance
import pandas as pd
import warnings
import gseapy as gp
import pandas as pd

from ..preprocessing._preprocessing import clone_size, joint_distribution

warnings.filterwarnings('ignore')


def clonotypic_entropy(adata, method="probabilistic"):
    phenotypic_joint_distribution(adata)
    jd = adata.uns["joint_distribution"].to_numpy() 
    jd = jd / np.sum(jd)
    marginal_pheno_probs = np.sum(jd, axis=1)  # Sum over phenotypes
    epsilon = np.finfo(float).eps  # Smallest positive float number
    pheno_entropy = -np.sum(marginal_pheno_probs * np.log(marginal_pheno_probs + epsilon))
    return pheno_entropy

def phenotypic_entropy(adata, method="probabilistic"):
    joint_distribution(adata, method=method)
    jd = adata.uns["joint_distribution"].to_numpy().T
    jd = jd / np.sum(jd)
    marginal_tcr_probs = np.sum(jd, axis=1)  # Sum over phenotypes
    epsilon = np.finfo(float).eps  # Smallest positive float number
    tcr_entropy = -np.sum(marginal_tcr_probs * np.log(marginal_tcr_probs + epsilon))
    return tcr_entropy

def clonotypic_entropies(adata, method="probabilistic", normalized=True):
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

def phenotypic_entropies(adata, method="probabilistic", normalized=True):
    joint_distribution(adata, method=method)
    phenotypes = adata.obs[adata.uns["tcri_phenotype_key"]].tolist()
    unique_phenotypes = np.unique(phenotypes)
    jd = adata.uns["joint_distribution"].to_numpy()
    jd = jd / np.sum(jd)
    phenotype_entropies = np.zeros(jd.shape[0])
    max_entropy = np.log(jd.shape[1])
    for i, phenotype_distribution in enumerate(jd):
        normalized_distribution = phenotype_distribution / np.sum(phenotype_distribution)
        epsilon = np.finfo(float).eps
        if normalized:
            phenotype_entropies[i] = -np.sum(normalized_distribution * np.log(normalized_distribution + epsilon)) / max_entropy
            print(normalized_distribution, phenotype_entropies[i])
        else:
            phenotype_entropies[i] = -np.sum(normalized_distribution * np.log(normalized_distribution + epsilon))
    phenotype_to_entropy_dict = dict(zip(unique_phenotypes, phenotype_entropies))
    return phenotype_to_entropy_dict

from scipy.stats import entropy
import numpy
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


def marker_enrichment(adata, markers):
    gene_entropies = transcriptional_clonotypic_entropy(adata)
    df = pd.DataFrame(list(gene_entropies.items()), columns=['Gene', 'Entropy'])
    df["Entropy"] = -1* df["Entropy"]
    pre_res = gp.prerank(rnk=df,
                        gene_sets=markers,
                        min_size=1,
                        max_size=1000,
                        permutation_num=2000,
                        outdir=None,
                        seed=42,
                        verbose=False)
    return pre_res

def rank_genes_by_clonotypic_entropy(adata,genes=None, probability=False):
    if genes == None:
        genes = adata.var.index.tolist()
    clone_dist = clonotype_distribution(adata, genes=genes, probability=probability)
    ents = entropy(clone_dist,base=2) / np.log2(clone_dist.shape[0])
    gene_entropy = pd.DataFrame.from_dict({"Gene":genes,"Entropy":np.nan_to_num(ents)})
    return gene_entropy.sort_values("Entropy",ascending=True)

def rank_phenotypes_by_clonotypic_entropy(adata, probability=True):
    phenotypes = adata.uns['phenotypic_joint_distribution'].index.tolist()
    clone_dist = clonotype_phenotype_distribution(adata, probability=probability)
    ents = entropy(clone_dist,base=2) / np.log2(clone_dist.shape[0])
    gene_entropy = pd.DataFrame.from_dict({"Phenotype":phenotypes,"Entropy":np.nan_to_num(ents)})
    return gene_entropy.sort_values("Entropy",ascending=True)
    
def rank_clones_by_transcriptional_entropy(adata):
    ce = dict(zip(adata.var.index.tolist(), transcriptional_distribution(adata,probability=True)))
    sorted_ce = [x[0] for x in sorted(ce.items(), key=operator.itemgetter(1))]
    return sorted_ce

def transcriptional_distribution(adata, clones=None, probability=False):
    if clones == None:
        clones = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    dist = adata.uns["joint_distribution"][clones].to_numpy()
    dist = dist.sum(axis=1)
    if probability:
        dist /= dist.sum()
    return np.nan_to_num(dist).T

def clonotype_distribution(adata, genes=None, probability=False):
    if genes == None:
        genes = adata.var.index.tolist()
    clone_counts = clone_size(adata, return_counts=True)
    dist = adata.uns["transcriptional_joint_distribution"].T[genes].to_numpy()
    order = adata.uns["transcriptional_joint_distribution"].columns.tolist()
    sizes = []
    for o in order:
        sizes.append(clone_counts[o])
    dist = dist.T / np.array(sizes)
    if probability:
        dist /= dist.sum()
    return dist.T

def clonotype_phenotype_distribution(adata, probability=False):
    clone_counts = clone_size(adata, return_counts=True)
    dist = adata.uns["phenotypic_joint_distribution"].T.to_numpy()
    order = adata.uns["phenotypic_joint_distribution"].columns.tolist()
    sizes = []
    for o in order:
        sizes.append(clone_counts[o])
    dist = dist.T / np.array(sizes)
    if probability:
        dist /= dist.sum()
    return dist.T

def flux_l1(adata, key, from_this, to_that, clones=None):
    this = adata[adata.obs[key] == from_this]
    that = adata[adata.obs[key] == to_that]
    this_distribution = transcriptional_distribution(this,clones=clones)
    that_distribution = transcriptional_distribution(that,clones=clones)
    return distance.cityblock(this_distribution, that_distribution)

def flux_dkl(adata, key, from_this, to_that, clones=None):
    this = adata[adata.obs[key] == from_this]
    that = adata[adata.obs[key] == to_that]
    this_distribution = transcriptional_distribution(this,clones=clones)
    that_distribution = transcriptional_distribution(that,clones=clones)
    return entropy(this_distribution, that_distribution, base=2, axis=0)

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
