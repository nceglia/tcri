from scipy.stats import entropy
import numpy as np
import operator
from scipy.spatial import distance
import pandas as pd
import warnings
import gseapy as gp

from ..preprocessing._preprocessing import clone_size, transcriptional_joint_distribution, phenotypic_joint_distribution

warnings.filterwarnings('ignore')

def empirical_phenotypic_entropy(adata,phenotype_key="genevector"):
    clonotype_key = adata.uns["tcri_clone_key"]
    df = adata.obs[[clonotype_key,phenotype_key]]
    prob_distributions = df.groupby(clonotype_key)[phenotype_key].value_counts(normalize=True)
    def calc_entropy(P, log_units = 2):
        P = P[P>0].to_numpy().flatten()
        return np.dot(P, -np.log(P))/np.log(log_units)
    grouped_entropy = prob_distributions.groupby(level=0).apply(calc_entropy)
    num_unique_values_col2 = df[phenotype_key].nunique()
    max_entropy = np.log(num_unique_values_col2)
    normalized_entropy = grouped_entropy / max_entropy
    return dict(normalized_entropy.abs())

def probabilistic_phenotypic_entropy(adata):
    phenotypic_joint_distribution(adata)
    clones = adata.uns["phenotypic_joint_distribution"].columns.tolist()
    dist = clonotype_phenotype_distribution(adata).T
    ent = entropy(dist, base=2) / np.log2(dist.shape[1])
    return dict(zip(clones,ent))

def transcriptional_distribution(adata, genes=None):
    if genes == None:
        genes = adata.var.index.tolist()
    clone_counts = clone_size(adata, return_counts=True)
    dist = adata.uns["transcriptional_joint_distribution"].T[genes].to_numpy()
    order = adata.uns["transcriptional_joint_distribution"].columns.tolist()
    sizes = [clone_counts[o] for o in order]
    dist = dist.T / np.array(sizes)
    return dist.T

def transcriptional_entropy(adata, genes=None):
    transcriptional_joint_distribution(adata)
    if genes == None:
        genes = adata.var.index.tolist()
    clones = adata.uns["transcriptional_joint_distribution"].columns.tolist()
    genes = set(genes).intersection(set(adata.var.index.tolist()))
    dist = transcriptional_distribution(adata, genes=list(genes)).T
    ent = entropy(dist, base=2) / np.log2(len(dist))
    return dict(zip(clones,ent))

def transcriptional_clonotypic_entropy(adata, genes=None):
    if genes == None:
        genes = adata.var.index.tolist()
    transcriptional_joint_distribution(adata)
    ent = dict()
    dist = adata.uns["transcriptional_joint_distribution"].T[genes].to_numpy().T
    for gene, d in zip(genes, dist):
        d = d[d > 0]
        ent[gene] = entropy(d, base=2) / np.log2(len(d))
    return ent

def probabilistic_clonotypic_entropy(adata):
    phenotypic_joint_distribution(adata)
    ent = dict()
    dist = adata.uns["phenotypic_joint_distribution"].to_numpy()
    for ph, d in zip(adata.uns["phenotypic_joint_distribution"].index, dist):
        d = d[d > 0]
        ent[ph] = entropy(d, base=2) / np.log2(len(d))
    return ent

def empirical_clonotypic_entropy(adata, phenotype_key="genevector"):
    clone_key = adata.uns["tcri_clone_key"]
    df = adata.obs[[clone_key,phenotype_key]]
    prob_distributions = df.groupby(phenotype_key)[clone_key].value_counts(normalize=True)
    def entropy(probs):
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    grouped_entropy = prob_distributions.groupby(level=0).apply(entropy)
    num_unique_values_col2 = df[clone_key].nunique()
    max_entropy = np.log2(num_unique_values_col2)
    normalized_entropy = grouped_entropy / max_entropy
    return dict(normalized_entropy)

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
