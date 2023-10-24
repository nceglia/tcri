from scipy.stats import entropy
import numpy as np
import operator
from scipy.spatial import distance
import pandas as pd
import warnings
import gseapy as gp

warnings.filterwarnings('ignore')

def clonotypic_entropy(adata, clones=None):
    if clones != None:
        clones = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    dist = transcriptional_distribution(adata, clones=clones)
    dist = dist[dist > 0]
    ent = entropy(dist, base=2) / np.log2(len(dist))
    return ent

def phenotypic_entropy(adata, genes=None):
    if genes == None:
        genes = adata.var.index.tolist()
    genes = set(genes).intersection(set(adata.var.index.tolist()))
    dist = clonotype_distribution(adata, genes=list(genes))
    dist = dist[dist > 0]
    ent = entropy(dist, base=2) / np.log2(len(dist))
    return ent

def marker_enrichment(adata, markers):
    df = rank_genes_by_clonotypic_entropy(adata,probability=True)
    pre_res = gp.prerank(rnk=df,
                        gene_sets=markers,
                        threads=4,
                        min_size=1,
                        max_size=1000,
                        permutation_num=2000, # reduce number to speed up testing
                        outdir=None, # don't write to disk
                        seed=6,
                        verbose=False, # see what's going on behind the scenes
                        )
    return pre_res

def rank_genes_by_clonotypic_entropy(adata, probability=False):
    ce = dict(zip(adata.var.index.tolist(), clonotype_distribution(adata,probability=probability)))
    gene = []
    entr = []
    for g in reversed(sorted(ce.items(), key=operator.itemgetter(1))):
        gene.append(g[0])
        entr.append(g[1])
    df = pd.DataFrame.from_dict({'Gene':gene,'Entropy':entr})
    return df

def rank_clones_by_phenotypic_entropy(adata):
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
    return np.nan_to_num(dist)

def clonotype_distribution(adata, genes=None, probability=False):
    if genes == None:
        genes = adata.var.index.tolist()
    dist = adata.uns["joint_distribution"].T[genes].to_numpy()
    dist = dist.sum(axis=1)
    if probability:
        dist /= dist.sum()
    return np.nan_to_num(dist)

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
