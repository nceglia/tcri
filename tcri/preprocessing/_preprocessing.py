from scipy.stats import entropy
import numpy as np
import tqdm
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def joint_distribution(adata, method='probabalistic'):
    if method == "probabilistic":
        matrix = adata.obs[adata.uns["probability_columns"]]
        matrix["clonotype"] = adata.obs[adata.uns["tcri_clone_key"]]
        jd = matrix.groupby("clonotype").sum().T
        jd.index = [i.replace(" Pseudo-probability","") for i in jd.index]
        adata.uns["joint_distribution"] = jd
    elif method == "empirical":
        tcr_sequences = adata.obs[adata.uns["tcri_clone_key"]].tolist()
        phenotypes = adata.obs[adata.uns['tcri_phenotype_key']].tolist()
        unique_tcrs = np.unique(tcr_sequences)
        unique_phenotypes = np.unique(phenotypes)
        joint_prob_matrix = np.zeros((len(unique_tcrs), len(unique_phenotypes)))
        for tcr, phenotype in zip(tcr_sequences, phenotypes):
            tcr_index = np.where(unique_tcrs == tcr)[0][0]
            phenotype_index = np.where(unique_phenotypes == phenotype)[0][0]
            joint_prob_matrix[tcr_index, phenotype_index] += 1
        for i in range(len(unique_tcrs)):
            joint_prob_matrix[i, :] /= np.sum(joint_prob_matrix[i, :])
        jd = pd.DataFrame(joint_prob_matrix.T,index=adata.uns["probability_columns"],columns=unique_tcrs)
        jd.index = [i.replace(" Pseudo-probability","") for i in jd.index]
        adata.uns["joint_distribution"] = jd
    else:
        raise ValueError("Method must be 'empirical' or 'probabalistic'.")

def register_tcr_key(adata, tcr_key):
    assert tcr_key in adata.obs, "Key {} not found.".format(tcr_key)
    adata.uns["tcri_clone_key"] = tcr_key

def register_phenotype_key(adata, phenotype_key):
    assert phenotype_key in adata.obs, "Key {} not found.".format(phenotype_key)
    adata.uns["tcri_phenotype_key"] = phenotype_key

def register_probability_columns(adata, probability_columns):
    adata.uns["probability_columns"] = probability_columns

def gene_entropy(adata, key_added="entropy"):
    X = adata.X.todense()
    X = np.array(X.T)
    gene_to_row = list(zip(adata.var.index.tolist(), X))
    entropies = []
    for _, exp in tqdm.tqdm(gene_to_row):
        counts = np.unique(exp, return_counts = True)
        entropies.append(entropy(counts[1][1:]))
    adata.var[key_added] = entropies

def clone_size(adata, key_added="clone_size", return_counts=False):
    df = adata.obs
    tcr_key = adata.uns["tcri_clone_key"]
    clone_sizes = dict()
    for clone in set(df[tcr_key]):
        clone_sizes[clone] = len(df[df[tcr_key]==clone].index)
    sizes = []
    for clone in adata.obs[tcr_key]:
        sizes.append(clone_sizes[clone])
    adata.obs[key_added] = sizes
    if return_counts:
        return clone_sizes