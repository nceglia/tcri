from scipy.stats import entropy
import numpy as np
import tqdm
import pandas as pd
import collections
import warnings
warnings.filterwarnings('ignore')

def joint_distribution(adata, method='probabilistic'):
    if method == "probabilistic":
        matrix = adata.obs[adata.uns["probability_columns"]]
        matrix["clonotype"] = adata.obs[adata.uns["tcri_clone_key"]]
        jd = matrix.groupby("clonotype").sum().T
        jd.index = [i.replace(" Pseudo-probability","") for i in jd.index]
        jd = np.round(jd,decimals=5)
        adata.uns["joint_distribution"] = jd
    elif method == "empirical":
        tcr_sequences = adata.obs[adata.uns["tcri_clone_key"]].tolist()
        phenotypes = adata.obs[adata.uns['tcri_phenotype_key']].tolist()
        unique_tcrs = adata.uns["tcri_unique_clonotypes"]
        unique_phenotypes = adata.uns["tcri_unique_phenotypes"]
        joint_prob_matrix = np.zeros((len(unique_tcrs), len(unique_phenotypes)))
        for tcr, phenotype in zip(tcr_sequences, phenotypes):
            tcr_index = np.where(unique_tcrs == tcr)[0][0]
            phenotype_index = np.where(unique_phenotypes == phenotype)[0][0]
            joint_prob_matrix[tcr_index, phenotype_index] += 1
        jd = pd.DataFrame(joint_prob_matrix.T,index=unique_phenotypes,columns=unique_tcrs)
        jd.index = [i.replace(" Pseudo-probability","") for i in jd.index]
        jd = np.round(jd,decimals=5)
        adata.uns["joint_distribution"] = jd
    else:
        raise ValueError("Method must be 'empirical' or 'probabalistic'.")

def register_phenotype_key(adata, phenotype_key, order=None):
    assert phenotype_key in adata.obs, "Key {} not found.".format(phenotype_key)
    if order==None:
        adata.uns["tcri_unique_phenotypes"] = np.unique(adata.obs[phenotype_key].tolist())
    adata.uns["tcri_phenotype_key"] = phenotype_key

def register_clonotype_key(adata, tcr_key):
    assert tcr_key in adata.obs, "Key {} not found.".format(tcr_key)
    adata.uns["tcri_clone_key"] = tcr_key
    adata.uns["tcri_unique_clonotypes"] = np.unique(adata.obs[tcr_key].tolist())


def register_probability_columns(adata, probability_columns):
    adata.uns["probability_columns"] = probability_columns

def gene_entropy(adata, key_added="entropy", batch_key=None, agg_function=None):
    if batch_key == None:
        X = adata.X.todense()
        X = np.array(X.T)
        gene_to_row = list(zip(adata.var.index.tolist(), X))
        entropies = []
        for _, exp in tqdm.tqdm(gene_to_row):
            counts = np.unique(exp, return_counts = True)
            entropies.append(entropy(counts[1][1:]))
        adata.var[key_added] = entropies
    else:
        if agg_function == None:
            agg_function = np.mean
        entropies = collections.defaultdict(list)
        for x in tqdm.tqdm(list(set(adata.obs[batch_key]))):
            sdata = adata[adata.obs[batch_key]==x]
            X = sdata.X.todense()
            X = np.array(X.T)
            gene_to_row = list(zip(sdata.var.index.tolist(), X))
            for symbol, exp in gene_to_row:
                counts = np.unique(exp, return_counts = True)
                entropies[symbol].append(entropy(counts[1][1:]))        
        aggregated_entropies = []
        for g in adata.var.index.tolist():
            ent = agg_function(entropies[g])
            aggregated_entropies.append(ent)
        adata.var[key_added] = aggregated_entropies

def clone_size(adata, key_added="clone_size", return_counts=False):
    tcr_key = adata.uns["tcri_clone_key"]
    res = np.unique(adata.obs[tcr_key].tolist(), return_counts=True)
    clone_sizes = dict(zip(res[0],res[1]))
    sizes = []
    for clone in adata.obs[tcr_key]:
        sizes.append(clone_sizes[clone])
    adata.obs[key_added] = sizes
    if return_counts:
        return clone_sizes