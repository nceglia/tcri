from scipy.stats import entropy
import numpy as np
import tqdm

import warnings
warnings.filterwarnings('ignore')

def joint_distribution(adata):
    matrix = adata.to_df()
    matrix["clonotype"] = adata.obs[adata.uns["tcri_clone_key"]]
    jd = matrix.groupby("clonotype").sum().T
    # jd /= jd.sum(axis=0)
    adata.uns["joint_distribution"] = jd

def add_tcr_key(adata,tcr_key):
    assert tcr_key in adata.obs, "Key {} not found.".fromat(tcr_key)
    adata.uns["tcri_clone_key"] = tcr_key

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