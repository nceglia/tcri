from scipy.stats import entropy
import numpy as np
import tqdm
import pandas as pd
import collections
import warnings
import torch
import torch.nn.functional as F
import datetime

warnings.filterwarnings('ignore')

def store_model_outputs(model, adata):
    # Assuming model.module provides these tensors
    adata.uns["tcri_p_ct"] = model.module.get_p_ct().cpu().numpy()  # (num_ct_pairs, num_phenotypes)
    adata.uns["tcri_ct_to_cov"] = model.module.ct_to_cov.cpu().numpy()  # (num_ct_pairs,)
    adata.uns["tcri_ct_to_c"] = model.module.ct_to_c.cpu().numpy()  # (num_ct_pairs,)

    # Store category labels explicitly
    cov_col = model.adata_manager.registry["covariate_col"]
    clone_col = model.adata_manager.registry["clonotype_col"]
    pheno_col = model.adata_manager.registry["phenotype_col"]

    adata.uns["tcri_covariate_categories"] = adata.obs[cov_col].astype("category").cat.categories.tolist()
    adata.uns["tcri_clonotype_categories"] = adata.obs[clone_col].astype("category").cat.categories.tolist()
    adata.uns["tcri_phenotype_categories"] = adata.obs[pheno_col].astype("category").cat.categories.tolist()

    # Store metadata keys for easy reference
    adata.uns["tcri_metadata"] = {
        "covariate_col": cov_col,
        "clonotype_col": clone_col,
        "phenotype_col": pheno_col,
        "timestamp": str(datetime.datetime.now())
    }


def joint_distribution(
    adata, covariate_label: str, temperature: float = 1.0
) -> pd.DataFrame:
    # Retrieve stored tensors and metadata
    p_ct = torch.tensor(adata.uns["tcri_p_ct"])  # shape (num_ct_pairs, num_phenotypes)
    ct_to_cov = torch.tensor(adata.uns["tcri_ct_to_cov"])  # shape (num_ct_pairs,)
    ct_to_c = torch.tensor(adata.uns["tcri_ct_to_c"])  # shape (num_ct_pairs,)

    # Retrieve categorical mappings
    covariate_categories = adata.uns["tcri_covariate_categories"]
    phenotype_categories = adata.uns["tcri_phenotype_categories"]
    clonotype_categories = adata.uns["tcri_clonotype_categories"]

    metadata = adata.uns["tcri_metadata"]
    covariate_col = metadata["covariate_col"]

    # A) Convert covariate_label to an integer index
    try:
        cov_value = covariate_categories.index(covariate_label)
    except ValueError:
        raise ValueError(f"Covariate label '{covariate_label}' not found in stored categories.")

    # B) Retrieve p_ct for that covariate
    chosen_mask = (ct_to_cov == cov_value)
    chosen_idx = chosen_mask.nonzero(as_tuple=True)[0]
    p_ct_for_cov = p_ct[chosen_mask]

    # C) Apply temperature scaling
    eps = 1e-8  # stability constant
    p_ct_for_cov = F.softmax(torch.log(p_ct_for_cov + eps) / temperature, dim=-1)

    # D) Map each row to its clonotype index
    clone_indices = ct_to_c[chosen_idx].numpy()

    # E) Build DataFrame
    p_ct_arr = p_ct_for_cov.numpy()
    if len(phenotype_categories) != p_ct_arr.shape[1]:
        raise ValueError("Phenotype category mismatch with p_ct columns.")

    df = pd.DataFrame(p_ct_arr, columns=phenotype_categories)
    df["clonotype_index"] = clone_indices

    # F) Map clonotype indices to actual clonotype IDs
    df["clonotype_id"] = df["clonotype_index"].apply(lambda i: clonotype_categories[i])
    df.index = df["clonotype_id"]
    df = df[[x for x in df.columns if "clonotype" not in x]]
    return df


def group_small_clones(adata, patient_key="",):
    ct = []
    for x, s, p in zip(adata.obs["trb"], adata.obs["clone_size"], adata.obs[patient_key]):
        if s < 4:
            ct.append("Singleton_{}".format(p))
        else:
            ct.append("{}_{}".format(x,p))
    adata.obs["trb_unique"] = ct


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