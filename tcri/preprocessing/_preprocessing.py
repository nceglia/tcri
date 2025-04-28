from scipy.stats import entropy
import numpy as np
import tqdm
import pandas as pd
import collections
import warnings
import torch
import torch.nn.functional as F
import datetime
import pyro.distributions as dist
from pyro.distributions import Dirichlet
import pyro

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Optional

from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

def register_phenotype_key(adata, phenotype_key, order=None):
    assert phenotype_key in adata.obs, "Key {} not found.".format(phenotype_key)
    if order==None:
        adata.uns["tcri_unique_phenotypes"] = np.unique(adata.obs[phenotype_key].tolist())
    adata.uns["tcri_phenotype_key"] = phenotype_key

def register_clonotype_key(adata, tcr_key):
    assert tcr_key in adata.obs, "Key {} not found.".format(tcr_key)
    adata.uns["tcri_clone_key"] = tcr_key
    adata.uns["tcri_unique_clonotypes"] = np.unique(adata.obs[tcr_key].tolist())

def group_singletons(adata,clonotype_key="trb",groupby="patient", target_col="trb_unique", min_clone_size=10):
    adata.obs["trb_candidate"] = adata.obs[clonotype_key].astype(str) + "_" + adata.obs[groupby].astype(str)
    clone_counts = adata.obs["trb_candidate"].value_counts()
    def collapse_singleton(row):
        candidate = row["trb_candidate"]
        if clone_counts[candidate] < min_clone_size:
            return f"Singleton_{row[groupby]}"
        else:
            return candidate
    adata.obs[target_col] = adata.obs.apply(collapse_singleton, axis=1)

def classify_phenotypes(adata, phenotype_prob_slot="X_tcri_phenotypes", phenotype_assignment_obs="tcri_phenotype"):
    print("Classifying phenotypes")
    phenotype_col = adata.uns["tcri_metadata"]["phenotype_col"]
    ct_array = adata.uns["tcri_ct_array_for_cells"]
    unique_cts = np.unique(ct_array)
    phenotype_probs_posterior = adata.uns["tcri_p_ct"]
    phenotypes = adata.uns["tcri_phenotype_categories"]
    latent_z = adata.obsm["X_tcri"]

    # Pre-compute phenotype archetype embeddings
    archetype_matrix = np.vstack([
        latent_z[adata.obs[phenotype_col].values == phenotype].mean(axis=0) 
        for phenotype in phenotypes
    ])

    all_probs = np.zeros((adata.n_obs, len(phenotypes)))

    # Iterate over each unique ct
    for ct in unique_cts:
        ct_indices = np.where(ct_array == ct)[0]
        ct_embeddings = latent_z[ct_indices]

        # Cosine similarity (cells x phenotypes)
        similarity = cosine_similarity(ct_embeddings, archetype_matrix)
        similarity = (similarity + 1) / 2  # Normalize cosine similarity to [0,1]

        # Adjust similarity scores by posterior phenotype probabilities
        adjusted_scores = similarity * phenotype_probs_posterior[ct]

        # Normalize to probabilities per cell
        probs_normalized = adjusted_scores / adjusted_scores.sum(axis=1, keepdims=True)
        all_probs[ct_indices] = probs_normalized

    # Store the normalized probabilities in AnnData
    adata.obsm[phenotype_prob_slot] = all_probs

    # Assign phenotype with highest probability
    assignments = all_probs.argmax(axis=1)
    adata.obs[phenotype_assignment_obs] = pd.Categorical.from_codes(
        assignments, categories=phenotypes
    )

@torch.no_grad()
def register_model(
    adata,
    model,
    phenotype_prob_slot="X_tcri_phenotypes",
    phenotype_assignment_obs="tcri_phenotype",
    latent_slot="X_tcri",
    batch_size=256,
    gate_prob=0.5
):
    """
    Register TCRi model outputs in the AnnData object.
    
    This function takes a trained TCRi model and stores all relevant model outputs
    and metadata in the AnnData object for downstream analysis. It stores the model's
    latent representations, phenotype probabilities, and other model-derived data.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to register model outputs in
    model : TCRIModel
        Trained TCRIModel object
    phenotype_prob_slot : str, default="X_tcri_phenotypes"
        Key in adata.obsm where phenotype probabilities will be stored
    phenotype_assignment_obs : str, default="tcri_phenotype"
        Key in adata.obs where the discrete phenotype assignments will be stored
    latent_slot : str, default="X_tcri"
        Key in adata.obsm where latent representations will be stored
    batch_size : int, default=256
        Batch size for model inference
        
    Returns
    -------
    AnnData
        The input AnnData object with model outputs registered
        
    Notes
    -----
    The function stores multiple items in adata.uns for use by downstream functions:
    - "tcri_p_ct": Clonotype-covariate phenotype distributions
    - "tcri_ct_to_cov": Mapping from clonotype-covariate indices to covariate indices
    - "tcri_ct_to_c": Mapping from clonotype-covariate indices to clonotype indices
    - "tcri_covariate_categories": List of covariate category names
    - "tcri_clonotype_categories": List of clonotype category names
    - "tcri_phenotype_categories": List of phenotype category names
    - "tcri_metadata": Dictionary of column names for different data types
    - "tcri_ct_array_for_cells": For each cell, which clonotype-covariate index it belongs to
    - "tcri_cov_array_for_cells": For each cell, which covariate index it has
    
    Examples
    --------
    >>> import tcri
    >>> # Train a model
    >>> model = tcri.TCRIModel(adata)
    >>> model.train()
    >>> 
    >>> # Register model outputs
    >>> adata = tcri.pp.register_model(
    ...     adata, 
    ...     model, 
    ...     phenotype_prob_slot="X_tcri_phenotypes",
    ...     phenotype_assignment_obs="tcri_phenotype"
    ... )
    """
    # Store model outputs
    adata.uns["tcri_p_ct"] = model.module.get_p_ct().cpu().numpy()
    adata.uns["tcri_ct_to_cov"] = model.module.ct_to_cov.cpu().numpy()
    adata.uns["tcri_ct_to_c"] = model.module.ct_to_c.cpu().numpy()
    # Get category labels from the model
    cov_col = model.adata_manager.registry["covariate_col"]
    clone_col = model.adata_manager.registry["clonotype_col"]
    phenotype_col = model.adata_manager.registry["phenotype_col"]
    batch_col = model.adata_manager.registry["batch_col"]

    adata.uns["tcri_covariate_categories"] = (
        adata.obs[cov_col].astype("category").cat.categories.tolist()
    )
    adata.uns["tcri_clonotype_categories"] = (
        adata.obs[clone_col].astype("category").cat.categories.tolist()
    )
    adata.uns["tcri_phenotype_categories"] = (
        adata.obs[phenotype_col].astype("category").cat.categories.tolist()
    )
    adata.uns["tcri_metadata"] = {
        "covariate_col": cov_col,
        "clone_col": clone_col,
        "   ": phenotype_col,
        "batch_col": batch_col,
    }
    adata.uns["tcri_local_scale"] = model.module.local_scale
    # Store per-cell arrays needed by downstream functions
    ct_array_for_cells = model.module.ct_array.cpu().numpy()
    adata.uns["tcri_ct_array_for_cells"] = ct_array_for_cells
    ct_to_cov = model.module.ct_to_cov.cpu().numpy()
    cov_array_for_cells = ct_to_cov[ct_array_for_cells]
    adata.uns["tcri_cov_array_for_cells"] = cov_array_for_cells
    # Store per-cell phenotype probabilities
    classify_phenotypes(adata, phenotype_prob_slot=phenotype_prob_slot, phenotype_assignment_obs=phenotype_assignment_obs)
    # Store latent representation
    latent_z = model.get_latent_representation(batch_size=batch_size)
    adata.obsm[latent_slot] = latent_z
    adata.uns["tcri_global_prior"] = model.module.clone_phen_prior.cpu().numpy()
    adata.uns["tcri_cov_prior"] = model.module.get_p_ct().cpu().numpy()
    register_phenotype_key(adata,"tcri_phenotype")
    register_clonotype_key(adata,"trb_unique")
    return adata

def joint_distribution(
    adata, 
    covariate_label: str, 
    temperature: float = 1.0, 
    n_samples: int = 0, 
    clones=None,
    weighted: bool = False,
) -> pd.DataFrame:
    """
    Calculate joint distribution of phenotypes and TCR clonotypes for a given covariate.
    
    This function retrieves phenotype probability distributions for each clonotype in a
    specified covariate condition (e.g., timepoint, tissue type). The resulting distribution
    shows how different clonotypes are associated with different phenotypes. Optionally,
    distributions can be weighted by clone size to give more influence to larger clones.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing TCRi model outputs. Must contain certain fields in
        adata.uns that are created by the register_model function
    covariate_label : str
        The covariate label (e.g., "Day0", "tumor") to calculate the joint distribution for
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions. Lower values make
        distributions more peaked, higher values make them more uniform
    n_samples : int, default=0
        If > 0, samples from the Dirichlet distribution n_samples times for uncertainty
        estimation. If 0, returns point estimates
    clones : list, optional
        List of clone IDs to restrict the analysis to. If None, uses all clones
    weighted : bool, default=False
        If True, weights each distribution by the number of cells in that clonotype and
        covariate, giving more influence to larger clones in the final distribution

    Returns
    -------
    pd.DataFrame
        DataFrame where each row is a clonotype and each column is a phenotype.
        Each row contains the probability distribution over phenotypes for that clonotype.
        If weighted=True, large clones contribute more to the overall distribution.
        
    Examples
    --------
    >>> import tcri
    >>> # Get joint distribution for Day0
    >>> jd = tcri.pp.joint_distribution(adata, "Day0")
    >>> 
    >>> # Get joint distribution with sampling for uncertainty estimation
    >>> jd_samples = tcri.pp.joint_distribution(adata, "tumor", n_samples=100)
    >>> 
    >>> # Get weighted joint distribution for specific clones
    >>> my_clones = ["CASSQETQYF", "CASSLGQAYEQYF"]
    >>> jd_weighted = tcri.pp.joint_distribution(
    ...     adata, "Day14", temperature=0.5, clones=my_clones, weighted=True
    ... )
    """
    # Retrieve stored tensors and metadata
    p_ct = torch.tensor(adata.uns["tcri_p_ct"])
    ct_to_cov = torch.tensor(adata.uns["tcri_ct_to_cov"])
    ct_to_c = torch.tensor(adata.uns["tcri_ct_to_c"])

    covariate_categories = adata.uns["tcri_covariate_categories"]
    phenotype_categories = adata.uns["tcri_phenotype_categories"]
    clonotype_categories = adata.uns["tcri_clonotype_categories"]

    metadata = adata.uns["tcri_metadata"]
    covariate_col = metadata["covariate_col"]

    # Convert covariate_label to index
    try:
        cov_value = covariate_categories.index(covariate_label)
    except ValueError:
        raise ValueError(f"Covariate label '{covariate_label}' not found among: {covariate_categories}")

    # Get data specific to this covariate
    chosen_mask = (ct_to_cov == cov_value)
    chosen_idx = chosen_mask.nonzero(as_tuple=True)[0]
    p_ct_for_cov = p_ct[chosen_mask]

    # Apply temperature scaling
    eps = 1e-8
    p_ct_for_cov = F.softmax(torch.log(p_ct_for_cov + eps) / temperature, dim=-1)

    # Get clonotype indices for each chosen ct
    clone_indices = ct_to_c[chosen_idx].numpy()

    # Get cell counts for each clonotype-covariate pair (for weighting)
    ct_array_for_cells = adata.uns["tcri_ct_array_for_cells"]
    cov_array_for_cells = adata.uns["tcri_cov_array_for_cells"]

    from collections import Counter
    cell_mask = (cov_array_for_cells == cov_value)
    cts_in_cov = ct_array_for_cells[cell_mask]
    ct_counts_dict = Counter(cts_in_cov.tolist())

    p_ct_arr = p_ct_for_cov.numpy()

    if n_samples == 0:
        # Build dataframe with point estimates (no sampling)
        df = pd.DataFrame(p_ct_arr, columns=phenotype_categories)
        df["clonotype_index"] = clone_indices
        df["clonotype_id"] = [clonotype_categories[i] for i in clone_indices]

        # Filter to requested clones
        if clones is not None:
            df = df[df["clonotype_id"].isin(clones)]

        # Apply clone size weighting if requested
        if weighted:
            counts = []
            for i, row in df.iterrows():
                ct_i = row["clonotype_index"]
                c_count = ct_counts_dict.get(ct_i, 0)
                counts.append(c_count)

            counts = np.array(counts, dtype=float)
            df.loc[:, phenotype_categories] = df[phenotype_categories].values * counts[:, None]

            total_mass = df[phenotype_categories].sum().sum()
            if total_mass > 0:
                df.loc[:, phenotype_categories] = df[phenotype_categories] / total_mass

        # Set the index and clean up columns
        df.index = df["clonotype_id"]
        df = df[[col for col in df.columns if "clonotype" not in col]]
        return df

    else:
        # Sample from Dirichlet distribution
        local_scale = adata.uns.get("tcri_local_scale", 1.0)
        conc = local_scale * p_ct_for_cov
        
        samples = Dirichlet(conc).sample((n_samples,))
        samples_np = samples.cpu().numpy()

        # Reshape samples for DataFrame creation
        num_chosen, num_pheno = p_ct_arr.shape
        samples_expanded = samples_np.transpose(1, 0, 2).reshape(-1, num_pheno)

        # Create arrays for sample tracking
        clonotype_indices_expanded = np.repeat(clone_indices, n_samples)
        clonotype_ids_expanded = [clonotype_categories[i] for i in clonotype_indices_expanded]
        sample_ids = np.tile(np.arange(n_samples), num_chosen)

        # Build dataframe
        df_samples = pd.DataFrame(samples_expanded, columns=phenotype_categories)
        df_samples["clonotype_index"] = clonotype_indices_expanded
        df_samples["clonotype_id"] = clonotype_ids_expanded
        df_samples["sample_id"] = sample_ids

        # Filter to requested clones
        if clones is not None:
            df_samples = df_samples[df_samples["clonotype_id"].isin(clones)]

        # Apply clone size weighting if requested
        if weighted:
            counts = []
            for i, row in df_samples.iterrows():
                ct_i = row["clonotype_index"]
                c_count = ct_counts_dict.get(ct_i, 0)
                counts.append(c_count)
            counts = np.array(counts, dtype=float)
            df_samples.loc[:, phenotype_categories] = (
                df_samples[phenotype_categories].values * counts[:, None]
            )
            total_mass = df_samples[phenotype_categories].sum().sum()
            if total_mass > 0:
                df_samples.loc[:, phenotype_categories] /= total_mass

        # Set the index and clean up columns
        df_samples.index = [
            f"{cid}_{sid}" for cid, sid in zip(df_samples["clonotype_id"], df_samples["sample_id"])
        ]
        df_samples = df_samples[[col for col in df_samples.columns if col not in ["clonotype_id","clonotype_index","sample_id"]]]
        return df_samples

def get_latent_embedding(
    adata, 
    latent_slot: str = "X_tcri",
    n_samples: int = 0,
    posterior_scale: float = 1.0
) -> "np.ndarray":
    mean_z = adata.obsm[latent_slot] 
    n_cells, latent_dim = mean_z.shape
    samples = np.random.normal(
        loc=mean_z, 
        scale=posterior_scale, 
        size=(n_samples, n_cells, latent_dim)
    )
    return samples


def group_small_clones(adata, patient_key=""):
    ct = []
    for x, s, p in zip(adata.obs["trb"], adata.obs["clone_size"], adata.obs[patient_key]):
        if s < 4:
            ct.append("Singleton_{}".format(p))
        else:
            ct.append("{}_{}".format(x,p))
    adata.obs["trb_unique"] = ct

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