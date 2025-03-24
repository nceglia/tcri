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


warnings.filterwarnings('ignore')

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

import torch
import pandas as pd

@torch.no_grad()
def register_model(
    adata,
    model,
    phenotype_prob_slot="X_tcri_phenotypes",
    phenotype_assignment_obs="tcri_phenotype",
    latent_slot="X_tcri",
    batch_size=256
):
    # --- Store core model outputs ---
    adata.uns["tcri_p_ct"] = model.module.get_p_ct().cpu().numpy()
    adata.uns["tcri_ct_to_cov"] = model.module.ct_to_cov.cpu().numpy()
    adata.uns["tcri_ct_to_c"] = model.module.ct_to_c.cpu().numpy()

    # Store category labels explicitly
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

    # Store metadata keys for easy reference
    adata.uns["tcri_metadata"] = {
        "covariate_col": cov_col,
        "clone_col": clone_col,
        "phenotype_col": phenotype_col,
        "batch_col": batch_col,
    }

    # Store local_scale
    adata.uns["tcri_local_scale"] = model.module.local_scale

    # ------------------------------------------------------------------------
    #  A) Per-cell arrays that some downstream functions need
    # ------------------------------------------------------------------------
    #  1) ct_array_for_cells: for each cell, which ct index it belongs to
    ct_array_for_cells = model.module.ct_array.cpu().numpy()
    adata.uns["tcri_ct_array_for_cells"] = ct_array_for_cells

    #  2) cov_array_for_cells: for each cell, which covariate index it has
    #     This is simply ct_to_cov[ct_array_for_cells].
    ct_to_cov = model.module.ct_to_cov.cpu().numpy()
    cov_array_for_cells = ct_to_cov[ct_array_for_cells]
    adata.uns["tcri_cov_array_for_cells"] = cov_array_for_cells

    #  If you also want the global clone index per cell, you can similarly do:
    #     c_array_for_cells = model.module.c_array.cpu().numpy()
    #     adata.uns["tcri_c_array_for_cells"] = c_array_for_cells
    #
    #  Just make sure 'c_array' is defined in the module (like 'module.ct_array').

    # ------------------------------------------------------------------------
    #  B) Compute and store per-cell phenotype probabilities
    # ------------------------------------------------------------------------
    phenotype_probs = model.get_cell_phenotype_probs(batch_size=batch_size)
    adata.obsm[phenotype_prob_slot] = phenotype_probs

    # ------------------------------------------------------------------------
    #  C) Compute and store phenotype assignments (argmax of probabilities)
    # ------------------------------------------------------------------------
    assignments = phenotype_probs.argmax(axis=1)
    adata.obs[phenotype_assignment_obs] = pd.Categorical.from_codes(
        assignments, categories=adata.uns["tcri_phenotype_categories"]
    )

    # ------------------------------------------------------------------------
    #  D) Compute and store latent representation (z)
    # ------------------------------------------------------------------------
    latent_z = model.get_latent_representation(batch_size=batch_size)
    adata.obsm[latent_slot] = latent_z

    adata.uns["tcri_global_prior"] = model.module.clone_phen_prior.cpu().numpy()
    adata.uns["tcri_cov_prior"] = model.module.get_p_ct().cpu().numpy()
    adata.uns["tcri_confusion_matrix"] = pyro.param("confusion_matrix").detach().cpu().numpy()

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
    Returns a DataFrame whose rows are local phenotype distributions for each
    clonotype in the specified covariate, with optional weighting by clone size.
    
    If weighted=True, each row is multiplied by (# cells in that clonotype+covariate).
    Then the entire matrix is normalized to sum to 1, giving a "joint" distribution
    where large clones have more mass.
    
    Parameters
    ----------
    adata : AnnData
        Must contain certain fields in `adata.uns`, including:
        - "tcri_p_ct" -> local distributions p_ct
        - "tcri_ct_to_cov" -> array mapping ct index -> covariate index
        - "tcri_ct_to_c" -> array mapping ct index -> clone index
        - "tcri_covariate_categories" -> list of covariate labels
        - "tcri_phenotype_categories" -> list of phenotype labels
        - "tcri_clonotype_categories" -> list of clonotype labels
        - "tcri_ct_array_for_cells" -> ct index per cell
        - "tcri_cov_array_for_cells" -> covariate index per cell
        - "tcri_metadata" -> includes keys "clone_col", "covariate_col", etc.
    covariate_label : str
        A label in `adata.uns["tcri_covariate_categories"]` that specifies
        which covariate we want the local distributions for.
    temperature : float
        Temperature to use when scaling p_ct via softmax(log(...)/temperature).
    n_samples : int
        If > 0, we sample from the Dirichlet. Otherwise, we return point estimates.
    clones : list of str
        If given, we filter to only those clonotype IDs.
    weighted : bool
        If True, multiply each row by the #cells in that clone+covariate, then do a
        global normalization so large clones have more total mass.

    Returns
    -------
    pd.DataFrame
        Columns = phenotype categories, each row = distribution for a clonotype (or sample).
        If weighted=True and n_samples=0, the row sums will NOT necessarily be 1—only
        the entire DataFrame sums to 1 across all rows.
    """
    # A) Retrieve stored tensors and metadata
    p_ct = torch.tensor(adata.uns["tcri_p_ct"])  # shape: (num_ct_pairs, num_phenotypes)
    ct_to_cov = torch.tensor(adata.uns["tcri_ct_to_cov"])  # shape: (num_ct_pairs,)
    ct_to_c = torch.tensor(adata.uns["tcri_ct_to_c"])      # shape: (num_ct_pairs,)

    covariate_categories = adata.uns["tcri_covariate_categories"]
    phenotype_categories = adata.uns["tcri_phenotype_categories"]
    clonotype_categories = adata.uns["tcri_clonotype_categories"]

    metadata = adata.uns["tcri_metadata"]
    covariate_col = metadata["covariate_col"]

    # B) Convert covariate_label to index
    try:
        cov_value = covariate_categories.index(covariate_label)
    except ValueError:
        raise ValueError(f"Covariate label '{covariate_label}' not found among: {covariate_categories}")

    # C) Mask to rows for this covariate
    chosen_mask = (ct_to_cov == cov_value)
    chosen_idx = chosen_mask.nonzero(as_tuple=True)[0]
    p_ct_for_cov = p_ct[chosen_mask]  # shape (num_chosen, num_phenotypes)

    # D) Apply temperature scaling to get distributions
    eps = 1e-8
    p_ct_for_cov = F.softmax(torch.log(p_ct_for_cov + eps) / temperature, dim=-1)

    # E) Get clonotype indices for each chosen ct
    clone_indices = ct_to_c[chosen_idx].numpy()  # shape (num_chosen,)

    # F) [Optional] If weighted=True, compute #cells in each ct index
    #    Then multiply each row by that count. We'll do the final normalization at the end.
    ct_array_for_cells = adata.uns["tcri_ct_array_for_cells"]  # shape (#cells,)
    cov_array_for_cells = adata.uns["tcri_cov_array_for_cells"]  # shape (#cells,)

    # Count how many cells in each ct for this covariate
    from collections import Counter
    cell_mask = (cov_array_for_cells == cov_value)
    cts_in_cov = ct_array_for_cells[cell_mask]
    ct_counts_dict = Counter(cts_in_cov.tolist())

    # G) Distinguish between no-sampling vs. sampling
    p_ct_arr = p_ct_for_cov.numpy()  # shape (num_chosen, num_phenotypes)

    if n_samples == 0:
        # (1) No-sampling, point estimates
        df = pd.DataFrame(p_ct_arr, columns=phenotype_categories)
        df["clonotype_index"] = clone_indices
        df["clonotype_id"] = [clonotype_categories[i] for i in clone_indices]

        # Filter clones if needed
        if clones is not None:
            df = df[df["clonotype_id"].isin(clones)]

        # If weighted, multiply each row by # cells and do a single global normalization
        if weighted:
            # Multiply each row
            counts = []
            for i, row in df.iterrows():
                ct_i = row["clonotype_index"]
                c_count = ct_counts_dict.get(ct_i, 0)  # # cells in that ct+cov
                counts.append(c_count)

            counts = np.array(counts, dtype=float)
            # Multiply distribution columns
            df.loc[:, phenotype_categories] = df[phenotype_categories].values * counts[:, None]

            # Now sum all elements across the entire matrix to get a global normalizer
            total_mass = df[phenotype_categories].sum().sum()
            if total_mass > 0:
                df.loc[:, phenotype_categories] = df[phenotype_categories] / total_mass

        # Set the index to clonotype_id
        df.index = df["clonotype_id"]
        # Optionally remove clonotype columns
        df = df[[col for col in df.columns if "clonotype" not in col]]
        return df

    else:
        # (2) We have n_samples>0, so sample from Dirichlet
        local_scale = adata.uns.get("tcri_local_scale", 1.0)
        conc = local_scale * p_ct_for_cov  # shape: (num_chosen, num_phenotypes)
        
        samples = Dirichlet(conc).sample((n_samples,))  # shape: (n_samples, num_chosen, num_phenotypes)
        samples_np = samples.cpu().numpy()

        # Reshape to (num_chosen * n_samples, num_phenotypes)
        # so each row is a single sample from that ct
        num_chosen, num_pheno = p_ct_arr.shape
        samples_expanded = samples_np.transpose(1, 0, 2).reshape(-1, num_pheno)

        # Make repeated arrays for clonotype indices
        clonotype_indices_expanded = np.repeat(clone_indices, n_samples)
        clonotype_ids_expanded = [clonotype_categories[i] for i in clonotype_indices_expanded]
        sample_ids = np.tile(np.arange(n_samples), num_chosen)

        df_samples = pd.DataFrame(samples_expanded, columns=phenotype_categories)
        df_samples["clonotype_index"] = clonotype_indices_expanded
        df_samples["clonotype_id"] = clonotype_ids_expanded
        df_samples["sample_id"] = sample_ids

        if clones is not None:
            df_samples = df_samples[df_samples["clonotype_id"].isin(clones)]

        # If weighted, multiply each row by the # cells in that ct, then globally normalize
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

        # Set a unique index for each (clonotype_id, sample_id)
        df_samples.index = [
            f"{cid}_{sid}" for cid, sid in zip(df_samples["clonotype_id"], df_samples["sample_id"])
        ]
        # Remove columns if desired
        df_samples = df_samples[[col for col in df_samples.columns if col not in ["clonotype_id","clonotype_index","sample_id"]]]
        return df_samples



def global_joint_distribution(
    adata, 
    temperature: float = 1.0, 
    n_samples: int = 0, 
) -> pd.DataFrame:
    # Retrieve global clonotype-level phenotype estimates and mappings.
    try:
        # p_c is expected to be of shape (num_clonotypes, num_phenotypes)
        p_c = torch.tensor(adata.uns["tcri_p_c"])
    except KeyError:
        raise KeyError("Global p_c not found in adata.uns. Please store it (e.g., under 'tcri_p_c').")
    
    clonotype_categories = adata.uns["tcri_clonotype_categories"]
    phenotype_categories = adata.uns["tcri_phenotype_categories"]

    # Apply temperature scaling (with a stability constant)
    eps = 1e-8
    p_c = F.softmax(torch.log(p_c + eps) / temperature, dim=-1)

    if n_samples == 0:
        # Point estimate: one row per clonotype.
        p_c_arr = p_c.numpy()
        df = pd.DataFrame(p_c_arr, columns=phenotype_categories)
        # Create a clonotype index (assumes rows are in the same order as clonotype_categories)
        indices = np.arange(len(clonotype_categories))
        df["clonotype_index"] = indices
        df["clonotype_id"] = [clonotype_categories[i] for i in indices]
        # Set the index to clonotype_id (each is unique)
        df.index = df["clonotype_id"]
        df = df[[col for col in df.columns if "clonotype" not in col]]
        return df

    else:
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0 when method is 'posterior'")
        # Retrieve a stored global scale (if available) for concentration parameters; default to 1.0
        global_scale = adata.uns.get("tcri_global_scale", 1.0)
        # Form concentration parameters using the global scale
        conc = global_scale * p_c  # shape: (num_clonotypes, num_phenotypes)
        # Sample n_samples draws for each clonotype
        samples = dist.Dirichlet(conc).sample((n_samples,))  # shape: (n_samples, num_clonotypes, num_phenotypes)
        samples_np = samples.cpu().numpy() if samples.device.type != "cpu" else samples.numpy()
        # Rearrange so each row corresponds to one sample draw:
        num_clonotypes, num_pheno = samples_np.shape[1], samples_np.shape[2]
        samples_expanded = samples_np.transpose(1, 0, 2).reshape(-1, num_pheno)
        
        # Expand the clonotype indices accordingly.
        indices = np.arange(num_clonotypes)
        indices_expanded = np.repeat(indices, n_samples)
        clonotype_ids_expanded = [clonotype_categories[i] for i in indices_expanded]
        # Also record a sample identifier for each draw.
        sample_ids = np.tile(np.arange(n_samples), num_clonotypes)
        
        # Build the expanded DataFrame.
        df_samples = pd.DataFrame(samples_expanded, columns=phenotype_categories)
        df_samples["clonotype_index"] = indices_expanded
        df_samples["clonotype_id"] = clonotype_ids_expanded
        df_samples["sample_id"] = sample_ids
        # Set a new index that combines clonotype_id and sample_id (not unique per clonotype)
        df_samples.index = df_samples["clonotype_id"].astype(str) + "_" + df_samples["sample_id"].astype(str)
        del df_samples["sample_id"]
        df_samples = df_samples[[col for col in df_samples.columns if "clonotype" not in col]]
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