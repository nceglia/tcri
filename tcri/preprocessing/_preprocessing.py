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

@torch.no_grad()
def register_model(adata, model,
                        phenotype_prob_slot="X_tcri_phenotypes",
                        phenotype_assignment_obs="tcri_phenotype",
                        latent_slot="X_tcri",
                        batch_size=256):
    # --- Store core model outputs ---
    adata.uns["tcri_p_ct"] = model.module.get_p_ct().cpu().numpy()  # (num_ct_pairs, num_phenotypes)
    adata.uns["tcri_ct_to_cov"] = model.module.ct_to_cov.cpu().numpy()  # (num_ct_pairs,)
    adata.uns["tcri_ct_to_c"] = model.module.ct_to_c.cpu().numpy()  # (num_ct_pairs,)
    # adata.uns["tcri_p_c"] = model.module.get_p_c.cpu().numpy()  # (num_ct_pairs,)

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
    
    # --- Store the local scale for Dirichlet concentration (needed for posterior sampling) ---
    adata.uns["tcri_local_scale"] = model.module.local_scale

    # --- Compute and store per-cell phenotype probabilities ---
    phenotype_probs = model.get_cell_phenotype_probs(batch_size=batch_size)
    adata.obsm[phenotype_prob_slot] = phenotype_probs

    # --- Compute and store phenotype assignments ---
    assignments = np.argmax(phenotype_probs, axis=1)
    phenotype_categories = adata.obs[pheno_col].astype("category").cat.categories
    assignment_labels = [phenotype_categories[i] for i in assignments]
    adata.obs[phenotype_assignment_obs] = assignment_labels

    # --- Compute and store latent representation and UMAP embedding ---
    latent_z = model.get_latent_representation(batch_size=batch_size)
    adata.obsm[latent_slot] = latent_z

    return adata

def joint_distribution(
    adata, 
    covariate_label: str, 
    temperature: float = 1.0, 
    n_samples: int = 0, 
    clones=None
) -> pd.DataFrame:
    # Retrieve stored tensors and metadata
    p_ct = torch.tensor(adata.uns["tcri_p_ct"])  # shape: (num_ct_pairs, num_phenotypes)
    ct_to_cov = torch.tensor(adata.uns["tcri_ct_to_cov"])  # shape: (num_ct_pairs,)
    ct_to_c = torch.tensor(adata.uns["tcri_ct_to_c"])  # shape: (num_ct_pairs,)

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

    # E) Build DataFrame for point estimates (if method is "point" or no sampling requested)
    p_ct_arr = p_ct_for_cov.numpy()
    if n_samples == 0:
        df = pd.DataFrame(p_ct_arr, columns=phenotype_categories)
        df["clonotype_index"] = clone_indices
        df["clonotype_id"] = [clonotype_categories[i] for i in clone_indices]
        if clones!=None:
            df = df[df["clonotype_id"].isin(clones)]
        # Set the index as the clonotype_id (it will be unique in this case)
        df.index = df["clonotype_id"]
        # Remove the extra columns if desired
        df = df[[col for col in df.columns if "clonotype" not in col]]
        return df

    # F) For "posterior" method, sample from the Dirichlet posterior and expand rows
    else:
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0 when method is 'posterior'")
        # Retrieve stored local_scale (set in store_model_outputs)
        local_scale = adata.uns.get("tcri_local_scale", 1.0)
        # Form Dirichlet concentration parameters using temperature-scaled probabilities
        conc = local_scale * p_ct_for_cov  # shape: (num_chosen, num_phenotypes)
        # Sample n_samples draws for each clonotype
        samples = dist.Dirichlet(conc).sample((n_samples,))  # shape: (n_samples, num_chosen, num_phenotypes)
        # Convert samples to numpy and reshape so each row corresponds to one sample draw
        samples_np = samples.cpu().numpy() if samples.device.type != "cpu" else samples.numpy()
        num_chosen, num_pheno = samples_np.shape[1], samples_np.shape[2]
        # Transpose so shape becomes (num_chosen, n_samples, num_pheno) then reshape to (num_chosen*n_samples, num_pheno)
        samples_expanded = samples_np.transpose(1, 0, 2).reshape(-1, num_pheno)
        
        # Expand clonotype indices accordingly: repeat each clonotype n_samples times
        clonotype_indices_expanded = np.repeat(clone_indices, n_samples)
        clonotype_ids_expanded = [clonotype_categories[i] for i in clonotype_indices_expanded]
        
        # Optionally, add a sample identifier per clonotype
        sample_ids = np.tile(np.arange(n_samples), num_chosen)
        
        # Build the expanded DataFrame
        df_samples = pd.DataFrame(samples_expanded, columns=phenotype_categories)
        df_samples["clonotype_index"] = clonotype_indices_expanded
        df_samples["clonotype_id"] = clonotype_ids_expanded
        df_samples["sample_id"] = sample_ids
        if clones!=None:
            df_samples = df_samples[df_samples["clonotype_id"].isin(clones)]
        # Set a new index that combines clonotype_id and sample_id to ensure uniqueness
        df_samples.index = df_samples["clonotype_id"].astype(str) + "_" + df_samples["sample_id"].astype(str)
        del df_samples["sample_id"]
        df_samples = df_samples[[col for col in df_samples.columns if "clonotype" not in col]]
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