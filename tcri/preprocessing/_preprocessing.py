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
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
import umap
import numpy as np, pandas as pd, torch, umap
from tqdm.auto import tqdm
from scvi import REGISTRY_KEYS

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
import numpy as np
import pandas as pd
from   scipy.special import softmax
from   torch.distributions import Dirichlet
import torch
import warnings

warnings.filterwarnings('ignore')

# ------------ simple ANSI helpers ------------ #
RESET  = "\x1b[0m"
BOLD   = "\x1b[1m"
DIM    = "\x1b[2m"
GREEN  = "\x1b[32m"
CYAN   = "\x1b[36m"
MAGENT = "\x1b[35m"
# ╭─ colour / pretty-print helpers ─────────────────────────────────────────╮
RESET  = "\x1b[0m";  BOLD  = "\x1b[1m";  DIM  = "\x1b[2m"
GRN = "\x1b[32m";  CYN = "\x1b[36m";  MAG = "\x1b[35m";  YLW = "\x1b[33m"; RED = "\x1b[31m"

def _ok(msg:str, quiet=False):    # success mark
    if not quiet: print(f"{GRN}✅ {msg}{RESET}")

def _info(key:str, txt:str, quiet=False):       # key-value info line
    if not quiet: print(f"   {CYN}🎯 {key:<22}{DIM}{txt}{RESET}")

def _warn(msg:str, quiet=False):   # warning line
    if not quiet: print(f"{YLW}⚠️  {msg}{RESET}")

def _fin(quiet=False):             # final flourish
    if not quiet: print(f"{MAG}✨  Done!{RESET}")

def _ascii_hist(samples, bins=25, width=40) -> str:
    hist, edges = np.histogram(samples, bins=bins)
    top = hist.max()
    lines=[]
    for h,e0,e1 in zip(hist, edges[:-1], edges[1:]):
        bar = "█"*int(width*h/top) if top else ""
        lines.append(f"{e0:7.3f}-{e1:7.3f} | {bar}")
    return "\n".join(lines)


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
    print("\t...classifying phenotypes...\n")
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

# ------------ helper to extract logits -------- #
@torch.no_grad()
def _compute_logits_and_prior(model, adata, batch_size=256, eps=1e-8):
    device   = next(model.module.parameters()).device
    loader   = model._make_data_loader(adata=adata, batch_size=batch_size)
    ct_arr   = model.module.ct_array.to(device)
    p_ct     = model.module.get_p_ct().to(device)

    logits_buf, prior_buf = [], []
    start = 0
    for tensors in tqdm(loader, desc="extracting logits", leave=False):
        x = tensors[REGISTRY_KEYS.X_KEY].to(device)
        b = tensors[REGISTRY_KEYS.BATCH_KEY].long().to(device)
        n = x.size(0)

        z_loc, _, _ = model.module.encoder(x, b)
        logits      = model.module.classifier(z_loc)
        prior_log   = torch.log(p_ct[ct_arr[start:start+n]] + eps)

        logits_buf.append(logits.cpu())
        prior_buf.append(prior_log.cpu())
        start += n

    return (torch.cat(logits_buf).numpy().astype("float32"),
            torch.cat(prior_buf).numpy().astype("float32"))

# ------------ main routine -------------------- #
@torch.no_grad()
def register_model(
    adata, model,
    phenotype_prob_slot="X_tcri_probabilities",
    phenotype_assignment_obs="tcri_phenotype",
    latent_slot="X_tcri",
    batch_size=256,
    store_logits=True,
    store_logposterior=True,
    compute_umap=False,
    umap_n_neighbors=50,
    umap_min_dist=1e-3,
    umap_metric="euclidean",
    umap_random_state=42,
    umap_output_metric="euclidean",
    clonotype_key="trb_unique",
):
    print(f"{BOLD}{MAGENT}🔗  Registering TCRi model outputs …{RESET}")

    # 1) priors & arrays -------------------------------------------------
    adata.uns["tcri_p_ct"]      = model.module.get_p_ct().cpu().numpy()
    adata.uns["tcri_ct_to_cov"] = model.module.ct_to_cov.cpu().numpy()
    adata.uns["tcri_ct_to_c"]   = model.module.ct_to_c.cpu().numpy()
    adata.uns["tcri_local_scale"] = model.module.local_scale
    _ok("stored hierarchical priors")
    for k in ("tcri_p_ct","tcri_ct_to_cov","tcri_ct_to_c"):
        _info(f"uns['{k}']", np.shape(adata.uns[k]))

    # 2) metadata --------------------------------------------------------
    meta = {
        "covariate_col": model.adata_manager.registry["covariate_col"],
        "clone_col":     model.adata_manager.registry["clonotype_col"],
        "phenotype_col": model.adata_manager.registry["phenotype_col"],
        "batch_col":     model.adata_manager.registry["batch_col"],
    }
    adata.uns["tcri_metadata"] = meta
    _ok("stored metadata dictionary")

    # categories
    for key, col in (("covariate","covariate_col"),
                     ("clonotype","clone_col"),
                     ("phenotype","phenotype_col")):
        cats = adata.obs[meta[col]].astype("category").cat.categories.tolist()
        adata.uns[f"tcri_{key}_categories"] = cats
        _info(f"uns['tcri_{key}_categories']", len(cats))

    # per-cell ct / cov arrays
    ct_arr = model.module.ct_array.cpu().numpy()
    adata.uns["tcri_ct_array_for_cells"] = ct_arr
    cov_arr = model.module.ct_to_cov.cpu().numpy()[ct_arr]
    adata.uns["tcri_cov_array_for_cells"] = cov_arr
    _ok("stored per-cell ct / cov indices")

    # 3) latent means ----------------------------------------------------
    z = model.get_latent_representation(batch_size=batch_size).astype("float32")
    adata.obsm[latent_slot] = z
    _ok("stored latent means")
    _info(f"obsm['{latent_slot}']", z.shape)

    # 4) logits & log-posterior -----------------------------------------
    cls_logits, prior_log = _compute_logits_and_prior(model, adata, batch_size)
    if store_logits:
        adata.obsm["X_tcri_logits"] = cls_logits
        _info("obsm['X_tcri_logits']", cls_logits.shape)
    if store_logposterior:
        adata.obsm["X_tcri_logposterior"] = cls_logits + prior_log
        _info("obsm['X_tcri_logposterior']", cls_logits.shape)
    _ok("computed logits & additive log-posterior")

    # 5) probabilities & hard labels ------------------------------------
    if phenotype_prob_slot not in adata.obsm:
        from scipy.special import softmax
        probs = softmax(cls_logits + prior_log, axis=1).astype("float32")
        adata.obsm[phenotype_prob_slot] = probs
        _info(f"obsm['{phenotype_prob_slot}']", probs.shape)

    adata.obs[phenotype_assignment_obs] = pd.Categorical.from_codes(
        adata.obsm[phenotype_prob_slot].argmax(1),
        categories=adata.uns["tcri_phenotype_categories"],
    )
    _ok("stored probabilities and hard labels")

    # 6) optional UMAP ---------------------------------------------------
    if compute_umap:
        print(f"{CYAN}🗺️  computing UMAP …{RESET}")
        reducer = umap.UMAP(
            n_neighbors=umap_n_neighbors, min_dist=umap_min_dist,
            metric=umap_metric, random_state=umap_random_state,
            output_metric=umap_output_metric,
        )
        adata.obsm["X_umap"] = reducer.fit_transform(z)
        _info("obsm['X_umap']", adata.obsm["X_umap"].shape)
    
    register_phenotype_key(adata,phenotype_assignment_obs)
    register_clonotype_key(adata,clonotype_key)
    
    print(f"{MAGENT}✨  All TCRi artefacts registered!{RESET}")
    return adata

def joint_distribution_posterior(
        adata, covariate_label, *, temperature=1.0, clones=None,
        weighted=False, combine_with_logits=True, precision=3, silent=False):

    meta      = adata.uns["tcri_metadata"];  cov_col = meta["covariate_col"]
    clone_col = meta["clone_col"];           ph_cats  = adata.uns["tcri_phenotype_categories"]
    cov_idx   = adata.uns["tcri_covariate_categories"].index(covariate_label)

    ct_per_cell  = adata.uns["tcri_ct_array_for_cells"]
    cov_per_cell = adata.uns["tcri_cov_array_for_cells"]
    clone_labels = adata.obs[clone_col].values

    idx_cov = np.nonzero(cov_per_cell == cov_idx)[0]
    if clones is not None:
        idx_cov = idx_cov[np.isin(clone_labels[idx_cov], clones)]

    _ok(f"selected {len(idx_cov):,} cells", silent)

    p_ct_mean   = torch.tensor(adata.uns["tcri_p_ct"])
    local_scale = adata.uns.get("tcri_local_scale", 1.0)
    p_ct_sample = Dirichlet(local_scale * p_ct_mean + 1e-8).sample().numpy()
    _ok("sampled one draw from posterior p_ct", silent)

    if combine_with_logits:
        if "X_tcri_logits" not in adata.obsm:
            raise RuntimeError("X_tcri_logits missing in adata.")
        logits     = adata.obsm["X_tcri_logits"][idx_cov]
        ct_idx_sel = ct_per_cell[idx_cov]
        log_prior  = np.log(p_ct_sample[ct_idx_sel] + 1e-8)
        probs_cell = softmax((logits + log_prior)/temperature, axis=1)
        _ok("combined logits with sampled prior", silent)
    else:
        probs_cell = p_ct_sample[ct_per_cell[idx_cov]]
        _ok("using sampled p_ct only", silent)

    df = (pd.DataFrame(probs_cell, columns=ph_cats,
                       index=clone_labels[idx_cov])
          .groupby(level=0).sum().astype(float))
    if not weighted:
        df = df.div(df.sum(1), axis=0).fillna(0.0)
    if clones is not None:
        df = df.reindex(clones).fillna(0.0)

    _info("resulting DataFrame", df.shape, silent); _fin(silent)
    return df.round(precision)

def remove_meaningless_genes(adata, include_mt=True, include_rp=True, include_mtrn=True, include_hsp=True, include_tcr=True):
    genes = [x for x in adata.var.index.tolist() if "RIK" not in x.upper()]
    genes = [x for x in genes if "GM" not in x]
    genes = [x for x in genes if "-" not in x or "HLA" in x]
    genes = [x for x in genes if "." not in x or "HLA" in x]
    genes = [x for x in genes if "LINC" not in x.upper()]
    if include_mtrn:
        genes = [x for x in adata.var.index.tolist() if "MTRN" not in x]
    if include_hsp:
        genes = [x for x in adata.var.index.tolist() if "HSP" not in x]
    if include_mt:
        genes = [x for x in genes if "MT-" not in x.upper()]
    if include_rp:
        genes = [x for x in genes if "RP" not in x.upper()]
    if include_tcr:
        genes = [x for x in genes if "TRAV" not in x]
        genes = [x for x in genes if "TRAJ" not in x]
        genes = [x for x in genes if "TRAD" not in x]

        genes = [x for x in genes if "TRBV" not in x]
        genes = [x for x in genes if "TRBJ" not in x]
        genes = [x for x in genes if "TRBD" not in x]

        genes = [x for x in genes if "TRGV" not in x]
        genes = [x for x in genes if "TRGJ" not in x]
        genes = [x for x in genes if "TRGD" not in x]

        genes = [x for x in genes if "TRDV" not in x]
        genes = [x for x in genes if "TRDJ" not in x]
        genes = [x for x in genes if "TRDD" not in x]
    adata = adata[:,genes]
    return adata.copy()

def joint_distribution(
    adata, 
    covariate_label: str, 
    temperature: float = 1.0, 
    n_samples: int = 0, 
    clones=None,
    weighted: bool = False,
) -> pd.DataFrame:

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
    import tqdm
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