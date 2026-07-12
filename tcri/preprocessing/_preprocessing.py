from scipy.stats import entropy
from .. import _keys as K
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

from .._console import _ok, _info, _warn, _fin

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
    phenotype_prob_slot=K.X_PROBABILITIES,
    phenotype_assignment_obs=K.PHENOTYPE,
    latent_slot=K.X_TCRI,
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
    adata.uns[K.P_CT]      = model.module.get_p_ct().cpu().numpy()
    adata.uns[K.CT_TO_COV] = model.module.ct_to_cov.cpu().numpy()
    adata.uns[K.CT_TO_C]   = model.module.ct_to_c.cpu().numpy()
    adata.uns[K.LOCAL_SCALE] = model.module.local_scale
    _ok("stored hierarchical priors")
    for k in (K.P_CT,K.CT_TO_COV,K.CT_TO_C):
        _info(f"uns['{k}']", np.shape(adata.uns[k]))

    # 2) metadata --------------------------------------------------------
    meta = {
        "covariate_col": model.adata_manager.registry["covariate_col"],
        "clone_col":     model.adata_manager.registry["clonotype_col"],
        "phenotype_col": model.adata_manager.registry["phenotype_col"],
        "batch_col":     model.adata_manager.registry["batch_col"],
    }
    adata.uns[K.METADATA] = meta
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
    adata.uns[K.CT_ARRAY] = ct_arr
    cov_arr = model.module.ct_to_cov.cpu().numpy()[ct_arr]
    adata.uns[K.COV_ARRAY] = cov_arr
    _ok("stored per-cell ct / cov indices")

    # 3) latent means ----------------------------------------------------
    z = model.get_latent_representation(batch_size=batch_size).astype("float32")
    adata.obsm[latent_slot] = z
    _ok("stored latent means")
    _info(f"obsm['{latent_slot}']", z.shape)

    # 4) logits & log-posterior -----------------------------------------
    cls_logits, prior_log = _compute_logits_and_prior(model, adata, batch_size)
    if store_logits:
        adata.obsm[K.X_LOGITS] = cls_logits
        _info("obsm['X_tcri_logits']", cls_logits.shape)
    if store_logposterior:
        adata.obsm[K.X_LOGPOSTERIOR] = cls_logits + prior_log
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
        categories=adata.uns[K.PHENOTYPE_CATEGORIES],
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

    meta      = adata.uns[K.METADATA];  cov_col = meta["covariate_col"]
    clone_col = meta["clone_col"];           ph_cats  = adata.uns[K.PHENOTYPE_CATEGORIES]
    cov_idx   = adata.uns[K.COVARIATE_CATEGORIES].index(covariate_label)

    ct_per_cell  = adata.uns[K.CT_ARRAY]
    cov_per_cell = adata.uns[K.COV_ARRAY]
    clone_labels = adata.obs[clone_col].values

    # Guard against filtered AnnData (view or subset copy). The per-cell arrays in
    # .uns are stored in the original full-cell space and are NOT subset when adata
    # is sliced, whereas .obs/.obsm ARE subset. Indexing one with positions derived
    # from the other then silently misaligns cells (Notion #4). Fail loudly instead
    # of returning wrong numbers.
    n_obs = adata.n_obs
    if len(ct_per_cell) != n_obs or len(cov_per_cell) != n_obs:
        raise ValueError(
            "joint_distribution_posterior received an AnnData whose per-cell "
            f"registration arrays (len {len(ct_per_cell)}) do not match adata.n_obs "
            f"({n_obs}). This happens when the function is called on a filtered "
            "AnnData view or subset: the 'tcri_*_array_for_cells' arrays in .uns "
            "remain in the original full-cell space while .obs/.obsm are subset, so "
            "cell indices silently misalign. Re-run register_model(...) on the "
            "filtered AnnData, or pass the full object and filter with `clones=`."
        )

    idx_cov = np.nonzero(cov_per_cell == cov_idx)[0]
    if clones is not None:
        idx_cov = idx_cov[np.isin(clone_labels[idx_cov], clones)]

    _ok(f"selected {len(idx_cov):,} cells", silent)

    p_ct_mean   = torch.tensor(adata.uns[K.P_CT])
    local_scale = adata.uns.get(K.LOCAL_SCALE, 1.0)
    bad = ~torch.isfinite(p_ct_mean)
    if bad.any():
        n_phen = p_ct_mean.shape[1]
        p_ct_mean = torch.where(bad, torch.ones_like(p_ct_mean) / n_phen, p_ct_mean)
    p_ct_sample = Dirichlet(local_scale * p_ct_mean + 1e-8).sample().numpy()
    _ok("sampled one draw from posterior p_ct", silent)

    if combine_with_logits:
        if K.X_LOGITS not in adata.obsm:
            raise RuntimeError("X_tcri_logits missing in adata.")
        logits     = adata.obsm[K.X_LOGITS][idx_cov]
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


def joint_distribution(
    adata, 
    covariate_label: str, 
    temperature: float = 1.0, 
    n_samples: int = 0, 
    clones=None,
    weighted: bool = False,
) -> pd.DataFrame:

    p_ct = torch.tensor(adata.uns[K.P_CT])
    ct_to_cov = torch.tensor(adata.uns[K.CT_TO_COV])
    ct_to_c = torch.tensor(adata.uns[K.CT_TO_C])

    covariate_categories = adata.uns[K.COVARIATE_CATEGORIES]
    phenotype_categories = adata.uns[K.PHENOTYPE_CATEGORIES]
    clonotype_categories = adata.uns[K.CLONOTYPE_CATEGORIES]

    metadata = adata.uns[K.METADATA]
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
    ct_array_for_cells = adata.uns[K.CT_ARRAY]
    cov_array_for_cells = adata.uns[K.COV_ARRAY]

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
        local_scale = adata.uns.get(K.LOCAL_SCALE, 1.0)
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





def clone_size(adata, key_added=K.CLONE_SIZE, return_counts=False):
    tcr_key = adata.uns["tcri_clone_key"]
    res = np.unique(adata.obs[tcr_key].tolist(), return_counts=True)
    clone_sizes = dict(zip(res[0],res[1]))
    sizes = []
    for clone in adata.obs[tcr_key]:
        sizes.append(clone_sizes[clone])
    adata.obs[key_added] = sizes
    if return_counts:
        return clone_sizes


