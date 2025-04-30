# Standard library imports
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc
from typing import Optional, List, Union

# Third-party imports
from scipy.stats import entropy
from scipy.spatial import distance
import gseapy as gp

# Local imports
from ..preprocessing._preprocessing import joint_distribution,joint_distribution_posterior

warnings.filterwarnings('ignore')

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
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭─ MI helper (single source of truth) ─────────────────────────────────────╮
def _mi_from_joint(pxy: np.ndarray, normalised: bool, mode: str="average") -> float:
    """
    Mutual information (optionally normalised) from an *already normalised*
    joint table pxy (shape C×P).
    """
    eps = 1e-15
    px  = pxy.sum(1, keepdims=True)
    py  = pxy.sum(0, keepdims=True)
    mi  = np.sum(pxy * np.log2((pxy+eps) / (px @ py + eps)))

    if not normalised:
        return mi
    h_c = -np.sum(px * np.log2(px+eps))
    h_p = -np.sum(py * np.log2(py+eps))
    denom = 0.5*(h_c+h_p) if mode == "average" else min(h_c, h_p)
    return mi/denom if denom > 0 else 0.0
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭─ tiny ASCII histogram (handy in notebooks/SSH) ──────────────────────────╮
def _ascii_hist(samples, bins=25, width=40) -> str:
    hist, edges = np.histogram(samples, bins=bins)
    top = hist.max()
    lines=[]
    for h,e0,e1 in zip(hist, edges[:-1], edges[1:]):
        bar = "█"*int(width*h/top) if top else ""
        lines.append(f"{e0:7.3f}-{e1:7.3f} | {bar}")
    return "\n".join(lines)
# ╰─────────────────


def dkl(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return entropy(p, q) 

def clonotypic_entropy(adata, covariate, phenotype, base=2, normalized=True, temperature=1.0, clones=None, n_samples=0):
    """
    Calculate the clonotypic entropy for a given phenotype and covariate.
    
    This function measures the diversity or uncertainty in the distribution of TCR clonotypes
    within a specific phenotype for a given covariate condition. Higher entropy indicates
    greater clonal diversity within the phenotype.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data with TCR and phenotype information
    covariate : str
        The covariate label (e.g., "Day0", "tumor") to calculate entropy for
    phenotype : str
        The phenotype for which to calculate clonotypic entropy
    base : int, default=2
        The base of the logarithm used for entropy calculation (base 2 gives entropy in bits)
    normalized : bool, default=True
        If True, normalizes the entropy by the maximum possible entropy (log of the number of clones)
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions
    clones : list, optional
        List of clone IDs to restrict the analysis to. If None, uses all clones
    n_samples : int, default=0
        Number of samples to use for Monte Carlo estimation. If 0, uses point estimates
        
    Returns
    -------
    float
        The clonotypic entropy value for the specified phenotype
        
    Examples
    --------
    >>> import tcri
    >>> # Calculate clonotypic entropy for naive T cells at Day0
    >>> entropy_value = tcri.tl.clonotypic_entropy(adata, "Day0", "naive")
    >>> 
    >>> # Calculate non-normalized entropy with custom parameters
    >>> entropy_value = tcri.tl.clonotypic_entropy(
    ...     adata, "tumor", "exhausted", base=10, normalized=False, temperature=0.5
    ... )
    """
    logf = lambda x : np.log(x) / np.log(base)
    jd = joint_distribution(adata, covariate, temperature=temperature, n_samples=n_samples, clones=clones).T
    res = jd.loc[phenotype].to_numpy()
    cent = entropy(res, base=base)
    if normalized:
        cent = cent / logf(len(res))
    return cent

def phenotypic_entropy(adata, covariate, clonotype, base=2, normalized=True, temperature=1.0, n_samples=0, clones=None):
    """
    Calculate the phenotypic entropy for a given clonotype and covariate.
    
    This function measures the diversity or uncertainty in the distribution of phenotypes
    for a specific TCR clonotype within a given covariate condition. Higher entropy indicates
    that the clonotype is associated with a broader range of phenotypes.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data with TCR and phenotype information
    covariate : str
        The covariate label (e.g., "Day0", "tumor") to calculate entropy for
    clonotype : str
        The clonotype ID for which to calculate phenotypic entropy
    base : int, default=2
        The base of the logarithm used for entropy calculation (base 2 gives entropy in bits)
    normalized : bool, default=True
        If True, normalizes the entropy by the maximum possible entropy (log of the number of phenotypes)
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions
    n_samples : int, default=0
        Number of samples to use for Monte Carlo estimation. If 0, uses point estimates
    clones : list, optional
        List of clone IDs to restrict the analysis to. If None, uses all clones
        
    Returns
    -------
    float
        The phenotypic entropy value for the specified clonotype
    """
    logf = lambda x : np.log(x) / np.log(base)
    jd = joint_distribution(adata, covariate, temperature=temperature, n_samples=n_samples, clones=clones).T
    res = jd.loc[clonotype].to_numpy()
    pent = entropy(res, base=base)
    if normalized:
        pent = pent / logf(len(res))
    return pent

def phenotypic_entropies(adata, covariate, base=2, normalized=True, temperature=1., n_samples=0):
    tcr_sequences = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    unique_tcrs = np.unique(tcr_sequences)
    tcr_to_entropy_dict = {
        tcr: phenotypic_entropy(
            adata, 
            covariate, 
            tcr, 
            base=base, 
            normalized=normalized, 
            temperature=temperature,
            n_samples=n_samples
        ) 
        for tcr in unique_tcrs
    }
    return tcr_to_entropy_dict

def clonotypic_entropies(adata, covariate, normalized=True, base=2, temperature=1., decimals=5, n_samples=0):
    unique_phenotypes = adata.uns["tcri_phenotype_categories"]
    phenotype_entropies = dict()
    for phenotype in unique_phenotypes:
        cent = clonotypic_entropy(adata, covariate, phenotype, temperature=temperature)
        phenotype_entropies[phenotype] = np.round(cent,decimals=decimals)
    return phenotype_entropies

def clonality(adata):
    phenotypes = adata.obs[adata.uns["tcri_phenotype_key"]].tolist()
    unique_phenotypes = np.unique(phenotypes)
    entropys = dict()
    df = adata.obs.copy()
    for phenotype in unique_phenotypes:
        pheno_df = df[df[adata.uns['tcri_phenotype_key']]==phenotype]
        clonotypes = pheno_df[adata.uns["tcri_clone_key"]].tolist()
        unique_clonotypes = np.unique(clonotypes)
        nums = []
        for tcr in unique_clonotypes:
            tcrs = pheno_df[pheno_df[adata.uns["tcri_clone_key"]]==tcr]
            nums.append(len(tcrs))
        clonality = 1 - entropy(np.array(nums),base=2) / np.log2(len(nums))
        entropys[phenotype] = np.nan_to_num(clonality)
    return entropys

def clone_fraction(adata, groupby):
    frequencies = dict()
    for group in set(adata.obs[groupby]):
        frequencies[group] = dict()
        sdata = adata[adata.obs[groupby] == group]
        total_cells = len(sdata.obs.index.tolist())
        clones = sdata.obs[sdata.uns["tcri_clone_key"]].tolist()
        for c in set(clones):
            frequencies[group][c] = clones.count(c) / total_cells
    return frequencies

def flux(adata, from_this, to_that, clones=None, temperature=1.0, distance_metric="l1", n_samples=0, weighted=False):
    distances = dict()
    jd_this = joint_distribution(adata, from_this, temperature=temperature, n_samples=n_samples, clones=clones,weighted=weighted)
    jd_that = joint_distribution(adata, to_that, temperature=temperature, n_samples=n_samples, clones=clones,weighted=weighted)
    common_indices = jd_this.index.intersection(jd_that.index)
    if distance_metric == "l1":
        distances = (jd_this.loc[common_indices] - jd_that.loc[common_indices]).abs().sum(axis=1)
    else:
        distances = pd.Series(
            {idx: dkl(jd_this.loc[idx], jd_that.loc[idx]) for idx in common_indices}
        )
    return distances

def mutual_information(
    adata,
    covariate: str,
    *,
    temperature: float = 1.0,
    n_samples: int = 0,
    clones: Optional[List[str]] = None,
    weighted: bool = False,
    normalised: bool = True,
    normalise_mode: str = "average",
    posterior: bool = True,
    combine_with_logits: bool = True,
    verbose: bool = True,
    graph: bool = False,
) -> Union[float, np.ndarray]:
    """
    MI between clonotype and phenotype at one covariate value.

    posterior=True  → Dirichlet draw of p_ct (+ optional logits)  
    posterior=False → prior-only (uses your original joint_distribution).
    """

    if verbose:
        print(f"{BOLD}{MAGENT}📊  MI for '{covariate}'{RESET}")
        _info("posterior",  posterior)
        _info("weighted",   weighted)
        _info("n_samples",  n_samples)

    # helper to obtain *one* joint table
    def _get_df(silent_flag: bool):
        if posterior:
            return joint_distribution_posterior(
                adata,
                covariate_label     = covariate,
                temperature         = temperature,
                clones              = clones,
                weighted            = weighted,
                combine_with_logits = combine_with_logits,
                silent              = silent_flag,
            )
        else:
            # fall back to prior-only version (not shown here)
            raise NotImplementedError("prior-only joint_distribution not included")

    # ---------- single draw ----------------------------------------
    if n_samples == 0:
        df  = _get_df(silent_flag=not verbose)
        pxy = df.to_numpy().astype(float)
        pxy /= pxy.sum()
        mi  = _mi_from_joint(pxy, normalised, normalise_mode)
        _ok("computed MI",           quiet=not verbose)
        _info("value", f"{mi:.4f}",  quiet=not verbose)
        return mi

    # ---------- multiple draws -------------------------------------
    mi_samples = np.empty(n_samples, dtype=float)
    iter_bar   = tqdm(range(n_samples), disable=not verbose, leave=False,
                      desc=f"sampling MI [{covariate}]")
    for i in iter_bar:
        df = _get_df(silent_flag=True)
        pxy = df.to_numpy().astype(float)
        pxy /= pxy.sum()
        mi_samples[i] = _mi_from_joint(pxy, normalised, normalise_mode)

    _ok("computed MI for all samples", quiet=not verbose)
    _info("mean ± sd",
          f"{mi_samples.mean():.4f} ± {mi_samples.std():.4f}",
          quiet=not verbose)
    _info("95% CI",
          f"[{np.percentile(mi_samples,2.5):.4f}, "
          f"{np.percentile(mi_samples,97.5):.4f}]",
          quiet=not verbose)

    if graph and n_samples > 1 and verbose:
        print(f"{DIM}\nASCII histogram of MI posterior:\n"
              f"{_ascii_hist(mi_samples)}{RESET}")

    return mi_samples