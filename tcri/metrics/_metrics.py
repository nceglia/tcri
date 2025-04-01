# Standard library imports
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc

# Third-party imports
from scipy.stats import entropy
from scipy.spatial import distance
import gseapy as gp

# Local imports
from ..preprocessing._preprocessing import joint_distribution

warnings.filterwarnings('ignore')

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
        
    Examples
    --------
    >>> import tcri
    >>> # Calculate phenotypic entropy for a specific TCR clonotype at Day0
    >>> entropy_value = tcri.tl.phenotypic_entropy(adata, "Day0", "CASSQETQYF")
    >>> 
    >>> # Calculate non-normalized entropy with custom parameters
    >>> entropy_value = tcri.tl.phenotypic_entropy(
    ...     adata, "tumor", "CASSLGQAYEQYF", base=10, normalized=False, temperature=0.5
    ... )
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
    """
    Calculate clonality for each phenotype in the dataset.
    
    Clonality is a measure of the inequality in clone size distribution within each phenotype.
    It ranges from 0 (all clones equally represented) to 1 (complete dominance by a single clone).
    Mathematically, it is defined as 1 - (Shannon entropy / maximum possible entropy).
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data with TCR and phenotype information.
        Must have tcri_phenotype_key and tcri_clone_key registered in adata.uns.
        
    Returns
    -------
    dict
        Dictionary mapping phenotype names to their clonality values
        
    Notes
    -----
    The function uses Shannon entropy to measure the diversity of the clone size distribution
    and normalizes it by the maximum possible entropy (log₂ of the number of unique clones).
    The formula is: Clonality = 1 - H/H_max, where H is the Shannon entropy.
    
    Examples
    --------
    >>> import tcri
    >>> # Calculate clonality for all phenotypes
    >>> clonality_dict = tcri.tl.clonality(adata)
    >>> 
    >>> # Print clonality for each phenotype
    >>> for phenotype, clonality_value in clonality_dict.items():
    ...     print(f"{phenotype}: {clonality_value:.3f}")
    """
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

def flux(adata, from_this, to_that, clones=None, temperature=1.0, distance_metric="l1", n_samples=0):
    """
    Calculate phenotypic flux between two covariate conditions for each clonotype.
    
    This function measures how much the phenotype distribution changes for each clonotype
    between two conditions (e.g., before and after treatment). Higher flux values indicate
    greater phenotypic changes for that clonotype.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data with TCR and phenotype information
    from_this : str
        The source covariate label (e.g., "Day0")
    to_that : str
        The target covariate label (e.g., "Day14")
    clones : list, optional
        List of clone IDs to restrict the analysis to. If None, uses all common clones
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions
    distance_metric : str, default="l1"
        The distance metric to use for calculating flux. Options:
        - "l1": Manhattan/L1 distance (sum of absolute differences)
        - "kl": Kullback-Leibler divergence
    n_samples : int, default=0
        Number of samples to use for Monte Carlo estimation. If 0, uses point estimates
        
    Returns
    -------
    pandas.Series
        Series mapping clonotype IDs to their flux values
        
    Notes
    -----
    Only clonotypes present in both conditions are included in the result.
    
    Examples
    --------
    >>> import tcri
    >>> # Calculate L1 phenotypic flux between Day0 and Day14
    >>> flux_values = tcri.tl.flux(adata, "Day0", "Day14")
    >>> 
    >>> # Calculate KL divergence-based flux with specific parameters
    >>> flux_values = tcri.tl.flux(
    ...     adata, "untreated", "treated", 
    ...     distance_metric="kl", temperature=0.5
    ... )
    """
    distances = dict()
    jd_this = joint_distribution(adata, from_this, temperature=temperature, n_samples=n_samples, clones=clones)
    jd_that = joint_distribution(adata, to_that, temperature=temperature, n_samples=n_samples, clones=clones)
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
    covariate,
    temperature=1.0,
    n_samples=0,
    clones=None,
    weighted=False
):
    """
    Compute mutual information between clonotypes and phenotypes.
    
    This function calculates the mutual information (MI) between TCR clonotypes and cell phenotypes 
    for a given covariate condition. Mutual information quantifies how much information one variable 
    provides about the other, effectively measuring the dependency between clonotypes and phenotypes.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data with TCR and phenotype information
    covariate : str
        The covariate label (e.g., "Day0", "tumor") to calculate MI for
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions in the joint distribution
    n_samples : int, default=0
        Number of samples to use for Monte Carlo estimation. If 0, uses point estimates
    clones : list, optional
        List of clone IDs to restrict the analysis to. If None, uses all clones
    weighted : bool, default=False
        If True, weights the MI calculation by clone size, giving more influence to larger clones
        
    Returns
    -------
    float
        The mutual information value between clonotypes and phenotypes
        
    Notes
    -----
    The calculation uses the joint distribution of clonotypes and phenotypes:
    MI(X;Y) = ∑∑ p(x,y) * log₂(p(x,y) / (p(x) * p(y)))
    
    Examples
    --------
    >>> import tcri
    >>> # Calculate MI for a specific timepoint
    >>> mi_value = tcri.tl.mutual_information(adata, "Day0")
    >>> 
    >>> # Calculate weighted MI using sampling
    >>> mi_value = tcri.tl.mutual_information(
    ...     adata, "tumor", temperature=0.5, n_samples=100, weighted=True
    ... )
    """
    # Get joint distribution matrix
    pxy_df = joint_distribution(
        adata=adata,
        covariate_label=covariate,
        temperature=temperature,
        n_samples=n_samples,
        clones=clones,
        weighted=weighted
    )
    pxy = pxy_df.to_numpy()
    
    # Normalize to ensure it's a proper joint distribution
    total = pxy.sum()
    if total < 1e-15:
        return 0.0
    pxy /= total

    # Calculate marginal distributions
    px = pxy.sum(axis=1, keepdims=True)  # p(clonotype)
    py = pxy.sum(axis=0, keepdims=True)  # p(phenotype)

    # Calculate product of marginals p(clonotype)p(phenotype)
    px_py = px @ py
    
    # Calculate MI with numerical stability
    eps = 1e-15
    mask = (pxy > eps)
    mi = np.sum(pxy[mask] * np.log2((pxy[mask] + eps) / (px_py[mask] + eps)))
    return mi