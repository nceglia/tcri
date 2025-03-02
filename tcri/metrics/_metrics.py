from scipy.stats import entropy
import numpy as np
import operator
from scipy.spatial import distance
import pandas as pd
import warnings
import gseapy as gp
import math
import pandas as pd
import torch
from scipy.stats import entropy
import numpy
import torch.nn.functional as F
import scanpy as sc

from ..preprocessing._preprocessing import clone_size, joint_distribution
from ..model._model import TCRIModel

warnings.filterwarnings('ignore')


class TCRIMetrics:
    """
    A helper class that wraps a trained UnifiedTCRTwoLevelModel and provides methods
    for retrieving various distributions and embeddings as well as loading them into
    the AnnData object.
    """
    def __init__(self, model: TCRIModel):
        """
        Initialize the helper with a trained model.

        Parameters
        ----------
        model : UnifiedTCRTwoLevelModel
            A trained instance of the high-level scVI model.
        """
        self.model = model
        self.module = model.module
        self.adata = model.adata
        self.adata_manager = model.adata_manager

    @torch.no_grad()
    def get_joint_distribution(
        self, covariate_label: str, device="cpu", temperature: float = 1.0
    ) -> pd.DataFrame:
        """
        Return a DataFrame for all (clonotype, covariate) combinations where
        the covariate matches the given covariate_label.

        The DataFrame contains columns corresponding to phenotype probabilities,
        as well as clonotype indices and clonotype IDs.
        """
        # A) Convert covariate_label to an integer code.
        covariate_col = self.adata_manager.registry["covariate_col"]
        cov_series = self.adata.obs[covariate_col].astype("category")
        try:
            cov_value = cov_series.cat.categories.get_loc(covariate_label)
        except KeyError:
            raise ValueError(
                f"Covariate label '{covariate_label}' not found in .obs['{covariate_col}']"
            )

        # B) Retrieve p_ct for that covariate.
        p_ct = self.module.get_p_ct().to(device)  # shape: (ct_count, P)
        ct_to_cov = self.module.ct_to_cov.to(device)  # shape: (ct_count,)
        chosen_mask = (ct_to_cov == cov_value)
        chosen_idx = chosen_mask.nonzero(as_tuple=True)[0]
        p_ct_for_cov = p_ct[chosen_mask]  # shape: (N, P)

        # C) Apply temperature scaling.
        p_ct_for_cov = F.softmax(torch.log(p_ct_for_cov + self.module.eps) / temperature, dim=-1)

        # D) Map each row to its clonotype index.
        ct_to_c = self.module.ct_to_c.to(device)
        clone_indices = ct_to_c[chosen_idx].cpu().numpy()

        # E) Build DataFrame.
        p_ct_arr = p_ct_for_cov.cpu().numpy()  # shape: (N, P)
        phenotype_col = self.adata_manager.registry["phenotype_col"]
        phenotype_categories = self.adata.obs[phenotype_col].astype("category").cat.categories
        if len(phenotype_categories) != p_ct_arr.shape[1]:
            raise ValueError("Mismatch: number of phenotype categories does not equal number of columns in p_ct.")
        df = pd.DataFrame(p_ct_arr, columns=list(phenotype_categories))
        df["clonotype_index"] = clone_indices

        # F) Map clonotype index to actual clonotype IDs.
        clonotype_col = self.adata_manager.registry["clonotype_col"]
        clone_series = self.adata.obs[clonotype_col].astype("category")
        clone_categories = clone_series.cat.categories
        df["clonotype_id"] = df["clonotype_index"].apply(lambda i: clone_categories[i])
        return df

    @torch.no_grad()
    def load_cell_phenotype_probs(self, slot_name: str = "X_tcri_phenotypes", batch_size: int = 256):
        """
        Compute the per-cell phenotype probabilities from the model and store them
        in the AnnData object under obsm[slot_name].

        Parameters
        ----------
        slot_name : str, optional
            The name of the obsm slot to store the probabilities (default: "X_tcri_phenotypes").
        batch_size : int, optional
            Batch size for processing the data (default: 256).

        Returns
        -------
        AnnData
            The AnnData object with an updated obsm slot.
        """
        probs = self.model.get_cell_phenotype_probs(batch_size=batch_size)
        self.adata.obsm[slot_name] = probs
        return self.adata

    @torch.no_grad()
    def load_cell_phenotype_assignments(self, obs_column: str, batch_size: int = 256):
        """
        Compute the per-cell phenotype probabilities, determine the most likely
        phenotype (argmax), map that to the actual phenotype label, and store the
        assignment in the AnnData obs under the specified column.

        Parameters
        ----------
        obs_column : str
            The name of the obs column where the assigned phenotype labels will be stored.
        batch_size : int, optional
            Batch size for processing the data (default: 256).

        Returns
        -------
        AnnData
            The AnnData object with the new phenotype assignment in obs[obs_column].
        """
        # Get per-cell phenotype probabilities (shape: [n_cells, n_phenotypes])
        probs = self.model.get_cell_phenotype_probs(batch_size=batch_size)
        # Compute the argmax to get integer assignments for each cell.
        assignments = np.argmax(probs, axis=1)
        # Retrieve the actual phenotype category labels.
        phenotype_col = self.adata_manager.registry["phenotype_col"]
        phenotype_categories = self.adata.obs[phenotype_col].astype("category").cat.categories
        # Map integer assignments to phenotype labels.
        assignment_labels = [phenotype_categories[i] for i in assignments]
        # Load these assignments into the obs dataframe.
        self.adata.obs[obs_column] = assignment_labels
        return self.adata

    @torch.no_grad()
    def load_latent_umap(self, slot_name: str = "X_tcri", batch_size: int = 256):
        """
        Compute the cell-level latent representation from the model, store it in
        adata.obsm[slot_name], and then perform neighbor graph construction and
        UMAP dimensionality reduction using Scanpy.

        Parameters
        ----------
        slot_name : str, optional
            The name of the obsm slot to store the latent representation (default: "X_tcri").
        batch_size : int, optional
            Batch size for processing the data (default: 256).

        Returns
        -------
        AnnData
            The AnnData object with updated obsm[slot_name] and UMAP coordinates in adata.obsm.
        """
        # Retrieve the latent representation (shape: [n_cells, n_latent])
        z = self.model.get_latent_representation(batch_size=batch_size)
        self.adata.obsm[slot_name] = z

        # Run Scanpy's neighbors and UMAP on the new representation.
        sc.pp.neighbors(self.adata, use_rep=slot_name)
        sc.tl.umap(self.adata)

        return self.adata

def phenotypic_entropy_delta(adata, groupby, key, from_this, to_that):
    clone = []
    entropy = []
    resp = []
    for x in set(adata.obs[groupby]):
        response = adata[adata.obs[groupby] == x]
        predata = response[response.obs[key] == from_this]
        joint_distribution(predata)
        postdata = response[response.obs[key] == to_that]
        joint_distribution(postdata)
        preents = phenotypic_entropies(predata,normalized=True)
        postents = phenotypic_entropies(postdata,normalized=True)
        common_tcr = set(preents.keys()).intersection(postents.keys())
        for c in common_tcr:
            diff = postents[c] - preents[c] / (preents[c] +0.00000000000001)
            resp.append(x)
            entropy.append(diff)
            clone.append(c)
    df = pd.DataFrame.from_dict({"Clone":clone,groupby:resp, "Delta Phenotypic Entropy":entropy})
    return df

def get_largest_clonotypes(adata, n=20):
    df = adata.obs[[adata.uns["tcri_clone_key"],"clone_size"]]
    return df.sort_values("clone_size",ascending=False)[adata.uns['tcri_clone_key']].unique().tolist()[:n]

def probability_distribution(adata, method="probabilistic"):
    joint_distribution(adata,method=method)
    if method == "probabilistic":
        joint_distribution(adata) 
        total = adata.uns["joint_distribution"].to_numpy().sum()
        pdist = adata.uns["joint_distribution"].sum(axis=1) / total
    else:
        raise NotImplementedError("In progress!")
    return pdist

def clonotypic_entropy(adata, covariate, phenotype, base=2, normalized=True, temperature=1.):
    logf = lambda x : np.log(x) / np.log(base)
    jd = joint_distribution(adata,covariate,temperature=temperature).T
    res = jd.loc[phenotype].to_numpy()
    cent = entropy(res,base=base)
    if normalized:
        cent = cent / logf(len(res))
    return cent

def phenotypic_entropy(adata, covariate, clonotype, base=2, normalized=True, temperature=1.):
    logf = lambda x : np.log(x) / np.log(base)
    jd = joint_distribution(adata,covariate, temperature=temperature)
    res = jd.loc[clonotype].to_numpy()
    pent = entropy(res,base=base)
    if normalized:
        pent = pent / logf(len(res))
    return pent

def phenotypic_entropies(adata, covariate, base=2, normalized=True, temperature=1.):
    tcr_sequences = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    unique_tcrs = np.unique(tcr_sequences)
    jd = joint_distribution(adata,covariate, temperature=temperature).to_numpy().T
    clonotype_entropies = np.zeros(jd.shape[1])
    max_entropy = np.log2(jd.shape[0])
    for i, clonotype_distribution in enumerate(jd.T):
        normalized_distribution = clonotype_distribution / np.sum(clonotype_distribution)
        epsilon = np.finfo(float).eps
        if normalized:
            clonotype_entropies[i] = -np.sum(normalized_distribution * np.log2(normalized_distribution + epsilon)) / max_entropy
        else:
            clonotype_entropies[i] = -np.sum(normalized_distribution * np.log2(normalized_distribution + epsilon))
    tcr_to_entropy_dict = dict(zip(unique_tcrs, clonotype_entropies))
    return tcr_to_entropy_dict

def clonotypic_entropies(adata, covariate, normalized=True, base=2, temperature=1., decimals=5):
    unique_phenotypes = adata.uns["tcri_phenotype_categories"]
    phenotype_entropies = dict()
    for phenotype in unique_phenotypes:
        cent = clonotypic_entropy(adata, covariate, phenotype, base=base, normalized=normalized, temperature=temperature)
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
        clonality = 1 - entropy(numpy.array(nums),base=2) / numpy.log2(len(nums))
        entropys[phenotype] = numpy.nan_to_num(clonality)
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

def marginal_phenotypic(adata, clones=None, probability=False, method="probabilistic"):
    joint_distribution(adata,method=method)
    if clones == None:
        clones = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    dist = adata.uns["joint_distribution"][clones].to_numpy()
    dist = dist.sum(axis=1)
    if probability:
        dist /= dist.sum()
    return np.nan_to_num(dist).T

def flux(adata, key, from_this, to_that, clones=None, method="probabilistic", distance_metric="l1"):
    this = adata[adata.obs[key] == from_this]
    that = adata[adata.obs[key] == to_that]
    this_clones = set(this.obs[this.uns["tcri_clone_key"]])
    that_clones = set(that.obs[this.uns["tcri_clone_key"]])
    clones = list(this_clones.intersection(that_clones))
    distances = dict()
    joint_distribution(this, method=method)
    joint_distribution(that, method=method)
    for clone in clones:
        this_distribution = marginal_phenotypic(this,clones=[clone], probability=True)
        that_distribution = marginal_phenotypic(that,clones=[clone], probability=True)
        if distance_metric == "l1":
            dist = distance.cityblock(this_distribution, that_distribution)
        elif distance_metric == "dkl":
            dist = entropy(this_distribution, that_distribution, base=2, axis=0)
        distances[clone] = dist
    return distances

def mutual_information(adata, method="probabilistic"):
    joint_distribution(adata,method=method)
    pxy = adata.uns['joint_distribution'].to_numpy()
    pxy = pxy / pxy.sum()
    px = np.sum(pxy, axis=1)
    px = px / px.sum()
    py = np.sum(pxy, axis=0)
    py = py / py.sum()
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log2((pxy[nzs] / px_py[nzs])))
    return mi

