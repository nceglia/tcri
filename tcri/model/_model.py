from typing import Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import dirichlet
from scipy.cluster.hierarchy import dendrogram, linkage
import anndata
import scanpy as sc
import torch
from torch.distributions import Categorical, kl_divergence
from torch.utils.data import DataLoader, Dataset
import pyro
import pyro.distributions as dist
from pyro.distributions import Dirichlet, Gamma, constraints
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import warnings


class TCRCooccurrenceDataset(Dataset):
    """
    A more general dataset class that can store two separate covariates:
    - The primary covariate (covariate_label), e.g. 'patient'
    - An optional second covariate (second_covariate_label), e.g. 'timepoint' or 'site'
    """
    def __init__(
        self, 
        adata: anndata.AnnData, 
        tcr_label: str, 
        covariate_label: str, 
        phenotype_probs: np.ndarray,
        min_expression: float = 1e-10,
        second_covariate_label: Optional[str] = None  # <--- renamed from timepoint_label
    ):
        """
        Initialize the dataset with phenotype probabilities.
        
        Args:
            adata: AnnData object containing gene expression data
            tcr_label: Column name in adata.obs for TCR labels
            covariate_label: Column name in adata.obs for 'patient' or other main covariate
            phenotype_probs: Matrix of shape (n_phenotypes, n_cells) with probabilities
            min_expression: Minimum expression value to avoid numerical issues
            second_covariate_label: (Optional) Another covariate in adata.obs (time, site, etc.)
        """
        # ----------------------------------------------------------------
        # 1) Validate input columns
        # ----------------------------------------------------------------
        if tcr_label not in adata.obs.columns:
            raise ValueError(f"TCR label '{tcr_label}' not found in adata.obs")
        if covariate_label not in adata.obs.columns:
            raise ValueError(f"Covariate label '{covariate_label}' not found in adata.obs")
        if second_covariate_label is not None and second_covariate_label not in adata.obs.columns:
            raise ValueError(f"Second covariate label '{second_covariate_label}' not found in adata.obs")

        # ----------------------------------------------------------------
        # 2) Phenotype probabilities
        # ----------------------------------------------------------------
        self.phenotype_probs = torch.tensor(phenotype_probs, dtype=torch.float32)
        
        # ----------------------------------------------------------------
        # 3) Remove cells with missing TCR or covariates
        # ----------------------------------------------------------------
        if adata.obs[tcr_label].isna().any():
            n_missing = adata.obs[tcr_label].isna().sum()
            warnings.warn(f"Found {n_missing} missing TCR labels. Removing those cells.")
            adata = adata[~adata.obs[tcr_label].isna()].copy()
        if adata.obs[covariate_label].isna().any():
            n_missing = adata.obs[covariate_label].isna().sum()
            warnings.warn(f"Found {n_missing} missing covariate labels. Removing those cells.")
            adata = adata[~adata.obs[covariate_label].isna()].copy()
        if second_covariate_label is not None and adata.obs[second_covariate_label].isna().any():
            n_missing = adata.obs[second_covariate_label].isna().sum()
            warnings.warn(f"Found {n_missing} missing '{second_covariate_label}' labels. Removing those cells.")
            adata = adata[~adata.obs[second_covariate_label].isna()].copy()
            
        # ----------------------------------------------------------------
        # 4) Process expression matrix => row normalization
        # ----------------------------------------------------------------
        cooccurrence_matrix = adata.to_df().copy()
        cooccurrence_matrix += min_expression
        row_sums = cooccurrence_matrix.sum(axis=1)
        cooccurrence_matrix = cooccurrence_matrix.div(row_sums, axis=0).clip(lower=min_expression)
        row_sums = cooccurrence_matrix.sum(axis=1)
        cooccurrence_matrix = cooccurrence_matrix.div(row_sums, axis=0)

        # Validate normalized matrix
        if not np.allclose(cooccurrence_matrix.sum(axis=1), 1.0, rtol=1e-5):
            raise ValueError("Normalization failed: row sums are not ~1")

        # Convert to tensor
        matrix_values = cooccurrence_matrix.values
        if np.isnan(matrix_values).any() or np.isinf(matrix_values).any():
            raise ValueError("Invalid values in normalized expression matrix")
        self.matrix = torch.tensor(matrix_values, dtype=torch.float32)
        
        # ----------------------------------------------------------------
        # 5) Process TCR and covariate labels
        # ----------------------------------------------------------------
        tcrs = pd.Categorical(adata.obs[tcr_label])
        covariates = pd.Categorical(adata.obs[covariate_label])

        # Build mappings to integer indices
        self.tcr_mapping = {cat: idx for idx, cat in enumerate(tcrs.categories)}
        self.covariate_mapping = {cat: idx for idx, cat in enumerate(covariates.categories)}
        
        tcr_indices = [self.tcr_mapping[cat] for cat in tcrs]
        covariate_indices = [self.covariate_mapping[cat] for cat in covariates]

        self.tcrs = torch.tensor(tcr_indices, dtype=torch.long)
        self.covariates = torch.tensor(covariate_indices, dtype=torch.long)

        # ----------------------------------------------------------------
        # 6) Second covariate (optional)
        # ----------------------------------------------------------------
        if second_covariate_label is not None:
            cov2_cats = pd.Categorical(adata.obs[second_covariate_label])
            self.covariate2_mapping = {cat: idx for idx, cat in enumerate(cov2_cats.categories)}
            cov2_indices = [self.covariate2_mapping[cat] for cat in cov2_cats]
            self.covariate2 = torch.tensor(cov2_indices, dtype=torch.long)
            self.C2 = len(self.covariate2_mapping)
        else:
            self.covariate2 = None
            self.covariate2_mapping = {}
            self.C2 = 0

        # ----------------------------------------------------------------
        # 7) Dataset dimensions
        # ----------------------------------------------------------------
        self.K = len(self.tcr_mapping)          # number of TCRs
        self.C = len(self.covariate_mapping)    # number of main covariates (patients)
        self.D = self.matrix.shape[1]           # number of genes
        self.n_phenotypes = phenotype_probs.shape[0]  # number of phenotypes

        self._validate_tensors()

    # ----------------------------------------------------------------
    # Utility checks
    # ----------------------------------------------------------------
    def _validate_tensors(self) -> None:
        """Validate all tensors including phenotype probabilities."""
        # 1) Expression matrix
        if torch.isnan(self.matrix).any() or torch.isinf(self.matrix).any():
            raise ValueError("Invalid values in expression matrix tensor")
        row_sums = self.matrix.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5):
            raise ValueError("Expression matrix rows do not sum to 1")
        
        # 2) Phenotype probabilities
        if torch.isnan(self.phenotype_probs).any() or torch.isinf(self.phenotype_probs).any():
            raise ValueError("Invalid values in phenotype probabilities")
        col_sums = self.phenotype_probs.sum(dim=0)
        if not torch.allclose(col_sums, torch.ones_like(col_sums), rtol=1e-5):
            raise ValueError("Phenotype probabilities do not sum to 1 across phenotypes")

        # 3) TCR / Covariates index checks
        if self.tcrs.max() >= self.K or self.covariates.max() >= self.C:
            raise ValueError("Invalid TCR or primary covariate indices found")
        if self.covariate2 is not None and self.covariate2.max() >= self.C2:
            raise ValueError("Invalid second covariate indices found")

    # ----------------------------------------------------------------
    # Basic dataset interface
    # ----------------------------------------------------------------
    def __len__(self) -> int:
        return self.matrix.shape[0]
        
    def __getitem__(self, idx: int):
        """
        Retrieve data for a single cell. 
        We'll always return:
          - gene_probs
          - tcr_idx
          - primary covariate idx
          - phenotype_probs
        If second covariate is present, return it; otherwise return -1.
        """
        gene_probs = self.matrix[idx]
        tcr_idx = self.tcrs[idx]
        cov_idx = self.covariates[idx]
        phen_probs = self.phenotype_probs[:, idx]

        if self.covariate2 is not None:
            cov2_idx = self.covariate2[idx]
        else:
            cov2_idx = torch.tensor(-1, dtype=torch.long)

        return gene_probs, tcr_idx, cov_idx, cov2_idx, phen_probs


class JointProbabilityDistribution:
    def __init__(
            self,
            adata: anndata.AnnData,
            tcr_label: str,
            covariate_label: str,
            n_phenotypes: int,
            phenotype_prior: Optional[Union[str, np.ndarray]] = None,
            second_covariate_label: Optional[str] = None,  # <--- new optional covariate
            marker_genes: Optional[Dict[int, List[str]]] = None,
            marker_prior: float = 2.0,
            batch_size: int = 32,
            learning_rate: float = 1e-4,
            # Additional hyperparams for the log-offset model:
            covariate_logoffset_scale: float = 1.0,   # e.g. sigma for Normal(0, sigma)
            second_covariate_logoffset_scale: float = 1.0,
            gene_profile_prior_strength: float = 5.0,
            gene_profile_prior_offset: float = 0.5,
            persistence_factor: float = 100.0,
            consistency_weight: float = 0.1,
            gene_concentration: float = 100.,
            # etc...
        ):
        """
        A simplified example where we have two covariates:
          1) covariate_label (required)
          2) second_covariate_label (optional)
        and we plan on using a log-offset approach for each.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adata = adata
        
        self._tcr_label = tcr_label
        self._covariate_label = covariate_label
        self._second_covariate_label = second_covariate_label
        self.n_phenotypes = n_phenotypes
        
        # Store hyperparams
        self.covariate_logoffset_scale = covariate_logoffset_scale
        self.second_covariate_logoffset_scale = second_covariate_logoffset_scale
        self.gene_profile_prior_strength = gene_profile_prior_strength
        self.gene_profile_prior_offset = gene_profile_prior_offset
        self.persistence_factor = persistence_factor
        self.consistency_weight = consistency_weight
        self.gene_concentration = gene_concentration
        
        # 1) Process phenotype prior => self.phenotype_probabilities
        # self.phenotype_probabilities = self._process_phenotype_prior(
        #     adata, phenotype_prior, n_phenotypes
        # )
        cell_phen_probs, clone_phen_probs = self._process_phenotype_prior(
            adata=adata,
            phenotype_prior=phenotype_prior,
            n_phenotypes=n_phenotypes,
            tcr_label=tcr_label
        )
        self.phenotype_probabilities = cell_phen_probs  # shape => (n_phenotypes, n_cells)
        self.clone_to_phenotype_prior = torch.tensor(
            clone_phen_probs, dtype=torch.float32, device=self.device
        )

        # 2) Build the dataset with two covariates (the second is optional)
        self.dataset = TCRCooccurrenceDataset(
            adata=adata,
            tcr_label=tcr_label,
            covariate_label=covariate_label,
            phenotype_probs=self.phenotype_probabilities,
            min_expression=1e-6,
            second_covariate_label=second_covariate_label  # <-- here's the new param
        )
        
        # 3) DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # 4) Store dimensions
        self.K = self.dataset.K     # number of TCRs
        self.D = self.dataset.D     # number of genes
        self.C = self.dataset.C     # number of primary covariate categories
        self.C2 = getattr(self.dataset, "C2", 0)  # second covariate categories
        self.n_phenotypes = n_phenotypes
        
        raw_profile = self._compute_empirical_gene_profile_prior(
                phenotype_column="phenotype"  # or None
            )
        self.gene_profile_prior = torch.tensor(raw_profile, dtype=torch.float32, device=self.device)

        # 6) Clear Pyro param store and build the SVI with your "simple log-offset" model & guide
        pyro.clear_param_store()

        # We'll define self._model and self._guide as the "simple log-offset" version that expects
        # two covariate indexes in the signature: (matrix, tcr_idx, cov1_idx, cov2_idx, phenotype_probs).
        # For example:
        self.svi = SVI(
            model=self._model_with_2covariates,  # You must define this method
            guide=self._guide_with_2covariates,  # You must define this method
            optim=pyro.optim.ClippedAdam({"lr": learning_rate}),
            loss=TraceMeanField_ELBO(num_particles=5)
        )

        # Print summary
        print("\nInitialization Summary:")
        print(f"  TCRs (K): {self.K}")
        print(f"  Genes (D): {self.D}")
        print(f"  Primary Covariate (C): {self.C}")
        print(f"  Second Covariate (C2): {self.C2}")
        print(f"  n_phenotypes: {self.n_phenotypes}")

    def _compute_empirical_gene_profile_prior(
        self,
        phenotype_column: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute an empirical gene-profile prior of shape (n_phenotypes, D)
        by averaging gene expression for each phenotype group (if a phenotype
        column is provided). If no column is provided, produce a uniform distribution.

        Returns a NumPy array of shape (n_phenotypes, D) that sums to 1 per row.
        """
        import numpy as np
        import pandas as pd

        # 1) Number of phenotypes, number of genes
        n_pheno = self.n_phenotypes
        D = self.dataset.D

        if phenotype_column is not None:
            # Ensure the column is in adata.obs
            if phenotype_column not in self.adata.obs.columns:
                raise ValueError(f"Column '{phenotype_column}' not found in adata.obs")

            # 2) Group the raw data by phenotype => mean gene expression per phenotype
            df = self.adata.to_df()
            df["tmp_phenotype"] = self.adata.obs[phenotype_column].values
            grouped = df.groupby("tmp_phenotype").mean()  # shape => (#unique_phenotypes, D)

            # 3) Check if #unique_phenotypes == self.n_phenotypes
            if grouped.shape[0] != n_pheno:
                raise ValueError(
                    f"Phenotype column '{phenotype_column}' has {grouped.shape[0]} unique values "
                    f"but n_phenotypes={n_pheno} in the model."
                )

            # Convert to NumPy => shape (#unique_phenotypes, D)
            gene_profile_prior = grouped.to_numpy(dtype=np.float32)

            # 4) Row-normalize so each phenotype row sums to 1
            row_sums = gene_profile_prior.sum(axis=1, keepdims=True)
            gene_profile_prior /= np.clip(row_sums, 1e-12, None)

        else:
            # 5) If no phenotype column is given, produce a uniform distribution
            gene_profile_prior = np.ones((n_pheno, D), dtype=np.float32)
            gene_profile_prior /= gene_profile_prior.sum(axis=1, keepdims=True)

        return gene_profile_prior



    def compute_log_likelihood(self, adata_hold) -> float:
        """
        Plug-in log-likelihood for hold-out:
        usage => local_clone_concentration_pct, shape (K*C*T, n_phenotypes).
        flatten index => k*(C*T) + c*T + t
        patient offset
        etc.
        """
        import numpy as np
        import torch
        import pyro
        import pyro.distributions as dist

        hold_pheno_probs = np.ones((self.n_phenotypes, adata_hold.n_obs)) / self.n_phenotypes
        hold_dataset = TCRCooccurrenceDataset(
            adata_hold,
            tcr_label=self._tcr_label,
            covariate_label=self._covariate_label,
            phenotype_probs=hold_pheno_probs,
            min_expression=1e-6,
            timepoint_label=self._timepoint_label
        )

        param_store = pyro.get_param_store()
        local_conc_pct = param_store["local_clone_concentration_pct"].detach()  # shape (K*C*T, P)

        raw_log = param_store["unconstrained_gene_profile_log"].detach()
        log_clamped = torch.clamp(raw_log, min=-10, max=10)
        gene_conc_raw = torch.exp(log_clamped)
        gene_profiles = gene_conc_raw / gene_conc_raw.sum(dim=1, keepdim=True)

        shape_vals = param_store["patient_variance_shape"].detach() if "patient_variance_shape" in param_store else None
        rate_vals  = param_store["patient_variance_rate"].detach() if "patient_variance_rate" in param_store else None
        if shape_vals is not None and rate_vals is not None:
            patient_offset_means = shape_vals / rate_vals
        else:
            patient_offset_means = None

        persistence_factor = self.persistence_factor
        total_logp = 0.0
        n_cells = len(hold_dataset)

        for i in range(n_cells):
            gene_probs, k_idx, c_idx, phen_probs, t_idx = hold_dataset[i]
            gene_probs = gene_probs.cpu()
            k = k_idx.item()
            c = c_idx.item()
            t = t_idx.item()

            # flatten (k,c,t)
            kct = k*(self.C*self.T) + c*self.T + t
            conc_row = local_conc_pct[kct]
            base_dist = conc_row / conc_row.sum()

            # cell-level phenotype => mean of Dirichlet
            alpha_vec = base_dist*persistence_factor + 1.0
            alpha_sum = alpha_vec.sum()
            cell_phen_mean = alpha_vec/alpha_sum

            # Weighted sum across phenotypes => shape(D,)
            mixed_profile = torch.sum(gene_profiles * cell_phen_mean.unsqueeze(-1), dim=0)

            # patient offset
            if patient_offset_means is not None:
                p_off = patient_offset_means[c]
            else:
                p_off = 1.0
            adjusted = mixed_profile * p_off

            exponentiated = torch.exp(adjusted)
            concentration = exponentiated*self.gene_concentration+ 1.0

            dirich = dist.Dirichlet(concentration)
            total_logp += float(dirich.log_prob(gene_probs))

        return total_logp

    # ------------------------------------------------------------------------
    # Process phenotype prior
    # ------------------------------------------------------------------------
    # def _process_phenotype_prior(
    #     self,
    #     adata: anndata.AnnData,
    #     phenotype_prior: Optional[Union[str, np.ndarray]],
    #     n_phenotypes: int
    # ) -> np.ndarray:
    #     """
    #     Process phenotype prior information into probability matrix.
    #     """
    #     n_cells = adata.n_obs
        
    #     # Case 1: Column name in adata.obs
    #     if isinstance(phenotype_prior, str):
    #         if phenotype_prior not in adata.obs.columns:
    #             raise ValueError(f"Column {phenotype_prior} not found in adata.obs")
            
    #         unique_phenotypes = adata.obs[phenotype_prior].unique()
    #         if len(unique_phenotypes) != n_phenotypes:
    #             raise ValueError(
    #                 f"Number of unique phenotypes in {phenotype_prior} "
    #                 f"({len(unique_phenotypes)}) doesn't match n_phenotypes ({n_phenotypes})"
    #             )
            
    #         probabilities = np.zeros((n_phenotypes, n_cells))
    #         for i, phenotype in enumerate(unique_phenotypes):
    #             mask = adata.obs[phenotype_prior] == phenotype
    #             probabilities[i, mask] = 1.0
                
    #         return probabilities
        
    #     # Case 2: Probability matrix
    #     elif isinstance(phenotype_prior, np.ndarray):
    #         if phenotype_prior.shape != (n_phenotypes, n_cells):
    #             raise ValueError(
    #                 f"Probability matrix shape {phenotype_prior.shape} doesn't match "
    #                 f"expected shape ({n_phenotypes}, {n_cells})"
    #             )
            
    #         if not np.allclose(phenotype_prior.sum(axis=0), 1.0):
    #             raise ValueError("Probabilities must sum to 1 for each cell")
    #         if not np.all((phenotype_prior >= 0) & (phenotype_prior <= 1)):
    #             raise ValueError("Probabilities must be between 0 and 1")
                
    #         return phenotype_prior
        
    #     # Case 3: None (uniform distribution)
    #     elif phenotype_prior is None:
    #         return np.ones((n_phenotypes, n_cells)) / n_phenotypes
        
    #     else:
    #         raise ValueError(
    #             "phenotype_prior must be either a column name (str), "
    #             "probability matrix (np.ndarray), or None"
    #         )


    def _process_phenotype_prior(
        self,
        adata: anndata.AnnData,
        phenotype_prior: Optional[str],
        n_phenotypes: int,
        tcr_label: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return both:
        - phenotype_probabilities: shape (n_phenotypes, n_cells)
        - clone_to_phenotype_prior: shape (K, n_phenotypes)

        so we can store them in self.phenotype_probabilities and self.clone_to_phenotype_prior.
        
        Args:
            adata: an AnnData with .obs columns for TCR and phenotype.
            phenotype_prior: the name of a column in adata.obs specifying the phenotype label
            n_phenotypes: the number of distinct phenotypes
            tcr_label: column name in adata.obs that identifies which TCR each cell belongs to

        Returns:
            (cell_phen_probs, clone_phen_probs)
                cell_phen_probs: shape (n_phenotypes, n_cells)
                clone_phen_probs: shape (K, n_phenotypes)
        """
        import numpy as np
        import pandas as pd

        n_cells = adata.n_obs

        # ----------------------------------------------------------------
        # 1) Build a cell-level matrix "phenotype_probabilities"
        # ----------------------------------------------------------------
        if phenotype_prior is None:
            # Uniform distribution for each cell
            phenotype_probabilities = np.ones((n_phenotypes, n_cells), dtype=np.float32)/n_phenotypes
        else:
            if phenotype_prior not in adata.obs.columns:
                raise ValueError(f"Column '{phenotype_prior}' not found in adata.obs")

            # ensure we have exactly n_phenotypes unique categories
            phen_labels = adata.obs[phenotype_prior].cat.categories
            if len(phen_labels) != n_phenotypes:
                raise ValueError(
                    f"Number of unique phenotypes in column '{phenotype_prior}' "
                    f"({len(phen_labels)}) doesn't match n_phenotypes={n_phenotypes}"
                )

            phenotype_probabilities = np.zeros((n_phenotypes, n_cells), dtype=np.float32)
            # for i in range(n_phenotypes):
            for i, phen in enumerate(phen_labels):
                mask = (adata.obs[phenotype_prior] == phen)
                phenotype_probabilities[i, mask] = 1.0

        # ----------------------------------------------------------------
        # 2) Build a clone->phenotype prior: shape (K, n_phenotypes)
        # ----------------------------------------------------------------
        # We assume TCR label is in adata.obs[tcr_label], and 
        # we do a cross-tab of TCR vs. phenotype to see how each TCR is distributed across phen.
        # Then we row-normalize => sum=1 for each TCR.
        # This only makes sense if TCR is categorical, so ensure it is.
        if tcr_label not in adata.obs.columns:
            raise ValueError(f"TCR label '{tcr_label}' not found in adata.obs")
        if not pd.api.types.is_categorical_dtype(adata.obs[tcr_label]):
            # Make it categorical if not already
            adata.obs[tcr_label] = pd.Categorical(adata.obs[tcr_label])

        # Now do a crosstab
        tcr_phen_crosstab = pd.crosstab(
            adata.obs[tcr_label],
            adata.obs[phenotype_prior],
            normalize="index"  # row-normalize per TCR
        )
        # shape => (#unique_TCRs, n_phenotypes)

        # Some TCR clones might not appear if there's a mismatch or 0 coverage, but typically
        # crosstab index matches the TCR categories, columns match phenotype categories
        # We want it in the same order as TCR categories
        tcr_indices = adata.obs[tcr_label].cat.categories  # all TCR clone categories
        phen_indices = adata.obs[phenotype_prior].cat.categories  # all phenotype categories

        # Reindex in case crosstab is missing any row/col
        tcr_phen_crosstab = tcr_phen_crosstab.reindex(
            index=tcr_indices,
            columns=phen_indices,
            fill_value=0.0
        )
        # Now convert to numpy => shape (#unique_TCRs, n_phenotypes)
        clone_to_phenotype_prior = tcr_phen_crosstab.to_numpy(dtype=np.float32)

        # There's a chance some TCR row sums to 0 if it doesn't appear, but
        # crosstab with normalize="index" should have 0 distribution if it has no cells.

        return phenotype_probabilities, clone_to_phenotype_prior


    def _get_patient_tcr_indices(self):
        # rename to a more general _get_covariate_tcr_indices
        patient_tcrs = {}
        tcr_indices = self.dataset.tcrs.numpy()
        covariate_indices = self.dataset.covariates.numpy()
        
        for patient_idx in range(self.C):
            patient_mask = covariate_indices == patient_idx
            patient_tcrs[patient_idx] = torch.from_numpy(
                np.unique(tcr_indices[patient_mask])
            )
        return patient_tcrs

    def _model_with_2covariates(self, 
                                gene_probs, 
                                tcr_idx, 
                                cov1_idx,   # primary covariate (patient)
                                cov2_idx,   # second covariate (time, site, etc.)
                                phenotype_probs):
        """
        Example "simple log-offset" model for two covariates.
        Each covariate modifies the TCR->phenotype distribution in log space.
        
        Args:
            gene_probs: shape (batch_size, D) = observed normalized gene expression
            tcr_idx: shape (batch_size,) = TCR index in [0..K-1]
            cov1_idx: shape (batch_size,) = primary covariate index in [0..C-1]
            cov2_idx: shape (batch_size,) = second covariate index in [0..C2-1], or -1 if absent
            phenotype_probs: (unused in this minimal example, but you have it if needed)
        """
        import torch
        import pyro
        import pyro.distributions as dist

        batch_size = gene_probs.shape[0]

        # ----------------------------------------------------------------
        # 1) Baseline TCR->phenotype distribution
        #    shape => (K, n_phenotypes)
        # ----------------------------------------------------------------
        # We define a prior, e.g. from self.clone_to_phenotype_prior, but for brevity:
        baseline_prior = (
            self.clone_to_phenotype_prior * self.clone_to_phenotype_prior_strength
            + 1.0
        )

        with pyro.plate("clone_plate", self.K):
            clone_baseline = pyro.sample(
                "clone_baseline",
                dist.Dirichlet(baseline_prior)
            )
            # shape => (K, n_phenotypes)

        # ----------------------------------------------------------------
        # 2) Covariate #1 log-offset => shape (C, n_phenotypes)
        # ----------------------------------------------------------------
        if self.C > 0:
            with pyro.plate("covariate1_plate", self.C):
                cov1_offset = pyro.sample(
                    "cov1_offset",
                    dist.Normal(
                        torch.zeros(self.n_phenotypes, device=self.device),
                        torch.ones(self.n_phenotypes, device=self.device)*self.covariate_logoffset_scale
                    ).to_event(1)
                )
        else:
            # If for some reason C=0, define a dummy
            cov1_offset = torch.zeros(1, self.n_phenotypes, device=self.device)

        # ----------------------------------------------------------------
        # 3) Covariate #2 log-offset => shape (C2, n_phenotypes)
        # ----------------------------------------------------------------
        if self.C2 > 0:
            with pyro.plate("covariate2_plate", self.C2):
                cov2_offset = pyro.sample(
                    "cov2_offset",
                    dist.Normal(
                        torch.zeros(self.n_phenotypes, device=self.device),
                        torch.ones(self.n_phenotypes, device=self.device)*self.second_covariate_logoffset_scale
                    ).to_event(1)
                )
        else:
            # If second covariate is absent, define zeros
            cov2_offset = torch.zeros(1, self.n_phenotypes, device=self.device)

        # ----------------------------------------------------------------
        # 4) Phenotype->gene profile => shape (n_phenotypes, D)
        # ----------------------------------------------------------------
        # We define a prior from self.gene_profile_prior etc.
        gene_prior = (
            self.gene_profile_prior*self.gene_profile_prior_strength 
            + self.gene_profile_prior_offset
        )
        gene_prior = torch.clamp(gene_prior, min=1e-4)

        with pyro.plate("phenotype_plate", self.n_phenotypes):
            gene_profile = pyro.sample(
                "gene_profile",
                dist.Dirichlet(gene_prior)
            )
            # shape => (n_phenotypes, D)

        # ----------------------------------------------------------------
        # 5) Cell-level sampling
        # ----------------------------------------------------------------
        with pyro.plate("cells", batch_size):
            # a) Baseline clone distribution => shape (batch_size, n_phenotypes)
            base_clone = clone_baseline[tcr_idx]

            # b) Log offsets for each covariate
            #    shape => (batch_size, n_phenotypes)
            #    If cov2_idx = -1 for some cells, clamp it to 0 offset.
            offset1 = cov1_offset[cov1_idx] if self.C>0 else 0.0
            if self.C2 > 0:
                # For cells with cov2_idx=-1, clamp them to offset=0
                valid_cov2_idx = torch.where(cov2_idx>=0, cov2_idx, torch.zeros_like(cov2_idx))
                offset2 = cov2_offset[valid_cov2_idx]
                # zero out those rows if cov2_idx was -1
                offset2 = torch.where(cov2_idx.unsqueeze(-1)>=0, offset2, torch.zeros_like(offset2))
            else:
                offset2 = torch.zeros_like(offset1)

            # c) Combine log offsets => shape (batch_size, n_phenotypes)
            alpha_log = offset1 + offset2
            alpha = base_clone * torch.exp(alpha_log) + 1.0

            # d) Cell-level phenotype distribution
            cell_phen = pyro.sample(
                "cell_phen",
                dist.Dirichlet(alpha*self.persistence_factor + 1.0)
            )
            # shape => (batch_size, n_phenotypes)

            # e) (Optional) clone consistency penalty if desired
            #    (similar to your existing logic)

            # f) Mix phenotype->gene
            mixed_profile = torch.sum(
                gene_profile * cell_phen.unsqueeze(-1),  # shape => (batch_size, D)
                dim=1
            )
            # shape => (batch_size, D)

            # g) Final gene expression => Dirichlet
            exponentiated = torch.exp(mixed_profile)
            concentration = exponentiated*self.gene_concentration + 1.0

            pyro.sample(
                "obs",
                dist.Dirichlet(concentration),
                obs=gene_probs
            )

    def _guide_with_2covariates(self, 
                                gene_probs, 
                                tcr_idx, 
                                cov1_idx, 
                                cov2_idx, 
                                phenotype_probs):
        """
        Matching guide for the two-covariate log-offset model.
        We'll param-ify each sample site from the model:
        - "clone_baseline"
        - "cov1_offset"
        - "cov2_offset"
        - "gene_profile"
        - "cell_phen"
        """
        import torch
        import pyro
        import pyro.distributions as dist
        from pyro.distributions import constraints

        batch_size = gene_probs.shape[0]

        # ----------------------------------------------------------------
        # 1) Clone baseline => Dirichlet
        # ----------------------------------------------------------------
        # shape => (K, n_phenotypes)
        baseline_init = (
            self.clone_to_phenotype_prior*self.clone_to_phenotype_prior_strength
            + 1.0
        )
        clone_baseline_param = pyro.param(
            "clone_baseline_param",
            baseline_init,
            constraint=constraints.greater_than(0.0)
        )
        with pyro.plate("clone_plate", self.K):
            pyro.sample(
                "clone_baseline",
                dist.Dirichlet(clone_baseline_param)
            )

        # ----------------------------------------------------------------
        # 2) cov1_offset => shape (C, n_phenotypes)
        # ----------------------------------------------------------------
        if self.C > 0:
            cov1_loc = pyro.param(
                "cov1_loc",
                torch.zeros(self.C, self.n_phenotypes, device=self.device),
                constraint=constraints.real
            )
            cov1_scale = self.covariate_logoffset_scale   # from user hyperparam
            with pyro.plate("covariate1_plate", self.C):
                pyro.sample(
                    "cov1_offset",
                    dist.Normal(cov1_loc, cov1_scale).to_event(1)
                )

        # ----------------------------------------------------------------
        # 3) cov2_offset => shape (C2, n_phenotypes)
        # ----------------------------------------------------------------
        if self.C2 > 0:
            cov2_loc = pyro.param(
                "cov2_loc",
                torch.zeros(self.C2, self.n_phenotypes, device=self.device),
                constraint=constraints.real
            )
            cov2_scale = self.second_covariate_logoffset_scale
            with pyro.plate("covariate2_plate", self.C2):
                pyro.sample(
                    "cov2_offset",
                    dist.Normal(cov2_loc, cov2_scale).to_event(1)
                )

        # ----------------------------------------------------------------
        # 4) gene_profile => shape (n_phenotypes, D)
        # ----------------------------------------------------------------
        gene_profile_init = (
            self.gene_profile_prior*self.gene_profile_prior_strength
            + self.gene_profile_prior_offset
        )
        gene_profile_init = torch.clamp(gene_profile_init, min=1e-4)
        # Typically we log-transform to get an unconstrained param
        unconstrained_gene = pyro.param(
            "unconstrained_gene_profile",
            torch.log(gene_profile_init),
            constraint=constraints.real
        )
        with pyro.plate("phenotype_plate", self.n_phenotypes):
            gene_profile_conc = torch.exp(torch.clamp(unconstrained_gene, -10, 10))
            pyro.sample(
                "gene_profile",
                dist.Dirichlet(gene_profile_conc)
            )

        # ----------------------------------------------------------------
        # 5) cell-level => shape (batch_size, n_phenotypes)
        # ----------------------------------------------------------------
        # If you want a mean-field approach for each cell:
        cell_init = torch.ones(batch_size, self.n_phenotypes, device=self.device)+0.1
        cell_param = pyro.param(
            "cell_phen_param",
            cell_init,
            constraint=constraints.greater_than(0.0)
        )
        with pyro.plate("cells", batch_size):
            pyro.sample(
                "cell_phen",
                dist.Dirichlet(cell_param)
            )
    def train(
        self, 
        num_epochs: int, 
        patience: int = 20,
        window_size: int = 10,
        min_delta: float = 1e-4,
        smoothing_alpha: float = 0.1,
        verbose: bool = True
    ):
        """
        Updated training loop to handle two covariates. We now expect
        the DataLoader to yield batches of (gene_probs, tcr_idx, cov1_idx, cov2_idx, phenotype_probs).
        """
        # Tracking variables
        self.losses = []
        self.smoothed_losses = []
        best_loss = float('inf')
        best_epoch = 0
        early_stop_epoch = None
        patience_counter = 0
        window_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Unpack 5 or 6 items from the DataLoader
            # If your dataset doesn't supply phenotype_probs, you can omit it, etc.
            for gene_probs, tcr_idx, cov1_idx, cov2_idx, phenotype_probs in self.dataloader:
                # Move everything to the same device (CPU/GPU)
                gene_probs = gene_probs.to(self.device)
                tcr_idx    = tcr_idx.to(self.device)
                cov1_idx   = cov1_idx.to(self.device)
                cov2_idx   = cov2_idx.to(self.device)
                phenotype_probs = phenotype_probs.to(self.device)

                # Optional check for NaN input data
                if torch.isnan(gene_probs).any():
                    if verbose:
                        print("Warning: NaN values in input data")
                    continue

                # Pass all items to self.svi.step(...)
                # (Note that your model/guide must match this signature)
                loss = self.svi.step(
                    gene_probs,
                    tcr_idx,
                    cov1_idx,
                    cov2_idx,
                    phenotype_probs
                )
                
                if not torch.isnan(torch.tensor(loss)):
                    epoch_loss += loss
                    num_batches += 1
            
            # After iterating through the batches
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                self.losses.append(avg_loss)
                
                # Exponential smoothing
                if not self.smoothed_losses:
                    smoothed_loss = avg_loss
                else:
                    smoothed_loss = (
                        smoothing_alpha * avg_loss
                        + (1 - smoothing_alpha) * self.smoothed_losses[-1]
                    )
                self.smoothed_losses.append(smoothed_loss)
                
                # Sliding window for early stopping
                window_losses.append(smoothed_loss)
                if len(window_losses) > window_size:
                    window_losses.pop(0)
                window_avg = sum(window_losses) / len(window_losses)
                
                # Improvement check
                if window_avg < best_loss - min_delta:
                    best_loss = window_avg
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Print progress every 10 epochs or final
                if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                    print(f"Epoch {epoch}: "
                        f"Loss = {avg_loss:.4f}, "
                        f"Smoothed = {smoothed_loss:.4f}, "
                        f"Window Avg = {window_avg:.4f}")
                
                # Early stopping
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch}")
                        print(f"Best epoch was {best_epoch} with window-avg loss {best_loss:.4f}")
                    early_stop_epoch = epoch
                    break
        
        return self.losses, self.smoothed_losses


    # ------------------------------------------------------------------------
    # Parameter retrieval --- some of these are defunct, use dataset class index mappings
    # ------------------------------------------------------------------------
    def get_params(self):
        """Get the learned parameters from Pyro's param store."""
        return {
            name: pyro.param(name).detach().cpu().numpy() 
            for name in pyro.get_param_store().keys()
        }
    
    def get_clone_phenotype_distributions(self) -> torch.Tensor:
        """Get current clone-level phenotype distributions."""
        return pyro.param("clone_phenotype_concentration").detach()
    
    def get_cell_phenotype_distributions(self) -> torch.Tensor:
        """Get current cell-level phenotype distributions."""
        return pyro.param("cell_phenotype_concentration").detach()
    
    def get_gene_profiles(self) -> torch.Tensor:
        """Get current gene expression profiles for each phenotype."""
        return pyro.param("gene_profile_concentration").detach()
    
    def get_patient_effects(self) -> torch.Tensor:
        """
        Get current patient variance parameters (Gamma shape and rate).
        """
        shape = pyro.param("patient_variance_shape").detach()
        rate = pyro.param("patient_variance_rate").detach()
        return shape, rate

    def map_tcr_index_to_label(self, idx: int) -> str:
        return self.reverse_tcr_mapping.get(idx, f"Unknown TCR {idx}")
    
    def map_phenotype_index_to_label(self, idx: int) -> str:
        return self.phenotype_mapping.get(idx, f"Unknown Phenotype {idx}")
    
    def map_tcr_label_to_index(self, label: str) -> int:
        return self.tcr_mapping.get(label, -1)
    
    def map_phenotype_label_to_index(self, label: str) -> int:
        return self.reverse_phenotype_mapping.get(label, -1)
    
    def map_clone_index_to_label(self, idx: int) -> str:
        return self.clone_mapping.get(idx, f"Unknown Clone {idx}")
    
    def map_clone_label_to_index(self, label: str) -> int:
        return self.reverse_clone_mapping.get(label, -1)

    def plot_training_curves(self, figsize=(12, 6)):
        """
        Plot raw and smoothed loss curves for the training process.
        
        Assumes that the class has the following attributes:
            self.losses: A list of raw loss values (one per epoch).
            self.smoothed_losses: A list of exponentially smoothed loss values.
            self.early_stop_epoch: (Optional) The epoch index where early stopping occurred.
            self.best_epoch: (Optional) The epoch index corresponding to the best (lowest) smoothed loss.
        """
        plt.figure(figsize=figsize)
        plt.plot(self.losses, 'b-', alpha=0.3, label='Raw Loss')
        plt.plot(self.smoothed_losses, 'r-', label='Smoothed Loss')
        
        # Mark early stopping and best epoch if they exist
        if hasattr(self, 'early_stop_epoch') and self.early_stop_epoch is not None:
            plt.axvline(x=self.early_stop_epoch, color='g', linestyle='--', label='Early Stop')
        if hasattr(self, 'best_epoch') and self.best_epoch is not None:
            plt.axvline(x=self.best_epoch, color='r', linestyle='--', label='Best Model')
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _compute_local_distribution(self, k_idx: int, c1_idx: int, c2_idx: int) -> torch.Tensor:
        """
        Recompute the final TCR->phenotype distribution for (k, c1, c2)
        from the learned guide parameters:
        - "clone_baseline_param" => shape (K, n_phenotypes)
        - "cov1_loc" => shape (C, n_phenotypes)
        - "cov2_loc" => shape (C2, n_phenotypes)
        and combine them like your model does:
            alpha = clone_baseline[k] * exp(cov1_offset[c1] + cov2_offset[c2]) + 1.0
            (plus any 'persistence_factor' if you want).
        We'll return alpha normalized => shape (n_phenotypes,).
        """
        import pyro
        import torch
        import math

        # 1) Retrieve the baseline
        clone_baseline_param = pyro.param("clone_baseline_param").detach()  # shape => (K, n_phenotypes)
        alpha_base = clone_baseline_param[k_idx]  # shape => (n_phenotypes,)

        # 2) Retrieve covariate1 offset
        if self.C > 0:
            cov1_loc = pyro.param("cov1_loc").detach()  # shape => (C, n_phenotypes)
            offset1 = cov1_loc[c1_idx]
        else:
            offset1 = torch.zeros_like(alpha_base)

        # 3) Retrieve covariate2 offset
        if self.C2 > 0:
            cov2_loc = pyro.param("cov2_loc").detach()  # shape => (C2, n_phenotypes)
            offset2 = cov2_loc[c2_idx]
        else:
            offset2 = torch.zeros_like(alpha_base)

        # 4) Combine in log space
        alpha_log = offset1 + offset2
        alpha = alpha_base * torch.exp(alpha_log) + 1.0  # shape => (n_phenotypes,)

        # (Optional) multiply by self.persistence_factor if your final model does that
        # alpha = alpha*self.persistence_factor + 1.0  # for example

        # 5) Return the normalized distribution
        # If you want the raw Dirichlet concentration, skip normalization. 
        # For flux/entropy, you typically want a probability distribution => alpha / alpha.sum().
        dist = alpha / alpha.sum()
        return dist

    def get_phenotypic_flux(
        self,
        clone_idx: int,
        cov1_idx: int,
        cov2_a: int,
        cov2_b: int,
        metric: str = "l1"
    ) -> float:
        """
        Compare the distribution for (k, cov1, cov2_a) vs. (k, cov1, cov2_b)
        using your chosen metric (L1 or DKL).
        """
        import torch

        p1 = self._compute_local_distribution(clone_idx, cov1_idx, cov2_a)  # shape (n_phenotypes,)
        p2 = self._compute_local_distribution(clone_idx, cov1_idx, cov2_b)

        if metric.lower() == "l1":
            return float(torch.sum(torch.abs(p1 - p2)))
        elif metric.lower() == "dkl":
            eps = 1e-12
            p1c = torch.clamp(p1, min=eps)
            p2c = torch.clamp(p2, min=eps)
            return float(torch.sum(p1c * torch.log(p1c / p2c)))
        else:
            raise ValueError(f"Unknown metric {metric}")

    def get_phenotypic_entropy_by_clone(
        self,
        cov1_idx: int,
        cov2_idx: int,
        normalize: bool = False
    ) -> Dict[int, float]:
        """
        For each clone k, compute H(Phenotype|Clone=k) at (cov1_idx,cov2_idx).
        """
        import torch
        import math

        phen_entropy_by_clone = {}
        eps = 1e-12

        for k in range(self.K):
            p_phi_given_k = self._compute_local_distribution(k, cov1_idx, cov2_idx)
            # H(Phenotype|Clone=k)
            entropy_val = -torch.sum(torch.clamp(p_phi_given_k, min=eps) * torch.log(torch.clamp(p_phi_given_k, min=eps)))

            if normalize and self.n_phenotypes > 1:
                entropy_val = entropy_val / math.log(self.n_phenotypes)

            phen_entropy_by_clone[k] = float(entropy_val.item())

        return phen_entropy_by_clone

    def get_phenotypic_entropy(
        self,
        cov1_idx: int,
        cov2_idx: int,
        weight_by_clone_freq: bool = False,
        normalize: bool = False
    ) -> float:
        """
        H(Clone|Phenotype) for the location (cov1_idx, cov2_idx).
        """
        import torch
        import math

        # We'll gather p(phi|k) for each clone k => stack => shape(K,n_phenotypes)
        dist_kp = []
        clone_freqs = []

        for k in range(self.K):
            p_phi_given_k = self._compute_local_distribution(k, cov1_idx, cov2_idx)
            dist_kp.append(p_phi_given_k)
            if weight_by_clone_freq:
                # placeholder
                clone_freqs.append(1.0)
            else:
                clone_freqs.append(1.0)

        dist_kp = torch.stack(dist_kp, dim=0)
        clone_freqs = torch.tensor(clone_freqs, dtype=torch.float)
        clone_freqs = clone_freqs / clone_freqs.sum()

        # p(phi) = sum_k p(k)*p(phi|k)
        p_phi = (dist_kp.T * clone_freqs).sum(dim=1)
        p_phi = p_phi / p_phi.sum()

        eps = 1e-12
        p_k_given_phi = (dist_kp * clone_freqs.unsqueeze(-1)) / (p_phi.unsqueeze(0) + eps)

        # H(Clone|phi)
        entropy_per_phi = []
        for phi_idx in range(self.n_phenotypes):
            p_k_phi = torch.clamp(p_k_given_phi[:, phi_idx], min=eps)
            h_clone_phi = -torch.sum(p_k_phi * torch.log(p_k_phi))
            entropy_per_phi.append(h_clone_phi)

        entropy_per_phi = torch.stack(entropy_per_phi, dim=0)
        phenotypic_entropy = torch.sum(p_phi * entropy_per_phi)

        if normalize and self.K > 1:
            phenotypic_entropy = phenotypic_entropy / math.log(self.K)

        return float(phenotypic_entropy.item())

    def get_mutual_information(
        self,
        cov1_idx: int,
        cov2_idx: int,
        weight_by_clone_freq: bool = False
    ) -> float:
        """
        I(Clone; Phenotype) at (cov1_idx,cov2_idx).
        """
        import torch
        import math

        dist_kp = []
        clone_freqs = []

        for k in range(self.K):
            p_phi_given_k = self._compute_local_distribution(k, cov1_idx, cov2_idx)
            dist_kp.append(p_phi_given_k)
            if weight_by_clone_freq:
                clone_freqs.append(1.0)
            else:
                clone_freqs.append(1.0)

        dist_kp = torch.stack(dist_kp, dim=0)  # (K,n_phenotypes)
        clone_freqs = torch.tensor(clone_freqs, dtype=torch.float)
        clone_freqs = clone_freqs / clone_freqs.sum()
        p_k_phi = dist_kp * clone_freqs.unsqueeze(1)
        p_k_phi = p_k_phi / p_k_phi.sum()

        p_k = p_k_phi.sum(dim=1)
        p_phi = p_k_phi.sum(dim=0)

        eps = 1e-12
        p_k_clamp = torch.clamp(p_k, min=eps)
        p_phi_clamp = torch.clamp(p_phi, min=eps)
        p_k_phi_clamp = torch.clamp(p_k_phi, min=eps)

        H_clone = -torch.sum(p_k_clamp * torch.log(p_k_clamp))
        H_phi   = -torch.sum(p_phi_clamp * torch.log(p_phi_clamp))
        H_k_phi = -torch.sum(p_k_phi_clamp * torch.log(p_k_phi_clamp))

        MI = (H_clone + H_phi - H_k_phi).item()
        return MI

    def get_clonotypic_entropy(
        self, 
        cov1_idx: int, 
        cov2_idx: int, 
        weight_by_clone_freq: bool = False,
        normalize: bool = False
    ) -> float:
        """
        Compute H(Phenotype | Clone) aggregated across *all* clones at (cov1_idx, cov2_idx).
        If `normalize=True`, we divide by ln(n_phenotypes).

        Returns:
            clonotypic_entropy (float) in [0,1] if normalize=True,
            else up to ln(n_phenotypes).
        """
        import torch
        import math

        # We'll gather the entropy of each clone's phenotype distribution,
        # then weight by some 'clone_freq' if desired.
        entropies = []
        clone_freqs = []

        eps = 1e-12

        # For each clone k
        for k in range(self.K):
            # 1) Retrieve p(phi|k, cov1_idx, cov2_idx)
            p_k = self._compute_local_distribution(k, cov1_idx, cov2_idx)  # shape (n_phenotypes,)
            p_k_clamped = torch.clamp(p_k, min=eps)

            # 2) Entropy for that clone: H(Phenotype|Clone=k)
            entropy_k = -torch.sum(p_k_clamped * torch.log(p_k_clamped))
            entropies.append(float(entropy_k.item()))

            # 3) Weighting—if we have real clone frequencies, use them. Otherwise 1.0
            if weight_by_clone_freq:
                clone_freqs.append(1.0)  # placeholder
            else:
                clone_freqs.append(1.0)

        # 4) Weighted average
        clone_freqs_t = torch.tensor(clone_freqs, dtype=torch.float)
        clone_freqs_t = clone_freqs_t / clone_freqs_t.sum()
        entropies_t = torch.tensor(entropies, dtype=torch.float)
        clonotypic_entropy = torch.sum(clone_freqs_t * entropies_t)

        # 5) Normalize by ln(n_phenotypes) if requested
        if normalize and self.n_phenotypes > 1:
            clonotypic_entropy = clonotypic_entropy / math.log(self.n_phenotypes)

        return float(clonotypic_entropy.item())


    def get_clonotypic_entropy_by_phenotype(
        self,
        cov1_idx: int,
        cov2_idx: int,
        weight_by_clone_freq: bool = False,
        normalize: bool = False
    ) -> Dict[int, float]:
        """
        For each phenotype phi, compute H(Clone | Phenotype=phi) at (cov1_idx, cov2_idx).
        If `normalize=True`, we divide by ln(K).

        Returns:
            A dict {phi: entropy_value}, 
            where 0 <= entropy_value <= ln(K) (or 1 if normalized).
        """
        import torch
        import math
        from typing import Dict

        # 1) We'll build p(k, phi) for k in [0..K-1], phi in [0..n_phenotypes-1].
        #    Then from that, compute H(Clone|phi).
        eps = 1e-12
        dist_kp = []
        clone_freqs = []

        for k in range(self.K):
            # 2) p(phi|k, c1, c2)
            p_phi_given_k = self._compute_local_distribution(k, cov1_idx, cov2_idx)
            dist_kp.append(p_phi_given_k)

            if weight_by_clone_freq:
                clone_freqs.append(1.0)  # placeholder
            else:
                clone_freqs.append(1.0)

        dist_kp_t = torch.stack(dist_kp, dim=0)  # shape => (K, n_phenotypes)
        clone_freqs_t = torch.tensor(clone_freqs, dtype=torch.float)
        clone_freqs_t = clone_freqs_t / clone_freqs_t.sum()

        # 3) p(k, phi) = p(k)* p(phi|k)
        p_k_phi = dist_kp_t * clone_freqs_t.unsqueeze(-1)  # shape => (K, n_phenotypes)
        p_k_phi = p_k_phi / p_k_phi.sum()  # ensure sum=1

        # 4) p(phi) = sum_k p(k,phi)
        p_phi = p_k_phi.sum(dim=0)  # shape => (n_phenotypes,)

        # For each phi, define p(k|phi) = p(k,phi)/p(phi)
        # Then compute H(Clone|phi).
        clonotypic_entropy_dict: Dict[int, float] = {}

        for phi_idx in range(self.n_phenotypes):
            denom = p_phi[phi_idx]
            if denom < eps:
                # If p(phi_idx) ~ 0, skip or set to 0
                clonotypic_entropy_dict[phi_idx] = 0.0
                continue

            p_k_given_phi = torch.clamp(p_k_phi[:, phi_idx]/denom, min=eps)
            H_clone_phi = -torch.sum(p_k_given_phi * torch.log(p_k_given_phi))

            if normalize and self.K > 1:
                H_clone_phi = H_clone_phi / math.log(self.K)

            clonotypic_entropy_dict[phi_idx] = float(H_clone_phi.item())

        return clonotypic_entropy_dict

    def generate_posterior_samples(
        self,
        cov1_label: str,
        cov2_label: Optional[str],
        tcr_label: str,
        n_samples: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Example: to get a posterior sample of "phenotype distribution" at (k, c1, c2).
        We'll gather the param store, compute baseline + offsets for (k,c1,c2),
        then sample from Dirichlet => gene expression, etc.
        """
        import pyro
        import pyro.distributions as dist
        import numpy as np

        # 1) map user labels -> k_idx, c1_idx, c2_idx
        k_idx = self.tcr_mapping[tcr_label]
        c1_idx = self.dataset.covariate_mapping[cov1_label]
        if cov2_label is not None:
            c2_idx = self.dataset.covariate2_mapping[cov2_label]
        else:
            c2_idx = 0  # or whatever

        # 2) compute final alpha => shape (n_phenotypes,)
        alpha = self._compute_local_distribution(k_idx, c1_idx, c2_idx)
        # alpha is already normalized. If you want the raw Dirichlet concentration,
        # skip the final normalization in `_compute_local_distribution`.
        # Then do: phenotype_dist = dist.Dirichlet(alpha).sample()

        # If your model also has a gene profile, you can get it from param store similarly,
        # then do a second Dirichlet sample for gene expression per phenotype, etc.

        # Example: just sample from Dirichlet of alpha for phenotype distribution
        phen_samples = np.zeros((n_samples, self.n_phenotypes), dtype=np.float32)
        for i in range(n_samples):
            phen = dist.Dirichlet(alpha).sample().numpy()
            phen_samples[i,:] = phen

        return {"phenotype_samples": phen_samples}
