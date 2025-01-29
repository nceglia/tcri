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
from pyro.infer import TraceEnum_ELBO
from pyro.infer.enum import config_enumerate


class TCRCooccurrenceDataset(Dataset):
    """
    Dataset for Shared Global Model with Covariate Weights and Phenotype Priors,
    now extended to optionally include a timepoint label.
    """
    def __init__(
        self, 
        adata, 
        tcr_label: str, 
        covariate_label: str, 
        phenotype_probs: np.ndarray,
        min_expression: float = 1e-10,
        timepoint_label: Optional[str] = None
    ):
        """
        Initialize the dataset with phenotype probabilities.
        
        Args:
            adata: AnnData object containing gene expression data
            tcr_label: Column name in adata.obs for TCR labels
            covariate_label: Column name in adata.obs for 'patient' or other main covariate
            phenotype_probs: Matrix of shape (n_phenotypes, n_cells) with probabilities
            min_expression: Minimum expression value to avoid numerical issues
            timepoint_label: (Optional) Column name in adata.obs for timepoint labels
        """
        # Validate input data
        if tcr_label not in adata.obs.columns:
            raise ValueError(f"TCR label '{tcr_label}' not found in adata.obs")
        if covariate_label not in adata.obs.columns:
            raise ValueError(f"Covariate label '{covariate_label}' not found in adata.obs")
        if timepoint_label is not None and timepoint_label not in adata.obs.columns:
            raise ValueError(f"Timepoint label '{timepoint_label}' not found in adata.obs")

        # Store phenotype probabilities
        self.phenotype_probs = torch.tensor(phenotype_probs, dtype=torch.float32)
        
        # Remove cells with missing TCR or main covariate
        if adata.obs[tcr_label].isna().any():
            n_missing = adata.obs[tcr_label].isna().sum()
            warnings.warn(f"Found {n_missing} missing TCR labels. Removing those cells.")
            adata = adata[~adata.obs[tcr_label].isna()].copy()
        if adata.obs[covariate_label].isna().any():
            n_missing = adata.obs[covariate_label].isna().sum()
            warnings.warn(f"Found {n_missing} missing covariate labels. Removing those cells.")
            adata = adata[~adata.obs[covariate_label].isna()].copy()
        if timepoint_label is not None and adata.obs[timepoint_label].isna().any():
            n_missing = adata.obs[timepoint_label].isna().sum()
            warnings.warn(f"Found {n_missing} missing timepoint labels. Removing those cells.")
            adata = adata[~adata.obs[timepoint_label].isna()].copy()
            
        # Process expression matrix
        cooccurrence_matrix = adata.to_df().copy()
        cooccurrence_matrix += min_expression
        row_sums = cooccurrence_matrix.sum(axis=1)
        cooccurrence_matrix = cooccurrence_matrix.div(row_sums, axis=0)
        cooccurrence_matrix = cooccurrence_matrix.clip(lower=min_expression)
        row_sums = cooccurrence_matrix.sum(axis=1)
        cooccurrence_matrix = cooccurrence_matrix.div(row_sums, axis=0)

        # Validate normalized matrix
        if not np.allclose(cooccurrence_matrix.sum(axis=1), 1.0, rtol=1e-5):
            raise ValueError("Normalization failed: row sums are not 1")
        
        # Convert to tensor
        matrix_values = cooccurrence_matrix.values
        if np.isnan(matrix_values).any() or np.isinf(matrix_values).any():
            raise ValueError("Invalid values in normalized expression matrix")
        self.matrix = torch.tensor(matrix_values, dtype=torch.float32)
        
        # Process TCR and covariate labels
        tcrs = pd.Categorical(adata.obs[tcr_label])
        covariates = pd.Categorical(adata.obs[covariate_label])
        
        self.tcr_mapping = {cat: idx for idx, cat in enumerate(tcrs.categories)}
        self.covariate_mapping = {cat: idx for idx, cat in enumerate(covariates.categories)}
        
        tcr_indices = [self.tcr_mapping[cat] for cat in tcrs]
        covariate_indices = [self.covariate_mapping[cat] for cat in covariates]
        
        self.tcrs = torch.tensor(tcr_indices, dtype=torch.long)
        self.covariates = torch.tensor(covariate_indices, dtype=torch.long)
        
        # If timepoint_label is provided, process timepoints
        if timepoint_label is not None:
            timepoints = pd.Categorical(adata.obs[timepoint_label])
            self.timepoint_mapping = {cat: idx for idx, cat in enumerate(timepoints.categories)}
            tp_indices = [self.timepoint_mapping[cat] for cat in timepoints]
            self.timepoints = torch.tensor(tp_indices, dtype=torch.long)
            self.T = len(self.timepoint_mapping)  # number of timepoints
        else:
            self.timepoints = None
            self.timepoint_mapping = {}
            self.T = 0  # or set to 1 if you want a default

        # Dataset dimensions
        self.K = len(self.tcr_mapping)  # number of TCRs
        self.C = len(self.covariate_mapping)  # number of main covariates (patients)
        self.D = self.matrix.shape[1]  # number of genes
        self.n_phenotypes = phenotype_probs.shape[0]  # number of phenotypes

        self._validate_tensors()
        
    def _validate_tensors(self) -> None:
        """Validate all tensors including phenotype probabilities."""
        # Check matrix
        if torch.isnan(self.matrix).any() or torch.isinf(self.matrix).any():
            raise ValueError("Invalid values in expression matrix tensor")
            
        row_sums = self.matrix.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5):
            raise ValueError("Expression matrix rows do not sum to 1")
            
        # Check phenotype probabilities
        if torch.isnan(self.phenotype_probs).any() or torch.isinf(self.phenotype_probs).any():
            raise ValueError("Invalid values in phenotype probabilities")
            
        col_sums = self.phenotype_probs.sum(dim=0)
        if not torch.allclose(col_sums, torch.ones_like(col_sums), rtol=1e-5):
            raise ValueError("Phenotype probabilities do not sum to 1 across phenotypes")

        # Check indices
        if self.tcrs.max() >= self.K or self.covariates.max() >= self.C:
            raise ValueError("Invalid TCR or covariate indices found")
        if self.timepoints is not None and self.timepoints.max() >= self.T:
            raise ValueError("Invalid timepoint indices found")
            
    def __len__(self) -> int:
        return self.matrix.shape[0]
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieve data for a single cell including phenotype probabilities,
        plus timepoint index if available.
        
        Returns:
            (gene_probs, tcr_idx, covariate_idx, phenotype_probs, timepoint_idx or None)
        """
        if self.timepoints is not None:
            return (
            self.matrix[idx],
            self.tcrs[idx],
            self.covariates[idx],
            self.phenotype_probs[:, idx],
            (self.timepoints[idx] if self.timepoints is not None else 0),
            idx
            )
        else:
            return (
                self.matrix[idx],
                self.tcrs[idx],
                self.covariates[idx],
                self.phenotype_probs[:, idx],
                torch.tensor(-1, dtype=torch.long)  # or None
            )

class JointProbabilityDistribution:
    def __init__(
        self,
        adata,
        tcr_label: str,
        covariate_label: str,
        n_phenotypes: int,
        phenotype_prior: Optional[Union[str, np.ndarray]] = None,
        marker_genes: Optional[Dict[int, List[str]]] = None,
        marker_prior: float = 2.0,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        particles: int = 1,
        # --- Hierarchical Dirichlet: Per-Clone Baseline + Dimension-wise Gamma Scale ---
        clone_to_phenotype_prior_strength: float = 10.0,
        gene_profile_prior_strength: float = 5.0,
        gene_profile_prior_offset: float = 0.5,
        patient_variance_shape_val: float = 4.0,
        patient_variance_rate_val: float = 4.0,
        gamma_scale_shape: float = 2.0,  # new: dimension-wise Gamma shape
        gamma_scale_rate: float = 2.0,   # new: dimension-wise Gamma rate
        persistence_factor: float = 1.0,
        beta: float = 1.0,
        consistency_weight: float = 1.0,
        gene_concentration: float = 100.0,
        local_concentration_offset: float = 1.0,
        timepoint_label: Optional[str] = None
    ):
        """
        Joint Probability Distribution with hierarchical Dirichlet modeling:
        - Baseline Dirichlet for each clone+patient
        - Dimension-wise Gamma scale factors per timepoint
        - Optional marker gene priors

        Args:
            adata: AnnData object with expression data
            tcr_label: Column in adata.obs with TCR/clone identity
            covariate_label: Column in adata.obs for patient (or other main covariate)
            n_phenotypes: Number of phenotype states
            phenotype_prior: Either a column name in adata.obs or a probability matrix 
                            of shape (n_phenotypes, n_cells) or None for uniform
            marker_genes: Optional dict mapping phenotype index -> list of marker genes
            marker_prior: Multiplier for marker genes in gene-profile prior
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            particles: Number of particles for TraceMeanField_ELBO
            clone_to_phenotype_prior_strength: How strongly to weight the prior linking clones -> phenotypes
            gene_profile_prior_strength: Strength for the gene-profile prior
            gene_profile_prior_offset: Small constant offset for the gene-profile prior
            patient_variance_shape_val: Patient-level Gamma shape parameter for expression scaling
            patient_variance_rate_val: Patient-level Gamma rate parameter
            gamma_scale_shape: Shape hyperparameter for dimension-wise Gamma scale
            gamma_scale_rate: Rate hyperparameter for dimension-wise Gamma scale
            persistence_factor: Multiplicative factor for cell-level Dirichlet
            beta: Scalar that multiplies baseline usage in local child distributions
            consistency_weight: Weight for the clone consistency penalty in the cell-level phenotype
            gene_concentration: Multiplier in the final gene-level Dirichlet
            local_concentration_offset: Offset added to the child distribution concentration
            timepoint_label: If provided, name of the column in adata.obs with timepoint info
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adata = adata

        self._tcr_label = tcr_label
        self._covariate_label = covariate_label
        self._timepoint_label = timepoint_label
        self.particles = particles

        # -------------------- Store hierarchical priors --------------------
        self.clone_to_phenotype_prior_strength = clone_to_phenotype_prior_strength
        self.gene_profile_prior_strength = gene_profile_prior_strength
        self.gene_profile_prior_offset = gene_profile_prior_offset
        self.patient_variance_shape_val = patient_variance_shape_val
        self.patient_variance_rate_val = patient_variance_rate_val
        self.gamma_scale_shape = gamma_scale_shape
        self.gamma_scale_rate = gamma_scale_rate

        # Other modeling parameters
        self.beta = beta
        self.gene_concentration = gene_concentration
        self.persistence_factor = persistence_factor
        self.consistency_weight = consistency_weight
        self.local_concentration_offset = local_concentration_offset

        # -------------------- Process phenotype prior --------------------
        self.phenotype_probabilities = self._process_phenotype_prior(
            adata, phenotype_prior, n_phenotypes
        )

        # -------------------- Build TCR mappings --------------------
        tcrs = pd.Categorical(adata.obs[tcr_label])
        self.tcr_mapping = {tcr: idx for idx, tcr in enumerate(tcrs.categories)}
        self.reverse_tcr_mapping = {v: k for k, v in self.tcr_mapping.items()}

        # -------------------- Phenotype mappings --------------------
        if phenotype_prior is not None and isinstance(phenotype_prior, str):
            phenotypes = pd.Categorical(adata.obs[phenotype_prior])
            self.phenotype_mapping = {
                idx: label for idx, label in enumerate(phenotypes.categories)
            }
        else:
            self.phenotype_mapping = {
                idx: f"Phenotype {idx}" for idx in range(n_phenotypes)
            }
        self.reverse_phenotype_mapping = {v: k for k, v in self.phenotype_mapping.items()}

        # -------------------- Clone labels --------------------
        self.clone_mapping = {idx: clone for idx, clone in enumerate(tcrs.categories)}
        self.reverse_clone_mapping = {v: k for k, v in self.clone_mapping.items()}

        # -------------------- Dataset & DataLoader --------------------
        self.dataset = TCRCooccurrenceDataset(
            adata,
            tcr_label=tcr_label,
            covariate_label=covariate_label,
            min_expression=1e-6,
            phenotype_probs=self.phenotype_probabilities,
            timepoint_label=timepoint_label
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        self._posterior_cache = dict()

        # -------------------- Dimensions --------------------
        self.K = self.dataset.K  # Number of TCRs/clones
        self.D = self.dataset.D  # Number of genes
        self.C = self.dataset.C  # Number of patients
        self.n_phenotypes = n_phenotypes
        self.T = getattr(self.dataset, "T", 0)  # Number of timepoints (if any)

        # -------------------- Gene profile prior --------------------
        df = adata.to_df()
        if phenotype_prior is not None and isinstance(phenotype_prior, str):
            df["phenotype"] = adata.obs[phenotype_prior]
            gene_profile_prior = df.groupby("phenotype").mean().to_numpy()
            gene_profile_prior /= gene_profile_prior.sum(axis=1, keepdims=True)
        else:
            gene_profile_prior = np.ones((n_phenotypes, self.D)) / self.D

        # If marker genes are specified, apply that prior
        if marker_genes is not None:
            for phenotype_name, genes in marker_genes.items():
                if phenotype_name not in self.reverse_phenotype_mapping:
                    raise ValueError(f"Unknown phenotype name: {phenotype_name}")
                phenotype_idx = self.reverse_phenotype_mapping[phenotype_name]
                for gene in genes:
                    if gene in adata.var_names:
                        gene_idx = adata.var_names.get_loc(gene)
                        gene_profile_prior[phenotype_idx, gene_idx] *= marker_prior
            gene_profile_prior /= gene_profile_prior.sum(axis=1, keepdims=True)

        self.gene_profile_prior = torch.tensor(
            gene_profile_prior, dtype=torch.float32, device=self.device
        )

        # -------------------- Clone-to-phenotype prior --------------------
        epsilon = 1e-6
        if phenotype_prior is not None and isinstance(phenotype_prior, str):
            clone_to_phenotype_prior = pd.crosstab(
                adata.obs[tcr_label],
                adata.obs[phenotype_prior],
                normalize="index"
            ).to_numpy()
        else:
            clone_to_phenotype_prior = np.ones((self.K, n_phenotypes)) / n_phenotypes

        clone_to_phenotype_prior += epsilon
        clone_to_phenotype_prior /= clone_to_phenotype_prior.sum(axis=1, keepdims=True)
        self.clone_to_phenotype_prior = torch.tensor(
            clone_to_phenotype_prior, dtype=torch.float32, device=self.device
        )

        self.patient_tcr_indices = self._get_patient_tcr_indices()
        # self._build_cell_groups()

        # -------------------- Pyro setup (optimizer + SVI) --------------------
        pyro.clear_param_store()
        self.optimizer = pyro.optim.ClippedAdam(
            {
                "lr": learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "clip_norm": 5.0,
                "weight_decay": 1e-4,
            }
        )

        self.svi = SVI(
            model=self._model,
            guide=self._guide,
            optim=self.optimizer,
            loss=TraceMeanField_ELBO(num_particles=self.particles),
        )
        # self.svi = SVI(
        #     model=config_enumerate(self._model, "parallel"),
        #     guide=config_enumerate(self._guide, "parallel"),
        #     optim=self.optimizer,
        #     loss=TraceEnum_ELBO(num_particles=self.particles)
        # )

        self.training_losses = []

        # -------------------- Initialization Summary --------------------
        print("\nInitialization Summary:")
        print(f"Number of TCRs/Clones (K): {self.K}")
        print(f"Number of Phenotypes: {self.n_phenotypes}")
        print(f"Number of Genes (D): {self.D}")
        print(f"Number of Patients (C): {self.C}")
        print(f"Number of Timepoints (T): {self.T}")
        print(f"\nPhenotype Categories: {list(self.phenotype_mapping.values())}")
        print(f"Clone-to-phenotype prior shape: {self.clone_to_phenotype_prior.shape}")
        print(f"Gene profile prior shape: {self.gene_profile_prior.shape}")
        print(f"Gamma scale shape: {self.gamma_scale_shape}, rate: {self.gamma_scale_rate}")


    # ------------------------------------------------------------------------
    # Process phenotype prior
    # ------------------------------------------------------------------------
    def _process_phenotype_prior(
        self,
        adata: anndata.AnnData,
        phenotype_prior: Optional[Union[str, np.ndarray]],
        n_phenotypes: int
    ) -> np.ndarray:
        """
        Process phenotype prior information into probability matrix.
        """
        n_cells = adata.n_obs
        
        # Case 1: Column name in adata.obs
        if isinstance(phenotype_prior, str):
            if phenotype_prior not in adata.obs.columns:
                raise ValueError(f"Column {phenotype_prior} not found in adata.obs")
            
            unique_phenotypes = adata.obs[phenotype_prior].unique()
            if len(unique_phenotypes) != n_phenotypes:
                raise ValueError(
                    f"Number of unique phenotypes in {phenotype_prior} "
                    f"({len(unique_phenotypes)}) doesn't match n_phenotypes ({n_phenotypes})"
                )
            
            probabilities = np.zeros((n_phenotypes, n_cells))
            for i, phenotype in enumerate(unique_phenotypes):
                mask = adata.obs[phenotype_prior] == phenotype
                probabilities[i, mask] = 1.0
                
            return probabilities
        
        # Case 2: Probability matrix
        elif isinstance(phenotype_prior, np.ndarray):
            if phenotype_prior.shape != (n_phenotypes, n_cells):
                raise ValueError(
                    f"Probability matrix shape {phenotype_prior.shape} doesn't match "
                    f"expected shape ({n_phenotypes}, {n_cells})"
                )
            
            if not np.allclose(phenotype_prior.sum(axis=0), 1.0):
                raise ValueError("Probabilities must sum to 1 for each cell")
            if not np.all((phenotype_prior >= 0) & (phenotype_prior <= 1)):
                raise ValueError("Probabilities must be between 0 and 1")
                
            return phenotype_prior
        
        # Case 3: None (uniform distribution)
        elif phenotype_prior is None:
            return np.ones((n_phenotypes, n_cells)) / n_phenotypes
        
        else:
            raise ValueError(
                "phenotype_prior must be either a column name (str), "
                "probability matrix (np.ndarray), or None"
            )

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
    
    def _build_cell_groups(self):
        """
        Builds a dict that maps (patient_idx, clone_idx, time_idx) -> list of cell indices.
        This lets us do a truly nested approach for the final cell plate.
        """
        from collections import defaultdict
        self.cell_groups = defaultdict(list)

        for i in range(len(self.dataset)):
            # dataset[i] => (gene_probs, tcr_idx, covariate_idx, phenotype_probs, time_idx)
            _, tcr_val, cov_val, _, t_val = self.dataset[i]
            c_idx = cov_val.item()
            k_idx = tcr_val.item()
            t_idx = t_val.item()
            self.cell_groups[(c_idx, k_idx, t_idx)].append(i)


    def _model(self, batch_data):
        """
        Model that:
        1) Samples global variables (patient variance, clone-to-phenotype distribution, gene profiles).
        2) For each cell in the mini-batch, samples continuous z_i from Dirichlet,
            adds a consistency factor with known phenotype_probs,
            and samples gene expression from a Dirichlet of dimension D.
        """

        (
            gene_probs_batch,
            tcr_idx_batch,
            cov_idx_batch,
            pheno_probs_batch,
            time_idx_batch,
            cell_idx_batch
        ) = batch_data

        # -------------------- 1) Patient Variance -------------------- #
        with pyro.plate("patient_global", self.C):
            patient_variance = pyro.sample(
                "patient_variance",
                dist.Gamma(
                    torch.full((1,), 4.0, device=self.device),  # example shape
                    torch.full((1,), 4.0, device=self.device)   # example rate
                )
            )
            # patient_variance.shape => (C,)

        # -------------------- 2) Clone-Patient-Time Distribution -------------------- #
        flat_size = self.K * self.C * max(self.T, 1)
        p_kct_final = None
        if flat_size > 0:
            with pyro.plate("clone_patient_time", flat_size):
                # Example: just a Dirichlet prior for each (k,c,t)
                p_kct_final = pyro.sample(
                    "p_kct_final",
                    dist.Dirichlet(
                        torch.ones(flat_size, self.n_phenotypes, device=self.device)
                    )
                )
            # p_kct_final.shape => (flat_size, n_phenotypes)

        # -------------------- 3) Phenotype -> Gene Distribution -------------------- #
        with pyro.plate("phenotype", self.n_phenotypes):
            clone_gene_profiles = pyro.sample(
                "clone_gene_profiles",
                dist.Dirichlet(
                    torch.ones(self.D, device=self.device)
                )
            )
            # shape => (n_phenotypes, D)

        # -------------------- 4) Local (Per-Cell in the Mini-Batch) -------------------- #
        full_size = len(self.dataset)
        batch_size = gene_probs_batch.shape[0]

        # Scale factor so the mini-batch log-prob approximates full dataset
        scale_factor = float(full_size) / float(batch_size)

        with pyro.plate("batch_cells", batch_size, dim=-1) as idx:
            # (a) Map each cell to the correct patient variance
            pat_scale = patient_variance[cov_idx_batch]  # shape => (batch_size,)

            # (b) If p_kct_final is not None, gather local phenotype distribution
            if p_kct_final is not None:
                kct_index = (
                    tcr_idx_batch * (self.C * max(self.T, 1))
                    + cov_idx_batch * max(self.T, 1)
                    + time_idx_batch
                )
                local_phen_dist = p_kct_final[kct_index]  # shape => (batch_size, n_phenotypes)
            else:
                # fallback if no time dimension
                local_phen_dist = torch.ones(batch_size, self.n_phenotypes, device=self.device)
                local_phen_dist = local_phen_dist / local_phen_dist.sum(dim=1, keepdim=True)

            # (c) z_i ~ Dirichlet(local_phen_dist + 0.1)
            z_i = pyro.sample(
                "z_i",
                dist.Dirichlet(local_phen_dist + 0.1)
            )
            # shape => (batch_size, n_phenotypes)

            # (d) Consistency factor with the provided phenotype_probs
            # negative cross-entropy for each cell => sum_{p} phenotype_probs * log(z_i)
            ce_term = (pheno_probs_batch * torch.log(z_i + 1e-12)).sum(dim=1)
            pyro.factor(
                "consistency_factor",
                scale_factor * 10.0 * ce_term  # example: consistency_weight=10
            )

            # (e) Gene expression => Dirichlet( mixture_profile * pat_scale + offset )
            # mixture_profile[b, d] = sum_p z_i[b,p] * clone_gene_profiles[p,d]
            mixture_profile = z_i @ clone_gene_profiles  # shape => (batch_size, D)
            final_conc = mixture_profile * pat_scale.unsqueeze(-1) + 0.1

            pyro.sample(
                "obs_genes",
                dist.Dirichlet(final_conc),
                obs=gene_probs_batch
            )

    # ---------------------------------------------------------------------

    def _guide(self, batch_data):
        """
        Mean-field guide with global + local (per-cell) parameters.
        """

        (
            gene_probs_batch,
            tcr_idx_batch,
            cov_idx_batch,
            pheno_probs_batch,
            time_idx_batch,
            cell_idx_batch
        ) = batch_data

        # We unconditionally define cell_z_concentration here
        N = len(self.dataset)
        cell_z_conc = pyro.param(
            "cell_z_concentration",
            torch.ones(N, self.n_phenotypes, device=self.device),
            constraint=constraints.greater_than(0.0)
        )

        # 1) patient variance params
        shape_param = pyro.param(
            "patient_variance_shape",
            torch.full((self.C,), 4.0, device=self.device),
            constraint=constraints.positive
        )
        rate_param = pyro.param(
            "patient_variance_rate",
            torch.full((self.C,), 4.0, device=self.device),
            constraint=constraints.positive
        )
        with pyro.plate("patient_global", self.C):
            pyro.sample(
                "patient_variance",
                dist.Gamma(shape_param, rate_param)
            )

        # 2) clone-patient-time Dirichlet
        flat_size = self.K * self.C * max(self.T, 1)
        if flat_size > 0:
            with pyro.plate("clone_patient_time", flat_size):
                p_kct_conc = pyro.param(
                    "p_kct_final_conc",
                    torch.ones(flat_size, self.n_phenotypes, device=self.device),
                    constraint=constraints.greater_than(0.0)
                )
                pyro.sample(
                    "p_kct_final",
                    dist.Dirichlet(p_kct_conc)
                )

        # 3) gene profiles
        gene_profile_conc = pyro.param(
            "gene_profile_conc",
            torch.ones(self.n_phenotypes, self.D, device=self.device),
            constraint=constraints.greater_than(0.0)
        )
        with pyro.plate("phenotype", self.n_phenotypes):
            pyro.sample(
                "clone_gene_profiles",
                dist.Dirichlet(gene_profile_conc)
            )

        # 4) local per-cell mixture
        # We slice from 'cell_z_concentration' using cell_idx_batch
        with pyro.plate("batch_cells", gene_probs_batch.shape[0]):
            local_conc = cell_z_conc[cell_idx_batch]  # shape => (batch_size, n_phenotypes)
            pyro.sample(
                "z_i",
                dist.Dirichlet(local_conc)
            )


    # def _model(self):
    #     # print("\n[DEBUG] Entering _model() [flattened approach].")

    #     # 1) patient_variance => shape (C,)
    #     with pyro.plate("patient_global", self.C):
    #         patient_variance = pyro.sample(
    #             "patient_variance",
    #             dist.Gamma(
    #                 torch.full((1,), float(self.patient_variance_shape_val), device=self.device),
    #                 torch.full((1,), float(self.patient_variance_rate_val), device=self.device)
    #             )
    #         )
    #     # print(f"[DEBUG] patient_variance shape in code = {patient_variance.shape}")

    #     flat_size = self.K*self.C*self.T
    #     # print(f"[DEBUG] flat_size= {flat_size} [K*C*T= {self.K}*{self.C}*{self.T}]")

    #     # Build local_baseline_prior => shape (flat_size, n_phen)
    #     if self.T>0:
    #         repeated_baseline = self.clone_to_phenotype_prior.repeat_interleave(self.C*self.T, dim=0)
    #         # print(f"[DEBUG] repeated_baseline shape= {repeated_baseline.shape}")
    #         local_baseline_prior = repeated_baseline*self.clone_to_phenotype_prior_strength + 1.0
    #         # print(f"[DEBUG] local_baseline_prior shape= {local_baseline_prior.shape}")
    #     else:
    #         local_baseline_prior = None

    #     beta_val = torch.tensor(self.beta, device=self.device)
    #     offset_val = torch.tensor(self.local_concentration_offset, device=self.device)

    #     if self.T>0:
    #         with pyro.plate("clone_patient_time", flat_size) as idx:
    #             # print("[DEBUG] Inside plate clone_patient_time, size =", flat_size)
    #             final_local_baseline = pyro.sample(
    #                 "final_local_baseline",
    #                 dist.Dirichlet(local_baseline_prior)
    #             )
    #             # print("[DEBUG] final_local_baseline shape in code =", final_local_baseline.shape)

    #             scale_factor_kct = pyro.sample(
    #                 "scale_factor_kct",
    #                 dist.Gamma(
    #                     torch.full((self.n_phenotypes,), float(self.gamma_scale_shape), device=self.device),
    #                     torch.full((self.n_phenotypes,), float(self.gamma_scale_rate), device=self.device)
    #                 ).to_event(1)
    #             )
    #             # print("[DEBUG] scale_factor_kct shape in code =", scale_factor_kct.shape)

    #             final_conc = beta_val*final_local_baseline * scale_factor_kct + offset_val
    #             # print("[DEBUG] final_conc shape in code =", final_conc.shape)
    #             p_kct_final = pyro.sample(
    #                 "p_kct_final",
    #                 dist.Dirichlet(final_conc)
    #             )
    #             # print("[DEBUG] p_kct_final shape in code =", p_kct_final.shape)
    #     else:
    #         final_local_baseline=None
    #         scale_factor_kct=None
    #         p_kct_final=None

    #     # gene_profiles
    #     gene_prior = self.gene_profile_prior*self.clone_to_phenotype_prior_strength + self.local_concentration_offset
    #     with pyro.plate("phenotype", self.n_phenotypes):
    #         clone_gene_profiles = pyro.sample(
    #             "clone_gene_profiles",
    #             dist.Dirichlet(gene_prior)
    #         )
    #     # print("[DEBUG] clone_gene_profiles shape in code =", clone_gene_profiles.shape)

    #     # cell-level
    #     for (c_val,k_val,t_val), cell_idxs in self.cell_groups.items():
    #         if len(cell_idxs)==0:
    #             continue
    #         kct_index = k_val*(self.C*self.T) + c_val*self.T + t_val
    #         # Plate for the group
    #         with pyro.plate(f"cell_plate_{c_val}_{k_val}_{t_val}", len(cell_idxs)):
    #             gene_mat = self.dataset.matrix[cell_idxs].to(self.device)  # shape => (n_cells, 5)
    #             if self.T>0:
    #                 final_phen_dist = p_kct_final[kct_index]
    #             else:
    #                 final_phen_dist = torch.ones(self.n_phenotypes)/float(self.n_phenotypes)

    #             # incorporate patient_variance => shape => (C,)
    #             p_effect = patient_variance[c_val]
    #             mixed_profile = torch.einsum("p,pd->d", final_phen_dist, clone_gene_profiles)
    #             adjusted_probs = mixed_profile * p_effect
    #             expanded = adjusted_probs.unsqueeze(0).expand(len(cell_idxs), -1)
    #             pyro.sample(
    #                 f"obs_{c_val}_{k_val}_{t_val}",
    #                 dist.Dirichlet(expanded),
    #                 obs=gene_mat
    #             )

    # def _guide(self):
    #     # print("\n[DEBUG] Entering _guide() [flattened approach].")

    #     # 1) patient_variance => shape (C,)
    #     patient_variance_shape = pyro.param(
    #         "patient_variance_shape",
    #         torch.full((self.C,), float(self.patient_variance_shape_val), device=self.device),
    #         constraint=constraints.greater_than(0.1)
    #     )
    #     patient_variance_rate = pyro.param(
    #         "patient_variance_rate",
    #         torch.full((self.C,), float(self.patient_variance_rate_val), device=self.device),
    #         constraint=constraints.greater_than(0.1)
    #     )
    #     with pyro.plate("patient_global", self.C):
    #         pyro.sample(
    #             "patient_variance",
    #             dist.Gamma(patient_variance_shape, patient_variance_rate)
    #         )

    #     flat_size = self.K*self.C*self.T if self.T>0 else 0
    #     if self.T>0:
    #         repeated_baseline = self.clone_to_phenotype_prior.repeat_interleave(self.C*self.T, dim=0)
    #         baseline_conc_init = repeated_baseline*self.clone_to_phenotype_prior_strength + 1.0

    #         with pyro.plate("clone_patient_time", flat_size):
    #             local_baseline_param = pyro.param(
    #                 "local_baseline_conc",
    #                 baseline_conc_init,
    #                 constraint=constraints.greater_than(1e-6)
    #             )
    #             pyro.sample(
    #                 "final_local_baseline",
    #                 dist.Dirichlet(local_baseline_param)
    #             )

    #             shape_init = torch.full((self.n_phenotypes,), float(self.gamma_scale_shape), device=self.device)
    #             rate_init  = torch.full((self.n_phenotypes,), float(self.gamma_scale_rate),  device=self.device)

    #             scale_shape_param = pyro.param(
    #                 "scale_factor_shape",
    #                 shape_init,
    #                 constraint=constraints.greater_than(0.0)
    #             )
    #             scale_rate_param = pyro.param(
    #                 "scale_factor_rate",
    #                 rate_init,
    #                 constraint=constraints.greater_than(0.0)
    #             )
    #             pyro.sample(
    #                 "scale_factor_kct",
    #                 dist.Gamma(scale_shape_param, scale_rate_param).to_event(1)
    #             )

    #             p_kct_conc_init = torch.full(
    #                 (flat_size, self.n_phenotypes),
    #                 1.5, device=self.device
    #             )
    #             p_kct_conc_param = pyro.param(
    #                 "p_kct_final_conc",
    #                 p_kct_conc_init,
    #                 constraint=constraints.greater_than(0.0)
    #             )
    #             pyro.sample(
    #                 "p_kct_final",
    #                 dist.Dirichlet(p_kct_conc_param)
    #             )

    #     # gene profiles
    #     gene_profile_concentration_init = (
    #         self.gene_profile_prior*self.clone_to_phenotype_prior_strength
    #         + self.local_concentration_offset
    #     )
    #     gene_profile_concentration_init = torch.clamp(gene_profile_concentration_init, min=1e-4)
    #     unconstrained_profile_init = torch.log(gene_profile_concentration_init)

    #     unconstrained_profile = pyro.param(
    #         "unconstrained_gene_profile_log",
    #         unconstrained_profile_init,
    #         constraint=constraints.real
    #     )
    #     with pyro.plate("phenotype", self.n_phenotypes):
    #         gene_profile_conc = torch.exp(torch.clamp(unconstrained_profile, -10, 10))
    #         pyro.sample(
    #             "clone_gene_profiles",
    #             dist.Dirichlet(gene_profile_conc)
    #         )

    def train(
        self, 
        num_epochs: int, 
        patience: int = 20,
        window_size: int = 10,
        min_delta: float = 1e-4,
        smoothing_alpha: float = 0.1,
        verbose: bool = True
    ):
        # Tracking variables
        self.losses = []
        self.smoothed_losses = []
        best_loss = float('inf')
        best_epoch = 0
        # early_stop_epoch = None
        patience_counter = 0
        window_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_data in self.dataloader:
                if self.device != "cpu":
                    batch_data = tuple(x.to(self.device) for x in batch_data)
                epoch_loss += self.svi.step(batch_data)
            
            avg_loss = epoch_loss / len(self.dataloader)
            self.losses.append(avg_loss)
            

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
            
            # Print progress
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
        """
        Get the *baseline* clone-level phenotype Dirichlet parameters for each (k,c).
        That is, the param 'local_clone_concentration_pc' of shape (K*C, n_phenotypes).

        Returns:
            A Torch tensor of shape (K*C, n_phenotypes).
        """
        # This replaces the old "clone_phenotype_concentration" param.
        # In the guide, we define param("local_clone_concentration_pc", ...).
        return pyro.param("local_clone_concentration_pc").detach().cpu()

    def get_cell_phenotype_distributions(self) -> torch.Tensor:
        """
        Get current cell-level Dirichlet parameters for each cell's phenotype distribution,
        i.e. 'cell_phenotype_concentration' of shape (N_cells, n_phenotypes).

        Returns:
            A Torch tensor of shape (batch_size, n_phenotypes) in the param store.
        """
        return pyro.param("cell_phenotype_concentration").detach().cpu()

    
    def get_gene_profiles(self) -> torch.Tensor:
        """
        Get current gene expression Dirichlet *concentrations* for each phenotype.

        In the guide, we define:
            unconstrained_profile = pyro.param("unconstrained_gene_profile_log", ...)
            gene_profile_concentration = exp(clamp(unconstrained_profile, -10, 10))
            pyro.sample("clone_gene_profiles", Dirichlet(gene_profile_concentration))

        Here, we return 'gene_profile_concentration' = exp(...) from the ParamStore,
        NOT normalized as a distribution, but as the raw Dirichlet concentration parameter.

        Returns:
            A Torch tensor of shape (n_phenotypes, n_genes).
        """
        unconstrained_profile = pyro.param("unconstrained_gene_profile_log").detach().cpu()
        log_clamped = torch.clamp(unconstrained_profile, -10, 10)
        gene_profile_conc = torch.exp(log_clamped)
        return gene_profile_conc

    
    def get_patient_effects(self) -> torch.Tensor:
        """
        Get the current patient variance parameters (Gamma shape and rate)
        from the ParamStore. These are arrays of shape (C,).

        Returns:
            A tuple (shape, rate), each a Torch tensor of length C (number of patients).
        """
        shape = pyro.param("patient_variance_shape").detach().cpu()
        rate = pyro.param("patient_variance_rate").detach().cpu()
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

    def _get_kct_index(self, k_idx: torch.Tensor, c_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """
        Flatten (TCR index = k, patient index = c, time index = t)
        => single index in [0..(K*C*T - 1)].
        We do: idx = k*(C*T) + c*T + t.
        """
        return k_idx*(self.C*self.T) + c_idx*self.T + t_idx

    def _reconstruct_single_distribution(self, clone_idx: int, patient_idx: int, time_idx: int) -> torch.Tensor:
        """
        Reconstruct a single point-estimate distribution p(phi| clone_idx, patient_idx, time_idx)
        via baseline + mean(Gamma scale) + offset. Return it as a torch.Tensor shape (n_phenotypes,).
        """
        import torch
        eps = 1e-12

        # Retrieve baseline => (K*C, n_phenotypes)
        baseline_param = pyro.param("local_clone_concentration_pc").detach().cpu()
        baseline_idx = clone_idx * self.C + patient_idx
        baseline_kc = baseline_param[baseline_idx]  # shape => (n_phenotypes,)

        beta_val = self.beta
        offset_val = self.local_concentration_offset

        if self.T > 0:
            # Retrieve shape & rate => (K*C*T, n_phenotypes)
            shape_all = pyro.param("scale_factor_shape").detach().cpu()
            rate_all  = pyro.param("scale_factor_rate").detach().cpu()
            kct_index = clone_idx*(self.C*self.T) + patient_idx*self.T + time_idx
            shape_kct = shape_all[kct_index]  # shape => (n_phenotypes,)
            rate_kct  = rate_all[kct_index]
            mean_scale = shape_kct / (rate_kct + eps)
        else:
            mean_scale = torch.ones_like(baseline_kc)

        child_conc = beta_val * baseline_kc * mean_scale + offset_val
        child_conc = torch.clamp(child_conc, min=eps)
        # (We do not strictly have to normalize here, but let's make it a distribution.)
        dist = child_conc / torch.clamp(child_conc.sum(), min=eps)
        return dist

    def get_phenotypic_flux_map(
        self,
        clone_idx: int,
        patient_idx: int,
        time1: int,
        time2: int,
        metric: str = "l1",
        temperature: float = 1.0
    ) -> float:
        """
        Compute phenotypic flux between time1 and time2 for a specific (clone_idx, patient_idx),
        using the MAP-like concentration 'p_kct_final_conc' from the guide. Then apply an
        optional temperature transform and measure L1 or KL divergence.

        Args:
            clone_idx: integer clone index in [0..K-1]
            patient_idx: integer patient index in [0..C-1]
            time1: integer time index in [0..T-1]
            time2: integer time index in [0..T-1]
            metric: one of {"l1", "dkl"} for flux distance
            temperature: float > 0, post-hoc exponent transform p^(1/tau)

        Returns:
            A float representing the distance between the two distributions, e.g. L1 or DKL.
        """
        import numpy as np
        import torch
        import math

        eps = 1e-12
        p_store = pyro.get_param_store()

        # Retrieve p_kct_final_conc => shape (K*C*T, n_phenotypes)
        p_kct_final_conc = p_store["p_kct_final_conc"].detach().cpu()

        # Flatten index => kct_idx = k*(C*T) + c*T + t
        def get_mean_distribution(k_idx, c_idx, t_idx):
            kct_index = k_idx*(self.C*self.T) + c_idx*self.T + t_idx
            conc = p_kct_final_conc[kct_index]  # shape => (n_phenotypes,)
            sum_conc = torch.sum(conc).clamp_min(eps)
            p_mean = conc / sum_conc
            return p_mean.numpy()

        # 1) Retrieve the two mean distributions
        p1 = get_mean_distribution(clone_idx, patient_idx, time1)
        p2 = get_mean_distribution(clone_idx, patient_idx, time2)

        # 2) Temperature transform
        if abs(temperature - 1.0) > 1e-9:
            p1 = np.clip(p1, eps, None)**(1.0 / temperature)
            p2 = np.clip(p2, eps, None)**(1.0 / temperature)

        # 3) Normalize
        p1 /= np.clip(p1.sum(), eps, None)
        p2 /= np.clip(p2.sum(), eps, None)

        # 4) Compute distance
        if metric.lower() == "l1":
            flux_val = float(np.sum(np.abs(p1 - p2)))
        elif metric.lower() == "dkl":
            # DKL(p1 || p2) = sum( p1[i] * ln( p1[i]/p2[i] ) )
            mask = (p1 > eps)
            flux_val = float(np.sum(p1[mask] * np.log(p1[mask] / np.clip(p2[mask], eps, None))))
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'l1' or 'dkl'.")

        return flux_val


    def get_phenotypic_flux_posterior(
        self,
        clone_idx: int,
        patient_idx: int,
        time1: int,
        time2: int,
        metric: str = "l1",
        n_samples: int = 100,
        temperature: float = 1.0
    ) -> float:
        """
        Bayesian approach to phenotypic flux. For a specific clone & patient at two timepoints,
        we:
        1) generate posterior samples => (n_samples, n_phenotypes) for time1
        2) average => distribution p1, apply temperature transform
        3) generate posterior samples => (n_samples, n_phenotypes) for time2
        4) average => distribution p2, apply temperature transform
        5) measure L1 or KL difference

        Args:
            clone_idx: int in [0..K-1]
            patient_idx: int in [0..C-1]
            time1: int in [0..T-1]
            time2: int in [0..T-1]
            metric: one of {"l1", "dkl"}
            n_samples: how many posterior draws for each time
            temperature: p^(1/tau) transform post-hoc

        Returns:
            flux_val: float, distance between the two posterior-averaged distributions.
        """
        import numpy as np
        import math

        eps = 1e-12

        # Helper: temperature transform + clamp + normalize
        def apply_temperature(vec: np.ndarray, tau: float) -> np.ndarray:
            v = np.clip(vec, eps, None)
            if abs(tau - 1.0) < 1e-9:
                s = v.sum()
                return v / (s if s > eps else 1.0)
            power = v**(1.0 / tau)
            denom = power.sum()
            if denom < eps:
                return np.ones_like(v) / len(v)
            return power / denom

        # 1) Posterior samples => time1 => shape (n_samples, n_phenotypes)
        post1 = self.generate_posterior_samples(
            patient_label=self.dataset.covariate_mapping_inv[patient_idx], # or some direct map
            tcr_label=self.clone_mapping[clone_idx],
            timepoint_label=(self.dataset.timepoint_mapping_inv[time1] if self.T>0 else None),
            n_samples=n_samples,
            include_patient_effects=True
        )
        # => post1["phenotype_samples"]. shape => (n_samples, n_phenotypes)
        p1_array = post1["phenotype_samples"]
        # average => distribution p1
        p1_mean = p1_array.mean(axis=0)

        # 2) Posterior samples => time2 => shape (n_samples, n_phenotypes)
        post2 = self.generate_posterior_samples(
            patient_label=self.dataset.covariate_mapping_inv[patient_idx],
            tcr_label=self.clone_mapping[clone_idx],
            timepoint_label=(self.dataset.timepoint_mapping_inv[time2] if self.T>0 else None),
            n_samples=n_samples,
            include_patient_effects=True
        )
        p2_array = post2["phenotype_samples"]
        p2_mean = p2_array.mean(axis=0)

        # 3) Apply temperature transform
        p1 = apply_temperature(p1_mean, temperature)
        p2 = apply_temperature(p2_mean, temperature)

        # 4) Distance
        if metric.lower() == "l1":
            flux_val = float(np.sum(np.abs(p1 - p2)))
        elif metric.lower() == "dkl":
            mask = (p1> eps)
            flux_val = float(np.sum(p1[mask]* np.log(p1[mask]/ np.clip(p2[mask], eps, None))))
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'l1' or 'dkl'.")

        return flux_val


    def get_phenotypic_entropy_by_clone_map(
        self,
        patient_idx: int,
        time_idx: int,
        weight_by_clone_freq: bool = False,
        normalize: bool = False,
        temperature: float = 1.0
    ) -> Dict[int, float]:
        """
        Deterministic (MAP-like) version of H(Phenotype | Clone=k) for each clone k at 
        a fixed (patient_idx, time_idx), reading the mean distribution from 'p_kct_final_conc'.

        Steps:
        1) Possibly gather real clone frequencies from adata.obs["clone_size"] => p(k).
        2) For each clone k:
            - Flatten index => kct_idx = k*(C*T) + patient_idx*T + time_idx => read p_kct_final_conc[kct_idx].
            - Normalize => p(phi|k), optionally apply temperature transform, weigh by p(k).
            - Compute H(Phenotype|k) in nats => -sum p(phi)*ln p(phi).
        3) If normalize=True, divide by ln(n_phenotypes).
        4) Return {clone_idx -> entropy_value}.

        Args:
            patient_idx: int in [0..C-1]
            time_idx:   int in [0..T-1], or 0 if T=0
            weight_by_clone_freq: if True, weight each clone's distribution by p(k) from adata.obs["clone_size"].
            normalize: if True, divide by ln(n_phenotypes)
            temperature: exponent transform p^(1/tau). <1 => sharpen, >1 => flatten

        Returns:
            A dict {clone_idx -> float} giving H(Phenotype|Clone=k) in nats.
        """
        import numpy as np
        import math
        import torch
        from pyro import get_param_store
        from typing import Dict

        eps = 1e-12
        K = self.K
        P = self.n_phenotypes

        # -----------------------------
        # 0) Possibly gather real clone frequencies => p(k)
        # -----------------------------
        if weight_by_clone_freq and ("clone_size" in self.adata.obs):
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values
            for i in range(len(self.dataset)):
                cell_pat = self.dataset.covariates[i].item()
                cell_time = (
                    self.dataset.timepoints[i].item()
                    if self.dataset.timepoints is not None
                    else 0
                )
                if (cell_pat == patient_idx) and (cell_time == time_idx):
                    c_id = self.dataset.tcrs[i].item()
                    freq_for_clone[c_id] += clone_sizes[i]
            total_size = freq_for_clone.sum()
            if total_size > 0:
                freq_for_clone /= total_size
            else:
                freq_for_clone[:] = 1.0 / K
        else:
            freq_for_clone = np.ones(K, dtype=np.float32) / K

        # Convert to torch if you want easy tensored math
        freq_for_clone_t = torch.tensor(freq_for_clone, dtype=torch.float)

        # -----------------------------
        # 1) Retrieve 'p_kct_final_conc' => shape (K*C*T, P)
        # -----------------------------
        p_store = get_param_store()
        p_kct_final_conc = p_store["p_kct_final_conc"].detach().cpu()  # (K*C*T, n_phenotypes)

        # Helper: temperature transform
        def temperature_transform(vec: np.ndarray, tau: float) -> np.ndarray:
            vec = np.clip(vec, eps, None)
            if tau == 1.0:
                s = vec.sum()
                return vec / (s if s > eps else 1.0)
            p_pow = vec**(1.0/tau)
            denom = p_pow.sum()
            if denom < eps:
                return np.ones_like(vec) / len(vec)
            return p_pow / denom

        phen_entropy_by_clone: Dict[int, float] = {}

        # -----------------------------
        # 2) For each clone => read single distribution p(phi|k,c,t), compute entropy
        # -----------------------------
        for k_idx in range(K):
            # flatten index => kct_idx = k*(C*T) + c*T + t
            kct_idx = k_idx*(self.C*self.T) + patient_idx*self.T + (time_idx if self.T>0 else 0)
            conc_kct = p_kct_final_conc[kct_idx, :]  # shape => (P,)
            conc_sum = float(torch.sum(conc_kct).item())
            if conc_sum < eps:
                # fallback distribution
                p_phi_given_k = np.ones(P, dtype=np.float32) / P
            else:
                # normalize
                p_phi_given_k = (conc_kct.numpy() / conc_sum).astype(np.float64)

            # optional weighting by clone freq
            if weight_by_clone_freq:
                p_phi_given_k *= freq_for_clone[k_idx]
                sum_val = p_phi_given_k.sum()
                if sum_val < eps:
                    p_phi_given_k[:] = 1.0 / P
                else:
                    p_phi_given_k /= sum_val

            # temperature
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)

            # clamp & renormalize
            sum_val2 = p_phi_given_k.sum()
            if sum_val2 < eps:
                p_phi_given_k[:] = 1.0 / P
            else:
                p_phi_given_k /= sum_val2

            # -----------------------------
            # 3) Compute entropy in nats => - sum p(phi) ln p(phi)
            # -----------------------------
            entropy_val = -np.sum(p_phi_given_k * np.log(p_phi_given_k + eps))

            # optionally normalize by ln(n_phenotypes)
            if normalize and P > 1:
                entropy_val /= math.log(P)

            phen_entropy_by_clone[k_idx] = float(entropy_val)

        return phen_entropy_by_clone

    def get_phenotypic_entropy_by_clone_posterior(
        self,
        patient_idx: int,
        time_idx: int,
        n_samples: int = 100,
        weight_by_clone_freq: bool = False,
        normalize: bool = False,
        temperature: float = 1.0
    ) -> Dict[int, float]:
        """
        For each clone k, compute the posterior-based entropy H(Phenotype | Clone=k)
        by sampling from the guide for (patient_idx, time_idx).

        Steps:
        1) Possibly gather real clone frequencies p(k) from adata.obs["clone_size"], else uniform.
        2) For each clone k:
            - Call generate_posterior_samples(...) => shape (n_samples, n_phenotypes).
            - Average => p(phi|k).
            - Optionally multiply by p(k), apply temperature transform, renormalize.
            - Compute H(Phenotype|Clone=k) = - sum_phi p(phi|k)* ln p(phi|k).
        3) If normalize=True, divide by ln(n_phenotypes).
        4) Return a dict {k_idx -> entropy_value} in nats (using ln).

        Args:
            patient_idx: int index in [0..C-1]
            time_idx: int index in [0..T-1], or 0 if T=0
            n_samples: number of posterior draws for each clone
            weight_by_clone_freq: multiply distribution by each clone’s frequency p(k)?
            normalize: if True, divides by ln(n_phenotypes)
            temperature: exponent transform p^(1/tau) to sharpen or flatten

        Returns:
            A dictionary {clone_idx -> entropy_value} in nats (unless you switch to log2).
        """
        import numpy as np
        import math
        from typing import Dict

        eps = 1e-12
        K = self.K
        P = self.n_phenotypes

        # 0) Possibly gather real clone frequencies => p(k)
        if weight_by_clone_freq and ("clone_size" in self.adata.obs):
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values
            for i in range(len(self.dataset)):
                cell_pat = self.dataset.covariates[i].item()
                cell_time = self.dataset.timepoints[i].item() if self.dataset.timepoints is not None else 0
                if (cell_pat == patient_idx) and (cell_time == time_idx):
                    c_id = self.dataset.tcrs[i].item()
                    freq_for_clone[c_id] += clone_sizes[i]
            total_size = freq_for_clone.sum()
            if total_size > 0:
                freq_for_clone /= total_size
            else:
                freq_for_clone[:] = 1.0 / K
        else:
            freq_for_clone = np.ones(K, dtype=np.float32) / K

        # 1) Map patient_idx/time_idx -> labels if T>0
        rev_patient_map = {v: k for k, v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx} in dataset.covariate_mapping.")

        if self.T > 0 and time_idx >= 0:
            rev_time_map = {v: k for k, v in self.dataset.timepoint_mapping.items()}
            time_label = rev_time_map.get(time_idx, None)
            if time_label is None:
                raise ValueError(f"Unknown time_idx={time_idx} in dataset.timepoint_mapping.")
        else:
            time_label = None

        # 2) We'll compute H(Phenotype|k) for each clone
        phen_entropy_by_clone: Dict[int, float] = {}

        def temperature_transform(vec: np.ndarray, tau: float) -> np.ndarray:
            vec = np.clip(vec, eps, None)
            if tau == 1.0:
                denom = vec.sum()
                return vec / (denom if denom > eps else 1.0)
            p_pow = vec ** (1.0 / tau)
            denom = p_pow.sum()
            if denom < eps:
                return np.ones_like(vec) / len(vec)
            return p_pow / denom

        # 3) For each clone => gather posterior samples => average => compute entropy
        for k_idx in range(K):
            tcr_label = self.clone_mapping.get(k_idx, None)
            if tcr_label is None:
                continue

            # generate posterior samples => shape (n_samples, P)
            post_res = self.generate_posterior_samples(
                patient_label=patient_label,
                tcr_label=tcr_label,
                timepoint_label=time_label,
                n_samples=n_samples,
                include_patient_effects=True
            )
            pheno_samples = post_res["phenotype_samples"]  # (n_samples, P)

            # average => p(phi|k)
            p_phi_given_k = pheno_samples.mean(axis=0)

            # optionally multiply by freq_for_clone[k]
            if weight_by_clone_freq:
                p_phi_given_k *= freq_for_clone[k_idx]
                sum_val = p_phi_given_k.sum()
                if sum_val < eps:
                    p_phi_given_k[:] = 1.0 / P
                else:
                    p_phi_given_k /= sum_val

            # temperature
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)
            # clamp & normalize
            sum_val2 = p_phi_given_k.sum()
            if sum_val2 < eps:
                p_phi_given_k[:] = 1.0 / P
            else:
                p_phi_given_k /= sum_val2

            # 4) Entropy in nats => - sum p(phi)* ln p(phi)
            entropy_val = -np.sum(p_phi_given_k * np.log(p_phi_given_k + eps))

            # if normalize => divide by ln(n_phenotypes)
            if normalize and P > 1:
                entropy_val /= math.log(P)

            phen_entropy_by_clone[k_idx] = float(entropy_val)

        return phen_entropy_by_clone


    def get_phenotypic_flux_posterior_pairwise(
        self,
        clone_idx: int,
        patient_idx: int,
        time1: int,
        time2: int,
        metric: str = "l1",
        n_samples: int = 100,
        temperature: float = 1.0
    ) -> float:
        """
        A fully Bayesian 'phenotypic flux' measure between time1 and time2 for:
        - TCR=clone_idx
        - patient=patient_idx
        by comparing *all* posterior draws pairwise (n_samples^2) and averaging
        the chosen distance.

        Steps:
        1) Map integer (patient_idx, time1, time2) -> string labels.
        2) Generate posterior samples for each timepoint:
            time1 => shape (n_samples, n_phenotypes)
            time2 => shape (n_samples, n_phenotypes)
        3) For each i in [0..n_samples-1], j in [0..n_samples-1]:
            - apply temperature transform, clamp, normalize
            - measure distance (L1 or DKL)
        4) average => final flux_value

        Args:
            clone_idx: which clone index in [0..K-1].
            patient_idx: which patient index in [0..C-1].
            time1: which time index in [0..T-1], or 0 if T=0.
            time2: which time index in [0..T-1], or 0 if T=0.
            metric: "l1" (sum of absolute differences) or "dkl" (Kullback-Leibler).
            n_samples: how many draws per distribution/timepoint.
            temperature: post-hoc exponent transform p^(1/tau).
                        <1 => sharpen, >1 => flatten.

        Returns:
            flux_value (float):
                The average pairwise distance across all sample pairs in [0..n_samples-1].
        """
        import numpy as np
        import math

        eps = 1e-12

        # --- 0) Convert (patient_idx -> patient_label), (time1, time2 -> time labels)
        rev_patient_map = {v: k for k, v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx} in dataset.covariate_mapping.")

        if self.T > 0:
            rev_time_map = {v: k for k, v in self.dataset.timepoint_mapping.items()}
            time_label1 = rev_time_map.get(time1, None)
            time_label2 = rev_time_map.get(time2, None)
            if time_label1 is None or time_label2 is None:
                raise ValueError(f"Unknown time indices {time1}, {time2} in dataset.timepoint_mapping.")
        else:
            time_label1 = None
            time_label2 = None

        # clone_idx => TCR label
        tcr_label = self.clone_mapping.get(clone_idx, None)
        if tcr_label is None:
            raise ValueError(f"Unknown clone_idx={clone_idx} in clone_mapping.")

        # --- 1) Posterior samples for time1 => shape (n_samples, n_phenotypes)
        post1 = self.generate_posterior_samples(
            patient_label=patient_label,
            tcr_label=tcr_label,
            timepoint_label=time_label1,
            n_samples=n_samples,
            include_patient_effects=True
        )
        pheno1 = post1["phenotype_samples"]  # (n_samples, n_phenotypes)

        # --- 2) Posterior samples for time2 => shape (n_samples, n_phenotypes)
        post2 = self.generate_posterior_samples(
            patient_label=patient_label,
            tcr_label=tcr_label,
            timepoint_label=time_label2,
            n_samples=n_samples,
            include_patient_effects=True
        )
        pheno2 = post2["phenotype_samples"]

        # A small helper for the temperature transform
        def process_distribution(vec: np.ndarray, tau: float) -> np.ndarray:
            # clamp
            vec = np.clip(vec, eps, None)
            if tau != 1.0:
                power = vec ** (1.0 / tau)
                denom = power.sum()
                if denom > eps:
                    vec = power / denom
                else:
                    vec = np.ones_like(vec) / len(vec)
            else:
                s = vec.sum()
                if s < eps:
                    vec = np.ones_like(vec) / len(vec)
                else:
                    vec = vec / s
            return vec

        # --- 3) Double loop => all pairs => measure distance
        total_dist = 0.0
        count = 0

        for i in range(n_samples):
            p1 = process_distribution(pheno1[i], temperature)
            for j in range(n_samples):
                p2 = process_distribution(pheno2[j], temperature)

                if metric.lower() == "l1":
                    dist_ij = np.sum(np.abs(p1 - p2))
                elif metric.lower() == "dkl":
                    # sum_{phi} p1[phi]* ln( p1[phi] / p2[phi] )
                    dist_ij = np.sum(p1 * np.log(p1 / p2))
                else:
                    raise ValueError(f"Unknown metric '{metric}', must be 'l1' or 'dkl'.")

                total_dist += dist_ij
                count += 1

        # average
        flux_value = float(total_dist / count)
        return flux_value

    def get_mutual_information_posterior(
        self,
        patient_idx: int,
        time_idx: int,
        n_samples: int = 100,
        weight_by_clone_freq: bool = False,
        temperature: float = 1.0
    ) -> float:
        """
        Bayesian approach to computing I(Clone; Phenotype) at (patient_idx, time_idx).
        We sample each clone's Dirichlet posterior p(phi|k,c,t) from the guide's param
        `p_kct_final_conc`.

        Steps:
        1) Convert integer (patient_idx,time_idx) -> their string labels (if needed).
        2) Possibly gather real clone frequencies p(k) from adata.obs["clone_size"].
        3) For each clone k in [0..K-1]:
        - Flatten index => kct_index => p_kct_final_conc[kct_index],
            build a Dirichlet(...).sample((n_samples,)).
        - Average => p(phi|k), apply temperature transform, clamp & normalize.
        - Multiply by p(k), accumulate into p(k,phi).
        4) Normalize p(k,phi) => sum_{k,phi} = 1, compute mutual information in bits.
        """
        import numpy as np
        import math
        import torch
        from pyro import get_param_store

        eps = 1e-12
        K = self.K
        P = self.n_phenotypes

        # ----------- 0) Convert (patient_idx,time_idx) to string labels if needed -----------
        # But typically we only need them if we do something with clone frequencies or checks.
        # We'll do a quick check for validity:
        if not (0 <= patient_idx < self.C):
            raise ValueError(f"Invalid patient_idx={patient_idx}, must be in [0..{self.C-1}]")
        if not (0 <= time_idx < max(self.T, 1)):
            raise ValueError(f"Invalid time_idx={time_idx}, must be in [0..{self.T-1}] if T>0")

        # ----------- 1) Possibly gather real clone frequencies p(k) -----------
        if weight_by_clone_freq and ("clone_size" in self.adata.obs):
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values
            for i in range(len(self.dataset)):
                cell_pat = self.dataset.covariates[i].item()
                cell_time = (
                    self.dataset.timepoints[i].item()
                    if self.dataset.timepoints is not None
                    else 0
                )
                if (cell_pat == patient_idx) and (cell_time == time_idx):
                    c_id = self.dataset.tcrs[i].item()
                    freq_for_clone[c_id] += clone_sizes[i]

            total_size = freq_for_clone.sum()
            if total_size > eps:
                freq_for_clone /= total_size
            else:
                freq_for_clone[:] = 1.0 / K
        else:
            # default uniform weighting
            freq_for_clone = np.ones(K, dtype=np.float32) / K

        # ----------- 2) We'll accumulate joint p(k,phi) => shape (K,P) -----------
        p_k_phi = np.zeros((K, P), dtype=np.float64)

        def temperature_transform(vec: np.ndarray, tau: float) -> np.ndarray:
            vec = np.clip(vec, eps, None)
            if tau == 1.0:
                denom = vec.sum()
                return vec / (denom if denom > eps else 1.0)
            pow_ = vec ** (1.0 / tau)
            denom = pow_.sum()
            if denom < eps:
                return np.ones_like(vec) / len(vec)
            return pow_ / denom

        # Get param store => read p_kct_final_conc => shape (K*C*T, P)
        p_store = get_param_store()
        p_kct_final_conc = p_store["p_kct_final_conc"].detach().cpu()

        # for each clone => sample from Dirichlet => average => build p(k,phi)
        for k_idx in range(K):
            # flatten index => kct_index
            # if T=0, treat time_idx=0 => kct_idx = k*C*(1) + c*(1) + 0 => we do an or if T=0
            # but let's do max(self.T,1)
            kct_index = k_idx * (self.C * max(self.T,1)) + patient_idx * max(self.T,1) + time_idx

            # concentration => shape => (P,)
            conc_kct = p_kct_final_conc[kct_index]  # shape => (P,)
            conc_kct_np = conc_kct.numpy()
            sum_conc = conc_kct_np.sum()

            if sum_conc < eps:
                # fallback
                p_phi_k_samples = np.ones((n_samples, P), dtype=np.float32) / P
            else:
                # sample from Dirichlet n_samples times
                # shape => (n_samples, P)
                dirich = torch.distributions.Dirichlet(torch.tensor(conc_kct_np, dtype=torch.float32))
                # in a loop or .sample((n_samples,)):
                samples = dirich.sample((n_samples,))  # shape => (n_samples, P)

                p_phi_k_samples = samples.numpy()

            # average => p_phi_given_k
            p_phi_given_k = p_phi_k_samples.mean(axis=0)

            # temperature
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)
            p_phi_given_k /= max(p_phi_given_k.sum(), eps)

            # multiply by p(k)
            p_k = freq_for_clone[k_idx]
            p_k_phi[k_idx, :] = p_k * p_phi_given_k

        # ----------- 3) Normalize p(k,phi) => sum_{k,phi}=1 -----------
        total_mass = p_k_phi.sum()
        if total_mass < eps:
            p_k_phi[:] = 1.0 / (K*P)
        else:
            p_k_phi /= total_mass

        # p(k)= sum_{phi} p(k,phi), p(phi)= sum_{k} p(k,phi)
        px = p_k_phi.sum(axis=1)  # shape (K,)
        py = p_k_phi.sum(axis=0)  # shape (P,)

        # ----------- 4) Compute mutual information in bits -----------
        mutual_info = 0.0
        for k_idx in range(K):
            for phi_idx in range(P):
                val = p_k_phi[k_idx, phi_idx]
                if val > eps:
                    denom = px[k_idx] * py[phi_idx]
                    if denom > eps:
                        mutual_info += val * math.log2(val / denom)

        return float(mutual_info)


    def get_mutual_information_map(
        self,
        patient_idx: int,
        time_idx: int,
        weight_by_clone_freq: bool = False,
        temperature: float = 1.0
    ) -> float:
        """
        Deterministic (MAP) version of I(Clone; Phenotype) at (patient_idx, time_idx).
        We read the 'p_kct_final_conc' param from the guide, flatten each row => p(phi|k),
        apply optional temperature transforms, weigh by p(k) if requested, and compute mutual info in bits.
        """

        import numpy as np
        import math
        import torch
        from pyro import get_param_store

        eps = 1e-12
        K = self.K
        P = self.n_phenotypes

        # Quick checks
        if not (0 <= patient_idx < self.C):
            raise ValueError(f"Invalid patient_idx={patient_idx}, must be in [0..{self.C-1}]")
        if not (0 <= time_idx < max(self.T,1)):
            raise ValueError(f"Invalid time_idx={time_idx}, must be in [0..{self.T-1}] if T>0")

        # 1) Possibly gather real clone frequencies
        if weight_by_clone_freq and ("clone_size" in self.adata.obs):
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values
            for i in range(len(self.dataset)):
                cell_pat = self.dataset.covariates[i].item()
                cell_time = (
                    self.dataset.timepoints[i].item()
                    if self.dataset.timepoints is not None
                    else 0
                )
                if (cell_pat == patient_idx) and (cell_time == time_idx):
                    c_id = self.dataset.tcrs[i].item()
                    freq_for_clone[c_id] += clone_sizes[i]
            total_size = freq_for_clone.sum()
            if total_size > eps:
                freq_for_clone /= total_size
            else:
                freq_for_clone[:] = 1.0 / K
        else:
            freq_for_clone = np.ones(K, dtype=np.float32) / K

        # 2) Retrieve 'p_kct_final_conc' from ParamStore => shape (K*C*T, P)
        p_store = get_param_store()
        p_kct_final_conc = p_store["p_kct_final_conc"].detach().cpu().numpy()

        # We'll accumulate p(k,phi)
        p_k_phi = np.zeros((K, P), dtype=np.float64)

        def temperature_transform(vec: np.ndarray, tau: float) -> np.ndarray:
            vec = np.clip(vec, eps, None)
            if tau == 1.0:
                denom = vec.sum()
                return vec / (denom if denom > eps else 1.0)
            power = vec ** (1.0 / tau)
            d_ = power.sum()
            if d_ < eps:
                return np.ones_like(vec)/len(vec)
            return power / d_

        for k_idx in range(K):
            # flatten => kct_index
            kct_index = k_idx*(self.C*max(self.T,1)) + patient_idx*max(self.T,1) + time_idx
            conc = p_kct_final_conc[kct_index]  # shape => (P,)
            s = conc.sum()
            if s < eps:
                p_phi_given_k = np.ones(P, dtype=np.float32)/P
            else:
                p_phi_given_k = conc / s

            # temperature
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)
            sum_ = p_phi_given_k.sum()
            if sum_ < eps:
                p_phi_given_k[:] = 1.0 / P
            else:
                p_phi_given_k /= sum_

            # multiply by p(k)
            p_k = freq_for_clone[k_idx]
            p_k_phi[k_idx, :] = p_k * p_phi_given_k

        # 3) Normalize => sum_{k,phi} = 1
        total_mass = p_k_phi.sum()
        if total_mass < eps:
            p_k_phi[:] = 1.0/(K*P)
        else:
            p_k_phi /= total_mass

        # p(k)= sum_{phi}, p(phi)= sum_{k}
        px = p_k_phi.sum(axis=1)  # shape (K,)
        py = p_k_phi.sum(axis=0)  # shape (P,)

        # 4) Mutual information => sum_{k,phi} p(k,phi)* log2( p(k,phi)/ [p(k)*p(phi)] )
        mutual_info = 0.0
        for k_idx in range(K):
            for phi_idx in range(P):
                val = p_k_phi[k_idx, phi_idx]
                if val > eps:
                    denom = px[k_idx]*py[phi_idx]
                    if denom > eps:
                        mutual_info += val * math.log2(val/denom)

        return float(mutual_info)


    def get_clonotypic_entropy_by_phenotype_map(
        self,
        patient_idx: int,
        time_idx: int,
        weight_by_clone_freq: bool = False,
        normalize: bool = False,
        temperature: float = 1.0
    ) -> Dict[int, float]:
        """
        Deterministic (MAP-like) version of H(Clone | Phenotype=phi) for each phenotype phi,
        at a fixed (patient_idx, time_idx), using the guide’s final param p_kct_final_conc.
        We return a dict {phi_idx -> entropy_in_bits}.

        Steps:
        1) Convert (patient_idx, time_idx) -> labels (if needed) but we only need them
            for weighting by clone frequencies.
        2) For each clone k, flatten index => kct_index = k*(C*T) + c*T + t => read
            p_kct_final_conc[kct_index] => shape (n_phenotypes,). Normalize => p(phi|k).
        3) Optionally apply temperature transform => p^(1/tau).
        4) Weight each clone by p(k), if weight_by_clone_freq=True => real frequencies,
            else uniform => 1/K => build p(k, phi) = p(k)* p(phi|k).
        5) Sum => p(phi) = sum_k p(k,phi). Then H(Clone|phi) = sum_k p(k|phi) log(1/p(k|phi)).
        6) Return in bits (log base 2). If normalize=True, divide by log2(K).

        Args:
            patient_idx: integer index in [0..C-1].
            time_idx: integer index in [0..T-1] (or 0 if T=0).
            weight_by_clone_freq: whether to gather clone frequencies from adata.obs["clone_size"].
            normalize: if True, divide each entropy by log2(K).
            temperature: exponent for post-hoc sharpening (temp<1) or flattening (temp>1).

        Returns:
            clonotypic_entropy_dict: dict {phi_idx -> float} where each value is H(Clone|phi)
                                    in bits (base-2).
        """
        import numpy as np
        import math
        import torch
        from pyro import get_param_store
        from typing import Dict

        eps = 1e-12
        K = self.K
        P = self.n_phenotypes

        # -----------------------------
        # 0) Possibly gather real clone frequencies p(k)
        # -----------------------------
        if weight_by_clone_freq and ("clone_size" in self.adata.obs):
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values
            for i in range(len(self.dataset)):
                cell_pat = self.dataset.covariates[i].item()
                cell_time = (
                    self.dataset.timepoints[i].item()
                    if self.dataset.timepoints is not None
                    else 0
                )
                if (cell_pat == patient_idx) and (cell_time == time_idx):
                    c_id = self.dataset.tcrs[i].item()
                    freq_for_clone[c_id] += clone_sizes[i]
            total_size = freq_for_clone.sum()
            if total_size > 0:
                freq_for_clone /= total_size
            else:
                freq_for_clone[:] = 1.0 / K
        else:
            freq_for_clone = np.ones(K, dtype=np.float32) / K

        # Convert freq_for_clone to a torch tensor for easy math
        freq_for_clone_t = torch.tensor(freq_for_clone, dtype=torch.float)

        # -----------------------------
        # 1) Retrieve p_kct_final_conc => shape (K*C*T, P)
        # -----------------------------
        p_store = get_param_store()
        p_kct_final_conc = p_store["p_kct_final_conc"].detach().cpu()  # (K*C*T, P)

        # We'll accumulate p(k, phi) => shape (K, P)
        p_k_phi = torch.zeros(K, P, dtype=torch.float)

        # helper: temperature transform
        def temperature_transform(vec: torch.Tensor, tau: float) -> torch.Tensor:
            vec = torch.clamp(vec, min=eps)
            if tau == 1.0:
                s = vec.sum()
                return vec / max(s, eps)
            pow_ = vec**(1.0 / tau)
            denom = pow_.sum()
            return pow_ / max(denom, eps)

        # -----------------------------
        # 2) For each clone => read single distribution => apply temperature => multiply by p(k)
        # -----------------------------
        for k_idx in range(K):
            # Flatten index => kct = k*(C*T) + patient_idx*T + time_idx
            kct_idx = k_idx*(self.C*self.T) + patient_idx*self.T + (time_idx if self.T > 0 else 0)
            conc = p_kct_final_conc[kct_idx, :]  # shape (P,)
            conc_sum = torch.sum(conc).clamp_min(eps)

            # p(phi|k) = conc / sum
            p_phi_given_k = conc / conc_sum

            # temperature
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)

            # re-normalize if needed
            p_sum = p_phi_given_k.sum().clamp_min(eps)
            p_phi_given_k = p_phi_given_k / p_sum

            # multiply by p(k)
            p_k = freq_for_clone_t[k_idx]
            p_k_phi[k_idx, :] = p_k * p_phi_given_k

        # -----------------------------
        # 3) Normalize => sum_{k,phi}=1
        # -----------------------------
        total_mass = p_k_phi.sum().item()
        if total_mass < eps:
            p_k_phi[:] = 1.0 / (K * P)
        else:
            p_k_phi /= total_mass

        # p(phi) = sum_k p(k, phi)
        p_phi = torch.sum(p_k_phi, dim=0)  # shape (P,)

        # -----------------------------
        # 4) For each phenotype => H(Clone|phi) in bits
        #     = - sum_k p(k|phi) log2 p(k|phi)
        # -----------------------------
        clonotypic_entropy_dict: Dict[int, float] = {}

        for phi_idx in range(P):
            p_phi_val = p_phi[phi_idx].item()
            if p_phi_val < eps:
                clonotypic_entropy_dict[phi_idx] = 0.0
                continue

            # p(k|phi) = p(k, phi)/ p(phi)
            p_k_given_phi = p_k_phi[:, phi_idx] / p_phi_val
            p_k_given_phi = torch.clamp(p_k_given_phi, min=eps)
            p_k_given_phi = p_k_given_phi / p_k_given_phi.sum().clamp_min(eps)

            # Entropy in bits
            H_k_phi = -torch.sum(p_k_given_phi * torch.log2(p_k_given_phi))

            # normalize by log2(K) if requested
            if normalize and K > 1:
                H_k_phi = H_k_phi / math.log2(K)

            clonotypic_entropy_dict[phi_idx] = float(H_k_phi.item())

        return clonotypic_entropy_dict

    def get_clonotypic_entropy_by_phenotype_posterior(
        self,
        patient_idx: int,
        time_idx: int,
        n_samples: int = 100,
        weight_by_clone_freq: bool = False,
        normalize: bool = False,
        temperature: float = 1.0
    ) -> Dict[int, float]:
        """
        A fully Bayesian approach to H(Clone | Phenotype=phi) for (patient_idx, time_idx),
        by sampling from the posterior for each clone’s phenotype distribution.

        Steps:
        1) Possibly gather clone frequencies p(k) from adata.obs["clone_size"], or use uniform.
        2) For each clone k:
            - Generate n_samples draws of p(phi|k,c,t) via generate_posterior_samples(...).
            - Average => p_phi_given_k (a distribution over phenotypes).
            - (Optionally) apply a temperature transform p^(1/tau).
            - Weight by p(k).
        3) Build p(k,phi) = p(k)* p(phi|k), sum => p(phi).
        4) H(Clone|phi) = sum_k p(k|phi)* log(1/p(k|phi)) in bits (log2).
        5) Return a dict {phi_idx -> entropy_val}, optionally normalized by log2(K).

        Args:
            patient_idx: which patient index in [0..C-1]
            time_idx: which time index in [0..T-1] (0 if T=0)
            n_samples: how many posterior draws for each clone
            weight_by_clone_freq: gather real clone frequencies from self.adata.obs["clone_size"]
            normalize: if True, divides each entropy by log2(K)
            temperature: post-hoc exponent transform (tau>1 => flatten, tau<1 => sharpen)

        Returns:
            A dict {phi_idx -> float}, giving H(Clone | phi) in bits for each phenotype index.
        """
        import numpy as np
        import math
        import torch
        from typing import Dict

        eps = 1e-12
        K = self.K
        P = self.n_phenotypes

        # -----------------------------
        # 0) Possibly gather clone frequencies p(k)
        # -----------------------------
        if weight_by_clone_freq and ("clone_size" in self.adata.obs):
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values
            for i in range(len(self.dataset)):
                cell_pat = self.dataset.covariates[i].item()
                cell_time = (
                    self.dataset.timepoints[i].item()
                    if self.dataset.timepoints is not None
                    else 0
                )
                if (cell_pat == patient_idx) and (cell_time == time_idx):
                    c_id = self.dataset.tcrs[i].item()
                    freq_for_clone[c_id] += clone_sizes[i]
            total_size = freq_for_clone.sum()
            if total_size > 0:
                freq_for_clone /= total_size
            else:
                freq_for_clone[:] = 1.0 / K
        else:
            freq_for_clone = np.ones(K, dtype=np.float32) / K

        # -----------------------------
        # 1) Build p(k, phi) => shape (K, P)
        # -----------------------------
        p_k_phi = np.zeros((K, P), dtype=np.float64)

        # A small helper for temperature transform
        def temperature_transform(vec: np.ndarray, tau: float) -> np.ndarray:
            vec = np.clip(vec, eps, None)
            if tau == 1.0:
                s = vec.sum()
                return vec / (s if s > eps else 1.0)
            power = vec ** (1.0 / tau)
            denom = power.sum()
            if denom < eps:
                return np.ones_like(vec) / len(vec)
            return power / denom

        # -----------------------------
        # 2) For each clone => gather posterior samples => average => p(phi|k)
        # -----------------------------
        # We'll map the (patient_idx, time_idx) -> string labels needed by generate_posterior_samples
        rev_pat_map = {v: k for k, v in self.dataset.covariate_mapping.items()}
        patient_label = rev_pat_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Invalid patient_idx={patient_idx} in dataset.covariate_mapping.")

        if self.T > 0 and time_idx >= 0:
            rev_time_map = {v: k for k, v in self.dataset.timepoint_mapping.items()}
            time_label = rev_time_map.get(time_idx, None)
            if time_label is None:
                raise ValueError(f"Invalid time_idx={time_idx} in dataset.timepoint_mapping.")
        else:
            time_label = None

        for k_idx in range(K):
            # TCR label
            tcr_label = self.clone_mapping.get(k_idx, None)
            if tcr_label is None:
                continue

            # 2a) Draw from the posterior => shape (n_samples, P)
            posterior_res = self.generate_posterior_samples(
                patient_label=patient_label,
                tcr_label=tcr_label,
                timepoint_label=time_label,
                n_samples=n_samples,
                include_patient_effects=True
            )
            pheno_samples = posterior_res["phenotype_samples"]  # shape => (n_samples, P)

            # 2b) average => p(phi|k)
            p_phi_given_k = pheno_samples.mean(axis=0)

            # 2c) temperature
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)

            # clamp & renormalize
            s2 = p_phi_given_k.sum()
            if s2 < eps:
                p_phi_given_k[:] = 1.0 / P
            else:
                p_phi_given_k /= s2

            # 2d) multiply by p(k)
            p_k_phi[k_idx, :] = freq_for_clone[k_idx] * p_phi_given_k

        # -----------------------------
        # 3) Normalize => sum_{k,phi} = 1
        # -----------------------------
        total_mass = p_k_phi.sum()
        if total_mass < eps:
            p_k_phi[:] = 1.0 / (K * P)
        else:
            p_k_phi /= total_mass

        # p(phi) = sum_k p(k, phi)
        p_phi = p_k_phi.sum(axis=0)  # shape (P,)

        # -----------------------------
        # 4) For each phenotype => H(Clone|phi) in bits
        # -----------------------------
        clonotypic_entropy_dict: Dict[int, float] = {}

        for phi_idx in range(P):
            p_phi_val = p_phi[phi_idx]
            if p_phi_val < eps:
                clonotypic_entropy_dict[phi_idx] = 0.0
                continue

            # p(k|phi) = p(k,phi)/ p(phi)
            p_k_given_phi = p_k_phi[:, phi_idx] / p_phi_val
            p_k_given_phi = np.clip(p_k_given_phi, eps, None)
            sum_k = p_k_given_phi.sum()
            if sum_k < eps:
                # fallback
                p_k_given_phi[:] = 1.0 / K
            else:
                p_k_given_phi /= sum_k

            # H(Clone|phi) = - sum_k p(k|phi)* log2 p(k|phi)
            H_val = -np.sum(p_k_given_phi * np.log2(p_k_given_phi))

            # optionally normalize by log2(K)
            if normalize and K > 1:
                H_val /= math.log2(K)

            clonotypic_entropy_dict[phi_idx] = float(H_val)

        return clonotypic_entropy_dict


    def get_renyi_entropy_by_clone_map(
        self,
        patient_idx: int,
        time_idx: int,
        alpha: float = 2.0,
        weight_by_clone_freq: bool = False,
        temperature: float = 1.0
    ) -> Dict[int, float]:
        """
        Compute the Rényi-alpha entropy of p(Phenotype|Clone=k) for each clone k
        in a *deterministic (MAP)* manner, by reading 'p_kct_final_conc' from the param store.

        H_alpha(p) = 1/(1-alpha)* ln [ sum_phi p(phi)^alpha ],
        for alpha != 1. (If alpha=1, you'd do the limit -> Shannon).

        Steps:
        1) Possibly gather real clone frequencies p(k) => weigh each distribution.
        2) Flatten index => p_kct_final_conc[kct_idx] => normalize => p(phi|k).
        3) Multiply by freq_for_clone[k], apply temperature p^(1/tau), re-normalize.
        4) Compute sum p(phi)^alpha => log => multiply by 1/(1-alpha).
        5) Return {k_idx -> float} for all clones.

        Args:
            patient_idx: int in [0..C-1]
            time_idx: int in [0..T-1], or 0 if T=0
            alpha: the Rényi alpha parameter (>0, alpha!=1)
            weight_by_clone_freq: if True, weigh distribution by real freq from adata.obs["clone_size"]
            temperature: exponent transform p^(1/tau).  <1 => sharpen, >1 => flatten

        Returns:
            dict {clone_idx -> renyi_entropy_value}, in natural log scale.
        """
        import numpy as np
        import math
        import torch
        from pyro import get_param_store
        from typing import Dict

        eps = 1e-12
        if abs(alpha - 1.0) < 1e-7:
            raise ValueError("For alpha ~ 1, use the Shannon-entropy function or limit approach.")

        K = self.K
        P = self.n_phenotypes

        # ------------------- (A) Gather clone frequencies p(k) if needed -------------------
        if weight_by_clone_freq and ("clone_size" in self.adata.obs):
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values
            for i in range(len(self.dataset)):
                cell_pat = self.dataset.covariates[i].item()
                cell_time = (
                    self.dataset.timepoints[i].item()
                    if self.dataset.timepoints is not None
                    else 0
                )
                if (cell_pat == patient_idx) and (cell_time == time_idx):
                    c_id = self.dataset.tcrs[i].item()
                    freq_for_clone[c_id] += clone_sizes[i]
            total_size = freq_for_clone.sum()
            if total_size > 0:
                freq_for_clone /= total_size
            else:
                freq_for_clone[:] = 1.0 / K
        else:
            freq_for_clone = np.ones(K, dtype=np.float32) / K

        # ------------------- (B) Retrieve param p_kct_final_conc => shape (K*C*T, P) -------------------
        p_store = get_param_store()
        p_kct_final_conc = p_store["p_kct_final_conc"].detach().cpu().numpy()  # (K*C*T, P)

        # helper
        def temperature_transform(vec: np.ndarray, tau: float) -> np.ndarray:
            vec = np.clip(vec, eps, None)
            if tau == 1.0:
                s = vec.sum()
                return vec / (s if s > eps else 1.0)
            p_pow = vec ** (1.0 / tau)
            denom = p_pow.sum()
            if denom < eps:
                return np.ones_like(vec) / len(vec)
            return p_pow / denom

        renyi_by_clone: Dict[int, float] = {}

        # ------------------- (C) For each clone => flatten index, normalize => p(phi|k) => compute H_alpha -------------------
        for k_idx in range(K):
            kct_idx = k_idx*(self.C*self.T) + patient_idx*self.T + (time_idx if self.T>0 else 0)
            conc_kct = p_kct_final_conc[kct_idx, :]  # shape => (P,)
            conc_sum = conc_kct.sum()
            if conc_sum < eps:
                p_phi_given_k = np.ones(P, dtype=np.float64)/P
            else:
                p_phi_given_k = (conc_kct / conc_sum).astype(np.float64)

            # optional weighting by clone freq
            if weight_by_clone_freq:
                p_phi_given_k *= freq_for_clone[k_idx]
                s_val = p_phi_given_k.sum()
                if s_val < eps:
                    p_phi_given_k[:] = 1.0 / P
                else:
                    p_phi_given_k /= s_val

            # temperature transform
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)

            # clamp & normalize
            sum_val = p_phi_given_k.sum()
            if sum_val < eps:
                p_phi_given_k[:] = 1.0 / P
            else:
                p_phi_given_k /= sum_val

            # ------------------- (D) Compute Rényi alpha-entropy => 1/(1-alpha)* ln( sum p^alpha ) -------------------
            p_pow_alpha = p_phi_given_k**alpha
            sum_pow = np.sum(p_pow_alpha)
            H_alpha = (1.0 / (1.0 - alpha)) * math.log(sum_pow + eps)

            renyi_by_clone[k_idx] = float(H_alpha)

        return renyi_by_clone

    def get_renyi_entropy_by_clone_posterior(
        self,
        patient_idx: int,
        time_idx: int,
        alpha: float = 2.0,
        n_samples: int = 100,
        weight_by_clone_freq: bool = False,
        temperature: float = 1.0
    ) -> Dict[int, float]:
        """
        Bayesian version of the Rényi-alpha entropy:
        H_alpha(p) = 1/(1-alpha)* ln [ sum p(phi)^alpha ].
        Instead of a single MAP distribution, we:
        - For each clone k => call generate_posterior_samples(...) => shape (n_samples, P).
        - Average => p(phi|k).
        - Possibly multiply by freq_for_clone[k], apply temperature => renormalize.
        - Then compute Rényi alpha-entropy of that final distribution.

        If alpha=1, you should use a Shannon approach or limit, not this direct formula.

        Args:
            patient_idx: int in [0..C-1]
            time_idx:   int in [0..T-1], or 0 if T=0
            alpha: the Rényi alpha parameter (>0, alpha != 1)
            n_samples: how many posterior draws per clone
            weight_by_clone_freq: weigh distribution by real clone frequencies?
            temperature: exponent transform p^(1/tau) to sharpen or flatten

        Returns:
            A dict {k_idx -> float} of H_alpha in nats (natural log),
            unless you switch to log2 in the code.
        """
        import numpy as np
        import math
        from typing import Dict

        eps = 1e-12
        if abs(alpha - 1.0) < 1e-7:
            raise ValueError("For alpha ~ 1, use Shannon or the limit approach.")

        K = self.K
        P = self.n_phenotypes

        # (A) Possibly gather real clone frequencies p(k)
        if weight_by_clone_freq and ("clone_size" in self.adata.obs):
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values
            for i in range(len(self.dataset)):
                cell_pat = self.dataset.covariates[i].item()
                cell_time = (
                    self.dataset.timepoints[i].item()
                    if self.dataset.timepoints is not None
                    else 0
                )
                if (cell_pat == patient_idx) and (cell_time == time_idx):
                    c_id = self.dataset.tcrs[i].item()
                    freq_for_clone[c_id] += clone_sizes[i]
            total_size = freq_for_clone.sum()
            if total_size > 0:
                freq_for_clone /= total_size
            else:
                freq_for_clone[:] = 1.0 / K
        else:
            freq_for_clone = np.ones(K, dtype=np.float32) / K

        # (B) Convert (patient_idx, time_idx) to labels if T>0
        rev_patient_map = {v: k for k, v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx} in dataset.covariate_mapping.")

        if self.T > 0 and time_idx >= 0:
            rev_time_map = {v: k for k, v in self.dataset.timepoint_mapping.items()}
            time_label = rev_time_map.get(time_idx, None)
            if time_label is None:
                raise ValueError(f"Unknown time_idx={time_idx} in dataset.timepoint_mapping.")
        else:
            time_label = None

        def temperature_transform(vec: np.ndarray, tau: float) -> np.ndarray:
            vec = np.clip(vec, eps, None)
            if tau == 1.0:
                s = vec.sum()
                return vec / (s if s>eps else 1.0)
            p_pow = vec**(1.0/tau)
            denom = p_pow.sum()
            if denom < eps:
                return np.ones_like(vec)/len(vec)
            return p_pow / denom

        renyi_by_clone: Dict[int, float] = {}

        # (C) For each clone => draw posterior => average => p(phi|k)
        for k_idx in range(K):
            tcr_label = self.clone_mapping.get(k_idx, None)
            if tcr_label is None:
                continue

            # posterior samples => shape (n_samples, P)
            post_res = self.generate_posterior_samples(
                patient_label=patient_label,
                tcr_label=tcr_label,
                timepoint_label=time_label,
                n_samples=n_samples,
                include_patient_effects=True
            )
            pheno_samples = post_res["phenotype_samples"]  # shape => (n_samples, P)

            # average => p_phi_given_k
            p_phi_given_k = pheno_samples.mean(axis=0)  # shape => (P,)

            # optional weighting by freq
            if weight_by_clone_freq:
                p_phi_given_k *= freq_for_clone[k_idx]
                s_val = p_phi_given_k.sum()
                if s_val < eps:
                    p_phi_given_k[:] = 1.0 / P
                else:
                    p_phi_given_k /= s_val

            # temperature
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)
            # clamp & normalize
            sum_val = p_phi_given_k.sum()
            if sum_val < eps:
                p_phi_given_k[:] = 1.0 / P
            else:
                p_phi_given_k /= sum_val

            # (D) Compute H_alpha => 1/(1-alpha)* ln(sum p^alpha)
            p_pow_alpha = p_phi_given_k**alpha
            sum_pow = np.sum(p_pow_alpha)
            H_alpha = (1.0/(1.0 - alpha)) * math.log(sum_pow + eps)

            renyi_by_clone[k_idx] = float(H_alpha)

        return renyi_by_clone

    def get_phenotypic_flux_posterior_crossentropy(
        self,
        clone_idx: int,
        patient_idx: int,
        time1: int,
        time2: int,
        n_samples: int = 100,
        temperature: float = 1.0,
        return_perplexity: bool = False
    ) -> float:
        """
        Posterior-based cross-entropy (and optional perplexity) for the distribution of
        phenotypes at two timepoints, p1 (time1) and p2 (time2):

        CrossEntropy(p1, p2) = - sum_phi p1(phi) * ln( p2(phi) )

        Steps:
        1) Generate posterior samples for (time1) => shape (n_samples, n_phenotypes)
            and (time2).
        2) Average each => p1, p2.
        3) Apply optional temperature transform p^(1/tau).
        4) CrossEntropy = - sum p1 log(p2).
        5) If return_perplexity=True => return exp(CrossEntropy).
            Otherwise return CrossEntropy (in nats).

        Args:
            clone_idx: which clone index in [0..K-1].
            patient_idx: which patient in [0..C-1].
            time1, time2: time indices in [0..T-1] (0 if T=0).
            n_samples: how many posterior draws per distribution.
            temperature: exponent transform p^(1/tau) for each distribution.
            return_perplexity: if True, return exp(cross_entropy).

        Returns:
            A float: Cross-entropy (in nats) or perplexity if return_perplexity=True.
        """
        import numpy as np
        import math

        eps = 1e-12

        # --- 1) Convert integers => labels
        rev_patient_map = {v: k for k, v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx} in dataset.covariate_mapping.")

        if self.T > 0:
            rev_time_map = {v: k for k, v in self.dataset.timepoint_mapping.items()}
            time_label1 = rev_time_map.get(time1, None)
            time_label2 = rev_time_map.get(time2, None)
            if time_label1 is None or time_label2 is None:
                raise ValueError(
                    f"Unknown time indices ({time1}, {time2}) in dataset.timepoint_mapping."
                )
        else:
            time_label1 = None
            time_label2 = None

        tcr_label = self.clone_mapping.get(clone_idx, None)
        if tcr_label is None:
            raise ValueError(f"Unknown clone_idx={clone_idx} in clone_mapping.")

        # --- 2) Posterior samples => average => p1, p2
        post1 = self.generate_posterior_samples(
            patient_label=patient_label,
            tcr_label=tcr_label,
            timepoint_label=time_label1,
            n_samples=n_samples,
            include_patient_effects=True
        )
        p1_array = post1["phenotype_samples"]  # shape => (n_samples, n_phenotypes)
        p1 = p1_array.mean(axis=0)

        post2 = self.generate_posterior_samples(
            patient_label=patient_label,
            tcr_label=tcr_label,
            timepoint_label=time_label2,
            n_samples=n_samples,
            include_patient_effects=True
        )
        p2_array = post2["phenotype_samples"]
        p2 = p2_array.mean(axis=0)

        # A small helper for temperature transform
        def temp_scale(vec: np.ndarray, tau: float) -> np.ndarray:
            vec = np.clip(vec, eps, None)
            if tau == 1.0:
                s = vec.sum()
                return vec / (s if s > eps else 1.0)
            v_pow = vec**(1.0 / tau)
            denom = v_pow.sum()
            if denom < eps:
                return np.ones_like(vec)/len(vec)
            else:
                return v_pow / denom

        # --- 3) Apply temperature transform => clamp & normalize
        p1 = temp_scale(p1, temperature)
        p2 = temp_scale(p2, temperature)

        # final normalization
        p1 = np.clip(p1, eps, None)
        p1 /= p1.sum()
        p2 = np.clip(p2, eps, None)
        p2 /= p2.sum()

        # --- 4) Cross-entropy H(p1,p2) = - sum p1 log p2
        cross_ent = -np.sum(p1 * np.log(p2 + eps))

        # If requested => perplexity = exp( cross-entropy )
        if return_perplexity:
            return float(np.exp(cross_ent))
        else:
            return float(cross_ent)

    def generate_posterior_samples(
        self,
        patient_label: str,
        tcr_label: str,
        timepoint_label: Optional[str] = None,
        n_samples: int = 1000,
        include_patient_effects: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Draw posterior samples for the distribution p(phenotypes | clone=k, patient=c, time=t)
        plus the associated gene expression mixture and (optionally) patient effect.

        We read from the ParamStore:
        - patient_variance_shape, patient_variance_rate => shape (C,)
        - p_kct_final_conc => shape (K*C*T, n_phenotypes)
        - gene_profile_conc => shape (n_phenotypes, D)

        Steps per sample:
        1) Sample a Dirichlet( p_kct_final_conc[kct_index] ) => shape (n_phenotypes,)
        2) Sample gene_profile_draw from Dirichlet(gene_profile_conc) => shape (n_phenotypes, D)
        3) Weighted sum => mixture_profile = sum_{p} z_i[p] * gene_profile_draw[p,:]
        4) If include_patient_effects => sample Gamma(patient_variance_shape[c], patient_variance_rate[c])
            => scale mixture_profile by that factor
        5) Normalize mixture_profile => final gene expression distribution
        """

        import numpy as np
        import torch
        import pyro
        import pyro.distributions as dist

        # 1) Convert string labels -> integer indices
        if patient_label not in self.dataset.covariate_mapping:
            raise ValueError(f"Patient label '{patient_label}' not found in dataset.covariate_mapping")
        patient_idx = self.dataset.covariate_mapping[patient_label]

        if tcr_label not in self.dataset.tcr_mapping:
            raise ValueError(f"TCR label '{tcr_label}' not found in dataset.tcr_mapping")
        k_idx = self.dataset.tcr_mapping[tcr_label]

        if self.T > 0 and timepoint_label is not None:
            if timepoint_label not in self.dataset.timepoint_mapping:
                raise ValueError(f"Timepoint label '{timepoint_label}' not found in dataset.timepoint_mapping")
            t_idx = self.dataset.timepoint_mapping[timepoint_label]
        else:
            t_idx = 0  # default if T=0 or none provided

        # 2) Flatten => kct_index
        kct_index = k_idx * (self.C * max(self.T, 1)) + patient_idx * max(self.T, 1) + t_idx

        # Retrieve param store references
        p_store = pyro.get_param_store()

        # a) Patient variance shape/rate => shape (C,)
        patient_variance_shape = p_store["patient_variance_shape"].detach().cpu()
        patient_variance_rate  = p_store["patient_variance_rate"].detach().cpu()

        # b) p_kct_final_conc => shape (K*C*T, n_phenotypes)
        p_kct_final_conc = p_store["p_kct_final_conc"].detach().cpu()
        if kct_index >= p_kct_final_conc.shape[0]:
            raise ValueError(f"Invalid kct_index={kct_index}. Check K*C*T size.")
        phen_conc_kct = p_kct_final_conc[kct_index]  # shape (n_phenotypes,)

        # c) gene_profile_conc => shape (n_phenotypes, D)
        gene_profile_conc = p_store["gene_profile_conc"].detach().cpu()

        # We'll accumulate samples
        phenotype_samples = np.zeros((n_samples, self.n_phenotypes), dtype=np.float32)
        gene_expression_samples = np.zeros((n_samples, self.D), dtype=np.float32)
        patient_effect_samples = None
        if include_patient_effects:
            patient_effect_samples = np.zeros(n_samples, dtype=np.float32)

        # For the requested patient c
        c_shape = float(patient_variance_shape[patient_idx].item())
        c_rate  = float(patient_variance_rate[patient_idx].item())

        # 3) Create distribution objects that do not change across samples
        dirichlet_kct = dist.Dirichlet(phen_conc_kct)
        dirichlet_gene = dist.Dirichlet(gene_profile_conc)

        if include_patient_effects:
            patient_gamma = dist.Gamma(
                torch.tensor(c_shape, dtype=torch.float32),
                torch.tensor(c_rate, dtype=torch.float32)
            )
        else:
            patient_gamma = None  # not used

        # 4) Sampling loop
        for i in range(n_samples):
            # (A) sample z_i => shape (n_phenotypes,)
            z_i = dirichlet_kct.sample()  # torch.Size([n_phenotypes])
            phenotype_samples[i, :] = z_i.numpy()

            # (B) sample gene profile matrix => shape (n_phenotypes, D)
            # Each phenotype row is drawn from Dirichlet(gene_profile_conc[phen,:]).
            # However, if you do dist.Dirichlet(gene_profile_conc), you sample (n_phenotypes, D) *independently*?
            # That is the typical mean-field approach. 
            # So .sample() => shape (n_phenotypes, D)
            gene_profile_draw = dirichlet_gene.sample()  # shape => (n_phenotypes, D)

            # Weighted sum => mixture => shape (D,)
            # z_i => shape (n_phenotypes,) 
            # gene_profile_draw => shape (n_phenotypes, D)
            mixture_profile = (z_i.unsqueeze(-1) * gene_profile_draw).sum(dim=0).numpy()

            # (C) sample patient effect if requested
            if include_patient_effects and patient_gamma is not None:
                pat_eff = patient_gamma.sample().item()
                mixture_profile *= pat_eff
                patient_effect_samples[i] = pat_eff

            # (D) Normalize gene distribution
            total = mixture_profile.sum() + 1e-12
            gene_expression_samples[i, :] = mixture_profile / total

        # 5) Package results
        result = {
            "phenotype_samples": phenotype_samples,
            "gene_expression_samples": gene_expression_samples,
            "phenotype_labels": [self.phenotype_mapping[i] for i in range(self.n_phenotypes)]
        }
        if include_patient_effects:
            result["patient_effect_samples"] = patient_effect_samples

        return result

    def save_model(self, path: str):
        """
        Save the current state of this JointProbabilityDistribution, including:
        - Pyro param store
        - Key attributes (like self.K, self.C, self.n_phenotypes, self.phenotype_mapping, etc.)
        - Anything else you'd want to recover exactly upon load
        """
        # 1) Grab Pyro's param store state
        param_store_state = pyro.get_param_store().get_state()

        # 2) Gather any relevant class-level data
        #    (some you might not need, but these are typical)
        class_attrs = {
            "K": self.K,
            "C": self.C,
            "T": self.T,
            "n_phenotypes": self.n_phenotypes,
            "phenotype_mapping": self.phenotype_mapping,
            "clone_mapping": self.clone_mapping,
            "reverse_clone_mapping": self.reverse_clone_mapping,
            "posterior_cache": self._posterior_cache,
        }

        # 3) Combine into one dictionary
        save_dict = {
            "param_store_state": param_store_state,
            "class_attrs": class_attrs
        }

        # 4) Use torch.save to serialize
        torch.save(save_dict, path)

        print(f"Model saved to {path}")


    @classmethod
    def load_model(cls, path: str, adata):
        """
        Load a JointProbabilityDistribution from disk. We typically pass `adata` 
        if we need a reference to the original AnnData or something. 
        If you only need to restore params, you might not need it.
        """
        import torch
        import pyro

        print(f"Loading model from {path}")
        load_dict = torch.load(path)
        param_store_state = load_dict["param_store_state"]
        class_attrs = load_dict["class_attrs"]

        # 1) Create a new instance. 
        #    We'll pass any needed parameters in the constructor. 
        #    For convenience, you can pass minimal placeholders, then override attributes:
        jpd = cls(
            adata=adata,
            tcr_label="placeholder",
            covariate_label="placeholder",
            n_phenotypes=class_attrs["n_phenotypes"]
        )
        # Or do a no-arg constructor if your class allows it.

        # 2) Overwrite class attributes 
        jpd.K = class_attrs["K"]
        jpd.C = class_attrs["C"]
        jpd.T = class_attrs["T"]
        jpd.n_phenotypes = class_attrs["n_phenotypes"]
        jpd.phenotype_mapping = class_attrs["phenotype_mapping"]
        jpd.clone_mapping = class_attrs["clone_mapping"]
        jpd.reverse_clone_mapping = class_attrs["reverse_clone_mapping"]
        # ... add others as needed ...

        # 3) Load the Pyro param store
        pyro.get_param_store().set_state(param_store_state)

        print("Model loaded successfully.")
        return jpd

    def get_gene_usage_by_phenotype(
        self,
        n_samples: int = 0,
        include_uncertainty: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve a phenotype->gene usage matrix, either MAP-like or posterior-based,
        from the Pyro param store. Each row = a phenotype, each column = a gene,
        with values ~ p(gene | phenotype).

        Args:
            n_samples (int): 
                - If 0 or less, we return a *deterministic* usage from the guide param (MAP approach).
                - If > 0, we sample from the posterior Dirichlet for "gene_profile_conc"
                n_samples times, average, then row-normalize. This yields a posterior-based
                usage profile.
            include_uncertainty (bool): 
                - If True and n_samples>0, we additionally return columns for the standard deviation
                across draws for each (phenotype,gene). This shows uncertainty in usage.

        Returns:
            A pandas DataFrame of shape (n_phenotypes, n_genes) if include_uncertainty=False,
            or (n_phenotypes, 2*n_genes) if include_uncertainty=True (one set of columns
            for 'mean' usage, one for 'std').
            Index = phenotype labels, columns = gene names (and possibly extra columns for std).
        """
        import pandas as pd
        import torch
        import numpy as np
        from pyro import get_param_store
        import pyro.distributions as dist

        eps = 1e-12
        p_store = get_param_store()

        # 1) Identify which parameter we have for gene usage:
        #    - "gene_profile_conc" is used in your new guide (Dirichlet param).
        #    - "unconstrained_gene_profile_log" is the older approach.
        if "gene_profile_conc" in p_store:
            # shape => (n_phenotypes, n_genes)
            gene_profile_conc = p_store["gene_profile_conc"].detach().cpu()
            n_phenotypes, n_genes = gene_profile_conc.shape

            if n_samples > 0:
                # Posterior-based approach:
                # Sample from Dirichlet(gene_profile_conc) => shape (n_samples, n_phenotypes, n_genes)
                # average => (n_phenotypes, n_genes)
                dirich = dist.Dirichlet(gene_profile_conc)
                samples = dirich.sample((n_samples,))  # shape => (n_samples, n_phen, n_genes)

                # convert to numpy => (n_samples, n_phen, n_genes)
                samples_np = samples.numpy()

                # mean usage => shape (n_phen, n_genes)
                usage_mean = np.mean(samples_np, axis=0)
                # row-normalize each row (should already be near 1, but let's be safe)
                row_sums = usage_mean.sum(axis=1, keepdims=True)
                usage_mean = np.clip(usage_mean, eps, None) / np.clip(row_sums, eps, None)

                if include_uncertainty:
                    usage_std = np.std(samples_np, axis=0)
                    # We can also re-normalize each sample row if we want, but typically
                    # Dirichlet samples are already normalized. So 'std' is the raw standard deviation
                    # across draws. You might row-normalize each sample first if that suits your interpretation.

                    # build final DataFrame with mean & std columns
                    # e.g. columns are [gene0_mean, gene0_std, gene1_mean, gene1_std, ...]
                    data_cat = []
                    col_names = []
                    for g_idx in range(n_genes):
                        data_cat.append(usage_mean[:, g_idx])
                        data_cat.append(usage_std[:, g_idx])
                        col_names.append(self.adata.var_names[g_idx] + "_mean")
                        col_names.append(self.adata.var_names[g_idx] + "_std")
                    data_cat = np.vstack(data_cat).T  # shape => (n_phen, 2*n_genes)

                    phenotype_labels = [self.phenotype_mapping[p] for p in range(n_phenotypes)]
                    df_usage = pd.DataFrame(data=data_cat, index=phenotype_labels, columns=col_names)
                    return df_usage
                else:
                    # just return the mean usage
                    phenotype_labels = [self.phenotype_mapping[p] for p in range(n_phenotypes)]
                    gene_names = list(self.adata.var_names)
                    df_usage = pd.DataFrame(
                        data=usage_mean,
                        index=phenotype_labels,
                        columns=gene_names
                    )
                    return df_usage

            else:
                # MAP-like approach:
                # gene_profile_conc is the Dirichlet concentration. We can exponentiate or we can
                # directly treat it as usage (it might be the final param). Actually, in a Dirichlet
                # setting, 'gene_profile_conc' is the raw Dirichlet alpha. We want the row-normalized version:
                usage = gene_profile_conc / torch.clamp(gene_profile_conc.sum(dim=1, keepdim=True), min=eps)
                usage_np = usage.numpy()

                phenotype_labels = [self.phenotype_mapping[p] for p in range(n_phenotypes)]
                gene_names = list(self.adata.var_names)
                df_usage = pd.DataFrame(
                    data=usage_np,
                    index=phenotype_labels,
                    columns=gene_names
                )
                return df_usage

        elif "unconstrained_gene_profile_log" in p_store:
            # fallback / older approach
            unconstrained_log = p_store["unconstrained_gene_profile_log"].detach().cpu()
            # shape => (n_phenotypes, n_genes)
            n_phenotypes, n_genes = unconstrained_log.shape

            # We'll do a MAP-like approach only, since the older code didn't store enough info
            # for easy posterior sampling.
            log_clamped = torch.clamp(unconstrained_log, -10, 10)
            concentration = torch.exp(log_clamped)
            row_sums = concentration.sum(dim=1, keepdim=True).clamp_min(eps)
            usage = concentration / row_sums
            usage_np = usage.numpy()

            phenotype_labels = [self.phenotype_mapping[p] for p in range(n_phenotypes)]
            gene_names = list(self.adata.var_names)
            df_usage = pd.DataFrame(
                data=usage_np,
                index=phenotype_labels,
                columns=gene_names
            )
            return df_usage

        else:
            raise ValueError(
                "Could not find 'gene_profile_conc' or 'unconstrained_gene_profile_log' in param store."
            )



    def compare_clone_phenotype_distributions(
        self,
        timepoint_label: str = "timepoint_simple",
        clone_label: str = "trb",
        phenotype_label_col: str = "cluster",
        temperature: float = 0.5,
        n_samples: int = 100,
        show_empirical: bool = True
    ):
        """
        Compare the learned posterior distribution over phenotypes for each
        (patient, timepoint, clone) triple to the empirical distribution 
        (if `phenotype_label_col` exists in self.adata.obs and show_empirical=True).

        Args:
            timepoint_label: column in adata.obs for timepoint
            clone_label: column in adata.obs for TCR/clones
            phenotype_label_col: column in adata.obs that stores the 'true' or 
                                'cluster' phenotype label (for empirical distribution).
            temperature: post-hoc exponent transform (<1 => sharpen, >1 => flatten),
                        on the posterior distribution.
            n_samples: how many posterior samples to draw for each triple
            show_empirical: if True, also compute and plot the empirical distribution
                            side by side.

        Returns:
            (m_matrix, e_matrix) if show_empirical=True, else (m_matrix, None)
            - m_matrix: shape (#rows, P), each row is a posterior distribution for 
                        the triple after temperature transform.
            - e_matrix: shape (#rows, P), if computed, each row is the empirical
                        distribution over the same phenotype labels.

        Side effect:
            Plots one or two heatmaps side by side, 
            labeling columns by self.phenotype_mapping.values().
        """

        # -------------------------------------------------------------------------
        # 1) Gather unique patients, timepoints, clones from adata
        # -------------------------------------------------------------------------
        adata = self.adata
        patients = sorted(set(adata.obs["patient"]))
        timepoints = sorted(set(adata.obs[timepoint_label])) if timepoint_label in adata.obs.columns else [None]
        clones = sorted(set(adata.obs[clone_label]))

        # We'll need a stable ordering of phenotype labels:
        # e.g. list(self.phenotype_mapping.values()) in ascending key order:
        # If your mapping is {0: 'PhenA', 1: 'PhenB', ...},
        # you might do something like:
        phen_keys = sorted(self.phenotype_mapping.keys())
        phenotype_labels = [self.phenotype_mapping[k] for k in phen_keys]
        P = len(phenotype_labels)

        # We'll accumulate rows of the model matrix and (optionally) the empirical matrix
        m_rows = []
        e_rows = []  # only if show_empirical=True

        # -------------------------------------------------------------------------
        # 2) Helper function: temperature transform
        # -------------------------------------------------------------------------
        def apply_temperature(vec: np.ndarray, tau: float) -> np.ndarray:
            eps = 1e-12
            vec = np.clip(vec, eps, None)
            s = vec.sum()
            if s < eps:
                # fallback uniform
                return np.ones_like(vec) / len(vec)
            if abs(tau - 1.0) < 1e-9:
                return vec / s
            power = vec ** (1.0 / tau)
            denom = power.sum()
            if denom < eps:
                return np.ones_like(vec) / len(vec)
            return power / denom

        # -------------------------------------------------------------------------
        # 3) For each (patient, timepoint, clone), build row in m_matrix / e_matrix
        # -------------------------------------------------------------------------
        for pat in patients:
            # subset to this patient
            patient_str = str(pat)  # 'patient' label
            pat_mask = (adata.obs["patient"] == pat)

            # possible timepoints or just [None] if not found
            for tpt in timepoints:
                if tpt is None:
                    # no timepoint column
                    tpt_mask = pat_mask
                    tpt_str = None
                else:
                    tpt_mask = pat_mask & (adata.obs[timepoint_label] == tpt)
                    tpt_str = str(tpt)

                # now for each clone
                for c in clones:
                    c_mask = (adata.obs[clone_label] == c)
                    full_mask = tpt_mask & c_mask
                    if not np.any(full_mask):
                        # no cells => skip
                        continue

                    # (A) Generate model distribution via posterior samples
                    # We'll call self.generate_posterior_samples
                    post_res = self.generate_posterior_samples(
                        patient_label=patient_str,
                        tcr_label=str(c),
                        timepoint_label=tpt_str,
                        n_samples=n_samples,
                        include_patient_effects=True
                    )
                    # shape => (n_samples, P)
                    phen_samples = post_res["phenotype_samples"]
                    model_dist = phen_samples.mean(axis=0)  # shape (P,)

                    # apply temperature
                    model_dist = apply_temperature(model_dist, temperature)

                    # re-order if needed. But typically, 
                    # "phenotype_samples" is in the same order as your self.phenotype_mapping keys.
                    # We'll assume the order is correct. If you need re-ordering, do it here.

                    # store in m_rows
                    m_rows.append(model_dist)

                    # (B) If show_empirical => gather empirical distribution
                    if show_empirical and (phenotype_label_col in adata.obs.columns):
                        sub_adata = adata[full_mask]
                        total_cells = len(sub_adata)
                        e_vec = np.zeros(P, dtype=np.float32)
                        if total_cells == 0:
                            # all zeros
                            e_rows.append(e_vec)
                            continue

                        # for each phenotype label in the same order as phenotype_labels
                        # we count how many cells in sub_adata.obs[phenotype_label_col]
                        # match that label
                        # e.g. sub_adata.obs["cluster"] might have strings that match
                        # each label in phenotype_labels
                        cluster_list = sub_adata.obs[phenotype_label_col].values.tolist()

                        for i, label in enumerate(phenotype_labels):
                            e_vec[i] = cluster_list.count(label)

                        # convert counts => fraction
                        e_vec /= total_cells
                        e_rows.append(e_vec)
                    elif show_empirical:
                        # no phenotype_label_col => we can't do empirical
                        # store a row of zeros
                        e_rows.append(np.zeros(P, dtype=np.float32))

        # -------------------------------------------------------------------------
        # 4) Convert lists => numpy arrays for easier plotting
        # -------------------------------------------------------------------------
        m_matrix = np.array(m_rows, dtype=np.float32) if len(m_rows) > 0 else np.zeros((0, P))
        e_matrix = None
        if show_empirical:
            if len(e_rows) == len(m_rows):
                e_matrix = np.array(e_rows, dtype=np.float32)
            else:
                # inconsistent rows => fallback
                e_matrix = None

        # -------------------------------------------------------------------------
        # 5) Plotting
        # -------------------------------------------------------------------------
        # We'll do one or two subplots depending on show_empirical & e_matrix existence
        if e_matrix is not None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            sns.heatmap(m_matrix, ax=ax[0])
            ax[0].set_title(f"Posterior Dist (temp={temperature})")
            ax[0].set_xlabel("Phenotypes")
            ax[0].set_xticks(np.arange(P) + 0.5)
            ax[0].set_xticklabels(phenotype_labels, rotation=90)

            sns.heatmap(e_matrix, ax=ax[1])
            ax[1].set_title("Empirical Distribution")
            ax[1].set_xlabel("Phenotypes")
            ax[1].set_xticks(np.arange(P) + 0.5)
            ax[1].set_xticklabels(phenotype_labels, rotation=90)

            fig.tight_layout()
            plt.show()
        else:
            # Just the model distribution
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            sns.heatmap(m_matrix, ax=ax)
            ax.set_title(f"Posterior Dist (temp={temperature})")
            ax.set_xlabel("Phenotypes")
            ax.set_xticks(np.arange(P) + 0.5)
            ax.set_xticklabels(phenotype_labels, rotation=90)
            fig.tight_layout()
            plt.show()

        return m_matrix, e_matrix
        

    def build_alluvial_flow(
        self,
        time1: int,
        time2: int,
        use_posterior: bool = False,
        n_samples: int = 100,
        flow_mode: str = "min",  # one of {"min", "outer-product"} etc. "min" is the naive approach
        weight_by_clone_size: bool = True,
        temperature: float = 1.0
    ) -> pd.DataFrame:
        """
        Construct an alluvial-flow / Sankey diagram matrix from partial phenotype
        memberships at two timepoints t1 and t2, aggregated across clones.

        Steps:
        1) For each clone c, retrieve p(f|c,t1) and p(f|c,t2).
        - If use_posterior=True, sample from the posterior and average => distribution
            for each time. If False, read MAP param p_kct_final_conc.
        2) Possibly apply a temperature transform p^(1/tau).
        3) Define a flow matrix for that clone c:
            flow_c(f1->f2) = min( p(f1|c,t1), p(f2|c,t2) )  (the naive approach)
        Then multiply by the clone size if weight_by_clone_size is True.
        4) Sum these flows across all clones => shape (P,P).
        5) Return a DataFrame with columns: ["f1","f2","flow_value"] for each pair (f1,f2).

        Args:
            time1 (int): index of the first time in [0..T-1]
            time2 (int): index of the second time in [0..T-1]
            use_posterior (bool): 
                - If True, we call self.generate_posterior_samples(...) for each clone+time,
                average => distribution. 
                - Otherwise, we read 'p_kct_final_conc' from the param store (MAP).
            n_samples (int): how many posterior draws if use_posterior=True.
            flow_mode (str): "min" is the default naive approach => flow_c(f1->f2) = min( p1[f1], p2[f2] ).
                Another possibility might be "outer-product", i.e. p1[f1]*p2[f2], but that doesn't
                strictly interpret “movement” from f1 to f2.
            weight_by_clone_size (bool): if True, multiply each clone's flow by the sum
                of that clone's cell_size in the dataset for (time1 union time2).
            temperature (float): optional post-hoc exponent transform p^(1/tau).

        Returns:
            A DataFrame with columns ["phenotype_from","phenotype_to","flow_value"] of length P*P,
            describing the global flow among phenotypes from time1 to time2.
        """
        eps = 1e-12

        # 1) Retrieve the number of clones, phenotypes
        K = self.K
        P = self.n_phenotypes
        # Reverse map from phenotype index -> label
        phen_labels = [self.phenotype_mapping[p] for p in range(P)]

        # 2) Optionally gather real clone frequencies from adata.obs["clone_size"] 
        #    or just treat each clone equally
        clone_sizes = None
        if weight_by_clone_size and ("clone_size" in self.adata.obs):
            # we'll accumulate total sizes for each clone, 
            # specifically for cells that appear at time1 or time2, if desired
            clone_sizes = np.zeros(K, dtype=np.float32)
            obs = self.adata.obs
            for i in range(len(obs)):
                c_id = self.dataset.tcrs[i].item()  # clone index
                # check if time is time1 or time2
                cell_time = self.dataset.timepoints[i].item() if self.dataset.timepoints is not None else 0
                if cell_time == time1 or cell_time == time2:
                    c_size = obs["clone_size"].iloc[i]
                    clone_sizes[c_id] += c_size
        else:
            clone_sizes = np.ones(K, dtype=np.float32)  # uniform weighting

        # Normalize clone_sizes if you prefer, or keep them raw:
        # e.g. pass

        # 3) For each clone => get distribution p(f|c,t1) and p(f|c,t2)
        # build local flow => sum in a global matrix => shape (P,P)
        global_flow = np.zeros((P,P), dtype=np.float64)

        def temperature_transform(vec: np.ndarray, tau: float) -> np.ndarray:
            v = np.clip(vec, eps, None)
            if abs(tau - 1.0) < 1e-9:
                denom = v.sum()
                return v / max(denom, eps)
            power = v ** (1.0 / tau)
            d_ = power.sum()
            if d_ < eps:
                return np.ones_like(v)/len(v)
            return power / d_

        # If not use_posterior => we retrieve p_kct_final_conc from param_store
        # otherwise we generate posterior draws
        p_store = pyro.get_param_store()
        if not use_posterior:
            # MAP approach => p_kct_final_conc shape => (K*C*T, P)
            p_kct_final_conc = p_store["p_kct_final_conc"].detach().cpu().numpy()
        else:
            p_kct_final_conc = None  # won't be used in posterior approach


        def get_distribution_map(k_idx, t_idx):
            # Flatten => kct_index
            kct_index = k_idx*(self.C*self.T) + (t_idx if self.T>0 else 0) + (self.C*self.T)*(0)  # careful
            # Actually we want => k_idx*(C*T) + patient*(T) + t
            # but we need patient index => But we might not have a single patient for each clone in your dataset?
            # Typically, each clone is stored for each patient. If your data is multi-patient, the approach changes.
            # Possibly we need a different approach if clones can appear in multiple patients. 
            # We'll assume 1 patient scenario or single global clone indexing. 
            # Otherwise, we must store p_kct_final_conc shape => (k_idx * C + c_idx)*T + t...
            # For a single-patient scenario, c_idx=0 => kct_index = k_idx*T + t_idx
            # But your flatten index is k*(C*T) + c*T + t.
            # We need the c index. Let's assume we have a single patient for demonstration.

            # For now, let's do an example for single patient c=0
            # if you do multi-patient, you need the c index from your dataset. 
            c_idx = 0
            flatten_idx = k_idx*(self.C*self.T) + c_idx*self.T + t_idx
            row = p_kct_final_conc[flatten_idx, :]
            sum_ = row.sum()
            if sum_ < eps:
                usage = np.ones(P)/P
            else:
                usage = row/sum_
            return usage

        def get_distribution_posterior(k_idx, t_idx):
            # We'll call self.generate_posterior_samples. We also need which patient or time
            # In your code, each clone belongs to a specific patient or multiple? 
            # We might need to store c_idx => We'll do c_idx=0 for demonstration. 
            c_idx = 0
            # we do time_label = self.dataset.timepoint_mapping_inv[t_idx] if T>0 else None
            # tcr_label = self.clone_mapping[k_idx]
            # patient_label = some approach. 
            # This approach is simplified. If your data actually has multi-patient, you must adapt code.

            # We'll skip the actual code call for demonstration:
            # post_res = self.generate_posterior_samples(
            #   patient_label= "0",
            #   tcr_label= tcr_label,
            #   timepoint_label = str(t_idx),
            #   n_samples= n_samples,
            #   include_patient_effects = True
            # )
            # pheno_samples => shape => (n_samples, P)
            # usage = pheno_samples.mean(axis=0)
            usage = np.ones(P)/P  # placeholder
            return usage

        for k_idx in range(K):
            # For each clone => get usage t1, usage t2
            if not use_posterior:
                p1 = get_distribution_map(k_idx, time1)
                p2 = get_distribution_map(k_idx, time2)
            else:
                p1 = get_distribution_posterior(k_idx, time1)
                p2 = get_distribution_posterior(k_idx, time2)

            # temperature
            p1 = temperature_transform(p1, temperature)
            p2 = temperature_transform(p2, temperature)

            # flow_c => shape (P,P)
            flow_c = np.zeros((P,P), dtype=np.float64)
            if flow_mode == "min":
                for f1 in range(P):
                    for f2 in range(P):
                        flow_c[f1,f2] = min(p1[f1], p2[f2])
            elif flow_mode == "outer-product":
                # e.g. flow_c[f1,f2] = p1[f1]*p2[f2]
                # that doesn't strictly interpret "movement" but is an alternative
                flow_c = np.outer(p1, p2)
            else:
                raise ValueError(f"Unknown flow_mode={flow_mode}. Use 'min' or 'outer-product' etc.")

            # multiply by clone_size if requested
            w = float(clone_sizes[k_idx])
            flow_c *= w

            # add to global_flow
            global_flow += flow_c

        # Now global_flow[f1,f2] => total flow from phenotype f1 at t1 to phenotype f2 at t2
        # Convert to a DataFrame => with columns = ["phenotype_from","phenotype_to","flow_value"]
        rows = []
        for f1 in range(P):
            for f2 in range(P):
                rows.append({
                    "phenotype_from": phen_labels[f1],
                    "phenotype_to": phen_labels[f2],
                    "flow_value": global_flow[f1,f2]
                })
        df_flow = pd.DataFrame(rows)
        return df_flow