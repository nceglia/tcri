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
                self.timepoints[idx]  # new
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

    def _model(self, matrix, tcr_idx, patient_idx, time_idx, phenotype_probs):

        batch_size = matrix.shape[0]

        # 1) Patient variance (Gamma) for gene-scaling
        with pyro.plate("patient_global", self.C):
            patient_variance = pyro.sample(
                "patient_variance",
                dist.Gamma(
                    torch.ones(1, device=self.device)*self.patient_variance_shape_val,
                    torch.ones(1, device=self.device)*self.patient_variance_rate_val
                )
            )

        # 2) Baseline usage (k,c): Dirichlet
        #    shape => (K*C, n_phenotypes)
        baseline_prior = (
            self.clone_to_phenotype_prior * self.clone_to_phenotype_prior_strength
            + 1.0
        )
        expanded_baseline_prior = baseline_prior.repeat_interleave(self.C, dim=0)

        with pyro.plate("clone_patient_baseline", self.K*self.C):
            local_clone_phenotype_pc = pyro.sample(
                "local_clone_phenotype_pc",
                dist.Dirichlet(expanded_baseline_prior)
            )

        # 3) Dimension-wise Gamma scale for each timepoint (k,c,t), if T>0
        #    shape => (K*C*T, n_phenotypes)
        if self.T > 0:
            with pyro.plate("clone_patient_time", self.K*self.C*self.T):
                scale_factor_kct = pyro.sample(
                    "scale_factor_kct",
                    dist.Gamma(
                        torch.ones(self.n_phenotypes, device=self.device)*self.gamma_scale_shape,
                        torch.ones(self.n_phenotypes, device=self.device)*self.gamma_scale_rate
                    ).to_event(1)
                )
        else:
            scale_factor_kct = None

        # 4) Phenotype->gene profile => Dirichlet
        gene_prior = (
            self.gene_profile_prior*self.gene_profile_prior_strength
            + self.gene_profile_prior_offset
        )
        gene_prior = torch.clamp(gene_prior, min=1e-4)
        with pyro.plate("phenotype", self.n_phenotypes):
            clone_gene_profiles = pyro.sample(
                "clone_gene_profiles",
                dist.Dirichlet(gene_prior)
            )

        # 5) Cell-level mixture => observed expression
        #    We'll combine (baseline + scale) directly in the cell distribution. No child variable is stored.
        with pyro.plate("cells", batch_size):
            # Flatten (k,c,t) => single index
            kct_index = self._get_kct_index(tcr_idx, patient_idx, time_idx)

            # Retrieve baseline for each cell => shape (batch_size, n_phenotypes)
            # We either replicate baseline if T>0, or just use the direct index
            # We'll unify them at the cell level
            if self.T > 0:
                # baseline => shape (K*C, n_phenotypes)
                # replicate => shape (K*C*T, n_phenotypes)
                # but we skip storing that replication; we just index carefully.
                # We'll do:
                baseline_for_cell = local_clone_phenotype_pc.repeat_interleave(self.T, dim=0)[kct_index]
                scale_for_cell = scale_factor_kct[kct_index]  # shape (batch_size, n_phenotypes)
            else:
                # T=0 => no scale
                baseline_for_cell = local_clone_phenotype_pc[kct_index]
                scale_for_cell = torch.ones_like(baseline_for_cell)

            # Combine baseline & scale => local concentration
            beta_val = torch.tensor(self.beta, device=self.device)
            local_conc_for_cell = beta_val * baseline_for_cell * scale_for_cell + self.local_concentration_offset

            # Then incorporate phenotype_probs & persistence
            cell_phenotype_dist = pyro.sample(
                "cell_phenotype_dist",
                dist.Dirichlet(local_conc_for_cell * phenotype_probs * self.persistence_factor + 1.0)
            )

            # (Optional) consistency penalty block can remain commented out
            # if self.consistency_weight > 0.0:
            #     ...

            # Mix phenotype->gene => final expression distribution
            mixed_profile = torch.sum(
                clone_gene_profiles * cell_phenotype_dist.unsqueeze(-1),
                dim=1
            )

            # Patient effect
            p_effect = patient_variance[patient_idx]
            adjusted_probs = mixed_profile * p_effect.unsqueeze(-1)

            exponentiated = torch.exp(adjusted_probs)
            concentration = exponentiated*self.gene_concentration + 1.0

            pyro.sample("obs", dist.Dirichlet(concentration), obs=matrix)

    def _guide(self, matrix, tcr_idx, patient_idx, time_idx, phenotype_probs):

        batch_size = matrix.shape[0]

        # 1) Patient variance => param
        patient_variance_shape = pyro.param(
            "patient_variance_shape",
            torch.ones(self.C, device=self.device)*self.patient_variance_shape_val,
            constraint=constraints.greater_than(0.1)
        )
        patient_variance_rate = pyro.param(
            "patient_variance_rate",
            torch.ones(self.C, device=self.device)*self.patient_variance_rate_val,
            constraint=constraints.greater_than(0.1)
        )
        with pyro.plate("patient_global", self.C):
            pyro.sample(
                "patient_variance",
                dist.Gamma(patient_variance_shape, patient_variance_rate)
            )

        # 2) (k,c) baseline param => Dirichlet
        baseline_clone_conc_init = (
            self.clone_to_phenotype_prior*self.clone_to_phenotype_prior_strength
            + 1.0
        )
        expanded_baseline_init = baseline_clone_conc_init.repeat_interleave(self.C, dim=0)
        local_clone_conc_pc = pyro.param(
            "local_clone_concentration_pc",
            expanded_baseline_init,
            constraint=constraints.greater_than(0.1)
        )
        with pyro.plate("clone_patient_baseline", self.K*self.C):
            pyro.sample(
                "local_clone_phenotype_pc",
                dist.Dirichlet(local_clone_conc_pc)
            )

        # 3) dimension-wise scale factors (Gamma)
        if self.T > 0:
            shape_init = torch.ones(
                self.K*self.C*self.T, self.n_phenotypes, device=self.device
            ) * self.gamma_scale_shape
            rate_init = torch.ones(
                self.K*self.C*self.T, self.n_phenotypes, device=self.device
            ) * self.gamma_scale_rate

            guide_shape = pyro.param(
                "scale_factor_shape",
                shape_init,
                constraint=constraints.greater_than(0.0)
            )
            guide_rate = pyro.param(
                "scale_factor_rate",
                rate_init,
                constraint=constraints.greater_than(0.0)
            )

            with pyro.plate("clone_patient_time", self.K*self.C*self.T):
                pyro.sample(
                    "scale_factor_kct",
                    dist.Gamma(guide_shape, guide_rate).to_event(1)
                )

        # 4) Phenotype->gene Dirichlet
        gene_profile_concentration_init = (
            self.gene_profile_prior*self.gene_profile_prior_strength
            + self.gene_profile_prior_offset
        )
        gene_profile_concentration_init = torch.clamp(gene_profile_concentration_init, min=1e-4)
        unconstrained_profile_init = torch.log(gene_profile_concentration_init)

        unconstrained_profile = pyro.param(
            "unconstrained_gene_profile_log",
            unconstrained_profile_init,
            constraint=constraints.real
        )
        with pyro.plate("phenotype", self.n_phenotypes):
            gene_profile_concentration = torch.exp(torch.clamp(unconstrained_profile, -10, 10))
            pyro.sample(
                "clone_gene_profiles",
                dist.Dirichlet(gene_profile_concentration)
            )

        # 5) cell-level param => Dirichlet (for cell_phenotype_dist)
        cell_phenotype_conc_init = torch.ones(batch_size, self.n_phenotypes, device=self.device) + 1.0
        cell_phenotype_concentration = pyro.param(
            "cell_phenotype_concentration",
            cell_phenotype_conc_init,
            constraint=constraints.greater_than(0.1)
        )
        with pyro.plate("cells", batch_size):
            pyro.sample(
                "cell_phenotype_dist",
                dist.Dirichlet(cell_phenotype_concentration * phenotype_probs * self.persistence_factor + 1.0)
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
            
            # Notice we now unpack 5 items:
            for gene_probs, tcr_idx, patient_idx, phenotype_probs, time_idx in self.dataloader:
                gene_probs = gene_probs.to(self.device)
                tcr_idx = tcr_idx.to(self.device)
                patient_idx = patient_idx.to(self.device)
                phenotype_probs = phenotype_probs.to(self.device)
                time_idx = time_idx.to(self.device)  # new

                if torch.isnan(gene_probs).any():
                    if verbose:
                        print("Warning: NaN values in input data")
                    continue

                # Now pass all 5 arguments to self.svi.step(...)
                loss = self.svi.step(
                    gene_probs, tcr_idx, patient_idx, time_idx, phenotype_probs
                )
                
                if not torch.isnan(torch.tensor(loss)):
                    epoch_loss += loss
                    num_batches += 1
            
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



    def get_phenotypic_flux(
        self,
        clone_idx: int,
        patient_idx: int,
        time1: int,
        time2: int,
        metric: str = "l1",
        temperature: float = 1.0
    ) -> float:
        """
        Compute the "phenotypic flux" between time1 and time2 for:
        - TCR=clone_idx
        - patient=patient_idx
        using a *single point estimate* of the child distribution for each timepoint:
        p(phi|k,c,t) = Dirichlet( baseline_kc * scale_mean_kct + offset ).

        Then we compute a chosen distance metric ("l1" or "dkl") between the two distributions.

        Args:
            clone_idx: index in [0..K-1]
            patient_idx: index in [0..C-1]
            time1, time2: time indices in [0..T-1]
            metric: "l1" or "dkl"
            temperature: post-hoc temperature factor (<1 => sharper, >1 => flatter)

        Returns:
            flux_value: float, the distance between the 2 distributions at (time1, time2).
        """
        import torch
        import pyro

        eps = 1e-12

        # 1) Reconstruct distribution at time1
        p1 = self._reconstruct_single_distribution(clone_idx, patient_idx, time1)
        # 2) Reconstruct distribution at time2
        p2 = self._reconstruct_single_distribution(clone_idx, patient_idx, time2)

        # 3) Optional temperature transform
        def temp_scale(tensor_dist, tau):
            if tau == 1.0:
                return tensor_dist / torch.clamp(tensor_dist.sum(), min=eps)
            power = tensor_dist ** (1.0 / tau)
            denom = torch.clamp(power.sum(), min=eps)
            return power / denom

        if temperature != 1.0:
            p1 = temp_scale(p1, temperature)
            p2 = temp_scale(p2, temperature)

        # Re-clamp & re-normalize
        p1 = p1.clamp_min(eps)
        p2 = p2.clamp_min(eps)
        p1 = p1 / p1.sum()
        p2 = p2 / p2.sum()

        # 4) Distance
        if metric.lower() == "l1":
            flux_value = float(torch.sum(torch.abs(p1 - p2)))
        elif metric.lower() == "dkl":
            flux_value = float(torch.sum(p1 * torch.log(p1 / p2)))
        else:
            raise ValueError(f"Unknown metric {metric}, use 'l1' or 'dkl'.")

        return flux_value


    def get_phenotypic_entropy_by_clone(
        self,
        patient_idx: int,
        time_idx: int,
        weight_by_clone_freq: bool = False,
        normalize: bool = False,
        temperature: float = 1.0
    ) -> Dict[int, float]:
        """
        For each clone k, compute H(Phenotype | Clone=k) in nats or bits (here, using ln or log).
        We do a single *point estimate* of p(phenotype|k) by:
        - Reconstructing child_concentration = beta * baseline + scale_mean + offset
        - Then normalizing => distribution over phenotypes.

        Optionally:
        - weight_by_clone_freq => multiply distribution by freq_for_clone[k]
        - temperature transform => p^(1/tau)
        - normalize => divide by ln(n_phenotypes)

        Returns:
            dict: {clone_idx -> entropy_value}
        """
        import torch
        import pyro
        import math
        import numpy as np

        eps = 1e-12

        # Helper: temperature transform
        def temperature_transform(p: torch.Tensor, tau: float) -> torch.Tensor:
            # p has shape (n_phenotypes,)
            if tau == 1.0:
                s = torch.clamp(p.sum(), min=eps)
                return p / s
            else:
                p_pow = p ** (1.0 / tau)
                denom = torch.clamp(p_pow.sum(), min=eps)
                return p_pow / denom

        # 1) Retrieve baseline => shape (K*C, n_phenotypes)
        baseline_param = pyro.param("local_clone_concentration_pc").detach().cpu()

        # If T>0 => retrieve scale param => (K*C*T, n_phenotypes)
        if self.T > 0:
            guide_shape = pyro.param("scale_factor_shape").detach().cpu()
            guide_rate  = pyro.param("scale_factor_rate").detach().cpu()
        else:
            guide_shape = None
            guide_rate  = None

        # Possibly gather real clone frequencies
        freq_for_clone = np.ones(self.K, dtype=np.float32)
        if weight_by_clone_freq and "clone_size" in self.adata.obs:
            freq_for_clone[:] = 0.0
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
                freq_for_clone[:] = 1.0 / self.K

        phen_entropy_by_clone = {}

        beta_val = torch.tensor(self.beta, dtype=torch.float)
        offset_val = torch.tensor(self.local_concentration_offset, dtype=torch.float)

        # 2) For each clone k => reconstruct p(phenotype|k)
        for k_idx in range(self.K):
            # flatten indices => baseline_idx = k*C + patient_idx
            baseline_idx = k_idx * self.C + patient_idx
            baseline_kc = baseline_param[baseline_idx]  # shape => (n_phenotypes,)

            if self.T > 0:
                # scale => index = k*(C*T) + patient_idx*T + time_idx
                kct = k_idx * (self.C * self.T) + patient_idx*self.T + time_idx
                shape_kct = guide_shape[kct]
                rate_kct  = guide_rate[kct]
                mean_scale = shape_kct / (rate_kct + eps)
            else:
                mean_scale = torch.ones_like(baseline_kc)

            child_conc = beta_val * baseline_kc * mean_scale + offset_val
            denom = torch.clamp(child_conc.sum(), min=eps)
            p_phi_given_k = child_conc / denom

            # optional weighting by clone freq
            if weight_by_clone_freq:
                p_phi_given_k = p_phi_given_k * freq_for_clone[k_idx]
                s2 = torch.clamp(p_phi_given_k.sum(), min=eps)
                p_phi_given_k = p_phi_given_k / s2

            # temperature transform
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)
            p_phi_given_k = torch.clamp(p_phi_given_k, min=eps)
            p_phi_given_k = p_phi_given_k / p_phi_given_k.sum()

            # Entropy = - sum p(phi|k)* log p(phi|k)
            entropy_val = -torch.sum(p_phi_given_k * torch.log(p_phi_given_k))
            # Convert to float
            entropy_val = entropy_val.item()

            # if normalize => divide by ln(n_phenotypes)
            if normalize and self.n_phenotypes > 1:
                entropy_val /= math.log(self.n_phenotypes)

            phen_entropy_by_clone[k_idx] = float(entropy_val)

        return phen_entropy_by_clone

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
        A Bayesian version of phenotypic flux: we call generate_posterior_samples(...)
        for (time1) and (time2), average each => p1, p2 => measure distance.

        Args:
            clone_idx, patient_idx, time1, time2
            metric: "l1" or "dkl"
            n_samples: how many draws per distribution
            temperature: post-hoc factor
        Returns:
            flux_value (float)
        """
        import numpy as np
        import math

        # 1) Map integer => labels
        rev_patient_map = {v:k for k,v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx} in covariate_mapping.")

        if self.T > 0:
            rev_time_map = {v:k for k,v in self.dataset.timepoint_mapping.items()}
            time_label1 = rev_time_map.get(time1, None)
            time_label2 = rev_time_map.get(time2, None)
            if time_label1 is None or time_label2 is None:
                raise ValueError(f"Unknown time indices {time1}, {time2} in timepoint_mapping.")
        else:
            time_label1 = None
            time_label2 = None

        # Map clone_idx => TCR label
        tcr_label = self.clone_mapping.get(clone_idx, None)
        if tcr_label is None:
            raise ValueError(f"Unknown clone_idx={clone_idx} in clone_mapping.")

        # 2) Posterior samples for time1 => shape (n_samples, n_phenotypes)
        posterior_res1 = self.generate_posterior_samples(
            patient_label=patient_label,
            tcr_label=tcr_label,
            timepoint_label=time_label1,
            n_samples=n_samples,
            include_patient_effects=True
        )
        pheno_samples1 = posterior_res1["phenotype_samples"]

        # 3) Posterior samples for time2
        posterior_res2 = self.generate_posterior_samples(
            patient_label=patient_label,
            tcr_label=tcr_label,
            timepoint_label=time_label2,
            n_samples=n_samples,
            include_patient_effects=True
        )
        pheno_samples2 = posterior_res2["phenotype_samples"]

        # 4) Average => p1, p2
        p1 = pheno_samples1.mean(axis=0)  # shape => (n_phenotypes,)
        p2 = pheno_samples2.mean(axis=0)

        # 5) Temperature
        import numpy as np
        eps = 1e-12

        def temp_scale(vec, tau):
            vec = np.clip(vec, eps, None)
            if tau == 1.0:
                s = vec.sum()
                return vec / s
            v_pow = vec**(1.0/tau)
            denom = v_pow.sum()
            if denom > eps:
                return v_pow / denom
            else:
                return np.ones_like(vec)/len(vec)

        if temperature != 1.0:
            p1 = temp_scale(p1, temperature)
            p2 = temp_scale(p2, temperature)

        # clamp & normalize
        p1 = np.clip(p1, eps, None)
        p2 = np.clip(p2, eps, None)
        p1 /= p1.sum()
        p2 /= p2.sum()

        # 6) distance
        if metric.lower() == "l1":
            flux_value = float(np.sum(np.abs(p1 - p2)))
        elif metric.lower() == "dkl":
            flux_value = float(np.sum(p1 * np.log(p1 / p2)))
        else:
            raise ValueError(f"Unknown metric {metric}, use 'l1' or 'dkl'.")

        return flux_value

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
        that compares *all* posterior draws pairwise (n_samples^2).
        Then we average the chosen distance.

        Steps:
        1) generate_posterior_samples(...) for (time1) => shape (n_samples, n_phenotypes)
            generate_posterior_samples(...) for (time2)
        2) For each sample i in time1, each j in time2 => apply temperature => distance
        3) average => final flux_value
        """
        import numpy as np
        import math

        # 0) Convert (clone_idx, patient_idx, timeX) => labels
        rev_patient_map = {v:k for k,v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx}")

        if self.T > 0:
            rev_time_map = {v:k for k,v in self.dataset.timepoint_mapping.items()}
            time_label1 = rev_time_map.get(time1, None)
            time_label2 = rev_time_map.get(time2, None)
            if time_label1 is None or time_label2 is None:
                raise ValueError(f"Unknown time indices {time1}, {time2} in timepoint_mapping.")
        else:
            time_label1 = None
            time_label2 = None

        # clone_idx => TCR label
        tcr_label = self.clone_mapping.get(clone_idx, None)
        if tcr_label is None:
            raise ValueError(f"Unknown clone_idx={clone_idx}")

        # 1) Posterior samples for time1 => (n_samples, n_phenotypes)
        post1 = self.generate_posterior_samples(
            patient_label=patient_label,
            tcr_label=tcr_label,
            timepoint_label=time_label1,
            n_samples=n_samples,
            include_patient_effects=True
        )
        pheno1 = post1["phenotype_samples"]

        # 2) Posterior samples for time2
        post2 = self.generate_posterior_samples(
            patient_label=patient_label,
            tcr_label=tcr_label,
            timepoint_label=time_label2,
            n_samples=n_samples,
            include_patient_effects=True
        )
        pheno2 = post2["phenotype_samples"]

        eps = 1e-12

        def process_distribution(vec: np.ndarray, tau: float) -> np.ndarray:
            vec = np.clip(vec, eps, None)
            if tau != 1.0:
                power = vec ** (1.0/tau)
                denom = power.sum()
                if denom > eps:
                    vec = power / denom
                else:
                    vec = np.ones_like(vec)/len(vec)
            else:
                s = vec.sum()
                if s < eps:
                    vec = np.ones_like(vec)/len(vec)
                else:
                    vec = vec/s
            return vec

        total_dist = 0.0
        count = 0

        # 3) Double loop => all pairs
        for i in range(n_samples):
            p1 = process_distribution(pheno1[i], temperature)
            for j in range(n_samples):
                p2 = process_distribution(pheno2[j], temperature)

                if metric.lower() == "l1":
                    dist_ij = np.sum(np.abs(p1 - p2))
                elif metric.lower() == "dkl":
                    dist_ij = np.sum(p1 * np.log(p1 / p2))
                else:
                    raise ValueError(f"Unknown metric {metric}")

                total_dist += dist_ij
                count += 1

        flux_value = float(total_dist / count)
        return flux_value


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
        Bayesian version of H(Phenotype | Clone=k). For each clone k, we:
        - call generate_posterior_samples(...) => (n_samples, n_phenotypes)
        - average => p(phi|k)
        - optionally multiply by freq_for_clone[k], re-normalize
        - temperature transform => re-normalize
        - compute - sum p(phi|k)* log p(phi|k)

        If normalize=True => divide by ln(n_phenotypes).

        Returns:
            dict {clone_idx -> float} 
        """
        import numpy as np
        import math
        from typing import Dict

        # 1) Map integer => labels for patient, time
        rev_patient_map = {v: k for k, v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx} in covariate_mapping.")

        if self.T > 0:
            rev_time_map = {v: k for k, v in self.dataset.timepoint_mapping.items()}
            time_label = rev_time_map.get(time_idx, None)
            if time_label is None:
                raise ValueError(f"Unknown time_idx={time_idx} in timepoint_mapping.")
        else:
            time_label = None

        K = self.K
        P = self.n_phenotypes
        eps = 1e-12

        # 2) Possibly gather real clone frequencies
        if weight_by_clone_freq and "clone_size" in self.adata.obs:
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
            freq_for_clone = np.ones(K, dtype=np.float32)

        def temperature_transform(vec: np.ndarray, tau: float) -> np.ndarray:
            vec = np.clip(vec, eps, None)
            if tau == 1.0:
                s = vec.sum()
                if s < eps:
                    return np.ones_like(vec) / len(vec)
                return vec / s
            else:
                power = vec ** (1.0 / tau)
                denom = power.sum()
                if denom > eps:
                    return power / denom
                else:
                    return np.ones_like(vec) / len(vec)

        phen_entropy_by_clone = {}

        # 3) For each clone => posterior draws => average => weighting => temperature => compute entropy
        for k_idx in range(K):
            tcr_label = self.clone_mapping.get(k_idx, None)
            if tcr_label is None:
                continue

            # generate posterior samples => (n_samples, n_phenotypes)
            posterior_result = self.generate_posterior_samples(
                patient_label=patient_label,
                tcr_label=tcr_label,
                timepoint_label=time_label,
                n_samples=n_samples,
                include_patient_effects=True
            )
            pheno_samples = posterior_result["phenotype_samples"]

            # average => p(phi|k)
            p_phi_given_k = pheno_samples.mean(axis=0)  # shape (P,)

            # multiply by freq_for_clone[k], then re-normalize if requested
            if weight_by_clone_freq:
                p_phi_given_k *= freq_for_clone[k_idx]
                s2 = p_phi_given_k.sum()
                if s2 < eps:
                    p_phi_given_k[:] = 1.0 / P
                else:
                    p_phi_given_k /= s2

            # temperature transform
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)

            # clamp => re-normalize
            p_phi_given_k = np.clip(p_phi_given_k, eps, None)
            sum_phi = p_phi_given_k.sum()
            if sum_phi < eps:
                p_phi_given_k[:] = 1.0 / P
            else:
                p_phi_given_k /= sum_phi

            # Entropy => - sum p(phi|k)* log p(phi|k)
            entropy_val = -np.sum(p_phi_given_k * np.log(p_phi_given_k))

            if normalize and P > 1:
                entropy_val /= math.log(P)

            phen_entropy_by_clone[k_idx] = float(entropy_val)

        return phen_entropy_by_clone


    def get_mutual_information(
        self,
        patient_idx: int,
        time_idx: int,
        weight_by_clone_freq: bool = False,
        temperature: float = 1.0
    ) -> float:
        """
        Compute I(Clone; Phenotype) for a given (patient_idx, time_idx), 
        using a *point estimate* of each clone's distribution derived from:
        - local_clone_concentration_pc (baseline)
        - scale_factor_{kct} shape/rate => mean scale
        - offset + beta
        Then we do:
        p(k,phi) = p(k)* p(phi|k),
        sum => p(phi), then compute MI in bits (log2).

        Args:
            patient_idx: int in [0..C-1]
            time_idx: int in [0..T-1], or 0 if T=0
            weight_by_clone_freq: whether to gather real clone frequencies from adata.obs["clone_size"]
            temperature: float, post-hoc temperature transform

        Returns:
            mutual_info (float) in bits (log2).
        """
        import torch
        import numpy as np
        import pyro

        eps = 1e-12

        # 1) Retrieve baseline param => shape (K*C, n_phenotypes)
        baseline_param = pyro.param("local_clone_concentration_pc").detach().cpu()

        # If T>0, retrieve gamma scale shape & rate => shape (K*C*T, n_phenotypes)
        if self.T > 0:
            scale_shape = pyro.param("scale_factor_shape").detach().cpu()
            scale_rate  = pyro.param("scale_factor_rate").detach().cpu()
        else:
            scale_shape = None
            scale_rate  = None

        # We'll build a distribution p(phi|k), and a clone freq array => p(k)
        dist_kp = []
        clone_freqs = []

        # Possibly gather real clone frequencies
        import numpy as np
        if weight_by_clone_freq and "clone_size" in self.adata.obs:
            freq_for_clone = np.zeros(self.K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values
            for i in range(len(self.dataset)):
                cell_pat = self.dataset.covariates[i].item()
                cell_time = self.dataset.timepoints[i].item() if (self.dataset.timepoints is not None) else 0
                if (cell_pat == patient_idx) and (cell_time == time_idx):
                    c_id = self.dataset.tcrs[i].item()
                    freq_for_clone[c_id] += clone_sizes[i]

            total_size = freq_for_clone.sum()
            if total_size > 0:
                freq_for_clone /= total_size
            else:
                freq_for_clone[:] = 1.0 / self.K
        else:
            freq_for_clone = np.ones(self.K, dtype=np.float32) / self.K

        beta_val = torch.tensor(self.beta, dtype=torch.float)
        offset_val = torch.tensor(self.local_concentration_offset, dtype=torch.float)

        # 2) For each clone k => reconstruct child distribution
        for k in range(self.K):
            # flatten index => baseline_idx = k*C + patient_idx
            baseline_idx = k * self.C + patient_idx
            baseline_kc = baseline_param[baseline_idx]  # shape => (n_phenotypes,)

            if self.T > 0:
                kct_index = k * (self.C*self.T) + patient_idx*self.T + time_idx
                shape_kct = scale_shape[kct_index]  # (n_phenotypes,)
                rate_kct  = scale_rate[kct_index]
                mean_scale = shape_kct / (rate_kct + eps)
            else:
                mean_scale = torch.ones_like(baseline_kc)

            # child_concentration => beta*baseline*mean_scale + offset
            child_conc = beta_val * baseline_kc * mean_scale + offset_val
            denom = torch.clamp(child_conc.sum(), min=eps)
            p_phi_given_k = child_conc / denom  # shape => (n_phenotypes,)

            dist_kp.append(p_phi_given_k)

            # p(k)
            p_k = freq_for_clone[k]  # either uniform or real freq
            clone_freqs.append(p_k)

        dist_kp = torch.stack(dist_kp, dim=0)  # shape => (K, n_phenotypes)

        # 3) Temperature transform
        if temperature != 1.0:
            dist_kp_pow = dist_kp ** (1.0 / temperature)
            row_sums = dist_kp_pow.sum(dim=1, keepdim=True).clamp_min(eps)
            dist_kp = dist_kp_pow / row_sums

        # 4) p(k) array => normalize
        clone_freqs = np.array(clone_freqs, dtype=np.float32)
        total_clone_freq = clone_freqs.sum()
        if total_clone_freq > 0:
            clone_freqs /= total_clone_freq
        else:
            clone_freqs[:] = 1.0 / self.K

        # build p(k, phi) = p(k)* p(phi|k)
        pxy = dist_kp * torch.tensor(clone_freqs, dtype=torch.float).unsqueeze(1)
        pxy = pxy.detach().numpy()
        pxy_sum = pxy.sum()
        if pxy_sum > eps:
            pxy /= pxy_sum

        # px = sum_phi p(k,phi), py = sum_k p(k,phi)
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        # 5) Compute mutual info => sum_{k,phi} p(k,phi)* log2 [ p(k,phi)/(p(k)* p(phi)) ]
        px_py = np.outer(px, py)
        nzs = (pxy > 1e-12)

        mutual_info = np.sum(
            pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs])
        )

        return float(mutual_info)

    def get_mutual_information_posterior(
        self,
        patient_idx: int,
        time_idx: int,
        n_samples: int = 100,
        weight_by_clone_freq: bool = False,
        temperature: float = 1.0
    ) -> float:
        """
        A Bayesian approach to computing I(Clone; Phenotype) for a given (patient_idx,time_idx),
        sampling from the posterior for each clone, then combining.

        Steps:
        1) For each clone k in [0..K-1], we call generate_posterior_samples(...) => (n_samples, n_phenotypes)
        2) Average => p(phi|k), apply temperature transform
        3) Possibly weight by real clone frequencies => p(k, phi)
        4) sum => p(phi), compute I(X;Y) in bits (log2).

        Args:
            patient_idx: which patient index in [0..C-1]
            time_idx: which time index in [0..T-1]
            n_samples: how many posterior draws
            weight_by_clone_freq: if True, we gather frequencies from adata.obs["clone_size"]
            temperature: post-hoc temperature factor
                        <1 => sharper, >1 => flatter

        Returns:
            mutual_info (float) in bits (log base 2)
        """
        import numpy as np
        import math

        # 0) Map integer indices -> string labels
        rev_patient_map = {v: k for k, v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx} in covariate_mapping.")

        if self.T > 0 and time_idx >= 0:
            rev_time_map = {v: k for k, v in self.dataset.timepoint_mapping.items()}
            time_label = rev_time_map.get(time_idx, None)
            if time_label is None:
                raise ValueError(f"Unknown time_idx={time_idx} in timepoint_mapping.")
        else:
            time_label = None

        K = self.K
        P = self.n_phenotypes
        eps = 1e-12

        # 1) Possibly gather real clone frequencies => p(k)
        if weight_by_clone_freq and "clone_size" in self.adata.obs:
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

        # 2) We'll accumulate p(k,phi) => shape (K,P)
        p_k_phi = np.zeros((K, P), dtype=np.float64)

        # 3) For each clone k => generate samples => average => temperature => build p(k,phi)
        for k_idx in range(K):
            tcr_label = self.clone_mapping.get(k_idx, None)
            if tcr_label is None:
                continue

            posterior_result = self.generate_posterior_samples(
                patient_label=patient_label,
                tcr_label=tcr_label,
                timepoint_label=time_label,
                n_samples=n_samples,
                include_patient_effects=True
            )
            # shape => (n_samples, n_phenotypes)
            pheno_samples = posterior_result["phenotype_samples"]

            # average => p(phi|k)
            p_phi_given_k = pheno_samples.mean(axis=0)

            # temperature transform
            if temperature != 1.0:
                power = p_phi_given_k ** (1.0 / temperature)
                denom = power.sum()
                if denom > eps:
                    p_phi_given_k = power / denom
                else:
                    p_phi_given_k = np.ones(P, dtype=np.float32) / P

            # p(k,phi) = p(k)* p(phi|k)
            for phi_idx in range(P):
                p_k_phi[k_idx, phi_idx] = freq_for_clone[k_idx] * p_phi_given_k[phi_idx]

        # 4) Normalize => sum_{k,phi}=1
        total_mass = p_k_phi.sum()
        if total_mass > eps:
            p_k_phi /= total_mass
        else:
            p_k_phi[:] = 1.0 / (K*P)

        # p(k)= sum_phi, p(phi)= sum_k
        px = p_k_phi.sum(axis=1)
        py = p_k_phi.sum(axis=0)

        # 5) I(X;Y) => sum_{k,phi} p(k,phi)* log2[ p(k,phi)/( p(k)* p(phi) ) ]
        mutual_info = 0.0
        for k_idx in range(K):
            for phi_idx in range(P):
                val = p_k_phi[k_idx, phi_idx]
                if val > eps:
                    denom = px[k_idx]* py[phi_idx]
                    if denom > eps:
                        mutual_info += val * math.log2(val / denom)

        return float(mutual_info)


    def get_clonotypic_entropy_by_phenotype(
        self,
        patient_idx: int,
        time_idx: int,
        weight_by_clone_freq: bool = False,
        normalize: bool = False,
        temperature: float = 1.0
    ) -> Dict[int, float]:
        """
        For each phenotype phi, compute H(Clone | Phenotype=phi) in bits (base 2).
        If `normalize=True`, we divide by log2(K).

        We do NOT rely on local_clone_concentration_pct. Instead, for each clone k:
        - We reconstruct p(phi|k) using:
            baseline_k + dimension-wise gamma scale (mean) + offset
        - Then build p(k, phi), sum over k => p(phi).

        Optionally:
        - weight_by_clone_freq: weight each clone by freq in adata.obs["clone_size"]
        - apply temperature transform p(phi|k)^(1/tau).

        Returns:
            dict: {phi_idx -> entropy_value_in_bits}
        """
        import torch
        import pyro
        import math
        from typing import Dict

        eps = 1e-12

        # -----------------------------
        # 1) Retrieve baseline => shape (K*C, n_phenotypes)
        # -----------------------------
        baseline_param = pyro.param("local_clone_concentration_pc").detach().cpu()

        # If T>0, also retrieve gamma scale shape/rate => shape (K*C*T, n_phenotypes)
        if self.T > 0:
            scale_shape = pyro.param("scale_factor_shape").detach().cpu()
            scale_rate  = pyro.param("scale_factor_rate").detach().cpu()
        else:
            scale_shape = None
            scale_rate  = None

        # Beta + offset
        beta_val = self.beta
        offset_val = self.local_concentration_offset

        # We'll accumulate p(k,phi) => shape (K, n_phenotypes)
        dist_kp = []
        # Possibly gather real clone frequencies
        import numpy as np
        if weight_by_clone_freq and "clone_size" in self.adata.obs:
            freq_for_clone = np.zeros(self.K, dtype=np.float32)
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
                freq_for_clone[:] = 1.0 / self.K
        else:
            freq_for_clone = np.ones(self.K, dtype=np.float32) / self.K

        # For each clone k in [0..K)
        for k in range(self.K):
            # 2) Flatten indices => baseline index => k*C + patient_idx
            baseline_idx = k * self.C + patient_idx
            baseline_kc = baseline_param[baseline_idx]  # shape (n_phenotypes,)

            # 3) If T>0, we have a time dimension => scale index => k*C*T + patient_idx*T + time_idx
            if self.T > 0:
                kct_idx = k * (self.C * self.T) + patient_idx * self.T + time_idx
                shape_kct = scale_shape[kct_idx]  # (n_phenotypes,)
                rate_kct  = scale_rate[kct_idx]
                # mean scale factor => shape_kct / rate_kct
                mean_scale = shape_kct / (rate_kct + eps)
            else:
                # T=0 => no scale factor
                mean_scale = torch.ones_like(baseline_kc)

            # child_conc => beta * baseline + offset
            child_conc = beta_val * baseline_kc * mean_scale + offset_val
            # convert to p(phi|k)
            denom = torch.clamp(child_conc.sum(), min=eps)
            p_phi_given_k = child_conc / denom

            dist_kp.append(p_phi_given_k)

        # shape => (K, n_phenotypes)
        dist_kp = torch.stack(dist_kp, dim=0)

        # 4) Temperature transform
        if temperature != 1.0:
            dist_kp_pow = dist_kp ** (1.0 / temperature)
            row_sums = dist_kp_pow.sum(dim=1, keepdim=True).clamp_min(eps)
            dist_kp = dist_kp_pow / row_sums

        # 5) Weighted distribution p(k) if requested
        import torch
        clone_freqs_t = torch.tensor(freq_for_clone, dtype=torch.float)
        clone_freqs_t = clone_freqs_t / torch.clamp(clone_freqs_t.sum(), min=eps)

        # p(k, phi) = p(k)* p(phi|k)
        p_k_phi = dist_kp * clone_freqs_t.unsqueeze(-1)
        total_mass = torch.clamp(p_k_phi.sum(), min=eps)
        p_k_phi = p_k_phi / total_mass

        # p(phi) = sum_k p(k, phi)
        p_phi = p_k_phi.sum(dim=0)

        # 6) For each phenotype => H(Clone|phi)
        clonotypic_entropy_dict: Dict[int, float] = {}
        for phi_idx in range(self.n_phenotypes):
            denom = p_phi[phi_idx]
            if denom < eps:
                clonotypic_entropy_dict[phi_idx] = 0.0
                continue

            # p(k|phi) = p(k, phi)/p(phi)
            p_k_given_phi = p_k_phi[:, phi_idx] / denom.clamp_min(eps)
            p_k_given_phi = p_k_given_phi.clamp_min(eps)
            p_k_given_phi = p_k_given_phi / p_k_given_phi.sum()

            H_clone_phi = -torch.sum(p_k_given_phi * torch.log2(p_k_given_phi))
            if normalize and self.K > 1:
                H_clone_phi = H_clone_phi / math.log2(self.K)

            clonotypic_entropy_dict[phi_idx] = float(H_clone_phi.item())

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
        A Bayesian version of H(Clone | Phenotype=phi). We:
        1) Convert (patient_idx,time_idx) to string labels
        2) For each clone k, call generate_posterior_samples(..., n_samples) 
            => shape (n_samples, n_phenotypes) 
        3) Average => p(phi|k), apply temperature, and combine across clones.

        If weight_by_clone_freq=True, use real frequencies from adata.obs["clone_size"] 
        to define p(k).

        Then for each phi: H(Clone|phi) in bits. If normalize=True => divide by log2(K).
        """
        import numpy as np
        import math
        from typing import Dict

        # 0) Map integer => string label
        rev_patient_map = {v: k for k, v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx} in covariate_mapping.")

        if self.T > 0 and time_idx >= 0:
            rev_time_map = {v: k for k, v in self.dataset.timepoint_mapping.items()}
            time_label = rev_time_map.get(time_idx, None)
            if time_label is None:
                raise ValueError(f"Unknown time_idx={time_idx} in timepoint_mapping.")
        else:
            time_label = None

        K = self.K
        P = self.n_phenotypes
        eps = 1e-12

        # 1) Possibly gather real clone frequencies => p(k)
        if weight_by_clone_freq and "clone_size" in self.adata.obs:
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values
            for i in range(len(self.dataset)):
                cell_pat = self.dataset.covariates[i].item()
                cell_time = self.dataset.timepoints[i].item() if (self.dataset.timepoints is not None) else 0
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

        # 2) We'll accumulate p(k,phi) in shape (K, P)
        p_k_phi_accum = np.zeros((K, P), dtype=np.float64)

        # 3) For each clone k => generate posterior samples => average => weight
        for k_idx in range(K):
            tcr_label = self.clone_mapping.get(k_idx, None)
            if tcr_label is None:
                continue

            # get posterior samples => shape (n_samples, n_phenotypes)
            posterior_result = self.generate_posterior_samples(
                patient_label=patient_label,
                tcr_label=tcr_label,
                timepoint_label=time_label,
                n_samples=n_samples,
                include_patient_effects=True
            )
            pheno_samples = posterior_result["phenotype_samples"]  # (n_samples, P)

            # average => p(phi|k)
            p_phi_given_k = pheno_samples.mean(axis=0)  # shape (P,)

            # temperature transform
            if temperature != 1.0:
                power = p_phi_given_k ** (1.0 / temperature)
                denom = power.sum()
                if denom > eps:
                    p_phi_given_k = power / denom
                else:
                    p_phi_given_k = np.ones(P, dtype=np.float32) / P

            # p(k,phi) = p(k) * p(phi|k)
            for phi_idx in range(P):
                p_k_phi_accum[k_idx, phi_idx] = freq_for_clone[k_idx] * p_phi_given_k[phi_idx]

        # 4) Normalize => sum_{k,phi} = 1
        total_mass = p_k_phi_accum.sum()
        if total_mass > 0:
            p_k_phi_accum /= total_mass
        else:
            p_k_phi_accum[:] = 1.0 / (K * P)

        # p(phi) = sum_k p(k,phi)
        p_phi = p_k_phi_accum.sum(axis=0)

        # 5) For each phenotype => H(Clone|phi)
        clonotypic_entropy_dict: Dict[int, float] = {}
        for phi_idx in range(P):
            denom = p_phi[phi_idx]
            if denom < eps:
                clonotypic_entropy_dict[phi_idx] = 0.0
                continue

            # p(k|phi) = p(k,phi)/p(phi)
            p_k_given_phi = p_k_phi_accum[:, phi_idx] / denom
            p_k_given_phi = np.clip(p_k_given_phi, eps, None)
            p_k_given_phi /= p_k_given_phi.sum()

            H_clone_phi = -np.sum(p_k_given_phi * np.log2(p_k_given_phi))
            if normalize and K > 1:
                H_clone_phi /= math.log2(K)

            clonotypic_entropy_dict[phi_idx] = float(H_clone_phi)

        return clonotypic_entropy_dict


    def generate_posterior_samples(
        self,
        patient_label: str,
        tcr_label: str,
        timepoint_label: Optional[str] = None,
        n_samples: int = 1000,
        include_patient_effects: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        A fully Bayesian version of posterior sampling for (patient_label, tcr_label, timepoint_label),
        where each draw re-samples:
        - dimension-wise scale factors from their Gamma shape/rate
        - child distribution from Dirichlet( baseline * scale + offset )
        - gene profiles from Dirichlet(unconstrained_gene_profile_log)
        - (optionally) patient effect from Gamma(patient_variance_shape, patient_variance_rate)

        This approach does not rely on 'local_clone_concentration_pct' param, since we do not store
        the child distribution explicitly. Instead, we reconstruct it from:
        (baseline, scale_factor) for every single sample.

        Returns a dict with:
            - "phenotype_samples": (n_samples, n_phenotypes)
            - "gene_expression_samples": (n_samples, self.D)
            - "patient_effect_samples": (n_samples,) if include_patient_effects
            - "phenotype_labels": list[str] of length self.n_phenotypes
        """
        import numpy as np
        import torch
        import pyro
        import pyro.distributions as dist

        # 1) Build a cache key that ignores n_samples
        cache_key = (patient_label, tcr_label, timepoint_label, include_patient_effects)

        # 2) Map labels -> indices
        patient_idx = self.dataset.covariate_mapping.get(patient_label)
        if patient_idx is None:
            raise ValueError(f"Patient label '{patient_label}' not found.")
        tcr_idx = self.tcr_mapping.get(tcr_label)
        if tcr_idx is None:
            raise ValueError(f"TCR label '{tcr_label}' not found.")

        if self.T > 0 and timepoint_label is not None:
            time_idx = self.dataset.timepoint_mapping.get(timepoint_label)
            if time_idx is None:
                raise ValueError(f"Timepoint label '{timepoint_label}' not found.")
        else:
            time_idx = 0

        kct_index = tcr_idx * (self.C * self.T) + patient_idx * self.T + time_idx

        # 3) Check cache
        cache_entry = self._posterior_cache.get(cache_key, None)
        cached_max = cache_entry["max_samples"] if cache_entry else 0
        need_to_sample = n_samples - cached_max

        if need_to_sample > 0:
            # Prepare arrays for new draws
            new_phenotype_samples = np.zeros((need_to_sample, self.n_phenotypes), dtype=np.float32)
            new_gene_expression_samples = np.zeros((need_to_sample, self.D), dtype=np.float32)
            new_patient_effect_samples = None
            if include_patient_effects:
                new_patient_effect_samples = np.zeros(need_to_sample, dtype=np.float32)

            # ---------------------------
            #  A) Retrieve baseline => local_clone_concentration_pc (K*C, n_phenotypes)
            # ---------------------------
            local_baseline = pyro.param("local_clone_concentration_pc").detach().cpu()
            # For (k,c), index = tcr_idx*C + patient_idx
            baseline_idx = tcr_idx * self.C + patient_idx
            baseline_kc = local_baseline[baseline_idx]  # shape (n_phenotypes,)

            # ---------------------------
            #  B) Retrieve scale factor shape/rate => shape (K*C*T, n_phenotypes)
            # ---------------------------
            if self.T > 0:
                guide_shape = pyro.param("scale_factor_shape").detach().cpu()
                guide_rate  = pyro.param("scale_factor_rate").detach().cpu()
                shape_kct = guide_shape[kct_index]  # shape (n_phenotypes,)
                rate_kct  = guide_rate[kct_index]
            else:
                shape_kct = None
                rate_kct = None

            # ---------------------------
            #  C) Retrieve gene profile => "unconstrained_gene_profile_log"
            # ---------------------------
            unconstrained_log = pyro.param("unconstrained_gene_profile_log").detach().cpu()
            log_clamped = torch.clamp(unconstrained_log, -10, 10)
            raw_concentration = torch.exp(log_clamped)  # (n_phenotypes, D)

            # ---------------------------
            #  D) Patient variance shape/rate if needed
            # ---------------------------
            if include_patient_effects:
                pv_shape_all = pyro.param("patient_variance_shape").detach().cpu()
                pv_rate_all  = pyro.param("patient_variance_rate").detach().cpu()
                pat_shape = pv_shape_all[patient_idx]
                pat_rate  = pv_rate_all[patient_idx]
            else:
                pat_shape = None
                pat_rate  = None

            # 4) Do the actual sampling for the needed draws
            beta_val = torch.tensor(self.beta, dtype=torch.float)
            offset_val = torch.tensor(self.local_concentration_offset, dtype=torch.float)

            for i in range(need_to_sample):
                # (1) Sample dimension-wise scale if T>0, else all ones
                if self.T > 0:
                    # shape => (n_phenotypes,)
                    scale_draw = dist.Gamma(shape_kct, rate_kct).sample()
                    # child_conc => beta * baseline * scale_draw + offset
                    child_conc = beta_val * baseline_kc * scale_draw + offset_val
                else:
                    # T=0 => no scale factor
                    child_conc = beta_val * baseline_kc + offset_val

                # (2) phenotype_dist => Dirichlet(child_conc)
                phenotype_dist = dist.Dirichlet(child_conc).sample().numpy()  # shape (n_phenotypes,)
                new_phenotype_samples[i] = phenotype_dist

                # (3) sample gene_profile => Dirichlet(raw_concentration) => shape (n_phenotypes, D)
                gene_profile_draw = dist.Dirichlet(raw_concentration).sample().numpy()
                # Weighted sum by phenotype_dist => final profile => shape (D,)
                mixed_profile = np.sum(gene_profile_draw * phenotype_dist[:, np.newaxis], axis=0)

                # (4) sample patient effect if requested
                if include_patient_effects:
                    p_effect = dist.Gamma(pat_shape, pat_rate).sample().item()
                    new_patient_effect_samples[i] = p_effect
                    mixed_profile *= p_effect

                # (5) normalize => gene expression
                denom = mixed_profile.sum() + 1e-12
                new_gene_expression_samples[i] = mixed_profile / denom

            # 5) Combine with cache
            if cache_entry is not None:
                phenotype_samples_full = np.concatenate(
                    [cache_entry["phenotype_samples"], new_phenotype_samples],
                    axis=0
                )
                gene_expr_full = np.concatenate(
                    [cache_entry["gene_expression_samples"], new_gene_expression_samples],
                    axis=0
                )
                if include_patient_effects:
                    pe_full = np.concatenate(
                        [cache_entry["patient_effect_samples"], new_patient_effect_samples],
                        axis=0
                    )
                else:
                    pe_full = None
            else:
                phenotype_samples_full = new_phenotype_samples
                gene_expr_full = new_gene_expression_samples
                pe_full = new_patient_effect_samples

            new_cache_entry = {
                "max_samples": cached_max + need_to_sample,
                "phenotype_samples": phenotype_samples_full,
                "gene_expression_samples": gene_expr_full,
                "patient_effect_samples": pe_full
            }
            self._posterior_cache[cache_key] = new_cache_entry

        # 6) Now sub-slice from cache
        final_cache = self._posterior_cache[cache_key]
        selected_phenotype = final_cache["phenotype_samples"][:n_samples]
        selected_gene_expr = final_cache["gene_expression_samples"][:n_samples]
        selected_patient_eff = None
        if include_patient_effects:
            selected_patient_eff = final_cache["patient_effect_samples"][:n_samples]

        # 7) Build the result
        result = {
            "phenotype_samples": selected_phenotype,
            "gene_expression_samples": selected_gene_expr
        }
        if include_patient_effects:
            result["patient_effect_samples"] = selected_patient_eff

        # 8) Add phenotype labels
        phenotype_labels_ordered = [self.phenotype_mapping[i] for i in range(self.n_phenotypes)]
        result["phenotype_labels"] = phenotype_labels_ordered

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
