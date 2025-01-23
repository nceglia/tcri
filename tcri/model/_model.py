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
            # -- Added arguments to control priors in model/guide --
            clone_to_phenotype_prior_strength: float = 10.0,
            gene_profile_prior_strength: float = 5.0,
            gene_profile_prior_offset: float = 0.5,
            patient_variance_shape_val: float = 4.0,
            patient_variance_rate_val: float = 4.0,
            persistence_factor: float = 1.0,
            beta = 1.0,
            consistency_weight=1,
            gene_concentration=100.,
            local_concentration_offset=1.,
            time_variance_prior=1.,
            timepoint_label: Optional[str] = None
        ):
        """
        Extended Joint Probability Distribution to handle an optional timepoint_label.
        
        Args:
            adata: AnnData object with expression data
            tcr_label: Name of TCR column in adata.obs
            covariate_label: Name of patient or main covariate column in adata.obs
            n_phenotypes: Number of phenotype states
            phenotype_prior: Name of phenotype column in adata.obs, or probability matrix
            marker_genes: Optional dict mapping phenotype indices to marker gene lists
            marker_prior: Prior strength for marker genes
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            
            # Additional prior controls...
            
            # New param:
            timepoint_label: Name of timepoint column in adata.obs (optional).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adata = adata
        
        self._tcr_label = tcr_label
        self._covariate_label = covariate_label
        self._timepoint_label = timepoint_label
        self.particles = particles

        # Store prior-related parameters
        self.local_concentration_offset = local_concentration_offset
        self.time_variance_prior = time_variance_prior
        self.clone_to_phenotype_prior_strength = clone_to_phenotype_prior_strength
        self.gene_profile_prior_strength = gene_profile_prior_strength
        self.gene_profile_prior_offset = gene_profile_prior_offset
        self.patient_variance_shape_val = patient_variance_shape_val
        self.patient_variance_rate_val = patient_variance_rate_val
        self.beta = beta
        self.gene_concentration = gene_concentration
        self.persistence_factor = persistence_factor
        self.consistency_weight = consistency_weight

        # Process phenotype prior
        self.phenotype_probabilities = self._process_phenotype_prior(
            adata, phenotype_prior, n_phenotypes
        )

        # TCR mappings
        tcrs = pd.Categorical(adata.obs[tcr_label])
        self.tcr_mapping = {tcr: idx for idx, tcr in enumerate(tcrs.categories)}
        self.reverse_tcr_mapping = {v: k for k, v in self.tcr_mapping.items()}
        
        # Phenotype mappings
        if phenotype_prior is not None and isinstance(phenotype_prior, str):
            phenotypes = pd.Categorical(adata.obs[phenotype_prior])
            self.phenotype_mapping = {idx: label for idx, label in enumerate(phenotypes.categories)}
        else:
            self.phenotype_mapping = {idx: f"Phenotype {idx}" for idx in range(n_phenotypes)}
        self.reverse_phenotype_mapping = {v: k for k, v in self.phenotype_mapping.items()}
        
        # Clone (TCR) labels
        self.clone_mapping = {idx: clone for idx, clone in enumerate(tcrs.categories)}
        self.reverse_clone_mapping = {v: k for k, v in self.clone_mapping.items()}
        
        # Initialize dataset, now passing `timepoint_label`
        self.dataset = TCRCooccurrenceDataset(
            adata,
            tcr_label=tcr_label,
            covariate_label=covariate_label,
            min_expression=1e-6,
            phenotype_probs=self.phenotype_probabilities,
            timepoint_label=timepoint_label
        )
        
        # Setup dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        self._posterior_cache = dict()
        # Store dimensions
        self.K = self.dataset.K  # number of TCRs/clones
        self.D = self.dataset.D  # number of genes
        self.C = self.dataset.C  # number of main covariates (patients)
        self.n_phenotypes = n_phenotypes
        
        # If timepoint_label was provided, store T if found, else 0
        self.T = getattr(self.dataset, "T", 0)  # number of timepoints
        
        # Calculate gene expression profile prior from data
        df = adata.to_df()
        if phenotype_prior is not None and isinstance(phenotype_prior, str):
            df["phenotype"] = adata.obs[phenotype_prior]
            gene_profile_prior = df.groupby("phenotype").mean().to_numpy()
            gene_profile_prior /= gene_profile_prior.sum(axis=1, keepdims=True)
        else:
            gene_profile_prior = np.ones((n_phenotypes, self.D)) / self.D

        # Apply marker gene prior if provided
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
        
        # clone-to-phenotype prior
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
        
        # Pyro init
        pyro.clear_param_store()
        self.optimizer = pyro.optim.ClippedAdam({
            "lr": learning_rate,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "clip_norm": 5.0,
            "weight_decay": 1e-4
        })
        
        self.svi = SVI(
            model=self._model,
            guide=self._guide,
            optim=self.optimizer,
            loss=TraceMeanField_ELBO(num_particles=self.particles)
        )
        
        self.training_losses = []
        
        # Print diagnostics
        print("\nInitialization Summary:")
        print(f"Number of TCRs/Clones (K): {self.K}")
        print(f"Number of Phenotypes: {self.n_phenotypes}")
        print(f"Number of Genes (D): {self.D}")
        print(f"Number of Patients (C): {self.C}")
        print(f"Number of Timepoints (T): {self.T}")
        print(f"\nPhenotype Categories: {list(self.phenotype_mapping.values())}")
        print(f"Clone-to-phenotype prior shape: {self.clone_to_phenotype_prior.shape}")
        print(f"Gene profile prior shape: {self.gene_profile_prior.shape}")


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

        # 2) Baseline usage for each (k,c): Dirichlet
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

        # 3) Per-timepoint random offset in log-space
        #    shape => (T, n_phenotypes)
        if self.T > 0:
            with pyro.plate("time_offset_plate", self.T):
                time_offset_log = pyro.sample(
                    "time_offset_log",
                    dist.Normal(
                        torch.zeros(self.n_phenotypes, device=self.device),
                        torch.ones(self.n_phenotypes, device=self.device) * self.time_variance_prior
                    ).to_event(1)
                )
        else:
            # If T=0 or we have no time dimension:
            time_offset_log = torch.zeros(1, self.n_phenotypes, device=self.device)

        # 4) Combine baseline + offset => local Dirichlet parameters
        #    (K*C) repeated over T => (K*C*T, n_phenotypes)
        expanded_pc = local_clone_phenotype_pc.repeat_interleave(self.T, dim=0)

        # Also replicate time_offset_log over (K*C) => same final shape (K*C*T, n_phenotypes)
        if self.T > 0:
            expanded_offset = time_offset_log.repeat_interleave(self.K*self.C, dim=0)
        else:
            expanded_offset = torch.zeros_like(expanded_pc)

        # Partial-pooling scalar beta
        beta_val = torch.tensor(self.beta, device=self.device)

        local_concentration_pct = beta_val * expanded_pc * torch.exp(expanded_offset) + self.local_concentration_offset

        with pyro.plate("clone_patient_time", self.K*self.C*self.T):
            local_clone_phenotype_pct = pyro.sample(
                "local_clone_phenotype_pct",
                dist.Dirichlet(local_concentration_pct)
            )

        # 5) Phenotype->gene profile => Dirichlet
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

        # 6) Cell-level mixture => observed expression
        with pyro.plate("cells", batch_size):
            # Flatten (k, c, t) => single index
            kct_index = self._get_kct_index(tcr_idx, patient_idx, time_idx)
            base_dist = local_clone_phenotype_pct[kct_index]

            cell_phenotype_dist = pyro.sample(
                "cell_phenotype_dist",
                dist.Dirichlet(base_dist * phenotype_probs * self.persistence_factor + 1.0)
            )
            # >>> Clone consistency penalty block <<<
            if self.consistency_weight > 0.0:
                unique_clones = torch.unique(tcr_idx)
                consistency_loss = 0.0
                for clone_id in unique_clones:
                    mask = (tcr_idx == clone_id)
                    if mask.sum() > 1:
                        # shape: (N_clonal_cells, n_phenotypes)
                        cells_clone = cell_phenotype_dist[mask]
                        mean_clone = cells_clone.mean(dim=0)
                        dev = torch.mean(torch.sum((cells_clone - mean_clone) ** 2, dim=1))
                        consistency_loss += dev
                pyro.factor("clone_consistency", -self.consistency_weight * consistency_loss)

            # Mix phenotype->gene
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

        # 3) time_offset_log param => shape (T, n_phenotypes)
        if self.T > 0:
            unconstrained_time_offset_log = pyro.param(
                "unconstrained_time_offset_log",
                torch.zeros(self.T, self.n_phenotypes, device=self.device),
                constraint=constraints.real
            )
            with pyro.plate("time_offset_plate", self.T):
                pyro.sample(
                    "time_offset_log",
                    dist.Normal(unconstrained_time_offset_log,
                                torch.ones(self.n_phenotypes, device=self.device) * self.time_variance_prior).to_event(1)
                )

        # 4) local_clone_phenotype_pct => shape (K*C*T, n_phenotypes)
        #    We can param it directly. (Some people fold 'beta' in as well.)
        local_clone_conc_pct_init = torch.ones(
            self.K*self.C*self.T, self.n_phenotypes, device=self.device
        ) + 1.0
        local_clone_conc_pct = pyro.param(
            "local_clone_concentration_pct",
            local_clone_conc_pct_init,
            constraint=constraints.greater_than(0.1)
        )
        with pyro.plate("clone_patient_time", self.K*self.C*self.T):
            pyro.sample(
                "local_clone_phenotype_pct",
                dist.Dirichlet(local_clone_conc_pct)
            )

        # 5) Phenotype->gene Dirichlet
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

        # 6) cell-level param => Dirichlet
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

    def _get_kct_index(self, k_idx: torch.Tensor, c_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """
        Flatten (TCR index = k, patient index = c, time index = t)
        => single index in [0..(K*C*T - 1)].
        We do: idx = k*(C*T) + c*T + t.
        """
        return k_idx*(self.C*self.T) + c_idx*self.T + t_idx

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
        Compute a Bayesian version of the 'phenotypic flux' between time1, time2 for:
        - TCR = clone_idx (integer index)
        - patient = patient_idx (integer index)
        by sampling from the posterior distributions at time1 and time2,
        then comparing those *averaged* distributions with a chosen metric.

        Args:
        clone_idx: int, which clone index in [0..K-1]
        patient_idx: int, which patient index in [0..C-1]
        time1, time2: int, which time indices in [0..T-1]
        metric: "l1" or "dkl" for D_KL(p1||p2). (You could add other metrics.)
        n_samples: number of posterior draws for each distribution
        temperature: post-hoc temperature scaling factor
            - <1 => sharper distribution, >1 => flatter, =1 => no change

        Returns:
        flux_value (float): the distance between the (averaged) distribution at time1 vs time2.
        """
        import numpy as np
        import torch
        import math

        # 1) Map integer indices -> labels
        rev_patient_map = {v: k for k, v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx} in covariate_mapping.")

        if self.T > 0:
            rev_time_map = {v: k for k, v in self.dataset.timepoint_mapping.items()}
            time_label1 = rev_time_map.get(time1, None)
            time_label2 = rev_time_map.get(time2, None)
            if time_label1 is None or time_label2 is None:
                raise ValueError(f"Unknown time indices {time1}, {time2} in timepoint_mapping.")
        else:
            time_label1 = None
            time_label2 = None

        # Map clone_idx -> TCR label
        tcr_label = self.clone_mapping.get(clone_idx, None)
        if tcr_label is None:
            raise ValueError(f"Unknown clone_idx={clone_idx} in clone_mapping.")

        # 2) Posterior samples for distribution at time1
        posterior_result1 = self.generate_posterior_samples(
            patient_label=patient_label,
            tcr_label=tcr_label,
            timepoint_label=time_label1,
            n_samples=n_samples,
            include_patient_effects=True
        )
        # shape (n_samples, n_phenotypes)
        pheno_samples1 = posterior_result1["phenotype_samples"]

        # 3) Posterior samples for distribution at time2
        posterior_result2 = self.generate_posterior_samples(
            patient_label=patient_label,
            tcr_label=tcr_label,
            timepoint_label=time_label2,
            n_samples=n_samples,
            include_patient_effects=True
        )
        pheno_samples2 = posterior_result2["phenotype_samples"]

        # 4) Average across draws => p1, p2
        p1 = pheno_samples1.mean(axis=0)  # shape (n_phenotypes,)
        p2 = pheno_samples2.mean(axis=0)

        # 5) Optional temperature scaling
        if temperature != 1.0:
            def temp_scale(vec, tau):
                v_power = vec ** (1.0 / tau)
                denom = v_power.sum()
                if denom > 1e-12:
                    return v_power / denom
                else:
                    return np.ones_like(vec) / len(vec)

            p1 = temp_scale(p1, temperature)
            p2 = temp_scale(p2, temperature)

        # 6) Compute distance
        eps = 1e-12
        p1 = np.clip(p1, eps, None)
        p2 = np.clip(p2, eps, None)
        p1 /= p1.sum()
        p2 /= p2.sum()

        if metric.lower() == "l1":
            # sum |p1 - p2|
            flux_value = float(np.sum(np.abs(p1 - p2)))
        elif metric.lower() == "dkl":
            # sum p1 log(p1/p2) = KL(p1||p2)
            flux_value = float(np.sum(p1 * np.log(p1 / p2)))
        else:
            raise ValueError(f"Unknown metric {metric}, use 'l1' or 'dkl'.")

        return flux_value


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
        Flux between time1, time2 for TCR=clone_idx, patient=patient_idx,
        using the single point estimate from local_clone_concentration_pct.
        Optionally apply a post-hoc temperature transform to each distribution.

        Args:
            clone_idx: index in [0..K-1]
            patient_idx: index in [0..C-1]
            time1, time2: indices in [0..T-1]
            metric: "l1" or "dkl"
            temperature: post-hoc temperature factor
                - 1.0 => no change (original distribution)
                - <1 => sharper
                - >1 => flatter

        Returns:
            float: flux value (distance) between the two distributions at (time1, time2).
        """
        import torch
        import pyro

        # Retrieve the single point estimate of Dirichlet concentration
        local_conc_pct = pyro.param("local_clone_concentration_pct").detach().cpu()
        # shape => (K*C*T, n_phenotypes)

        # Flatten index => k*(C*T) + c*T + t
        idx1 = clone_idx*(self.C*self.T) + patient_idx*self.T + time1
        idx2 = clone_idx*(self.C*self.T) + patient_idx*self.T + time2

        conc1 = local_conc_pct[idx1]
        conc2 = local_conc_pct[idx2]

        # Convert to distributions
        eps = 1e-12
        p1 = conc1 / torch.clamp(conc1.sum(), min=eps)
        p2 = conc2 / torch.clamp(conc2.sum(), min=eps)

        # Post-hoc temperature scaling if needed
        def temp_scale(tensor_dist, tau):
            if tau == 1.0:
                return tensor_dist
            power = tensor_dist ** (1.0 / tau)
            denom = torch.clamp(power.sum(), min=eps)
            return power / denom

        if temperature != 1.0:
            p1 = temp_scale(p1, temperature)
            p2 = temp_scale(p2, temperature)

        # Re-clamp for safety
        p1 = torch.clamp(p1, min=eps)
        p2 = torch.clamp(p2, min=eps)

        # Final metric
        if metric.lower() == "l1":
            return float(torch.sum(torch.abs(p1 - p2)))
        elif metric.lower() == "dkl":
            return float(torch.sum(p1 * torch.log(p1 / p2)))
        else:
            raise ValueError(f"Unknown metric {metric}, use 'l1' or 'dkl'.")


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
        - TCR=clone_idx (int index)
        - patient=patient_idx (int index)
        that compares *all* posterior draws pairwise, then averages.

        Steps:
        1) Map integer (patient_idx, timeX, clone_idx) -> labels
        2) generate_posterior_samples(...) for time1 => shape (n_samples, n_phenotypes)
            generate_posterior_samples(...) for time2 => shape (n_samples, n_phenotypes)
        3) For each sample i from time1, each sample j from time2:
            - Possibly apply temperature transform
            - Normalize distribution
            - Compute chosen distance (l1 or dkl)
            Sum over i,j => average by dividing by (n_samples^2).
        4) Return that scalar flux measure.

        Args:
        clone_idx: which clone index in [0..K-1]
        patient_idx: which patient index in [0..C-1]
        time1, time2: which time indices in [0..T-1]
        metric: "l1" or "dkl"
        n_samples: number of posterior draws for each time distribution
        temperature: post-hoc temperature factor (<1 => sharper, >1 => flatter, =1 => no change)

        Returns:
        flux_value: float, the average pairwise distance between draws from time1, time2.
        """
        import numpy as np
        import math

        # 1) Map integer -> labels for patient, time
        rev_patient_map = {v: k for k, v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx} in covariate_mapping.")

        if self.T > 0:
            rev_time_map = {v: k for k, v in self.dataset.timepoint_mapping.items()}
            time_label1 = rev_time_map.get(time1, None)
            time_label2 = rev_time_map.get(time2, None)
            if time_label1 is None or time_label2 is None:
                raise ValueError(f"Unknown time indices {time1}, {time2} in timepoint_mapping.")
        else:
            time_label1 = None
            time_label2 = None

        # Map clone_idx -> TCR label
        tcr_label = self.clone_mapping.get(clone_idx, None)
        if tcr_label is None:
            raise ValueError(f"Unknown clone_idx={clone_idx} in clone_mapping.")

        # 2) Generate posterior samples for time1, time2
        #    shape => (n_samples, n_phenotypes) each
        posterior_result1 = self.generate_posterior_samples(
            patient_label=patient_label,
            tcr_label=tcr_label,
            timepoint_label=time_label1,
            n_samples=n_samples,
            include_patient_effects=True
        )
        pheno_samples1 = posterior_result1["phenotype_samples"]

        posterior_result2 = self.generate_posterior_samples(
            patient_label=patient_label,
            tcr_label=tcr_label,
            timepoint_label=time_label2,
            n_samples=n_samples,
            include_patient_effects=True
        )
        pheno_samples2 = posterior_result2["phenotype_samples"]

        # 3) Define a helper for temperature transform + normalization
        eps = 1e-12
        def process_distribution(vec: np.ndarray, tau: float) -> np.ndarray:
            # clamp
            vec = np.clip(vec, eps, None)
            if tau != 1.0:
                v_power = vec ** (1.0 / tau)
                denom = v_power.sum()
                if denom > eps:
                    vec = v_power / denom
                else:
                    vec = np.ones_like(vec) / len(vec)
            else:
                # no temperature transform, just renormalize
                s = vec.sum()
                if s < eps:
                    vec = np.ones_like(vec) / len(vec)
                else:
                    vec = vec / s
            return vec

        # 4) Pairwise distances
        total_dist = 0.0
        count = 0

        for i in range(n_samples):
            p1 = process_distribution(pheno_samples1[i], temperature)  # shape (n_phenotypes,)
            for j in range(n_samples):
                p2 = process_distribution(pheno_samples2[j], temperature)

                if metric.lower() == "l1":
                    dist_ij = np.sum(np.abs(p1 - p2))
                elif metric.lower() == "dkl":
                    # KL(p1||p2) = sum p1 * log(p1/p2)
                    dist_ij = np.sum(p1 * np.log(p1 / p2))
                else:
                    raise ValueError(f"Unknown metric {metric}, use 'l1' or 'dkl'.")

                total_dist += dist_ij
                count += 1

        flux_value = float(total_dist / count)
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
        For each clone k, compute H(Phenotype | Clone=k) from the single point estimate
        in local_clone_concentration_pct, optionally:
        - weighting the distribution by clone frequency
        - applying a post-hoc temperature transform
        - normalizing the entropy by ln(n_phenotypes).

        Returns:
        dict: {clone_idx: entropy_value}.
        """
        import torch
        import pyro
        import math
        import numpy as np

        eps = 1e-12

        # Helper: apply temperature transform to a distribution
        def temperature_transform(p: torch.Tensor, tau: float) -> torch.Tensor:
            if tau == 1.0:
                # Just ensure nonzero and re-normalize
                s = torch.clamp(p.sum(), min=eps)
                return p / s
            else:
                p_power = p**(1.0 / tau)
                denom = torch.clamp(p_power.sum(), min=eps)
                return p_power / denom

        # 1) Retrieve local_clone_concentration_pct => shape (K*C*T, n_phenotypes)
        local_conc_pt = pyro.param("local_clone_concentration_pct").detach().cpu()

        # 2) Possibly gather real clone frequencies
        freq_for_clone = np.ones(self.K, dtype=np.float32)
        if weight_by_clone_freq:
            freq_for_clone[:] = 0.0
            clone_sizes = self.adata.obs["clone_size"].values  # shape (#cells,)

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
                freq_for_clone[:] = 1.0 / self.K

        # 3) For each clone, read out the distribution => apply freq => temperature => compute entropy
        phen_entropy_by_clone = {}

        for k in range(self.K):
            kct = k * (self.C * self.T) + patient_idx * self.T + time_idx
            conc_k = local_conc_pt[kct]  # shape (n_phenotypes,)

            # Convert to distribution p(phi|k)
            denom = torch.clamp(conc_k.sum(), min=eps)
            p_phi_given_k = conc_k / denom

            # (Optional) multiply by freq_for_clone[k], then re-normalize
            if weight_by_clone_freq:
                p_phi_given_k = p_phi_given_k * freq_for_clone[k]

            # Post-hoc temperature transform
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)
            # Now it sums to ~1

            # H(Phenotype | Clone=k) = - sum_phi p(phi|k) ln p(phi|k)
            p_phi_given_k = torch.clamp(p_phi_given_k, min=eps)
            entropy_val = -torch.sum(p_phi_given_k * torch.log(p_phi_given_k))

            # Normalize by ln(n_phenotypes)?
            if normalize and self.n_phenotypes > 1:
                entropy_val = entropy_val / math.log(self.n_phenotypes)

            phen_entropy_by_clone[k] = float(entropy_val.item())

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
        A Bayesian version of H(Phenotype | Clone=k) for each clone k in [0..K-1],
        by sampling each clone's posterior distribution at (patient_idx, time_idx),
        optionally multiplying by clone frequency, applying post-hoc temperature,
        and computing entropy in nats (or normalized by ln(n_phenotypes)).

        Steps:
        1) Possibly gather clone frequencies freq_for_clone[k].
        2) For each clone k:
            - generate_posterior_samples(...) => shape (n_samples, n_phenotypes)
            - average => p(phi|k)
            - multiply by freq_for_clone[k] => re-normalize if weight_by_clone_freq=True
            - apply temperature => p^(1/tau), re-normalize
            - compute H(Phenotype|Clone=k) = - sum_phi p(phi|k)* log p(phi|k)
        3) If normalize=True, divide by ln(n_phenotypes).
        4) Return dict {k: entropy_val}.

        Returns:
        phen_entropy_by_clone: dict {clone_idx: float} in [0..ln(n_phenotypes)] or [0..1] if normalized.
        """
        import numpy as np
        import math

        # ------------------- 0) Map integer => labels (patient, time) -------------------
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

        # ------------------- 1) Possibly gather clone frequencies -------------------
        if weight_by_clone_freq:
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values  # (#cells,)

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
            freq_for_clone = np.ones(K, dtype=np.float32)  # no weighting

        # ------------------- 2) A helper for temperature transform + re-normalize -------------------
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

        # ------------------- 3) For each clone => posterior draws => average => weighting => temperature => entropy -------------------
        phen_entropy_by_clone = {}

        for k_idx in range(K):
            tcr_label = self.clone_mapping.get(k_idx, None)
            if tcr_label is None:
                continue

            # 3a) Generate posterior samples for this (patient, tcr, time)
            posterior_result = self.generate_posterior_samples(
                patient_label=patient_label,
                tcr_label=tcr_label,
                timepoint_label=time_label,
                n_samples=n_samples,
                include_patient_effects=True
            )
            # shape => (n_samples, n_phenotypes)
            pheno_samples = posterior_result["phenotype_samples"]

            # 3b) average => p(phi|k)
            p_phi_given_k = pheno_samples.mean(axis=0)  # shape (P,)

            # 3c) multiply by freq_for_clone => re-normalize if weighting
            if weight_by_clone_freq:
                p_phi_given_k *= freq_for_clone[k_idx]
            s2 = p_phi_given_k.sum()
            if s2 < eps:
                p_phi_given_k[:] = 1.0 / P
            else:
                p_phi_given_k /= s2

            # 3d) temperature transform
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)

            # 3e) H(Phenotype|Clone=k) = - sum_phi p(phi|k)* ln p(phi|k)
            p_phi_given_k = np.clip(p_phi_given_k, eps, None)
            p_phi_given_k /= p_phi_given_k.sum()  # ensure sum=1
            entropy_val = -np.sum(p_phi_given_k * np.log(p_phi_given_k))

            # 3f) optionally normalize by ln(n_phenotypes)
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
        Compute I(Clone; Phenotype) for a given (patient_idx, time_idx), optionally
        applying a post-hoc temperature transform to each phenotype distribution p(phi|k).

        If weight_by_clone_freq=True, we use actual clone frequencies from adata.obs["clone_size"].
        Otherwise, clones are weighted uniformly.

        temperature: float
            - 1.0 => no change (use the model's distribution)
            - <1.0 => "sharpen" distribution (more peaked)
            - >1.0 => "flatten" distribution (less peaked)

        Returns:
            mutual_info (float) in bits (log base 2).
        """
        import torch
        import numpy as np
        import pyro

        # Helper function to apply temperature scaling to a 2D batch of distributions
        def apply_temperature_scaling_2d(p, tau):
            """
            p: Tensor of shape (K, n_phenotypes), each row sums to ~1
            tau: temperature factor. tau < 1 => sharper, tau > 1 => flatter
            """
            if tau == 1.0:
                return p  # no change
            p_power = p ** (1.0 / tau)               # raise each prob to 1/tau
            row_sums = p_power.sum(dim=1, keepdim=True)
            row_sums = torch.clamp(row_sums, min=1e-12)
            return p_power / row_sums

        # -------------------------------------------------------------
        # 1) Retrieve local_clone_concentration_pct and build p(phi|k)
        # -------------------------------------------------------------
        local_conc_pt = pyro.param("local_clone_concentration_pct").detach().cpu()  
        # shape => (K*C*T, n_phenotypes)

        # We'll build a distribution p(phi|k), plus p(k)
        dist_kp = []
        clone_freqs = []

        # -------------------------------------------------------------
        # 2) If weight_by_clone_freq, gather frequencies from adata
        # -------------------------------------------------------------
        if weight_by_clone_freq:
            freq_for_clone = np.zeros(self.K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values  # shape => (#cells,)

            for i in range(len(self.dataset)):
                cell_patient = self.dataset.covariates[i].item()
                cell_time = self.dataset.timepoints[i].item() if self.dataset.timepoints is not None else 0
                if cell_patient == patient_idx and cell_time == time_idx:
                    clone_id = self.dataset.tcrs[i].item()
                    freq_for_clone[clone_id] += clone_sizes[i]

            total_size = freq_for_clone.sum()
            if total_size > 0:
                freq_for_clone /= total_size
            else:
                freq_for_clone[:] = 1.0 / self.K

        # -------------------------------------------------------------
        # 3) Construct p(phi|k) for each clone k
        # -------------------------------------------------------------
        for k in range(self.K):
            # Flatten index => k*(C*T) + patient_idx*T + time_idx
            kct = k * (self.C * self.T) + patient_idx * self.T + time_idx

            conc_k = local_conc_pt[kct]
            # unscaled distribution
            p_phi_given_k = conc_k / torch.clamp(conc_k.sum(), min=1e-12)

            dist_kp.append(p_phi_given_k)

            # p(k)
            if weight_by_clone_freq:
                p_k = freq_for_clone[k]
            else:
                p_k = 1.0  # uniform weighting
            clone_freqs.append(p_k)

        # Stack => shape (K, n_phenotypes)
        dist_kp = torch.stack(dist_kp, dim=0)

        # -------------------------------------------------------------
        # 4) Apply post-hoc temperature scaling
        # -------------------------------------------------------------
        dist_kp = apply_temperature_scaling_2d(dist_kp, temperature)  # shape (K, n_phenotypes)

        # -------------------------------------------------------------
        # 5) Build p(k) array and normalize
        # -------------------------------------------------------------
        clone_freqs = np.array(clone_freqs, dtype=np.float32)
        clone_freqs_sum = clone_freqs.sum()
        if clone_freqs_sum > 0:
            clone_freqs /= clone_freqs_sum
        else:
            clone_freqs[:] = 1.0 / self.K

        # p(k, phi) = p(k)* p(phi|k)
        pxy = dist_kp * torch.tensor(clone_freqs, dtype=torch.float).unsqueeze(1)
        pxy = pxy.detach().numpy()
        pxy_sum = pxy.sum()
        if pxy_sum > 0:
            pxy /= pxy_sum

        # p(k) = sum_phi p(k, phi)
        px = pxy.sum(axis=1)
        # p(phi) = sum_k p(k, phi)
        py = pxy.sum(axis=0)

        # -------------------------------------------------------------
        # 6) I(X; Y) = sum_{x,y} p(x,y) log2( p(x,y) / [ p(x) p(y) ] )
        # -------------------------------------------------------------
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
        A Bayesian approach to computing I(Clone; Phenotype) for a given (patient_idx, time_idx),
        sampling from the posterior for each clone.

        Steps:
        1) Map integer patient_idx -> patient_label, same for time_idx -> time_label.
        2) Loop over k in [0..K-1], map k -> tcr_label.
            - Call self.generate_posterior_samples(patient_label, tcr_label, timepoint_label, 
            n_samples).
            - That yields shape (n_samples, n_phenotypes) for "phenotype_samples."
        3) Average these samples => p(phi|k).
        4) Apply optional temperature transform p(phi|k)^(1/tau).
        5) Weight by p(k) if weight_by_clone_freq else uniform => p(k, phi).
        6) Sum => p(phi). Then compute I(Clone; Phenotype) = sum_{k,phi} p(k,phi) log2 [ p(k,phi)/(p(k)*p(phi)) ].

        Args:
        patient_idx: integer index of patient
        time_idx: integer index of time
        n_samples: how many posterior draws for each clone
        weight_by_clone_freq: whether to scale p(k) by real clone frequencies
        temperature: <1 => sharpen p(phi|k), >1 => flatten, 1 => no change

        Returns:
        mutual_info (float) in bits (log2).
        """
        import numpy as np
        import math

        # -----------------------------------------------------------------
        # 0) Map integer -> label for patient, time
        # -----------------------------------------------------------------
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
            time_label = None  # or "0"

        K = self.K
        P = self.n_phenotypes

        # We'll accumulate p(k,phi) in shape (K, P).
        p_k_phi = np.zeros((K, P), dtype=np.float64)

        # -----------------------------------------------------------------
        # 1) Possibly gather real clone frequencies if requested
        # -----------------------------------------------------------------
        if weight_by_clone_freq:
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values  # shape (#cells,)

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

        # -----------------------------------------------------------------
        # 2) Loop over clones k => generate posterior samples => average => apply temperature => build p(k,phi)
        # -----------------------------------------------------------------
        for k_idx in range(K):
            # Map k_idx -> tcr_label
            tcr_label = self.clone_mapping.get(k_idx, None)
            if tcr_label is None:
                continue

            # Generate samples for this (patient, tcr, time)
            posterior_result = self.generate_posterior_samples(
                patient_label=patient_label,
                tcr_label=tcr_label,
                timepoint_label=time_label,
                n_samples=n_samples,
                include_patient_effects=True
            )
            # shape => (n_samples, n_phenotypes)
            pheno_samples = posterior_result["phenotype_samples"]

            # Average => p(phi|k)
            p_phi_given_k = pheno_samples.mean(axis=0)  # shape (n_phenotypes,)

            # Temperature transform
            if temperature != 1.0:
                p_power = p_phi_given_k ** (1.0 / temperature)
                denom = p_power.sum()
                if denom > 1e-12:
                    p_phi_given_k = p_power / denom
                else:
                    p_phi_given_k = np.ones(P, dtype=np.float32) / P

            # p(k, phi) = p(k)* p(phi|k)
            for phi_idx in range(P):
                p_k_phi[k_idx, phi_idx] = freq_for_clone[k_idx] * p_phi_given_k[phi_idx]

        # -----------------------------------------------------------------
        # 3) Normalize => sum_{k,phi} = 1
        # -----------------------------------------------------------------
        total_mass = p_k_phi.sum()
        if total_mass > 0:
            p_k_phi /= total_mass
        else:
            p_k_phi[:] = 1.0 / (K*P)

        # -----------------------------------------------------------------
        # 4) p(k) = sum_phi p(k,phi), p(phi) = sum_k p(k,phi)
        # -----------------------------------------------------------------
        px = p_k_phi.sum(axis=1)   # shape (K,)
        py = p_k_phi.sum(axis=0)   # shape (P,)

        # -----------------------------------------------------------------
        # 5) I(X;Y) = sum_{k,phi} p(k,phi) log2 [ p(k,phi)/( p(k)* p(phi) ) ]
        # -----------------------------------------------------------------
        eps = 1e-12
        mutual_info = 0.0
        for k_idx in range(K):
            for phi_idx in range(P):
                val = p_k_phi[k_idx, phi_idx]
                if val > eps:
                    denom = px[k_idx]* py[phi_idx]
                    # denom could be 0 if px or py=0 => skip or clamp
                    if denom < eps:
                        continue
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

        Optionally apply a post-hoc temperature transform to each p(phi|k), i.e.
            p_scaled(phi|k) = normalize( p(phi|k)^(1/temperature) ).
        temperature < 1 => sharper / more peaked
        temperature > 1 => flatter

        Returns a dict: {phi_idx: entropy_value_in_bits}.
        """
        import torch
        import pyro
        import math
        from typing import Dict

        # Helper: apply temperature scaling to a 2D matrix of distributions
        def apply_temperature_scaling_2d(p, tau):
            """
            p: Tensor of shape (K, n_phenotypes), each row sums ~1
            tau: float, temperature factor
            """
            if tau == 1.0:
                return p  # no change
            p_power = p ** (1.0 / tau)
            row_sums = p_power.sum(dim=1, keepdim=True).clamp_min(1e-12)
            return p_power / row_sums

        # -------------------------------------------------------------
        # 1) Retrieve local_clone_concentration_pct => shape (K*C*T, n_phenotypes)
        # -------------------------------------------------------------
        local_conc_pt = pyro.param("local_clone_concentration_pct").detach().cpu()

        # Build p(phi|k) distribution for each clone k
        dist_kp = []
        clone_freqs = []

        # -------------------------------------------------------------
        # 2) Possibly gather real clone frequencies if weight_by_clone_freq
        #    (placeholder logic here)
        # -------------------------------------------------------------
        # For now, we just store 1.0 for each clone
        # In practice, you'd compute it similarly to your other function
        for k in range(self.K):
            clone_freqs.append(1.0)

        # -------------------------------------------------------------
        # 3) Construct p(phi|k) for each clone k
        # -------------------------------------------------------------
        for k in range(self.K):
            kct = k * (self.C * self.T) + patient_idx * self.T + time_idx
            conc_k = local_conc_pt[kct]
            p_phi_given_k = conc_k / torch.clamp(conc_k.sum(), min=1e-12)
            dist_kp.append(p_phi_given_k)

        dist_kp = torch.stack(dist_kp, dim=0)  # shape (K, n_phenotypes)

        # -------------------------------------------------------------
        # 4) Apply post-hoc temperature transform
        # -------------------------------------------------------------
        dist_kp = apply_temperature_scaling_2d(dist_kp, temperature)

        # -------------------------------------------------------------
        # 5) Weight clones if requested (placeholder)
        # -------------------------------------------------------------
        clone_freqs = torch.tensor(clone_freqs, dtype=torch.float)
        clone_freqs = clone_freqs / clone_freqs.sum()  # shape => (K,)

        # p(k, phi) = p(k) * p(phi|k)
        p_k_phi = dist_kp * clone_freqs.unsqueeze(1)
        p_k_phi_sum = torch.clamp(p_k_phi.sum(), min=1e-12)
        p_k_phi = p_k_phi / p_k_phi_sum  # ensure total mass = 1

        # p(phi) = sum_k p(k, phi)
        p_phi = p_k_phi.sum(dim=0)  # shape => (n_phenotypes,)

        eps = 1e-12
        clonotypic_entropy_dict: Dict[int, float] = {}

        # -------------------------------------------------------------
        # 6) For each phenotype phi, compute H(Clone | Phi=phi) in bits
        # -------------------------------------------------------------
        for phi_idx in range(self.n_phenotypes):
            denom = p_phi[phi_idx]
            if denom < eps:
                # If p(phi_idx) ~ 0 => no data
                clonotypic_entropy_dict[phi_idx] = 0.0
                continue

            # p(k|phi_idx) = p(k, phi_idx) / p(phi_idx)
            p_k_given_phi = (p_k_phi[:, phi_idx] / denom).clamp_min(eps)

            # H(Clone|phi) = - sum_k p(k|phi) log2 [p(k|phi)]
            H_clone_phi = -torch.sum(p_k_given_phi * torch.log2(p_k_given_phi))

            # Normalize by log2(K) if requested
            if normalize and self.K > 1:
                H_clone_phi = H_clone_phi / math.log2(self.K)

            clonotypic_entropy_dict[phi_idx] = float(H_clone_phi.item())

        return clonotypic_entropy_dict

    def get_clonotypic_entropy_by_phenotype(
        self,
        patient_idx: int,
        time_idx: int,
        n_samples: int = 100,
        weight_by_clone_freq: bool = False,
        normalize: bool = False,
        temperature: float = 1.0
    ) -> Dict[int, float]:
        """
        A Bayesian version of H(Clone | Phenotype), using posterior samples for each clone
        and optionally applying a post-hoc temperature transform.

        Steps:
        1) Convert (patient_idx, time_idx) -> their string labels
        2) For each clone k in [0..K-1], map -> TCR label
            then call generate_posterior_samples(..., n_samples=?),
            which returns shape (n_samples, n_phenotypes).
        3) Average across draws => p(phi|k), then apply temperature scaling if needed.
        4) Weighted by p(k) if weight_by_clone_freq else uniform => p(k,phi).
        5) Compute p(phi) = sum_k p(k,phi) => H(Clone|phi) in bits, 
            optionally normalized by log2(K).

        Returns:
        A dict {phi_idx: H(Clone|phi_idx)} in bits.
        """
        import numpy as np
        import math
        from typing import Dict

        # ------------------ 0) Map integer -> label ------------------
        rev_patient_map = {v:k for k,v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx} in covariate_mapping.")

        if self.T > 0 and time_idx >= 0:
            rev_time_map = {v:k for k,v in self.dataset.timepoint_mapping.items()}
            time_label = rev_time_map.get(time_idx, None)
            if time_label is None:
                raise ValueError(f"Unknown time_idx={time_idx} in timepoint_mapping.")
        else:
            time_label = None  # or "0"

        # -------------- 1) Setup for loops & arrays --------------
        K = self.K
        P = self.n_phenotypes

        # We'll accumulate p(k, phi) in a 2D array shape (K, P).
        p_k_phi_accum = np.zeros((K, P), dtype=np.float64)

        # -------------- 2) Possibly gather clone frequencies --------------
        if weight_by_clone_freq:
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values  # (#cells,)

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

        # -------------- 3) Loop clones k => (patient_label, tcr_label, time_label) --------------
        for k in range(K):
            tcr_label = self.clone_mapping.get(k, None)
            if tcr_label is None:
                continue

            # Generate posterior samples for this (patient, tcr, time)
            posterior_result = self.generate_posterior_samples(
                patient_label=patient_label,
                tcr_label=tcr_label,
                timepoint_label=time_label,
                n_samples=n_samples,
                include_patient_effects=True
            )
            # shape => (n_samples, n_phenotypes)
            pheno_samples = posterior_result["phenotype_samples"]

            # 4) Average across draws => p(phi|k)
            p_phi_given_k = pheno_samples.mean(axis=0)  # shape (n_phenotypes,)

            # 5) Temperature transform (post-hoc)
            if temperature != 1.0:
                p_power = p_phi_given_k ** (1.0 / temperature)
                denom = p_power.sum()
                if denom > 1e-12:
                    p_phi_given_k = p_power / denom
                else:
                    # fallback uniform
                    p_phi_given_k = np.ones(P, dtype=np.float32) / P

            # p(k, phi) = p(k) * p(phi|k)
            for phi_idx in range(P):
                p_k_phi_accum[k, phi_idx] = freq_for_clone[k] * p_phi_given_k[phi_idx]

        # 6) Normalize => sum_{k,phi} p(k,phi)=1
        total_mass = p_k_phi_accum.sum()
        if total_mass > 0:
            p_k_phi_accum /= total_mass
        else:
            p_k_phi_accum[:] = 1.0 / (K * P)

        # 7) p(phi)= sum_k p(k,phi)
        p_phi = p_k_phi_accum.sum(axis=0)  # shape (P,)

        # 8) For each phenotype, compute H(Clone|phi) in bits
        eps = 1e-12
        clonotypic_entropy_dict: Dict[int, float] = {}
        for phi_idx in range(P):
            denom = p_phi[phi_idx]
            if denom < eps:
                clonotypic_entropy_dict[phi_idx] = 0.0
                continue

            # p(k|phi) = p(k,phi)/p(phi)
            p_k_given_phi = p_k_phi_accum[:, phi_idx] / denom
            # clamp
            p_k_given_phi = np.clip(p_k_given_phi, eps, None)
            p_k_given_phi /= p_k_given_phi.sum()

            # H(Clone|phi) = - sum_k p(k|phi)* log2[p(k|phi]]
            H_clone_phi = -np.sum(p_k_given_phi * np.log2(p_k_given_phi))

            if normalize and K > 1:
                H_clone_phi /= math.log2(K)

            clonotypic_entropy_dict[phi_idx] = float(H_clone_phi)

        return clonotypic_entropy_dict

    def generate_posterior_samples_progressive(
        self,
        patient_label: str,
        tcr_label: str,
        timepoint_label: Optional[str] = None,
        n_samples: int = 1000,
        include_patient_effects: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        A 'progressive caching' version of posterior sampling:
        - We store the *largest* number of samples so far for each combo
        (patient_label, tcr_label, timepoint_label, include_patient_effects).
        - If user requests fewer, we sub-slice.
        - If user requests more, we sample only the difference, then store the new total.

        Args:
            patient_label: Label of the patient in adata.obs
            tcr_label: Label of the TCR in adata.obs
            timepoint_label: (Optional) Label of the timepoint in adata.obs.
                            If not provided or self.T==0, defaults to 0.
            n_samples: Number of posterior samples to return
            include_patient_effects: Whether to sample patient-specific offsets

        Returns:
            A dict with:
                - "phenotype_samples": shape (n_samples, self.n_phenotypes)
                - "gene_expression_samples": shape (n_samples, self.D)
                - "patient_effect_samples": shape (n_samples,) if include_patient_effects=True
                - "phenotype_labels": list of length self.n_phenotypes
        """
        import numpy as np
        import torch
        import pyro
        import pyro.distributions as dist

        # 1) Build a cache key that ignores n_samples
        #    (We store and track the *maximum* number for that key.)
        cache_key = (patient_label, tcr_label, timepoint_label, include_patient_effects)

        # 2) Map user labels -> indices
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

        # 3) See if we have a cache entry
        cache_entry = self._posterior_cache.get(cache_key, None)

        # 4) If we do not have any cached samples, we sample from scratch
        #    Otherwise we only sample the difference if n_samples > cached 'max_samples'
        cached_max = 0
        if cache_entry is not None:
            cached_max = cache_entry["max_samples"]

        need_to_sample = n_samples - cached_max

        # 5) Retrieve pyro params
        local_conc_pt = pyro.param("local_clone_concentration_pct").detach().cpu()
        local_dirichlet_conc = local_conc_pt[kct_index]  # shape (n_phenotypes,)

        unconstrained_log = pyro.param("unconstrained_gene_profile_log").detach().cpu()
        log_clamped = torch.clamp(unconstrained_log, min=-10.0, max=10.0)
        raw_concentration = torch.exp(log_clamped)  # shape (n_phenotypes, D)

        if include_patient_effects:
            patient_var_shape = pyro.param("patient_variance_shape")[patient_idx].detach().cpu()
            patient_var_rate  = pyro.param("patient_variance_rate")[patient_idx].detach().cpu()
        else:
            patient_var_shape = None
            patient_var_rate  = None

        # 6) If need_to_sample > 0, we do the sampling loop for that many draws
        if need_to_sample > 0:
            # Prepare arrays for the new draws
            new_phenotype_samples = np.zeros((need_to_sample, self.n_phenotypes), dtype=np.float32)
            new_gene_expression_samples = np.zeros((need_to_sample, self.D), dtype=np.float32)
            new_patient_effect_samples = None
            if include_patient_effects:
                new_patient_effect_samples = np.zeros(need_to_sample, dtype=np.float32)

            for i in range(need_to_sample):
                phenotype_dist = dist.Dirichlet(local_dirichlet_conc).sample().numpy()
                new_phenotype_samples[i] = phenotype_dist

                gene_profile_draw = dist.Dirichlet(raw_concentration).sample().numpy()
                mixed_profile = np.sum(gene_profile_draw * phenotype_dist[:, np.newaxis], axis=0)

                if include_patient_effects:
                    p_effect = dist.Gamma(patient_var_shape, patient_var_rate).sample().item()
                    new_patient_effect_samples[i] = p_effect
                    mixed_profile *= p_effect

                denom = mixed_profile.sum() + 1e-12
                new_gene_expression_samples[i] = mixed_profile / denom

            # If we have an existing cache entry, we append
            if cache_entry is not None:
                # Append new samples to existing
                phenotype_samples_full = np.concatenate(
                    [cache_entry["phenotype_samples"], new_phenotype_samples],
                    axis=0
                )
                gene_expression_full = np.concatenate(
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
                # No existing cache => these new samples are the full set
                phenotype_samples_full = new_phenotype_samples
                gene_expression_full   = new_gene_expression_samples
                pe_full = new_patient_effect_samples

            # Build a new or updated cache entry
            new_cache_entry = {
                "max_samples": cached_max + need_to_sample,
                "phenotype_samples": phenotype_samples_full,
                "gene_expression_samples": gene_expression_full,
                "patient_effect_samples": pe_full
            }
            self._posterior_cache[cache_key] = new_cache_entry

        # 7) Now we can build the result by sub-slicing the first n_samples
        final_cache = self._posterior_cache[cache_key]
        # sub-slice
        selected_phenotype = final_cache["phenotype_samples"][:n_samples]
        selected_gene_expr = final_cache["gene_expression_samples"][:n_samples]
        selected_patient_eff = None
        if include_patient_effects:
            selected_patient_eff = final_cache["patient_effect_samples"][:n_samples]

        # 8) Build the final result
        result = {
            "phenotype_samples": selected_phenotype,
            "gene_expression_samples": selected_gene_expr
        }
        if include_patient_effects:
            result["patient_effect_samples"] = selected_patient_eff

        # Add phenotype labels
        phenotype_labels_ordered = []
        for i in range(self.n_phenotypes):
            phenotype_labels_ordered.append(self.phenotype_mapping[i])
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
