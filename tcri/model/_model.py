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
            # -- Added arguments to control priors in model/guide --
            clone_to_phenotype_prior_strength: float = 10.0,
            gene_profile_prior_strength: float = 5.0,
            gene_profile_prior_offset: float = 0.5,
            persistence_factor: float = 100.0,
            consistency_weight: float = 0.1,
            patient_variance_shape_val: float = 4.0,
            patient_variance_rate_val: float = 4.0,
            beta = 1.0,
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

        # Store prior-related parameters
        self.local_concentration_offset = local_concentration_offset
        self.time_variance_prior = time_variance_prior
        self.clone_to_phenotype_prior_strength = clone_to_phenotype_prior_strength
        self.gene_profile_prior_strength = gene_profile_prior_strength
        self.gene_profile_prior_offset = gene_profile_prior_offset
        self.persistence_factor = persistence_factor
        self.consistency_weight = consistency_weight
        self.patient_variance_shape_val = patient_variance_shape_val
        self.patient_variance_rate_val = patient_variance_rate_val
        self.beta = beta
        self.gene_concentration = gene_concentration

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
            for phenotype_idx, genes in marker_genes.items():
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
            loss=TraceMeanField_ELBO(num_particles=5)
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
        import torch
        import pyro
        import pyro.distributions as dist

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
                dist.Dirichlet(base_dist*self.persistence_factor + 1.0)
            )

            # (Optional) Clone consistency penalty
            unique_clones = torch.unique(tcr_idx)
            consistency_loss = 0.0
            for clone_id in unique_clones:
                mask = (tcr_idx == clone_id)
                if mask.sum() > 1:
                    cells_clone = cell_phenotype_dist[mask]
                    mean_clone = torch.mean(cells_clone, dim=0)
                    dev = torch.mean(torch.sum((cells_clone - mean_clone)**2, dim=1))
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
        import torch
        import pyro
        import pyro.distributions as dist
        from pyro.distributions import constraints

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
                    dist.Normal(unconstrained_time_offset_log, 1.0).to_event(1)
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
                dist.Dirichlet(cell_phenotype_concentration)
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

    def get_phenotypic_flux(
        self,
        clone_idx: int,
        patient_idx: int,
        time1: int,
        time2: int,
        metric: str = "l1"
    ) -> float:
        """
        Flux between time1, time2 for TCR=clone_idx, patient=patient_idx,
        referencing local_clone_concentration_pct shape (K*C*T, ...).
        """
        import torch
        import pyro

        local_conc_pct = pyro.param("local_clone_concentration_pct").detach().cpu()
        idx1 = clone_idx*(self.C*self.T) + patient_idx*self.T + time1
        idx2 = clone_idx*(self.C*self.T) + patient_idx*self.T + time2

        conc1 = local_conc_pct[idx1]
        conc2 = local_conc_pct[idx2]

        p1 = conc1/conc1.sum()
        p2 = conc2/conc2.sum()

        if metric.lower()=="l1":
            return float(torch.sum(torch.abs(p1 - p2)))
        elif metric.lower()=="dkl":
            eps=1e-12
            p1c = torch.clamp(p1, min=eps)
            p2c = torch.clamp(p2, min=eps)
            return float(torch.sum(p1c * torch.log(p1c/p2c)))
        else:
            raise ValueError(f"Unknown metric {metric}")


    def get_phenotypic_entropy_by_clone(
        self,
        patient_idx: int,
        time_idx: int,
        weight_by_clone_freq: bool = False,
        normalize: bool = False
    ) -> Dict[int, float]:
        """
        For each clone k, compute H(Phenotype | Clone=k). If `normalize=True`, 
        we divide by ln(n_phenotypes).

        Returns a dict: {k: H(Phenotype|Clone=k) in [0..(ln(n_phenotypes) or 1 if normalized)]}.
        """
        import torch
        import pyro
        import math

        local_conc_pt = pyro.param("local_clone_concentration_pct").detach().cpu()
        # shape (K*C*T, n_phenotypes)

        phen_entropy_by_clone = {}
        eps = 1e-12

        for k in range(self.K):
            kct = k * (self.C * self.T) + patient_idx * self.T + time_idx
            conc_k = local_conc_pt[kct]  # shape (n_phenotypes,)

            # p(phi|k)
            p_phi_given_k = conc_k / conc_k.sum()
            p_phi_given_k = torch.clamp(p_phi_given_k, min=eps)

            # H(Phenotype|Clone=k) = - sum_phi p(phi|k) log p(phi|k)
            entropy_val = -torch.sum(p_phi_given_k * torch.log(p_phi_given_k))

            if normalize and self.n_phenotypes > 1:
                entropy_val = entropy_val / math.log(self.n_phenotypes)

            phen_entropy_by_clone[k] = entropy_val.item()

        return phen_entropy_by_clone


    def get_phenotypic_entropy(
        self,
        patient_idx: int,
        time_idx: int,
        weight_by_clone_freq: bool = False,
        normalize: bool = False
    ) -> float:
        """
        Computes H(Clone | Phenotype) aggregated across all phenotypes for 
        (patient_idx, time_idx). If `normalize=True`, the value is divided by ln(K).

        Returns:
            phenotypic_entropy (float) in [0, 1] if normalize=True.
        """
        import torch
        import pyro
        import math
        
        local_conc_pt = pyro.param("local_clone_concentration_pct").detach().cpu()
        dist_kp = []
        clone_freqs = []
        
        for k in range(self.K):
            kct = k * (self.C * self.T) + patient_idx * self.T + time_idx
            conc_k = local_conc_pt[kct]
            # Convert to distribution p(phi|k)
            p_phi_given_k = conc_k / conc_k.sum()
            dist_kp.append(p_phi_given_k)
            clone_freqs.append(1.0)  # or real frequencies...

        dist_kp = torch.stack(dist_kp, dim=0)  # (K, n_phenotypes)
        clone_freqs = torch.tensor(clone_freqs, dtype=torch.float)
        clone_freqs = clone_freqs / clone_freqs.sum()

        # p(phi) = sum_k [ p(k) * p(phi|k) ]
        p_phi = (dist_kp.T * clone_freqs).sum(dim=1)
        p_phi = p_phi / p_phi.sum()

        eps = 1e-12
        p_k_given_phi = (dist_kp * clone_freqs.unsqueeze(-1)) / (p_phi.unsqueeze(0) + eps)

        # H(Clone|phi) = -sum_k p(k|phi) log p(k|phi)
        entropy_per_phi = []
        for phi_idx in range(self.n_phenotypes):
            p_k_phi = torch.clamp(p_k_given_phi[:, phi_idx], min=eps)
            h_clone_phi = -torch.sum(p_k_phi * torch.log(p_k_phi))
            entropy_per_phi.append(h_clone_phi)

        entropy_per_phi = torch.stack(entropy_per_phi, dim=0)
        phenotypic_entropy = torch.sum(p_phi * entropy_per_phi)

        # Normalize by ln(K)
        if normalize and self.K > 1:
            phenotypic_entropy = phenotypic_entropy / math.log(self.K)

        return phenotypic_entropy.item()

    
    def get_mutual_information(
        self,
        patient_idx: int,
        time_idx: int,
        weight_by_clone_freq: bool = False
    ) -> float:
        """
        Compute I(Clone; Phenotype) for a given (patient, time).
        
        We'll estimate using `local_clone_concentration_pt` in the param store:
        - For each clone k in [0..K-1], get p(phi|k) from the normalized concentration
        - Assign p(k) either uniform or from a placeholder if weight_by_clone_freq=True
        - Then compute:
            p(k,phi) = p(k) * p(phi|k)
            p(phi) = sum_k p(k,phi)
            p(k) = sum_phi p(k,phi)
        
        From these, we do:
        H(Clone) = -sum_k p(k) log p(k)
        H(Phenotype) = -sum_phi p(phi) log p(phi)
        H(Clone,Phenotype) = -sum_{k,phi} p(k,phi) log p(k,phi)
        
        And:
        I(Clone;Phenotype) = H(Clone) + H(Phenotype) - H(Clone,Phenotype).
        
        Args:
            patient_idx: Index of the patient in [0..C-1]
            time_idx: Index of the time in [0..T-1]
            weight_by_clone_freq: If True, use non-uniform p(k) (currently placeholder).
        
        Returns:
            mutual_information (float)
        """
        import torch
        import pyro

        local_conc_pt = pyro.param("local_clone_concentration_pct").detach().cpu()  # (K*C*T, n_phenotypes)

        dist_kp = []
        clone_freqs = []
        
        for k in range(self.K):
            kct = k * (self.C * self.T) + patient_idx * self.T + time_idx
            conc_k = local_conc_pt[kct]
            p_phi_given_k = conc_k / conc_k.sum()
            dist_kp.append(p_phi_given_k)
            
            # p(k) .... placeholder
            if weight_by_clone_freq:
                clone_freqs.append(1.0)
            else:
                clone_freqs.append(1.0)
        
        dist_kp = torch.stack(dist_kp, dim=0)  # shape (K, n_phenotypes)
        clone_freqs = torch.tensor(clone_freqs, dtype=torch.float)
        clone_freqs = clone_freqs / clone_freqs.sum()  # shape (K,)
        
        # 2) p(k, phi) = p(k)* p(phi|k)
        p_k_phi = dist_kp * clone_freqs.unsqueeze(1)  # (K, n_phenotypes)
        total_mass = p_k_phi.sum()
        
        # Normalize to be safe
        p_k_phi = p_k_phi / total_mass
        
        # 3) p(k) = sum_phi p(k,phi)
        p_k = p_k_phi.sum(dim=1)
        # 4) p(phi) = sum_k p(k,phi)
        p_phi = p_k_phi.sum(dim=0)
        
        eps = 1e-12
        p_k_clamp = torch.clamp(p_k, min=eps)
        p_phi_clamp = torch.clamp(p_phi, min=eps)
        p_k_phi_clamp = torch.clamp(p_k_phi, min=eps)
        
        # H(Clone)
        H_clone = -torch.sum(p_k_clamp * torch.log(p_k_clamp))
        # H(Phenotype)
        H_phi = -torch.sum(p_phi_clamp * torch.log(p_phi_clamp))
        # H(Clone, Phenotype)
        H_k_phi = -torch.sum(p_k_phi_clamp * torch.log(p_k_phi_clamp))
        
        # I(Clone; Phenotype)
        mutual_info = (H_clone + H_phi - H_k_phi).item()
        return mutual_info

    def get_clonotypic_entropy(
        self, 
        patient_idx: int, 
        time_idx: int, 
        weight_by_clone_freq: bool = False,
        normalize: bool = False
    ) -> float:
        """
        Compute H(Phenotype | Clone) aggregated across clones for (patient_idx, time_idx).
        If `normalize=True`, we divide by ln(n_phenotypes).

        Returns:
            clonotypic_entropy (float) in [0, 1] if normalize=True.
        """
        import torch
        import pyro
        import math

        local_conc_pt = pyro.param("local_clone_concentration_pct").detach().cpu()
        entropies = []
        clone_freqs = []

        for k in range(self.K):
            kct = k * (self.C * self.T) + patient_idx * self.T + time_idx
            conc_k = local_conc_pt[kct]  # shape (n_phenotypes,)
            
            dist_k = conc_k / conc_k.sum()
            eps = 1e-12
            p_k = torch.clamp(dist_k, min=eps)
            entropy_k = -torch.sum(p_k * torch.log(p_k))
            entropies.append(entropy_k.item())

            # weighting if we have fequencies
            clone_freqs.append(1.0)

        clone_freqs = torch.tensor(clone_freqs, dtype=torch.float)
        clone_freqs = clone_freqs / clone_freqs.sum()
        entropies = torch.tensor(entropies, dtype=torch.float)

        clonotypic_entropy = torch.sum(clone_freqs * entropies)

        # Normalize by ln(n_phenotypes) 
        if normalize and self.n_phenotypes > 1:
            clonotypic_entropy = clonotypic_entropy / math.log(self.n_phenotypes)

        return clonotypic_entropy.item()

    def get_clonotypic_entropy_by_phenotype(
        self,
        patient_idx: int,
        time_idx: int,
        weight_by_clone_freq: bool = False,
        normalize: bool = False
    ) -> Dict[int, float]:
        """
        For each phenotype phi, compute H(Clone | Phenotype=phi). 
        If `normalize=True`, we divide by ln(K).

        Returns a dict: {phi: entropy_value}, 
        where entropy_value = - sum_k p(k|phi) log p(k|phi).
        """
        import torch
        import pyro
        import math
        from typing import Dict
        
        local_conc_pt = pyro.param("local_clone_concentration_pct").detach().cpu()
        # shape (K*C*T, n_phenotypes)

        # Build p(k,phi) = p(k)*p(phi|k).
        # some weighting if weight_by_clone_freq is True. 
        dist_kp = []
        clone_freqs = []
        for k in range(self.K):
            kct = k * (self.C * self.T) + patient_idx * self.T + time_idx
            conc_k = local_conc_pt[kct]
            p_phi_given_k = conc_k / conc_k.sum()
            dist_kp.append(p_phi_given_k)
            if weight_by_clone_freq:
                clone_freqs.append(1.0)  # placeholder
            else:
                clone_freqs.append(1.0)
        
        dist_kp = torch.stack(dist_kp, dim=0)  # (K, n_phenotypes)
        clone_freqs = torch.tensor(clone_freqs, dtype=torch.float)
        clone_freqs = clone_freqs / clone_freqs.sum()

        p_k_phi = dist_kp * clone_freqs.unsqueeze(1)
        p_k_phi = p_k_phi / p_k_phi.sum()  # shape (K, n_phenotypes)

        p_phi = p_k_phi.sum(dim=0)  # shape (n_phenotypes,)

        eps = 1e-12
        clonotypic_entropy_dict: Dict[int, float] = {}

        for phi_idx in range(self.n_phenotypes):
            denom = p_phi[phi_idx]
            if denom < eps:
                # If p(phi_idx) is ~0, no data
                clonotypic_entropy_dict[phi_idx] = 0.0
                continue

            # p(k|phi_idx) = p(k,phi_idx)/p(phi_idx)
            p_k_given_phi = torch.clamp(p_k_phi[:, phi_idx] / denom, min=eps)
            H_clone_phi = -torch.sum(p_k_given_phi * torch.log(p_k_given_phi))
            
            # Normalize by ln(K) if requested
            if normalize and self.K > 1:
                H_clone_phi = H_clone_phi / math.log(self.K)

            clonotypic_entropy_dict[phi_idx] = H_clone_phi.item()

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
        Generate posterior samples for a given (patient, TCR, [optional timepoint]) combination
        under the partial-pooling model that uses a log-reparameterized 
        'unconstrained_gene_profile_log' in the guide.

        Args:
            patient_label: Label of the patient in adata.obs
            tcr_label: Label of the TCR in adata.obs
            timepoint_label: (Optional) Label of the timepoint in adata.obs. 
                            If not provided or self.T==0, defaults to 0.
            n_samples: Number of posterior samples to generate
            include_patient_effects: Whether to sample patient-specific Gamma offsets

        Returns:
            Dictionary with:
                - "phenotype_samples": shape (n_samples, self.n_phenotypes)
                - "gene_expression_samples": shape (n_samples, self.D)
                - "patient_effect_samples": shape (n_samples,) if include_patient_effects=True
        """
        import numpy as np
        import torch
        import pyro
        import pyro.distributions as dist

        # ------------------------------------------------------------------------
        # 1) Map user labels -> indices (patient, TCR, [timepoint])
        # ------------------------------------------------------------------------
        patient_idx = self.dataset.covariate_mapping.get(patient_label)
        if patient_idx is None:
            raise ValueError(f"Patient label '{patient_label}' not found in dataset.covariate_mapping.")

        tcr_idx = self.tcr_mapping.get(tcr_label)
        if tcr_idx is None:
            raise ValueError(f"TCR label '{tcr_label}' not found in tcr_mapping.")

        if self.T > 0 and timepoint_label is not None:
            time_idx = self.dataset.timepoint_mapping.get(timepoint_label)
            if time_idx is None:
                raise ValueError(f"Timepoint label '{timepoint_label}' not found in dataset.timepoint_mapping.")
        else:
            time_idx = 0  # default if no timepoint or T=0

        # Flatten clone/patient/time => single index
        kct_index = tcr_idx * (self.C * self.T) + patient_idx * self.T + time_idx

        # ------------------------------------------------------------------------
        # 2) Retrieve the relevant Pyro params from the param store
        # ------------------------------------------------------------------------
        # local clone->phenotype partial pooling
        local_conc_pt = pyro.param("local_clone_concentration_pct").detach().cpu()
        local_dirichlet_conc = local_conc_pt[kct_index]  # shape (n_phenotypes,)

        # gene profile log param => exponentiate & clamp => "gene_profile_concentration"
        unconstrained_log = pyro.param("unconstrained_gene_profile_log").detach().cpu()
        # shape: (n_phenotypes, D)

        # clamp in [min, max] to avoid extreme or NaN
        log_clamped = torch.clamp(unconstrained_log, min=-10.0, max=10.0)
        # exponentiate
        raw_concentration = torch.exp(log_clamped)
        # => shape (n_phenotypes, D)

        # We do NOT row-normalize here, because the model's guide uses:
        #   pyro.sample("clone_gene_profiles", dist.Dirichlet(gene_profile_concentration))
        # which expects a Dirichlet concentration (not a distribution).
        # So we pass `raw_concentration` to Dirichlet, same as the guide.

        # patient variance if requested
        if include_patient_effects:
            # shape & rate for that single patient
            patient_var_shape = pyro.param("patient_variance_shape")[patient_idx].detach().cpu()
            patient_var_rate  = pyro.param("patient_variance_rate")[patient_idx].detach().cpu()
        else:
            patient_var_shape = None
            patient_var_rate  = None

        # ------------------------------------------------------------------------
        # 3) Prepare output arrays
        # ------------------------------------------------------------------------
        phenotype_samples = np.zeros((n_samples, self.n_phenotypes), dtype=np.float32)
        gene_expression_samples = np.zeros((n_samples, self.D), dtype=np.float32)
        patient_effect_samples = None
        if include_patient_effects:
            patient_effect_samples = np.zeros(n_samples, dtype=np.float32)

        # ------------------------------------------------------------------------
        # 4) Sample in a loop
        # ------------------------------------------------------------------------
        # For each sample:
        #   1) phenotype_dist ~ Dirichlet(local_dirichlet_conc)
        #   2) gene_profile_draw ~ Dirichlet(raw_concentration) => shape (n_phenotypes, D)
        #   3) Weighted sum across phenotypes
        #   4) (Optional) multiply by patient effect ~ Gamma(...)
        #   5) Normalize
        # ------------------------------------------------------------------------
        for i in range(n_samples):
            # 4a) phenotype distribution
            phenotype_dist = dist.Dirichlet(local_dirichlet_conc).sample().numpy()
            phenotype_samples[i] = phenotype_dist

            # 4b) gene profile draw => shape (n_phenotypes, D)
            gene_profile_draw = dist.Dirichlet(raw_concentration).sample().numpy()

            # Weighted sum => shape (D,)
            mixed_profile = np.sum(gene_profile_draw * phenotype_dist[:, np.newaxis], axis=0)

            # 4c) optional patient effect
            if include_patient_effects:
                p_effect = dist.Gamma(patient_var_shape, patient_var_rate).sample().item()
                patient_effect_samples[i] = p_effect
                mixed_profile *= p_effect

            # 4d) normalize => final gene expression distribution
            denom = mixed_profile.sum() + 1e-12
            gene_expression_samples[i] = mixed_profile / denom

        # ------------------------------------------------------------------------
        # 5) Return the result dict
        # ------------------------------------------------------------------------
        result = {
            "phenotype_samples": phenotype_samples,
            "gene_expression_samples": gene_expression_samples
        }
        if include_patient_effects:
            result["patient_effect_samples"] = patient_effect_samples

        return result
