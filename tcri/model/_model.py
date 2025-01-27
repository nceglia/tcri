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
        self._build_cell_groups()

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


    def _model(self):
        # print("\n[DEBUG] Entering _model() [flattened approach].")

        # 1) patient_variance => shape (C,)
        with pyro.plate("patient_global", self.C):
            patient_variance = pyro.sample(
                "patient_variance",
                dist.Gamma(
                    torch.full((1,), float(self.patient_variance_shape_val), device=self.device),
                    torch.full((1,), float(self.patient_variance_rate_val), device=self.device)
                )
            )
        # print(f"[DEBUG] patient_variance shape in code = {patient_variance.shape}")

        flat_size = self.K*self.C*self.T
        # print(f"[DEBUG] flat_size= {flat_size} [K*C*T= {self.K}*{self.C}*{self.T}]")

        # Build local_baseline_prior => shape (flat_size, n_phen)
        if self.T>0:
            repeated_baseline = self.clone_to_phenotype_prior.repeat_interleave(self.C*self.T, dim=0)
            # print(f"[DEBUG] repeated_baseline shape= {repeated_baseline.shape}")
            local_baseline_prior = repeated_baseline*self.clone_to_phenotype_prior_strength + 1.0
            # print(f"[DEBUG] local_baseline_prior shape= {local_baseline_prior.shape}")
        else:
            local_baseline_prior = None

        beta_val = torch.tensor(self.beta, device=self.device)
        offset_val = torch.tensor(self.local_concentration_offset, device=self.device)

        if self.T>0:
            with pyro.plate("clone_patient_time", flat_size) as idx:
                # print("[DEBUG] Inside plate clone_patient_time, size =", flat_size)
                final_local_baseline = pyro.sample(
                    "final_local_baseline",
                    dist.Dirichlet(local_baseline_prior)
                )
                # print("[DEBUG] final_local_baseline shape in code =", final_local_baseline.shape)

                scale_factor_kct = pyro.sample(
                    "scale_factor_kct",
                    dist.Gamma(
                        torch.full((self.n_phenotypes,), float(self.gamma_scale_shape), device=self.device),
                        torch.full((self.n_phenotypes,), float(self.gamma_scale_rate), device=self.device)
                    ).to_event(1)
                )
                # print("[DEBUG] scale_factor_kct shape in code =", scale_factor_kct.shape)

                final_conc = beta_val*final_local_baseline * scale_factor_kct + offset_val
                # print("[DEBUG] final_conc shape in code =", final_conc.shape)
                p_kct_final = pyro.sample(
                    "p_kct_final",
                    dist.Dirichlet(final_conc)
                )
                # print("[DEBUG] p_kct_final shape in code =", p_kct_final.shape)
        else:
            final_local_baseline=None
            scale_factor_kct=None
            p_kct_final=None

        # gene_profiles
        gene_prior = self.gene_profile_prior*self.clone_to_phenotype_prior_strength + self.local_concentration_offset
        with pyro.plate("phenotype", self.n_phenotypes):
            clone_gene_profiles = pyro.sample(
                "clone_gene_profiles",
                dist.Dirichlet(gene_prior)
            )
        # print("[DEBUG] clone_gene_profiles shape in code =", clone_gene_profiles.shape)

        # cell-level
        for (c_val,k_val,t_val), cell_idxs in self.cell_groups.items():
            if len(cell_idxs)==0:
                continue
            kct_index = k_val*(self.C*self.T) + c_val*self.T + t_val
            # Plate for the group
            with pyro.plate(f"cell_plate_{c_val}_{k_val}_{t_val}", len(cell_idxs)):
                gene_mat = self.dataset.matrix[cell_idxs].to(self.device)  # shape => (n_cells, 5)
                if self.T>0:
                    final_phen_dist = p_kct_final[kct_index]
                else:
                    final_phen_dist = torch.ones(self.n_phenotypes)/float(self.n_phenotypes)

                # incorporate patient_variance => shape => (C,)
                p_effect = patient_variance[c_val]
                mixed_profile = torch.einsum("p,pd->d", final_phen_dist, clone_gene_profiles)
                adjusted_probs = mixed_profile * p_effect
                expanded = adjusted_probs.unsqueeze(0).expand(len(cell_idxs), -1)
                pyro.sample(
                    f"obs_{c_val}_{k_val}_{t_val}",
                    dist.Dirichlet(expanded),
                    obs=gene_mat
                )

    def _guide(self):
        # print("\n[DEBUG] Entering _guide() [flattened approach].")

        # 1) patient_variance => shape (C,)
        patient_variance_shape = pyro.param(
            "patient_variance_shape",
            torch.full((self.C,), float(self.patient_variance_shape_val), device=self.device),
            constraint=constraints.greater_than(0.1)
        )
        patient_variance_rate = pyro.param(
            "patient_variance_rate",
            torch.full((self.C,), float(self.patient_variance_rate_val), device=self.device),
            constraint=constraints.greater_than(0.1)
        )
        with pyro.plate("patient_global", self.C):
            pyro.sample(
                "patient_variance",
                dist.Gamma(patient_variance_shape, patient_variance_rate)
            )

        flat_size = self.K*self.C*self.T if self.T>0 else 0
        if self.T>0:
            repeated_baseline = self.clone_to_phenotype_prior.repeat_interleave(self.C*self.T, dim=0)
            baseline_conc_init = repeated_baseline*self.clone_to_phenotype_prior_strength + 1.0

            with pyro.plate("clone_patient_time", flat_size):
                local_baseline_param = pyro.param(
                    "local_baseline_conc",
                    baseline_conc_init,
                    constraint=constraints.greater_than(1e-6)
                )
                pyro.sample(
                    "final_local_baseline",
                    dist.Dirichlet(local_baseline_param)
                )

                shape_init = torch.full((self.n_phenotypes,), float(self.gamma_scale_shape), device=self.device)
                rate_init  = torch.full((self.n_phenotypes,), float(self.gamma_scale_rate),  device=self.device)

                scale_shape_param = pyro.param(
                    "scale_factor_shape",
                    shape_init,
                    constraint=constraints.greater_than(0.0)
                )
                scale_rate_param = pyro.param(
                    "scale_factor_rate",
                    rate_init,
                    constraint=constraints.greater_than(0.0)
                )
                pyro.sample(
                    "scale_factor_kct",
                    dist.Gamma(scale_shape_param, scale_rate_param).to_event(1)
                )

                p_kct_conc_init = torch.full(
                    (flat_size, self.n_phenotypes),
                    1.5, device=self.device
                )
                p_kct_conc_param = pyro.param(
                    "p_kct_final_conc",
                    p_kct_conc_init,
                    constraint=constraints.greater_than(0.0)
                )
                pyro.sample(
                    "p_kct_final",
                    dist.Dirichlet(p_kct_conc_param)
                )

        # gene profiles
        gene_profile_concentration_init = (
            self.gene_profile_prior*self.clone_to_phenotype_prior_strength
            + self.local_concentration_offset
        )
        gene_profile_concentration_init = torch.clamp(gene_profile_concentration_init, min=1e-4)
        unconstrained_profile_init = torch.log(gene_profile_concentration_init)

        unconstrained_profile = pyro.param(
            "unconstrained_gene_profile_log",
            unconstrained_profile_init,
            constraint=constraints.real
        )
        with pyro.plate("phenotype", self.n_phenotypes):
            gene_profile_conc = torch.exp(torch.clamp(unconstrained_profile, -10, 10))
            pyro.sample(
                "clone_gene_profiles",
                dist.Dirichlet(gene_profile_conc)
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
                loss = self.svi.step()
                
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
        Compute the distance between the mean phenotype distributions at two timepoints
        for (clone_idx, patient_idx).
        
        Args:
            clone_idx: which clone (0 <= clone_idx < K)
            patient_idx: which patient (0 <= patient_idx < C)
            time1: which time index in [0..T-1]
            time2: which time index in [0..T-1]
            metric: "l1" (sum of absolute differences) or "dkl" (Kullback-Leibler)
            temperature: optional post-hoc scaling, p^(1/temperature).

        Returns:
            A float representing the flux distance between the two phenotype distributions.
        """
        import torch
        import numpy as np
        import math

        eps = 1e-12
        p_store = pyro.get_param_store()

        # Retrieve p_kct_final_conc => shape (K*C*T, n_phenotypes)
        p_kct_final_conc = p_store["p_kct_final_conc"].detach().cpu()

        def get_mean_distribution(k_idx, c_idx, t_idx):
            flat_idx = k_idx * (self.C*self.T) + c_idx*self.T + t_idx
            conc = p_kct_final_conc[flat_idx]  # shape (n_phenotypes,)
            conc_sum = torch.sum(conc).clamp_min(eps)
            p_mean = conc / conc_sum
            return p_mean.numpy()

        # 1) Get the two mean distributions
        p1 = get_mean_distribution(clone_idx, patient_idx, time1)
        p2 = get_mean_distribution(clone_idx, patient_idx, time2)

        # 2) Temperature transform
        if temperature != 1.0:
            p1 = p1**(1.0/temperature)
            p2 = p2**(1.0/temperature)

        # 3) Normalize
        p1 = p1 / np.clip(p1.sum(), eps, None)
        p2 = p2 / np.clip(p2.sum(), eps, None)

        # 4) Distance
        if metric.lower() == "l1":
            flux_val = float(np.sum(np.abs(p1 - p2)))
        elif metric.lower() == "dkl":
            # KL divergence: sum( p1 * log(p1 / p2) )
            mask = (p1 > eps)
            flux_val = float(np.sum(p1[mask] * np.log(p1[mask] / p2[mask])))
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
        A Bayesian approach to computing I(Clone; Phenotype) for a given (patient_idx, time_idx),
        by sampling from the posterior for each clone, then combining into a joint distribution.

        Steps:
        1) Convert (patient_idx, time_idx) -> corresponding patient_label, timepoint_label.
        2) For each clone k in [0..K-1], call generate_posterior_samples(...)
            to get (n_samples, n_phenotypes).
        3) Average these samples => p(phi|k), optionally apply a temperature transform.
        4) Possibly weight each clone by its frequency => p(k).
        5) Construct p(k, phi) = p(k)* p(phi|k), sum => p(phi), then compute
            I(X;Y) in bits (base-2 log).

        Args:
            patient_idx: integer index in [0..C-1] for the patient.
            time_idx: integer index in [0..T-1] for the timepoint (0 if T=0).
            n_samples: how many posterior draws per clone.
            weight_by_clone_freq: whether to gather real clone frequencies from
                                adata.obs["clone_size"] for that (patient, time).
            temperature: optional exponent for post-hoc sharpening (temperature < 1)
                        or flattening (temperature > 1).

        Returns:
            mutual_info (float): the estimated mutual information in bits.
        """
        import numpy as np
        import math

        eps = 1e-12
        K = self.K
        P = self.n_phenotypes

        # ----------- 0) Map integer indices -> string labels -------------
        # patient_idx -> patient_label
        rev_patient_map = {v: k for k, v in self.dataset.covariate_mapping.items()}
        patient_label = rev_patient_map.get(patient_idx, None)
        if patient_label is None:
            raise ValueError(f"Unknown patient_idx={patient_idx} in dataset.covariate_mapping.")

        # time_idx -> time_label
        if self.T > 0 and time_idx >= 0:
            rev_time_map = {v: k for k, v in self.dataset.timepoint_mapping.items()}
            time_label = rev_time_map.get(time_idx, None)
            if time_label is None:
                raise ValueError(f"Unknown time_idx={time_idx} in dataset.timepoint_mapping.")
        else:
            # If T=0 or time_idx not provided properly, we treat time_label as None
            time_label = None

        # ----------- 1) Possibly gather real clone frequencies p(k) -------------
        if weight_by_clone_freq and ("clone_size" in self.adata.obs):
            freq_for_clone = np.zeros(K, dtype=np.float32)
            clone_sizes = self.adata.obs["clone_size"].values
            # loop over the entire dataset
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
            # default uniform weighting
            freq_for_clone = np.ones(K, dtype=np.float32) / K

        # ----------- 2) We'll accumulate joint distribution p(k, phi) -------------
        p_k_phi = np.zeros((K, P), dtype=np.float64)

        # A helper to do the temperature transform
        def temperature_transform(vec: np.ndarray, tau: float) -> np.ndarray:
            vec = np.clip(vec, eps, None)
            if tau == 1.0:
                s = vec.sum()
                return vec / (s if s > eps else 1.0)
            power = vec**(1.0 / tau)
            denom = power.sum()
            if denom < eps:
                return np.ones_like(vec) / len(vec)
            return power / denom

        # ----------- 3) For each clone => posterior sampling => average => build p(k,phi) -------------
        for k_idx in range(K):
            # Map clone index -> TCR label
            tcr_label = self.clone_mapping.get(k_idx, None)
            if tcr_label is None:
                continue

            # Draw from the posterior for that clone, patient_label, time_label
            posterior_result = self.generate_posterior_samples(
                patient_label=patient_label,
                tcr_label=tcr_label,
                timepoint_label=time_label,
                n_samples=n_samples,
                include_patient_effects=True
            )

            # shape => (n_samples, P)
            pheno_samples = posterior_result["phenotype_samples"]

            # average across samples => p(phi|k)
            p_phi_given_k = pheno_samples.mean(axis=0)

            # temperature transform
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)

            # final distribution clamp
            sum_phi = p_phi_given_k.sum()
            if sum_phi < eps:
                p_phi_given_k[:] = 1.0 / P
            else:
                p_phi_given_k /= sum_phi

            # multiply by p(k)
            p_k = freq_for_clone[k_idx]
            p_k_phi[k_idx, :] = p_k * p_phi_given_k

        # ----------- 4) Normalize p(k,phi) => sum_{k,phi} = 1 -------------
        total_mass = p_k_phi.sum()
        if total_mass < eps:
            p_k_phi[:] = 1.0 / (K * P)
        else:
            p_k_phi /= total_mass

        # p(k) = sum_{phi}, p(phi) = sum_{k}
        px = p_k_phi.sum(axis=1)  # shape (K,)
        py = p_k_phi.sum(axis=0)  # shape (P,)

        # ----------- 5) Compute mutual information I(X;Y) in bits -------------
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
        Deterministic (MAP-like) version of I(Clone; Phenotype) for a given (patient_idx, time_idx),
        using only the mean distribution from the guide's parameter 'p_kct_final_conc' for each clone.

        Steps:
        1) Convert (patient_idx, time_idx) -> corresponding patient_label, timepoint_label.
        2) For each clone k, read p_kct_final_conc -> normalize => p(phi|k).
        3) Optionally apply a temperature transform, and weight by clone frequency => p(k).
        4) Construct p(k, phi), sum => p(phi), compute I(X;Y) in bits (base-2 log).

        Args:
            patient_idx: integer index of the patient in [0..C-1].
            time_idx: integer index of the timepoint in [0..T-1], or 0 if T=0.
            weight_by_clone_freq: if True, gather real clone frequencies from adata.obs["clone_size"].
            temperature: exponent for post-hoc sharpening (temp<1) or flattening (temp>1).

        Returns:
            mutual_info: a float, the mutual information in bits (base-2).
        """
        import numpy as np
        import math
        import torch
        from pyro import get_param_store

        eps = 1e-12
        K = self.K
        P = self.n_phenotypes

        # ----------- 0) Map integer indices -> string labels -------------
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

        # ----------- 1) Possibly gather real clone frequencies p(k) -------------
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

        # ----------- 2) We'll read p_kct_final_conc and accumulate p(k, phi) -------------
        p_store = get_param_store()
        # shape => (K*C*T, n_phenotypes)
        p_kct_final_conc = p_store["p_kct_final_conc"].detach().cpu().numpy()

        p_k_phi = np.zeros((K, P), dtype=np.float64)

        # helper function for temperature
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

        # flatten index => kct_index = k*(C*T) + c*T + t
        for k_idx in range(K):
            kct_index = k_idx*(self.C*self.T) + patient_idx*self.T + (time_idx if self.T > 0 else 0)
            conc = p_kct_final_conc[kct_index]  # shape (P,)
            sum_conc = conc.sum()
            if sum_conc < eps:
                # fallback
                p_phi_given_k = np.ones(P, dtype=np.float32) / P
            else:
                p_phi_given_k = conc / sum_conc

            # apply temperature
            p_phi_given_k = temperature_transform(p_phi_given_k, temperature)
            # ensure it sums to 1
            p_sum = p_phi_given_k.sum()
            if p_sum < eps:
                p_phi_given_k[:] = 1.0 / P
            else:
                p_phi_given_k /= p_sum

            # multiply by p(k)
            p_k = freq_for_clone[k_idx]
            p_k_phi[k_idx, :] = p_k * p_phi_given_k

        # ----------- 3) Normalize p(k, phi) => sum_{k,phi} = 1 -------------
        total_mass = p_k_phi.sum()
        if total_mass < eps:
            p_k_phi[:] = 1.0 / (K*P)
        else:
            p_k_phi /= total_mass

        # p(k)= sum_phi p(k, phi), p(phi)= sum_k p(k, phi)
        px = p_k_phi.sum(axis=1)
        py = p_k_phi.sum(axis=0)

        # ----------- 4) Compute mutual information => sum p(k,phi) * log2 [ p(k,phi)/(p(k)*p(phi)) ] -------------
        mutual_info = 0.0
        for k_idx in range(K):
            for phi_idx in range(P):
                val = p_k_phi[k_idx, phi_idx]
                if val > eps:
                    denom = px[k_idx]* py[phi_idx]
                    if denom > eps:
                        mutual_info += val * math.log2(val / denom)

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
        
        Returns:
            dict with keys:
            - "phenotype_samples": (n_samples, n_phenotypes)
            - "gene_expression_samples": (n_samples, D)
            - "patient_effect_samples": (n_samples,) if include_patient_effects=True
            - "phenotype_labels": list of length n_phenotypes
        """
        import numpy as np
        import torch
        import pyro
        import pyro.distributions as dist

        # 1) Convert string labels -> indices
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
            # If no timepoints, or None is passed, default to 0
            t_idx = 0

        # 2) Flatten (k, c, t) -> single index in [0..(K*C*T - 1)]
        kct_index = k_idx * (self.C * self.T) + patient_idx * self.T + t_idx

        # Retrieve param store references
        p_store = pyro.get_param_store()

        # --- Gamma parameters for patient variance ---
        patient_variance_shape_param = p_store["patient_variance_shape"].detach().cpu()  # shape (C,)
        patient_variance_rate_param  = p_store["patient_variance_rate"].detach().cpu()   # shape (C,)

        # --- Dirichlet parameters for p_kct_final ---
        p_kct_final_conc = p_store["p_kct_final_conc"].detach().cpu()  # shape (K*C*T, n_phenotypes)
        if kct_index >= p_kct_final_conc.shape[0]:
            raise ValueError(f"Invalid kct_index={kct_index}. Check K*C*T size.")
        phen_conc_kct = p_kct_final_conc[kct_index]  # shape (n_phenotypes,)

        # --- Dirichlet parameters for gene profiles ---
        unconstrained_log = p_store["unconstrained_gene_profile_log"].detach().cpu()  # shape (n_phenotypes, D)
        log_clamped = torch.clamp(unconstrained_log, -10, 10)
        gene_profile_conc = torch.exp(log_clamped)  # shape (n_phenotypes, D)

        # We'll accumulate samples in arrays
        phenotype_samples = np.zeros((n_samples, self.n_phenotypes), dtype=np.float32)
        gene_expression_samples = np.zeros((n_samples, self.D), dtype=np.float32)
        patient_effect_samples = None
        if include_patient_effects:
            patient_effect_samples = np.zeros(n_samples, dtype=np.float32)

        # For the requested patient c
        pat_shape = patient_variance_shape_param[patient_idx].item()
        pat_rate  = patient_variance_rate_param[patient_idx].item()

        # 3) Perform sampling
        gamma_shape = torch.tensor(pat_shape, dtype=torch.float32)
        gamma_rate  = torch.tensor(pat_rate,  dtype=torch.float32)

        # Pre-construct distribution objects that do not change in the loop
        # for speed (only shape/rate or conc changes when we vary over kct, here it’s constant)
        dirichlet_kct = dist.Dirichlet(phen_conc_kct)
        dirichlet_gene = dist.Dirichlet(gene_profile_conc)

        for i in range(n_samples):
            # (A) Sample p(phenotypes|k,c,t)
            p_kct_sample = dirichlet_kct.sample()  # shape (n_phenotypes,)
            phenotype_samples[i, :] = p_kct_sample.numpy()

            # (B) Sample gene profiles: shape (n_phenotypes, D)
            #    Each phenotype row is a Dirichlet. We do a single draw from the product-of-Dirichlets
            #    in the mean-field guide? Actually in a mean-field setting, each phenotype is independent.
            #    But we only have one param for the entire n_phenotypes. So .sample() returns (n_phenotypes, D).
            gene_profile_draw = dirichlet_gene.sample().numpy()  # shape (n_phenotypes, D)

            # Weighted sum => mixture over phenotypes => shape (D,)
            mixed_profile = (p_kct_sample.unsqueeze(-1) * torch.tensor(gene_profile_draw)).sum(dim=0).numpy()

            # (C) Sample patient effect if requested
            if include_patient_effects:
                pat_eff = dist.Gamma(gamma_shape, gamma_rate).sample().item()
                patient_effect_samples[i] = pat_eff
                mixed_profile = mixed_profile * pat_eff

            # (D) Normalize the gene distribution
            total = np.sum(mixed_profile) + 1e-12
            gene_expression_samples[i, :] = mixed_profile / total

        # 4) Package results
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
