"""TCRI target API contract (frozen) — the machine-checkable public surface.

The conformance test (``tests/test_contract_conformance.py``) checks each
*implemented* function's live signature against its declaration here; declared-
but-absent functions are the refactor worklist. Prose spec:
``docs/contract/tcri_api_and_responsibilities.md``.

RULES this file encodes:
- Only the KEPT surface is declared. A symbol NOT in this file must NOT be public
  after the refactor (see the Removal Ledger in ``docs/contract/REFACTOR_AGENDA.md``).
- ``tl``/``pp``/``pl``/``diag``/``ut`` are container classes purely for namespacing
  (so the ``tl``/``pl`` twins can share a name); ``TCRIModel`` is the real model class.
- Decisions baked in: ``n_samples=250`` default (sampling; ``0`` = opt-in point
  estimate); ``weighted=False`` default (kept as a dial); ``use_logits`` (not
  ``posterior=``); ``normalize_mode="min"``; American spelling; keyword-only options.
"""
from __future__ import annotations
from typing import Any, Optional
from anndata import AnnData
import pandas as pd


# ── model (ml) ───────────────────────────────────────────────────────────────
class TCRIModel:
    @classmethod
    def setup_anndata(
        cls, adata: AnnData, *, layer: Optional[str] = ...,
        clonotype_key: str = ..., phenotype_key: str = ...,
        covariate_key: str = ..., batch_key: str = ..., **kwargs: Any,
    ) -> None: ...
    def train(
        self, max_epochs: int = ..., batch_size: int = ..., lr: float = ...,
        reconstruction_loss_scale: float = ..., n_steps_kl_warmup: int = ..., **kwargs: Any,
    ) -> None: ...
    def get_latent_representation(
        self, adata: Optional[AnnData] = ..., indices: Any = ...,
        batch_size: Optional[int] = ...,
    ) -> Any: ...
    def predict(
        self, adata: Optional[AnnData] = ..., *, batch_size: int = ..., eps: float = ...,
    ) -> pd.DataFrame: ...
    def get_p_ct(self) -> Any: ...
    def to_anndata(
        self, adata: Optional[AnnData] = ..., *, batch_size: int = ..., compute_umap: bool = ...,
    ) -> AnnData: ...


# ── preprocessing (pp) ───────────────────────────────────────────────────────
class pp:
    def group_singletons(
        adata: AnnData, *, clonotype_key: str = ..., groupby: str = ...,
        target_col: str = ..., min_clone_size: int = ...,
    ) -> None: ...
    def clone_size(adata: AnnData, *, key_added: str = ..., return_counts: bool = ...) -> Any: ...


# ── tools / metrics (tl) ─────────────────────────────────────────────────────
class tl:
    def joint_distribution(
        adata: AnnData, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        n_samples: int = ..., use_logits: bool = ..., weighted: bool = ...,
        clones: Any = ..., temperature: float = ..., random_state: Any = ...,
    ) -> pd.DataFrame: ...
    def clonotypic_entropy(
        adata_or_jd: Any, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ..., random_state: Any = ...,
    ) -> Any: ...
    def phenotypic_entropy(
        adata_or_jd: Any, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ..., random_state: Any = ...,
    ) -> Any: ...
    def mutual_information(
        adata_or_jd: Any, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ...,
        normalize_mode: str = ..., random_state: Any = ...,
    ) -> Any: ...
    def phenotypic_flux(
        adata: AnnData, *, cov_from: str, cov_to: str, groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., distance_metric: str = ..., random_state: Any = ...,
    ) -> Any: ...
    def compare_groups(
        df: pd.DataFrame, *, value: str, splitby: str, reference: Optional[str] = ...,
        paired: bool = ..., pair_on: Optional[str] = ..., hdi_prob: float = ...,
        alternative: str = ...,
    ) -> pd.DataFrame: ...


# ── plotting (pl) — twins mirror tl by name ──────────────────────────────────
class pl:
    def clonotypic_entropy(
        adata: AnnData, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ...,
        order: Any = ..., hue_order: Any = ..., palette: Any = ..., ax: Any = ...,
        figsize: Any = ..., save: Any = ..., show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def phenotypic_entropy(
        adata: AnnData, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ...,
        order: Any = ..., hue_order: Any = ..., palette: Any = ..., ax: Any = ...,
        figsize: Any = ..., save: Any = ..., show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def mutual_information(
        adata: AnnData, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ..., normalize_mode: str = ...,
        order: Any = ..., hue_order: Any = ..., palette: Any = ..., ax: Any = ...,
        figsize: Any = ..., save: Any = ..., show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def phenotypic_flux(
        adata: AnnData, *, order: Any, groupby: Optional[str] = ..., splitby: Optional[str] = ...,
        n_samples: int = ..., temperature: float = ..., clones: Any = ..., weighted: bool = ...,
        distance_metric: str = ..., palette: Any = ..., ax: Any = ..., figsize: Any = ...,
        save: Any = ..., show: Any = ..., return_axes: bool = ...,
    ) -> Any: ...
    def resolve_palette(adata: AnnData, columns: Any) -> Any: ...


# ── diagnostics (diag) — returns DataFrames ──────────────────────────────────
class diag:
    def joint_distribution_ppc(
        adata: AnnData, *, covariate: Optional[str] = ..., distance_metric: str = ...,
        temperature: float = ..., clones: Any = ..., random_state: Any = ...,
    ) -> pd.DataFrame: ...
    def phenotype_calibration(adata: AnnData, *, n_bins: int = ...) -> pd.DataFrame: ...
    def reconstruction_ppc(
        model: Any, adata: Optional[AnnData] = ..., *, n_sims: int = ..., random_state: Any = ...,
    ) -> pd.DataFrame: ...
    def permutation_null(
        adata: AnnData, *, metric: str = ..., covariate: Optional[str] = ...,
        groupby: Optional[str] = ..., n_perm: int = ..., random_state: Any = ...,
    ) -> pd.DataFrame: ...
    def loss(model: Any, *, log_scale: bool = ..., ax: Any = ..., save: Any = ...) -> Any: ...
    def archetypes(model: Any, *, ax: Any = ..., save: Any = ...) -> Any: ...


# ── utils / session (ut) ─────────────────────────────────────────────────────
class ut:
    def save_tcri_session(
        model: Any, adata: AnnData, out_dir: str, *, save_adata: bool = ..., compression: str = ...,
    ) -> dict: ...
    def load_tcri_session(
        run_dir: str, *, adata_path: Optional[str] = ..., map_location: Any = ..., layer: Optional[str] = ...,
    ) -> Any: ...
