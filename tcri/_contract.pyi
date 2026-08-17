"""TCRI target API contract (frozen) — the machine-checkable public surface.

The conformance test (``tests/test_contract_conformance.py``) checks each
*implemented* function's live signature against its declaration here; declared-
but-absent functions are the refactor worklist. Prose spec:
``docs/contract/tcri_api_and_responsibilities.md``.

This freezes the public *interface*. Its sibling — ``tcri/model/_model_contract.py``
(prose: ``docs/contract/MODEL_CONTRACT.md``) — freezes the model's *mathematics*
(Supplementary Note 1).

RULES this file encodes:
- **The scope principle** (api doc, "The scope principle"): a comparison belongs in the API
  when producing it requires applying a metric at a level this surface does not already
  expose; when it is arithmetic on values already computed, it belongs to the user. So a
  metric with an item axis (clone / phenotype) gets cross-covariate functions, and one
  without gets none — a "delta" of a repertoire-level scalar is a subtraction, not a metric.
  Corollary: always know what a metric reduces to. If a proposed function cannot name its
  unit of reduction, that is the signal to re-examine whether it belongs here.
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
        covariate_key: str = ..., batch_key: str = ...,
        replicate: Optional[str] = ..., **kwargs: Any,
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
        adata: AnnData, *, covariate: Optional[str] = ...,
        n_samples: int = ..., use_logits: bool = ..., weighted: bool = ...,
        clones: Any = ..., temperature: float = ..., random_state: Any = ...,
        device: Any = ..., key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...
    def clonotypic_entropy(
        adata: AnnData, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ...,
        n_clones_ref: Any = ..., random_state: Any = ..., device: Any = ...,
            key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...
    def phenotypic_entropy(
        adata: AnnData, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ..., random_state: Any = ...,
        device: Any = ...,
            key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...
    def mutual_information(
        adata: AnnData, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ...,
        normalize_mode: str = ..., random_state: Any = ..., device: Any = ...,
            key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...
    def phenotypic_flux(
        adata: AnnData, *, cov_from: str, cov_to: str, groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., distance_metric: str = ..., random_state: Any = ...,
        device: Any = ...,
            key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...
    # compare_groups was here. It is now internal (tools/_compare.py): `splitby` produces the
    # contrast as part of the metric, in the `stats` slot, with the replicate unit already
    # resolved. A separate public step meant the caller picked the replicate unit themselves.


# ── plotting (pl) — twins mirror tl by name ──────────────────────────────────
#
# Every pl signature is now (adata, key=, display args). The metric arguments are GONE: a
# twin reads the result tl stored, so the covariate, groupby, splitby, n_samples and distance
# it renders are the ones tl actually used. `key=` selects a non-default `key_added`.
class pl:
    def clonotypic_entropy(
        adata: AnnData, *, key: Optional[str] = ..., order: Any = ..., hue_order: Any = ...,
        palette: Any = ..., ax: Any = ..., figsize: Any = ..., save: Any = ...,
        show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def phenotypic_entropy(
        adata: AnnData, *, key: Optional[str] = ..., order: Any = ..., hue_order: Any = ...,
        palette: Any = ..., ax: Any = ..., figsize: Any = ..., save: Any = ...,
        show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def mutual_information(
        adata: AnnData, *, key: Optional[str] = ..., order: Any = ..., hue_order: Any = ...,
        palette: Any = ..., ax: Any = ..., figsize: Any = ..., save: Any = ...,
        show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def phenotypic_flux(
        adata: AnnData, *, key: Optional[str] = ..., order: Any = ..., hue_order: Any = ...,
        palette: Any = ..., ax: Any = ..., figsize: Any = ..., save: Any = ...,
        show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def resolve_colors(
        adata: AnnData, cat_key: str, categories: Any = ..., *, palette: Any = ...,
        persist: bool = ...,
    ) -> dict: ...


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
        groupby: Optional[str] = ..., normalize_mode: str = ..., n_perm: int = ...,
        random_state: Any = ...,
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
    # The AUROC pair. Implemented and tested since PR1 but reachable only by importing the
    # private `tcri._stats`, so nothing in the package or the notebooks ever called them.
    # They are complementary and should be used together: a permutation p-value against
    # shuffled labels, and a bootstrap CI on the AUC itself.
    def auc_and_label_permutation(
        scores: Any, labels: Any, pos_label: Any = ..., n_perm: int = ..., seed: int = ...,
        max_exact: int = ...,
    ) -> tuple: ...
    def bootstrap_auc(
        scores: Any, labels: Any, pos_label: Any = ..., n_boot: int = ..., seed: int = ...,
    ) -> tuple: ...


# ── synthetic data (datasets) ────────────────────────────────────────────────
# Declared in the contract because it is already a shipped public namespace. It was
# outside the contract entirely until the conformance test moved from a hand-maintained
# allowlist to set equality against the live surface, which is what surfaced it.
class datasets:
    def simulate_tcri(
        *, n_clones: int = ..., n_phenotypes: int = ..., n_genes: int = ...,
        n_cells: int = ..., n_covariates: int = ..., n_factors: int = ...,
        omega_concentration: float = ..., pi_concentration: float = ...,
        fuzziness: float = ..., label_error_rate: float = ..., seed: int = ...,
    ) -> AnnData: ...
    def simulate_from_fit_params(
        params: Any, *, n_cells: int = ..., temperature: float = ...,
        fuzziness: float = ..., label_error_rate: float = ..., seed: int = ...,
    ) -> AnnData: ...
    def temperature_scale(P: Any, T: Any, eps: float = ...) -> Any: ...
    def mi_from_joint_oracle(joint: Any) -> dict: ...
