"""AnnData key constants — every literal tcri writes or reads lives here.

Re-exported as ``tcri._keys`` for backwards compatibility; the canonical import is
``from tcri._state import keys as K``.
"""
# ── uns: metadata + learned priors ───────────────────────────────────────────
METADATA = "tcri_metadata"                 # {covariate_col, clone_col, phenotype_col, batch_col}
P_CT = "tcri_p_ct"                         # learned posterior-mean p_ct, shape (n_ct, P)
LOCAL_SCALE = "tcri_local_scale"           # Dirichlet total-concentration scale (legacy draw)
CONC_CT = "tcri_conc_ct"                   # guide concentration lambda'_m, shape (n_ct, P)
GATE_PROB = "tcri_gate_prob"               # NEW (Phase 4): classifier/prior gate, scalar or None
CLASSIFIER_TEMPERATURE = "tcri_classifier_temperature"  # NEW (Phase 4): classifier temperature

CT_TO_COV = "tcri_ct_to_cov"               # ct -> covariate index
CT_TO_C = "tcri_ct_to_c"                   # ct -> clonotype index
CT_ARRAY = "tcri_ct_array_for_cells"       # per-cell ct index
COV_ARRAY = "tcri_cov_array_for_cells"     # per-cell covariate index

COVARIATE_CATEGORIES = "tcri_covariate_categories"
CLONOTYPE_CATEGORIES = "tcri_clonotype_categories"
PHENOTYPE_CATEGORIES = "tcri_phenotype_categories"

# ── obsm ─────────────────────────────────────────────────────────────────────
X_TCRI = "X_tcri"                          # latent posterior mean z
X_LOGITS = "X_tcri_logits"                 # per-cell classifier logits
X_LOGPOSTERIOR = "X_tcri_logposterior"     # logits + log prior
X_PROBABILITIES = "X_tcri_probabilities"   # per-cell phenotype probabilities
X_UMAP = "X_umap"

# ── obs ──────────────────────────────────────────────────────────────────────
PHENOTYPE = "tcri_phenotype"               # hard phenotype label
CLONE_SIZE = "clone_size"
INDICES = "indices"                        # scvi registration glue (kept)

# ── metadata sub-keys (values inside uns[METADATA]) ──────────────────────────
COVARIATE_COL = "covariate_col"
CLONE_COL = "clone_col"
PHENOTYPE_COL = "phenotype_col"
BATCH_COL = "batch_col"

# ── legacy ───────────────────────────────────────────────────────────────────
# The shadow keys `tcri_clone_key` / `tcri_phenotype_key` and the old
# `X_tcri_phenotypes` obsm slot are GONE: `to_anndata` no longer writes them and
# nothing reads them (`pp.clone_size`, the last reader, now uses METADATA).
# `LEGACY_MANAGER` stays because it still does defensive work — `save_tcri_session`
# pops it so a stray non-picklable AnnDataManager can never be serialized.
LEGACY_MANAGER = "tcri_manager"                    # popped defensively before save


# ── tl result blobs (uns) ────────────────────────────────────────────────────
# One namespaced key per metric, written by @tl_result and read by tcri.get.*.
# Named `tcri_<tool>` so the key IS the tool name — the same scheme grafiti uses.
JOINT_DISTRIBUTION = "tcri_joint_distribution"
MUTUAL_INFORMATION = "tcri_mutual_information"
CLONOTYPIC_ENTROPY = "tcri_clonotypic_entropy"
PHENOTYPIC_ENTROPY = "tcri_phenotypic_entropy"
PHENOTYPIC_FLUX = "tcri_phenotypic_flux"

#: The layer recorded by setup_anndata. Previously a bare literal in two places.
LAYER = "tcri_layer"

#: Oracle payload written by the synthetic generators.
TRUTH = "tcri_truth"


class Config:
    """Field names INSIDE ``uns[METADATA]`` — the effective values resolved at setup.

    ``REPLICATE`` is the column a metric uses when ``groupby`` is left implicit, so a user
    registers it once at ``setup_anndata`` rather than retyping it at every call. It is
    deliberately separate from ``BATCH_COL``: scvi's ``batch_key`` conditions the encoder and
    decoder (one-hot into every hidden layer), which is a modelling decision, not a statement
    about what an independent replicate is. They coincide when batches are patients and diverge
    the moment they are sequencing runs.
    """

    COVARIATE_COL = "covariate_col"
    CLONE_COL = "clone_col"
    PHENOTYPE_COL = "phenotype_col"
    BATCH_COL = "batch_col"
    REPLICATE = "replicate"
    LAYER = "layer"


def colors(cat_key: str) -> str:
    """The scanpy ``uns`` colors key for a categorical ``obs`` column.

    ``colors("response") -> "response_colors"``. A formatter so ``"_colors"`` is never spelled
    inline, and so tcri routes INTO scanpy's convention rather than beside it.
    """
    return f"{cat_key}_colors"
