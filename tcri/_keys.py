"""Canonical AnnData key registry — the single source of truth for every
``uns`` / ``obsm`` / ``obs`` string tcri reads or writes.

Import as ``from tcri import _keys as K`` and use ``K.P_CT`` etc. **Never** write
a ``"tcri_*"`` string literal in a function signature or body** — a PR1
conformance test forbids it. Migrating a reader/writer means swapping the
literal for the constant here.

The two legacy shadow keys (``tcri_clone_key`` / ``tcri_phenotype_key``) and the
legacy ``X_tcri_phenotypes`` obsm slot are listed only so the removal step can
find and delete them; new code uses ``METADATA`` + ``X_PROBABILITIES``.
"""

# ── uns: metadata + learned priors ───────────────────────────────────────────
METADATA = "tcri_metadata"                 # {covariate_col, clone_col, phenotype_col, batch_col}
P_CT = "tcri_p_ct"                         # learned posterior-mean p_ct, shape (n_ct, P)
LOCAL_SCALE = "tcri_local_scale"           # Dirichlet total-concentration scale
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

# ── legacy — declared ONLY so the removal step can find + delete them ─────────
LEGACY_MANAGER = "tcri_manager"                    # non-picklable AnnDataManager stash → drop
LEGACY_CLONE_KEY = "tcri_clone_key"                # shadow of METADATA[CLONE_COL] → drop
LEGACY_PHENOTYPE_KEY = "tcri_phenotype_key"        # shadow of METADATA[PHENOTYPE_COL] → drop
LEGACY_X_PHENOTYPES = "X_tcri_phenotypes"          # old prob slot → X_PROBABILITIES
