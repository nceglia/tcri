"""AnnData key constants — every literal tcri writes or reads lives here.

The canonical -- and now only -- import is
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
#: The paired (cov_from -> cov_to) forms. Only metrics with an ITEM axis get one -- see the
#: scope principle: a delta of a repertoire-level scalar is a subtraction, not a metric.
DELTA_CLONOTYPIC_ENTROPY = "tcri_delta_clonotypic_entropy"
DELTA_PHENOTYPIC_ENTROPY = "tcri_delta_phenotypic_entropy"
#: The in-silico perturbation (``tcri.perturb.gene_importance``): a query on the fitted model,
#: stored with the same ``{table, result, stats}`` shape the metrics use plus a ``shift`` slot.
GENE_IMPORTANCE = "tcri_gene_importance"

#: The layer recorded by setup_anndata. Previously a bare literal in two places.
LAYER = "tcri_layer"

#: Oracle payload written by the synthetic generators.
TRUTH = "tcri_truth"


# ── per-fit provenance (0.12) ─────────────────────────────────────────────────
# One AnnData can carry several fits: the main one and, beside it, any null or alternative
# model written with `to_anndata(fit=...)`. Each fit's arrays live under a prefixed key built by
# `fit_key`; these three carry what a fit IS, as opposed to what it learned.
#
# They are spelled `tcri_*` deliberately. `fit_key` inserts the fit name after a known prefix,
# and a bare base like "permutation" would fall through to its last branch and put an
# unprefixed key in `uns`.
PERMUTATION = "tcri_permutation"        # the integer permutation a null was built from
FIT_SETTINGS = "tcri_fit_settings"      # kind, strata, seed, train args, parent, joinability
BUFFERS = "tcri_buffers"                # the module's non-parameter state (BatchNorm, archetypes)

#: Inside `uns[METADATA]`: the fits this object carries, main first as ``None``.
FITS = "fits"


def fit_key(base: str, fit=None) -> str:
    """Insert a fit name after the namespace prefix of ``base``.

    ``fit=None`` returns ``base`` unchanged, which is what keeps every 0.11 object and every
    0.11 call byte-identical. Otherwise the fit name goes *after* the prefix rather than in
    front of it, so the keys still sort together and still read as tcri's::

        fit_key("tcri_p_ct", "null.phenotype")      -> "tcri_null.phenotype_p_ct"
        fit_key("X_tcri_logits", "null.phenotype")  -> "X_tcri_null.phenotype_logits"
        fit_key("X_tcri", "null.phenotype")         -> "X_tcri_null.phenotype"
        fit_key("my_key", "null.phenotype")         -> "my_key_null.phenotype"

    The last branch exists because this is also used on user-supplied ``key_added`` values,
    which need not start with ``tcri_``; without it the function would be partial over its own
    input.
    """
    if fit is None:
        return base
    fit = str(fit)
    if base == "X_tcri":
        return f"X_tcri_{fit}"
    for prefix in ("X_tcri_", "tcri_"):
        if base.startswith(prefix):
            return f"{prefix}{fit}_{base[len(prefix):]}"
    return f"{base}_{fit}"


def fits(adata) -> list:
    """The fit names this object carries, excluding the main fit.

    Written against the ROUND-TRIPPED object, not the in-memory one: h5ad stores a list of
    strings as a numpy array of them, so the idiomatic ``meta.get(FITS) or []`` raises "the
    truth value of an array with more than one element is ambiguous" on any object that has
    been through disk -- which is every object a reference is read from in practice.
    """
    meta = adata.uns.get(METADATA)
    names = None if meta is None else meta.get(FITS)
    return [] if names is None else [str(n) for n in names]


def resolve_fit(adata, fit):
    """Canonical fit name, accepting the bare kind.

    ``None`` stays ``None`` (the main fit). Otherwise the name is matched against the fits the
    object actually carries: literally first, then as ``null.{fit}``, so ``"phenotype"`` and
    ``"null.phenotype"`` are the same call — the way scanpy's ``basis="umap"`` finds
    ``X_umap``.

    Two differences from scanpy's version, both deliberate. Resolution is against the KNOWN
    LIST rather than against key existence, because a fit name is an infix in a dozen keys
    across ``uns``, ``obsm`` and ``obs`` rather than one key. And an ambiguous name raises
    instead of silently preferring the literal, which a two-branch fallback cannot do.
    """
    if fit is None:
        return None
    fit = str(fit)
    known = fits(adata)
    candidates = [c for c in (fit, f"null.{fit}") if c in known]
    if len(candidates) > 1:
        raise KeyError(
            f"fit={fit!r} is ambiguous: this object carries both {candidates[0]!r} and "
            f"{candidates[1]!r}. Name the one you mean."
        )
    if candidates:
        return candidates[0]
    raise KeyError(
        f"no fit named {fit!r} in this object; it carries {known or 'only the main fit'}. "
        f"Build one with tcri.null.all(model, adata), or pass fit=None for the main fit."
    )


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
