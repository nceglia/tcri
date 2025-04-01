
def _sample_zinb_counts(
    base_counts: np.ndarray,
    dispersion: float,
    dropout_prob: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample integer counts from a Zero-Inflated Negative Binomial distribution,
    given a base count vector for a single cell.

    Parameters
    ----------
    base_counts : np.ndarray
        Nonnegative integer vector (shape = [n_genes,]) representing
        the original cell's raw counts.
    dispersion : float
        Global NB dispersion parameter (> 0).
        Lower => less overdispersion, Higher => more overdispersion.
    dropout_prob : float
        Probability of dropout (zero inflation).
    rng : np.random.Generator
        A numpy random generator for reproducibility.

    Returns
    -------
    sampled_counts : np.ndarray, shape = [n_genes,]
        Integer counts sampled from ZINB.
    """
    # For each gene g, we interpret base_counts[g] as the "mean" mu_g.
    # Then NB(dispersion, mu_g) with "mu+disp" parameterization can be:
    #   total_count = dispersion
    #   p = dispersion / (dispersion + mu_g)
    # or we do the "mean-disp" parameterization:
    #   mean = mu_g, var = mean + mean^2 / dispersion
    # For simplicity, let's do "r, p" param:
    #   r = dispersion
    #   p = r / (r + mu_g)
    # Then the distribution is:
    #   NB(X; r, p)
    # with mean = r*(1-p)/p = mu_g, so:
    #   p = r/(r + mu_g)

    mu = base_counts.astype(float)
    n_genes = mu.shape[0]

    # Precompute r and p for each gene
    r = dispersion  # The same for all genes (global), you could also vary it by gene
    # p_g = r / (r + mu_g)
    # If mu_g = 0 for some genes, p_g = 1 => mostly zeros, which might be okay
    # but let's ensure we do it carefully.
    p = np.zeros(n_genes, dtype=float)
    mask_nonzero = (mu > 0)
    p[mask_nonzero] = r / (r + mu[mask_nonzero])
    p[~mask_nonzero] = 0.999999  # if mu=0, high p => mostly 0 from NB

    # Step 1: sample from NB
    # NB(r, p) => we can do an efficient approach gene-by-gene
    # or a loop. We'll do a loop for clarity:

    nb_counts = np.zeros(n_genes, dtype=int)
    for g in range(n_genes):
        # number of successes = r,
        # prob of success = p[g]
        # np.random.negative_binomial draws: # of failures before r successes
        # with success prob p[g]. The mean is r*(1-p[g])/p[g]
        # which matches mu[g].
        nb_counts[g] = rng.negative_binomial(r, p[g])

    # Step 2: apply zero inflation
    # with probability dropout_prob, set gene count = 0
    # We do it gene-wise.
    zeros_mask = rng.random(n_genes) < dropout_prob
    nb_counts[zeros_mask] = 0

    return nb_counts


def create_synthetic_data_limited_phen_subset(
    adata: AnnData,
    patient_col: str = "patient",
    covariate_col: str = "timepoint",
    phenotype_col: str = "phenotype",
    # Optional: single-patient subset
    patient_to_keep: Optional[str] = None,

    # Noise scale per phenotype
    phenotype_noise_scale: Union[float, Dict[str, float]] = 0.2,

    # ZINB dispersion / dropout
    zinb_dispersion: float = 10.0,
    zinb_dropout_prob: float = 0.2,

    # Clonal structure
    n_stable_clones: int = 5,
    n_divergent_clones: int = 5,
    # distinctness_factor now means "bigger => more difference across covariates"
    # We'll do alpha_vec = base_dist / distinctness_factor + eps
    distinctness_factor: float = 2.0,
    min_clone_size: int = 50,
    max_clone_size: int = 200,
    ablation_prob: float = 0.3,
    ablation_fraction: float = 0.5,

    # Phenotype subset logic:
    # e.g. "at most 2 phenotypes" or a distribution for how many phen.
    # We'll illustrate a simple approach:
    #  - p(1 phen) = 0.5
    #  - p(2 phen) = 0.3
    #  - p(3 phen) = 0.2  (only if # phen >=3)
    subset_size_probs = (0.5, 0.3, 0.2),

    mislabel_fraction: float = 0.1,

    # Optional scRNA postprocessing
    do_postprocessing: bool = True,
    n_hvg: int = 2000,
    n_pcs: int = 20,

    random_seed: int = 42,
) -> AnnData:
    """
    Create a synthetic AnnData:
      1) Single patient (optional).
      2) Each phenotype -> compute mean expression vector.
      3) Each clone is assigned a subset of phenotypes. We do *not* do a full Dirichlet
         across all phen. Non-chosen phen are truly zero. 
      4) "Stable" clones keep that subset distribution across covariates;
         "Divergent" clones re-sample a distribution for each cov with logic:
             alpha_vec = base_dist / distinctness_factor
         => bigger distinctness_factor => smaller alpha => more variation
      5) Partial ablation in one cov for each ablated clone
      6) ZINB generation for each cell from phen-specific mean + gene-level noise
      7) optional scRNA postprocessing

    The key changes are:
      - We invert distinctness_factor => bigger => more cross-covariate difference
      - We pick a subset of phenotypes for each clone => truly zero for unchosen phen.
    """
    rng = np.random.default_rng(random_seed)

    # -------------------------------------------------------------------------
    # 1) Single-patient subset
    # -------------------------------------------------------------------------
    if patient_col not in adata.obs:
        raise ValueError(f"Column '{patient_col}' not found in `adata.obs`.")
    patient_counts = adata.obs[patient_col].value_counts()
    if len(patient_counts) == 0:
        raise ValueError("No patients found.")
    if patient_to_keep is None:
        best_pt = patient_counts.index[0]
    else:
        best_pt = patient_to_keep
        if best_pt not in patient_counts.index:
            raise ValueError(f"Requested patient '{best_pt}' not in data.")
    mask_best = (adata.obs[patient_col] == best_pt)
    adata_sub = adata[mask_best].copy()

    # -------------------------------------------------------------------------
    # 2) Phenotype means
    # -------------------------------------------------------------------------
    if phenotype_col not in adata_sub.obs:
        raise ValueError(f"Column '{phenotype_col}' not in `adata_sub.obs`.")
    ph_array = adata_sub.obs[phenotype_col].astype(str).values
    unique_phens = np.unique(ph_array)

    if sp.issparse(adata_sub.X):
        expr = adata_sub.X.A
    else:
        expr = adata_sub.X
    if not np.issubdtype(expr.dtype, np.integer):
        warnings.warn("adata_sub.X not integer => might not be raw counts?")

    # compute means
    phen2mean = {}
    for ph in unique_phens:
        idx_ = np.where(ph_array==ph)[0]
        if len(idx_)>0:
            phen2mean[ph] = expr[idx_, :].mean(axis=0)
        else:
            phen2mean[ph] = np.zeros(expr.shape[1], dtype=float)

    # parse phenotype_noise_scale
    if isinstance(phenotype_noise_scale, dict):
        for ph in unique_phens:
            if ph not in phenotype_noise_scale:
                phenotype_noise_scale[ph] = 0.0
    else:
        global_scale = float(phenotype_noise_scale)
        phenotype_noise_scale = {ph: global_scale for ph in unique_phens}

    # covariates
    if covariate_col not in adata_sub.obs:
        adata_sub.obs[covariate_col] = "cov0"
        unique_covs = ["cov0"]
    else:
        unique_covs = adata_sub.obs[covariate_col].unique().tolist()

    # -------------------------------------------------------------------------
    # 3) Clones: stable vs. divergent
    #    => pick subset of phen for each clone => partial Dirichlet only there
    # -------------------------------------------------------------------------
    total_clones = n_stable_clones + n_divergent_clones

    def sample_subset_of_phen(all_phens, subset_probs=(0.5, 0.3, 0.2)):
        """
        We interpret subset_probs as probabilities for picking subset_size=1,2,3,... 
        up to the min(#phen, len(subset_probs)).

        Then we pick that many phen from all_phens => Dirichlet among them => fill base_dist
        => other phen=0
        """
        n_ph = len(all_phens)
        max_size = min(n_ph, len(subset_probs))
        # e.g. if n_ph=6, subset_probs=(0.5,0.3,0.2) => up to 3 phen
        # norm them
        arr_ = np.array(subset_probs[:max_size], dtype=float)
        arr_ /= arr_.sum()
        chosen_size = rng.choice(range(1, max_size+1), p=arr_)
        # pick that many phen
        chosen_phen = rng.choice(all_phens, size=chosen_size, replace=False)
        # Dirichlet among them
        alpha_ = rng.gamma(1.0, 1.0, size=chosen_size)
        alpha_ /= alpha_.sum()
        base_dist = np.zeros(n_ph, dtype=float)
        # fill chosen
        for i, cph in enumerate(chosen_phen):
            idx_ph = np.where(all_phens==cph)[0][0]
            base_dist[idx_ph] = alpha_[i]
        return base_dist

    def generate_dirichlet_divergent(base_dist):
        """
        We interpret distinctness_factor s.t. bigger => more difference from base_dist.
        => alpha_vec = base_dist / distinctness_factor
        => if distinctness_factor=1 => alpha=base_dist => same scale
        => if distinctness_factor=2 => alpha=base_dist/2 => smaller => more variation
        => if distinctness_factor> base_dist => can get alpha<1 => spikier draws
        """
        eps_ = 1e-3
        alpha_vec = (base_dist + eps_) / distinctness_factor
        # sample Dirichlet
        x_ = np.random.gamma(alpha_vec, 1.0)
        x_ /= x_.sum()
        return x_

    clones_info = []
    for c_ in range(total_clones):
        st_ = "stable" if c_ < n_stable_clones else "divergent"
        c_size = np.random.randint(min_clone_size, max_clone_size+1)
        # pick subset for phen
        base_dist = sample_subset_of_phen(unique_phens, subset_size_probs)
        clones_info.append({
            "clone_id": f"clone_{c_}",
            "stability": st_,
            "approx_size": c_size,
            "base_dist": base_dist
        })

    # ablation
    clone2abl_cov = {}
    for c_ in clones_info:
        if len(unique_covs)>1 and np.random.rand()<ablation_prob:
            ab_cov = np.random.choice(unique_covs)
        else:
            ab_cov = None
        clone2abl_cov[c_["clone_id"]] = ab_cov

    # -------------------------------------------------------------------------
    # 4) Actually generate cells
    # -------------------------------------------------------------------------
    def generate_cells_from_phenotype(ph_name, n_cells):
        """
        ZINB from phen2mean[ph_name] plus multiplicative noise
        """
        if n_cells<=0:
            return np.zeros((0, expr.shape[1]), dtype=int)
        base_mu = phen2mean[ph_name]
        scale_ = phenotype_noise_scale[ph_name]
        out_list = []
        for _ in range(n_cells):
            log_factors = np.random.normal(0.0, scale_, size=base_mu.size)
            mu_pert = base_mu * np.exp(log_factors)
            # NB
            c_ = np.zeros_like(mu_pert, dtype=int)
            for g_ in range(len(mu_pert)):
                mg = mu_pert[g_]
                if mg<1e-9:
                    c_[g_] = 0
                else:
                    p_g = zinb_dispersion / (zinb_dispersion+mg)
                    c_[g_] = np.random.negative_binomial(zinb_dispersion, p_g)
            # zero infl
            dropmask = np.random.rand(len(mu_pert)) < zinb_dropout_prob
            c_[dropmask] = 0
            out_list.append(c_)
        return np.vstack(out_list)

    results_expr_list = []
    results_meta_list = []

    for cinfo in clones_info:
        cid = cinfo["clone_id"]
        st_ = cinfo["stability"]
        c_size = cinfo["approx_size"]
        base_dist = cinfo["base_dist"]
        ab_cov = clone2abl_cov[cid]

        # partition c_size among cov
        portion = c_size//len(unique_covs)
        leftover = c_size - portion*len(unique_covs)
        cov_counts = {}
        for iC, cv_ in enumerate(unique_covs):
            cov_counts[cv_] = portion
        for iC in range(leftover):
            cov_counts[unique_covs[iC]] += 1

        # partial ablation
        if ab_cov in cov_counts:
            old_num = cov_counts[ab_cov]
            keep_ = int(round(old_num*(1.0 - ablation_fraction)))
            cov_counts[ab_cov] = max(keep_, 0)

        # stable => same subset distribution
        # divergent => for each cov => sample new distribution around base_dist
        cov2dist = {}
        for cv_ in unique_covs:
            if st_=="stable":
                cov2dist[cv_] = base_dist
            else:
                cov2dist[cv_] = generate_dirichlet_divergent(base_dist)

        # now generate actual cells
        for cv_ in unique_covs:
            n_cov = cov_counts[cv_]
            if n_cov<=0:
                continue
            phen_dist = cov2dist[cv_]
            # phen_dist might have zeros for phen not in subset
            # sample how many cells per phen
            phen_counts = np.random.multinomial(n_cov, phen_dist)
            for iPH, ph_ in enumerate(unique_phens):
                c_ = phen_counts[iPH]
                if c_>0:
                    # generate c_ cells
                    newmat = generate_cells_from_phenotype(ph_, c_)
                    for _ in range(c_):
                        results_meta_list.append({
                            "clone_id": cid,
                            "covariate": cv_,
                            "true_phenotype": ph_
                        })
                    results_expr_list.append(newmat)

    if results_expr_list:
        final_expr = np.vstack(results_expr_list)
    else:
        final_expr = np.zeros((0, expr.shape[1]), dtype=int)

    final_obs = pd.DataFrame(results_meta_list).reset_index(drop=True)

    # mislabel
    if len(final_obs)>0:
        ph_col = final_obs["true_phenotype"].values.astype(object)
        nm_ = int(len(ph_col)*mislabel_fraction)
        if nm_>0:
            mis_idx = np.random.choice(len(ph_col), size=nm_, replace=False)
            uphens = np.unique(ph_col)
            for mI in mis_idx:
                cph = ph_col[mI]
                alt = [p for p in uphens if p!=cph]
                if alt:
                    ph_col[mI] = np.random.choice(alt)
        final_obs["observed_phenotype"] = ph_col
    else:
        final_obs["observed_phenotype"] = []

    # stable_or_divergent
    st_map = {x["clone_id"]: x["stability"] for x in clones_info}
    final_obs["stable_or_divergent"] = final_obs["clone_id"].map(st_map)

    # set covariate_col
    if covariate_col not in final_obs.columns and "covariate" in final_obs.columns:
        final_obs[covariate_col] = final_obs["covariate"]

    # build adata
    new_adata = anndata.AnnData(
        X=final_expr,
        obs=final_obs,
        var=adata_sub.var.copy(),
        uns=adata_sub.uns.copy()
    )
    new_adata.layers["raw_counts"] = new_adata.X.copy()

    # store metadata => clone_cov_phen_distribution
    clone_cov_phen_dist = {}
    for c_ in clones_info:
        cid = c_["clone_id"]
        base_ = c_["base_dist"]
        ab_ = clone2abl_cov[cid]
        for cv_ in unique_covs:
            if ab_==cv_:
                # partial ablation => store "abl_d" as scaled
                abl_d = base_.copy()*(1.0-ablation_fraction)
            else:
                abl_d = base_
            clone_cov_phen_dist[(cid, cv_)] = {
                "true_distribution": base_,
                "ablated_distribution": abl_d
            }

    new_adata.uns["clone_metadata"] = pd.DataFrame(clones_info)
    new_adata.uns["clone_cov_phen_distribution"] = clone_cov_phen_dist
    new_adata.uns["synthetic_patient_chosen"] = best_pt

    # -------------------------------------------------------------------------
    # optional scRNA postprocessing
    # -------------------------------------------------------------------------
    if do_postprocessing and SCANPY_AVAILABLE and new_adata.n_obs>0:
        sc_raw = new_adata.copy()
        sc_raw.X = sc_raw.layers["raw_counts"].copy()
        sc.pp.normalize_total(sc_raw, target_sum=1e4)
        sc.pp.log1p(sc_raw)
        sc.pp.highly_variable_genes(sc_raw, n_top_genes=n_hvg, flavor="seurat_v3", inplace=True)
        sc_raw = sc_raw[:, sc_raw.var["highly_variable"]].copy()
        sc.pp.scale(sc_raw, max_value=10)
        sc.tl.pca(sc_raw, n_comps=n_pcs)
        sc.pp.neighbors(sc_raw, n_pcs=n_pcs)
        sc.tl.umap(sc_raw)

        new_adata.obsm["X_pca"] = sc_raw.obsm["X_pca"]
        new_adata.obsm["X_umap"] = sc_raw.obsm["X_umap"]
        new_adata.uns["umap_params"] = {"n_hvg": n_hvg, "n_pcs": n_pcs}

    return new_adata