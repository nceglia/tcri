"""Self-contained synthetic generator with an **exact mutual-information oracle**.

Implements the semi-synthetic generative story from Supplementary Note 1
("Generative Model for Semi-Synthetic Simulations")::

    pi        ~ Dirichlet(alpha_pi)              # clone abundance P(c)
    omega_c   ~ Dirichlet(alpha_omega)           # P(phi | c)
    z_i       ~ Categorical(pi)                  # clone of cell i
    phi_i|z_i ~ Categorical(omega[z_i])          # phenotype of cell i
    x_i|phi_i ~ Poisson(U_i @ V),  U_i ~ Gamma(program of phi_i)

Because ``pi`` and ``omega`` are known, the **population** mutual information
I(c;phi) is available in closed form — which is what makes statistical *recovery*
testing possible at all. Nothing else in the test suite has an oracle: the contract
tests check structure and identities, never accuracy.

Two oracles are reported, and the distinction matters for writing honest tests:

``true_mi_*``
    the **population** value implied by ``(pi, omega)`` — what an estimator should
    approach as ``n_cells -> inf``.
``empirical_mi_*``
    the value implied by the **realized** ``(clone, phenotype)`` counts in this
    particular sample. A perfect estimator applied to *this* dataset returns this,
    not the population value. They differ by finite-sample noise plus the plug-in
    estimator's upward bias, roughly ``(C-1)(P-1) / (2 N ln2)`` bits.

Both are given under both normalizations (``min`` and ``average``): tcri defaults to
``normalize_mode="min"`` while the note's benchmark used the mean denominator, so a
like-for-like comparison has to pick deliberately.

Unlike the original ``sc_simulator``, this needs no real dataset to fit — the gene
programs are generated directly — so it is importable, seeded, and fast.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

__all__ = ["simulate_tcri", "simulate_cohort", "mi_from_joint_oracle",
           "simulate_from_fit_params",
           "temperature_scale"]


def mi_from_joint_oracle(joint: np.ndarray) -> dict:
    """Exact MI / entropies (bits) of a clone x phenotype **probability** table.

    ``joint`` must sum to 1. Returns ``mi``, ``h_clone``, ``h_phenotype`` and both
    normalized variants. This is the oracle — deliberately a small, independent
    implementation so it cannot drift with the package's own metric code (a test
    that computes the expected value with the code under test proves nothing).
    """
    P = np.asarray(joint, dtype=np.float64)
    total = P.sum()
    if total <= 0:
        raise ValueError("joint has no mass")
    P = P / total
    p_c = P.sum(1)
    p_ph = P.sum(0)

    nz = P > 0
    outer = np.outer(p_c, p_ph)
    mi = float(np.sum(P[nz] * np.log2(P[nz] / outer[nz])))

    h_c = float(-np.sum(p_c[p_c > 0] * np.log2(p_c[p_c > 0])))
    h_p = float(-np.sum(p_ph[p_ph > 0] * np.log2(p_ph[p_ph > 0])))

    denom_min = min(h_c, h_p)
    denom_avg = 0.5 * (h_c + h_p)
    return {
        "mi": mi,
        "h_clone": h_c,
        "h_phenotype": h_p,
        "nmi_min": mi / denom_min if denom_min > 0 else 0.0,
        "nmi_average": mi / denom_avg if denom_avg > 0 else 0.0,
    }


def _phenotype_programs(rng, n_phenotypes, n_factors, fuzziness):
    """Per-phenotype Gamma params over latent factors, blended by ``fuzziness``.

    ``fuzziness=0`` leaves the phenotypes fully distinct; ``fuzziness=1`` collapses
    them to a common program so phenotype is **unidentifiable from expression** while
    the clone->phenotype coupling (and hence the true MI) is untouched. That
    separation is the point: it varies estimation difficulty at fixed ground truth.

    Interpolation is in natural-parameter space ``theta = [alpha-1, -beta]``, matching
    the original ``interpolate_gamma_params``.
    """
    alpha = rng.uniform(1.5, 6.0, size=(n_phenotypes, n_factors))
    beta = rng.uniform(1.0, 3.0, size=(n_phenotypes, n_factors))

    # DE-20. Supplementary Note 1 interpolates with a CONCAVE mapping g(f), not with f:
    #
    #     theta'_k = (1 - g(f)) theta_k + g(f) theta_bar,
    #     "in the reported experiments, we use g(f) = sqrt(f)"
    #
    # The code used g(f) = f, which under-mixes at every f in (0, 1) -- at f=0.1 the note
    # blends 0.316 toward the mean where this blended 0.100. The endpoints f=0 and f=1 agree,
    # so only the interior of the sweep was affected. The note permits any concave g on
    # [0, 1]; sqrt is what the reported experiments use and is therefore the default here.
    g = float(np.sqrt(fuzziness))

    theta = np.concatenate([alpha - 1.0, -beta], axis=1)
    theta = (1.0 - g) * theta + g * theta.mean(0, keepdims=True)

    alpha_f = theta[:, :n_factors] + 1.0
    beta_f = -theta[:, n_factors:]
    # keep the Gamma valid after blending
    return np.clip(alpha_f, 1e-3, None), np.clip(beta_f, 1e-3, None)


def simulate_tcri(
    *,
    n_clones: int = 30,
    n_phenotypes: int = 4,
    n_genes: int = 60,
    n_cells: int = 2000,
    n_covariates: int = 1,
    n_factors: int = 8,
    omega_concentration: float = 0.35,
    pi_concentration: float = 2.0,
    fuzziness: float = 0.0,
    label_error_rate: float = 0.0,
    seed: int = 0,
) -> AnnData:
    """Simulate a TCR+RNA dataset whose true mutual information is known exactly.

    Parameters
    ----------
    n_clones, n_phenotypes, n_genes, n_cells
        Problem size. ``n_cells`` drives how close the realized sample sits to the
        population oracle.
    n_covariates
        Cells are split across covariates. ``omega`` is shared, so each covariate's
        population MI equals the global one — which lets a per-covariate estimate be
        compared against the same oracle.
    omega_concentration
        Dirichlet concentration for ``P(phi|c)`` and **the knob that sets the true
        MI**: small (<1) gives near-one-hot rows and MI near ``H(phi)``; large gives
        near-uniform rows and MI near 0.
    pi_concentration
        Dirichlet concentration for clone abundance.
    fuzziness
        In ``[0,1]``. Blends the per-phenotype expression programs, making phenotype
        harder to read off expression **without changing the true MI**.
    label_error_rate
        Fraction of cells whose recorded phenotype is replaced by a uniform draw.
        This DOES change the realized coupling, so ``empirical_*`` is recomputed from
        the corrupted labels while ``true_*`` keeps the uncorrupted population value.
    seed
        Seeds everything.

    Returns
    -------
    AnnData
        ``X`` / ``layers['counts']`` integer counts; ``obs`` with ``clone_id``,
        ``phenotype`` (possibly corrupted), ``true_phenotype``, ``covariate``,
        ``batch``; ``uns['tcri_truth']`` holding ``omega``, ``pi``, the population
        oracle (``true_mi``, ``true_nmi_min``, ``true_nmi_average``, entropies) and
        the realized-sample oracle (``empirical_*``), plus the generating settings.
    """
    if not 0.0 <= fuzziness <= 1.0:
        raise ValueError("fuzziness must be in [0, 1]")
    if not 0.0 <= label_error_rate <= 1.0:
        raise ValueError("label_error_rate must be in [0, 1]")
    if n_covariates < 1:
        raise ValueError("n_covariates must be >= 1")

    rng = np.random.default_rng(seed)

    # ── ground truth: clone abundance and the clone -> phenotype coupling ────
    pi = rng.dirichlet(np.full(n_clones, pi_concentration))
    omega = rng.dirichlet(np.full(n_phenotypes, omega_concentration), size=n_clones)

    # ── population oracle (closed form, independent of any sampling) ─────────
    truth = mi_from_joint_oracle(pi[:, None] * omega)

    # ── sample cells ────────────────────────────────────────────────────────
    z = rng.choice(n_clones, size=n_cells, p=pi)
    phi_true = np.array([rng.choice(n_phenotypes, p=omega[c]) for c in z])

    phi = phi_true.copy()
    if label_error_rate > 0:
        flip = rng.random(n_cells) < label_error_rate
        phi[flip] = rng.integers(0, n_phenotypes, size=int(flip.sum()))

    # ── expression: x_i ~ Poisson(U_i @ V), U_i ~ Gamma(program of phi_i) ────
    alpha, beta = _phenotype_programs(rng, n_phenotypes, n_factors, fuzziness)
    V = rng.gamma(2.0, 1.0, size=(n_factors, n_genes))
    U = rng.gamma(alpha[phi_true], 1.0 / beta[phi_true])      # (n_cells, n_factors)
    X = rng.poisson(U @ V).astype("float32")

    # ── realized-sample oracle, from the labels actually recorded ───────────
    counts = np.zeros((n_clones, n_phenotypes), dtype=np.float64)
    np.add.at(counts, (z, phi), 1.0)
    empirical = mi_from_joint_oracle(counts) if counts.sum() > 0 else dict.fromkeys(truth, np.nan)

    _PHEN_LEVELS = [f"phen_{p}" for p in range(n_phenotypes)]
    obs = pd.DataFrame(
        {
            # DE-13: declare the label space. Without `categories=`, pandas infers it from
            # the values it happens to see and sorts LEXICOGRAPHICALLY, so at K>=10
            # 'phen_2' gets code 4 and 'phen_11' code 3 -- codes stop matching the integer
            # phenotype index they were built from. A phenotype with zero sampled cells
            # also vanishes from the level set, silently shrinking P.
            "clone_id": pd.Categorical([f"clone_{i}" for i in z],
                                       categories=[f"clone_{i}" for i in range(n_clones)]),
            "phenotype": pd.Categorical([f"phen_{p}" for p in phi],
                                        categories=_PHEN_LEVELS),
            "true_phenotype": pd.Categorical([f"phen_{p}" for p in phi_true],
                                             categories=_PHEN_LEVELS),
            "covariate": pd.Categorical(
                [f"cov_{i}" for i in rng.integers(0, n_covariates, size=n_cells)],
                categories=[f"cov_{i}" for i in range(n_covariates)],
            ),
            "batch": pd.Categorical(["batch_0"] * n_cells, categories=["batch_0"]),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    adata = AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{g}" for g in range(n_genes)]),
    )
    adata.layers["counts"] = adata.X.copy()

    adata.uns["tcri_truth"] = {
        "omega": omega,
        "pi": pi,
        "true_mi": truth["mi"],
        "true_nmi_min": truth["nmi_min"],
        "true_nmi_average": truth["nmi_average"],
        "true_h_clone": truth["h_clone"],
        "true_h_phenotype": truth["h_phenotype"],
        "empirical_mi": empirical["mi"],
        "empirical_nmi_min": empirical["nmi_min"],
        "empirical_nmi_average": empirical["nmi_average"],
        "settings": {
            "n_clones": n_clones,
            "n_phenotypes": n_phenotypes,
            "n_genes": n_genes,
            "n_cells": n_cells,
            "n_covariates": n_covariates,
            "n_factors": n_factors,
            "omega_concentration": omega_concentration,
            "pi_concentration": pi_concentration,
            "fuzziness": fuzziness,
            "label_error_rate": label_error_rate,
            "seed": seed,
        },
    }
    return adata


# ── benchmark reproduction: generate from an empirical fit ──────────────────

def temperature_scale(P, T, eps=1e-12):
    """Sharpen/flatten a row-stochastic matrix: ``P**(1/T)`` renormalized.

    Verbatim behaviour of ``sc_simulator.temperature_scale_conditional``. ``T<1``
    sharpens (raising I(c;phi)), ``T>1`` flattens. This is the axis the published
    benchmark sweeps, and it changes the GROUND TRUTH, not just the difficulty.
    """
    T = float(T)
    # Supplementary Note 1, "Temperature Scaling of Conditional Distributions", specifies
    # T > 0. Outside that the function used to fail three different silent ways:
    #   T = 0      -> ZeroDivisionError
    #   T = nan    -> an all-NaN matrix, no error
    #   T = -1.0   -> finite, plausible-looking numbers that INVERT the distribution
    # The last is the dangerous one: a negative T produced a valid-looking row-stochastic
    # matrix and would have propagated into a benchmark as though it meant something.
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(
            f"temperature must be finite and > 0 (Supplementary Note 1, 'Temperature "
            f"Scaling of Conditional Distributions'); got T={T!r}"
        )

    P = np.clip(np.asarray(P, dtype=float), eps, None)
    Pp = P ** (1.0 / T)

    # float64 underflow: once (1/T)*log10(p) < -308 every entry of a row becomes exactly
    # 0.0 and the renormalisation below is 0/0. Measured on a [0.7, 0.2, 0.1] row: fine at
    # T=1e-3, all-NaN at T=1e-4. Raising beats returning NaN, which the caller would have
    # to notice.
    row_sums = Pp.sum(axis=1, keepdims=True)
    dead = ~np.isfinite(row_sums) | (row_sums <= 0.0)
    if dead.any():
        raise ValueError(
            f"temperature T={T!r} underflows float64: {int(dead.sum())} of {len(row_sums)} "
            f"row(s) collapsed to all-zero under P**(1/T), so the result would be NaN. "
            f"The smallest usable T depends on the smallest entry of P -- underflow starts "
            f"once (1/T)*log10(min(P)) < -308."
        )
    return Pp / row_sums


def simulate_from_fit_params(
    params,
    *,
    n_cells: int = 1000,
    temperature: float = 1.0,
    fuzziness: float = 0.0,
    label_error_rate: float = 0.0,
    seed: int = 0,
) -> AnnData:
    """Simulate from an **empirically fitted** ``(pi, omega, gamma_params, V)``.

    Reproduces ``sc_simulator.simulate_dataset``: ``z ~ Cat(pi)``,
    ``phi|z ~ Cat(omega[z])``, ``U ~ Gamma(alpha_phi, 1/beta_phi)``,
    ``x ~ Poisson(U @ V)``.

    Use this — rather than :func:`simulate_tcri` — whenever the point is to compare
    against the published benchmark. A symmetric-Dirichlet ``omega`` cannot
    reproduce the benchmark's true-NMI anchors: its response to temperature has the
    wrong SHAPE (sharpening ratio 4.22x vs the true 2.86x), so no reparameterization
    of the synthetic generator suffices. The empirical fit matches all three anchors
    exactly (0.520 / 0.316 / 0.182 at T = 0.1 / 0.5 / 1.0).

    Parameters
    ----------
    params
        Path to a ``fit_params.pkl``, or the already-unpickled dict. Needs
        ``pi``, ``omega``, ``gamma_params``, ``V``, ``L``.
    temperature
        Applied to ``omega`` BEFORE sampling, so it moves the ground truth.
    fuzziness
        Blends the per-phenotype Gamma programs toward their mean — difficulty only,
        the truth is untouched.
    """
    import pickle

    if not isinstance(params, dict):
        with open(params, "rb") as fh:
            params = pickle.load(fh)
    for key in ("pi", "omega", "gamma_params", "V"):
        if key not in params:
            raise KeyError(f"fit params missing {key!r}; got {sorted(params)}")

    rng = np.random.default_rng(seed)
    pi = np.asarray(params["pi"], dtype=float)
    pi = pi / pi.sum()
    omega = np.asarray(params["omega"], dtype=float)
    if temperature != 1.0:
        omega = temperature_scale(omega, temperature)
    omega = omega / omega.sum(axis=1, keepdims=True)

    # NB the fit stores V as (D, L) and the reference sampler does U @ V.T
    V = np.asarray(params["V"], dtype=float)
    if V.shape[0] != int(params.get("L", V.shape[1])):
        V = V.T                                        # -> (L, D)
    L, D = V.shape
    n_clones, P = omega.shape

    truth = mi_from_joint_oracle(pi[:, None] * omega)

    z = rng.choice(n_clones, size=n_cells, p=pi)
    phi_true = np.array([rng.choice(P, p=omega[c]) for c in z])
    phi = phi_true.copy()
    if label_error_rate > 0:
        flip = rng.random(n_cells) < label_error_rate
        phi[flip] = rng.integers(0, P, size=int(flip.sum()))

    # per-phenotype Gamma programs, optionally blended toward the mean
    gp = params["gamma_params"]
    keys = sorted(gp.keys())
    alpha = np.stack([np.asarray(gp[k]["alpha"], dtype=float) for k in keys])
    beta = np.stack([np.asarray(gp[k]["beta"], dtype=float) for k in keys])
    if fuzziness > 0:
        theta = np.concatenate([alpha - 1.0, -beta], axis=1)
        theta = (1.0 - fuzziness) * theta + fuzziness * theta.mean(0, keepdims=True)
        alpha = np.clip(theta[:, :L] + 1.0, 1e-3, None)
        beta = np.clip(-theta[:, L:], 1e-3, None)

    U = rng.gamma(alpha[phi_true], 1.0 / beta[phi_true])
    X = rng.poisson(U @ V).astype("float32")

    counts = np.zeros((n_clones, P), dtype=np.float64)
    np.add.at(counts, (z, phi), 1.0)
    empirical = mi_from_joint_oracle(counts) if counts.sum() > 0 else dict.fromkeys(truth, np.nan)

    clone_levels = params.get("clone_levels")
    clone_names = ([str(clone_levels[i]) for i in z] if clone_levels is not None
                   else [f"clone_{i}" for i in z])

    # DE-13, as above: declare the label space rather than letting pandas infer it.
    phen_levels = [f"phen_{p}" for p in range(n_phenotypes)]
    clone_levels_all = ([str(c) for c in clone_levels] if clone_levels is not None
                        else [f"clone_{i}" for i in range(n_clones)])
    obs = pd.DataFrame({
        "clone_id": pd.Categorical(clone_names, categories=clone_levels_all),
        "phenotype": pd.Categorical([f"phen_{p}" for p in phi], categories=phen_levels),
        "true_phenotype": pd.Categorical([f"phen_{p}" for p in phi_true],
                                         categories=phen_levels),
        "covariate": pd.Categorical(["cov_0"] * n_cells, categories=["cov_0"]),
        "batch": pd.Categorical(["batch_0"] * n_cells, categories=["batch_0"]),
    }, index=[f"cell_{i}" for i in range(n_cells)])

    adata = AnnData(X=X, obs=obs,
                    var=pd.DataFrame(index=[f"gene_{g}" for g in range(D)]))
    adata.layers["counts"] = adata.X.copy()
    adata.uns["tcri_truth"] = {
        "omega": omega, "pi": pi,
        "true_mi": truth["mi"],
        "true_nmi_min": truth["nmi_min"],
        "true_nmi_average": truth["nmi_average"],
        "true_h_clone": truth["h_clone"],
        "true_h_phenotype": truth["h_phenotype"],
        "empirical_mi": empirical["mi"],
        "empirical_nmi_min": empirical["nmi_min"],
        "empirical_nmi_average": empirical["nmi_average"],
        "settings": {"source": "empirical fit", "n_cells": n_cells,
                     "temperature": temperature, "fuzziness": fuzziness,
                     "label_error_rate": label_error_rate, "seed": seed,
                     "n_clones": n_clones, "n_phenotypes": P, "n_genes": D, "L": L},
    }
    return adata


def _zipf_weights(n, exponent, rng):
    """Clone-size weights following a power law — the shape real repertoires have.

    TCR clone sizes are heavy-tailed: a handful of large expanded clones and a long tail of
    singletons. The standard model is a power law (Zipf; Pareto in continuous form),
    ``P(size) ~ size**-alpha`` with ``alpha`` near 2. ``simulate_tcri``'s own ``pi`` is a
    symmetric Dirichlet, which is not heavy-tailed at all -- so a cohort that wants realistic
    clone sizes has to impose them, which is what this does.

    Ranks are shuffled so clone *identity* is unrelated to clone *size*; without that, low
    clone ids would always be the expanded ones.
    """
    ranks = np.arange(1, n + 1, dtype=float)
    w = ranks ** (-float(exponent))
    rng.shuffle(w)
    return w / w.sum()


def simulate_cohort(
    *,
    n_patients: int = 8,
    conditions=("pre", "post"),
    responder_fraction: float = 0.5,
    n_clones=(12, 30),
    n_phenotypes: int = 4,
    n_genes: int = 40,
    n_cells_per_sample: int = 300,
    clone_size_distribution: str = "powerlaw",
    clone_size_exponent: float = 2.0,
    responder_enrichment: float = 12.0,
    nonresponder_enrichment: float = 1.1,
    omega_concentration: float = 0.9,
    seed: int = 0,
):
    """A multi-patient, multi-condition cohort in one line.

    The shape most analyses actually have — patients as replicates, an ordered condition axis
    *within* each patient, and a response label *between* them — which the single-sample
    :func:`simulate_tcri` cannot express.

    >>> adata = simulate_cohort(n_patients=10, conditions=("pre", "mid", "post"))

    **Clones are paired across conditions.** Simulating each condition independently and
    giving the runs matching clone names does *not* pair them: the ids line up but the
    underlying clone->phenotype relationship is unrelated, so ``phenotypic_flux`` and every
    ``delta_*`` would have nothing to measure. Each patient is simulated **once**, fixing that
    patient's clone->phenotype structure and its clone sizes, and every condition is drawn
    from that one population.

    **Condition progression.** The first condition is an unbiased draw. Later ones oversample
    each clone's dominant phenotype, ramping linearly to the arm's full enrichment at the last
    condition — so responders' clones commit over time and non-responders' barely move.
    Nothing is relabelled, so a cell's phenotype still matches the expression it was generated
    with; only the clone->phenotype *concentration* changes.

    Parameters
    ----------
    n_patients
        Total patients. Each contributes one sample per condition.
    conditions
        Ordered condition labels, two or more (``("pre", "post")``, or a timeseries).
    responder_fraction
        Fraction assigned to the ``"R"`` arm; the rest are ``"NR"``.
    n_clones
        Clones per patient. An ``int`` for a fixed count, or ``(lo, hi)`` to draw each
        patient's count uniformly — real cohorts are ragged, and a metric normalized by
        ``log2(C)`` is not comparable across patients with different ``C``, which is exactly
        what ``n_clones_ref`` exists to pin.
    clone_size_distribution
        ``"powerlaw"`` (default) or ``"uniform"``. Power law is what repertoires look like:
        a few large expanded clones over a long tail of singletons.
    clone_size_exponent
        The power-law exponent ``alpha`` in ``P(size) ~ size**-alpha``. ~2 is the usual
        repertoire regime; larger is more skewed toward singletons. It is a **target**: cells
        are drawn without replacement from a finite pool, so a clone whose target share
        exceeds its pool supply is capped and the realized tail comes out shallower. Measured
        at the default (40 clones, 1200 cells/sample): requested 2.0 -> realized log-log slope
        about -1.5, Gini 0.70, largest clone ~22% of cells. Still firmly heavy-tailed; just
        not the exact exponent asked for.
    responder_enrichment, nonresponder_enrichment
        How hard the final condition oversamples each clone's dominant phenotype. ``1.0`` is
        no enrichment. Jittered +/-15% per patient so replicates are not identical.
    omega_concentration
        Passed to :func:`simulate_tcri` per patient. Lower = sharper coupling at baseline;
        the default is deliberately diffuse so there is room to concentrate.
    seed
        Seeds everything.

    Returns
    -------
    AnnData
        ``obs`` with ``clone_id``, ``phenotype``, ``condition``, ``patient``, ``response``;
        ``layers['counts']``; and ``uns['tcri_truth']`` carrying ``per_sample`` — the
        empirical NMI of each (patient, condition) from the labels alone, computed **per
        patient** because that is the unit a per-patient metric estimates — plus ``per_arm``
        and the generating ``settings``.

    Notes
    -----
    ``per_sample`` is the **plug-in estimate on the observed labels**, not a target a fitted
    model should reproduce. Two reasons it sits above what ``tl.mutual_information`` reports,
    and neither is a defect:

    * the plug-in is upward-biased at finite N, by roughly ``(C-1)(P-1) / (2 N ln2)`` bits
      (see this module's header) — it is the quantity the model is trying to see *past*;
    * the model shrinks toward a covariate-free ``omega_c`` (Note 1 eq 2), deliberately, since
      the cells in hand are a sample of a much larger unobserved repertoire. Conservative is
      the intent.

    Use it to check that arms are ordered as constructed and separate from a permutation null
    — not to score agreement in absolute bits.
    """
    conditions = tuple(str(c) for c in conditions)
    if len(conditions) < 2:
        raise ValueError(f"conditions needs at least two labels, got {conditions}")
    if len(set(conditions)) != len(conditions):
        raise ValueError(f"conditions must be unique, got {conditions}")
    if n_patients < 2:
        raise ValueError("n_patients must be >= 2")
    if clone_size_distribution not in ("powerlaw", "uniform"):
        raise ValueError("clone_size_distribution must be 'powerlaw' or 'uniform', "
                         f"got {clone_size_distribution!r}")
    for name, value in (("responder_enrichment", responder_enrichment),
                        ("nonresponder_enrichment", nonresponder_enrichment)):
        if value < 1.0:
            raise ValueError(f"{name} must be >= 1.0 (1.0 = no enrichment), got {value}")

    lo, hi = (n_clones, n_clones) if isinstance(n_clones, (int, np.integer)) else n_clones
    if lo < 2:
        raise ValueError(f"n_clones must be >= 2, got {n_clones}")

    rng = np.random.default_rng(seed)
    n_responders = max(1, min(n_patients - 1, round(n_patients * responder_fraction)))
    width = max(2, len(str(n_patients)))
    n_steps = len(conditions) - 1

    blocks = []
    for i in range(n_patients):
        patient = f"P{i + 1:0{width}d}"
        arm = "R" if i < n_responders else "NR"
        target = responder_enrichment if arm == "R" else nonresponder_enrichment
        target = float(target * rng.uniform(0.85, 1.15))
        n_c = int(rng.integers(lo, hi + 1))

        # ONE simulation per patient fixes its clone -> phenotype structure. Oversampled so
        # the clone-size reshaping below has cells to draw on for the expanded clones.
        pool = simulate_tcri(
            n_clones=n_c, n_phenotypes=n_phenotypes, n_genes=n_genes,
            n_cells=6 * n_cells_per_sample, n_covariates=1,
            omega_concentration=omega_concentration, seed=seed + i,
        )
        clones = pool.obs["clone_id"].astype(str).to_numpy()
        uniq = np.array(sorted(set(clones)))
        size_w = (_zipf_weights(len(uniq), clone_size_exponent, rng)
                  if clone_size_distribution == "powerlaw"
                  else np.full(len(uniq), 1.0 / len(uniq)))
        # per-cell weight from its clone's target share, divided by how many cells that clone
        # has in the pool -- so the realized clone SIZES follow the target, not the pool's
        pool_counts = pd.Series(clones).value_counts()
        share = dict(zip(uniq, size_w))
        base_w = np.array([share[c] / pool_counts[c] for c in clones])

        modal = pool.obs.groupby("clone_id", observed=True)["phenotype"].agg(
            lambda x: x.value_counts().idxmax())
        is_modal = (pool.obs["clone_id"].map(modal).astype(str)
                    == pool.obs["phenotype"].astype(str)).to_numpy()

        for step, condition in enumerate(conditions):
            # linear ramp: unbiased at the first condition, full enrichment at the last
            enrichment = 1.0 + (target - 1.0) * (step / n_steps)
            w = base_w * np.where(is_modal, enrichment, 1.0)
            idx = rng.choice(pool.n_obs, size=n_cells_per_sample, replace=False, p=w / w.sum())

            block = pool[idx].copy()
            block.obs["clone_id"] = block.obs["clone_id"].astype(str) + "@" + patient
            block.obs["condition"] = condition
            block.obs["patient"] = patient
            block.obs["response"] = arm
            block.obs_names = [f"{patient}_{condition}_{k}" for k in range(block.n_obs)]
            blocks.append(block)

    adata = ad.concat(blocks, join="outer", label=None)
    # `covariate` and `batch` come from the per-patient sims and are single-valued there;
    # `condition` and `patient` supersede them, and leaving them would point someone at the
    # wrong column when they reach for setup_anndata(covariate_key=...)
    adata.obs = adata.obs.drop(columns=["covariate", "batch"], errors="ignore")
    for col in ("clone_id", "phenotype", "condition", "patient", "response"):
        adata.obs[col] = adata.obs[col].astype("category")
    adata.obs["condition"] = adata.obs["condition"].cat.reorder_categories(list(conditions))
    adata.layers["counts"] = adata.X.copy()

    # The oracle, per (patient, condition). PER PATIENT on purpose: pooling an arm's patients
    # first mixes clones across patients and inflates the NMI, so comparing that against a
    # per-patient estimate would be a unit mismatch rather than a benchmark.
    rows = []
    for (patient, arm, condition), g in adata.obs.groupby(
            ["patient", "response", "condition"], observed=True):
        crosstab = pd.crosstab(g["clone_id"], g["phenotype"]).to_numpy(dtype=float)
        oracle = mi_from_joint_oracle(crosstab)
        rows.append({"patient": patient, "response": arm, "condition": condition,
                     "n_clones": int(g["clone_id"].nunique()),
                     "empirical_mi": oracle["mi"],
                     "empirical_nmi_min": oracle["nmi_min"],
                     "empirical_nmi_average": oracle["nmi_average"]})
    per_sample = pd.DataFrame(rows)

    adata.uns["tcri_truth"] = {
        "per_sample": per_sample,
        "per_arm": (per_sample.groupby(["response", "condition"], observed=True)
                    ["empirical_nmi_min"].mean().unstack("condition")[list(conditions)]),
        # LISTS, not tuples: h5py has no writer for a tuple, so a tuple anywhere in `uns`
        # makes the whole object unwritable to .h5ad -- which surfaces far from here, as an
        # IORegistryError out of save_tcri_session.
        "settings": {
            "n_patients": n_patients, "n_responders": n_responders,
            "conditions": list(conditions),
            "n_clones": list(n_clones) if not isinstance(n_clones, (int, np.integer))
                        else int(n_clones),
            "n_phenotypes": n_phenotypes, "n_genes": n_genes,
            "n_cells_per_sample": n_cells_per_sample,
            "clone_size_distribution": clone_size_distribution,
            "clone_size_exponent": clone_size_exponent,
            "responder_enrichment": responder_enrichment,
            "nonresponder_enrichment": nonresponder_enrichment,
            "omega_concentration": omega_concentration, "seed": seed,
        },
    }
    return adata
