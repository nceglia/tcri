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

import numpy as np
import pandas as pd
from anndata import AnnData

__all__ = ["simulate_tcri", "mi_from_joint_oracle", "simulate_from_fit_params",
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
