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

__all__ = ["simulate_tcri", "mi_from_joint_oracle"]


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

    theta = np.concatenate([alpha - 1.0, -beta], axis=1)
    theta = (1.0 - fuzziness) * theta + fuzziness * theta.mean(0, keepdims=True)

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

    obs = pd.DataFrame(
        {
            "clone_id": pd.Categorical([f"clone_{i}" for i in z]),
            "phenotype": pd.Categorical([f"phen_{p}" for p in phi]),
            "true_phenotype": pd.Categorical([f"phen_{p}" for p in phi_true]),
            "covariate": pd.Categorical(
                [f"cov_{i}" for i in rng.integers(0, n_covariates, size=n_cells)]
            ),
            "batch": pd.Categorical(["batch_0"] * n_cells),
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
