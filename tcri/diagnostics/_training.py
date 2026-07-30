"""``tcri.diag`` training diagnostics (§9.2) — the two model plots relocated off
``TCRIModel`` (``plot_loss`` → :func:`loss`, ``plot_archetypes`` → :func:`archetypes`)."""
from __future__ import annotations

import numpy as np

__all__ = ["loss", "archetypes"]


def _series(hist, key):
    """Pull a 1-D value array from an scvi ``history_`` entry (DataFrame/Series/list)."""
    v = hist.get(key) if hasattr(hist, "get") else None
    if v is None:
        return []
    if hasattr(v, "values"):
        arr = np.asarray(v.values).ravel()
        return arr.tolist()
    return list(v)


def loss(model, *, log_scale=False, ax=None, save=None):
    """Plot training/validation ELBO and prior-KL from ``model.history_``."""
    import matplotlib.pyplot as plt

    hist = getattr(model, "history_", {}) or {}
    elbo_train = _series(hist, "elbo_train")
    elbo_val = _series(hist, "elbo_validation")
    kl_train = _series(hist, "kl_divergence_with_prior_train_epoch")
    kl_val = _series(hist, "kl_divergence_with_prior_val")

    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    else:
        fig = ax.figure
        axes = [ax, ax.figure.add_subplot(212)] if len(ax.figure.axes) < 2 else ax.figure.axes[:2]

    axes[0].plot(elbo_train, label="train ELBO")
    if elbo_val:
        axes[0].plot(elbo_val, label="val ELBO")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("ELBO"); axes[0].set_title("ELBO"); axes[0].legend()
    if kl_train or kl_val:
        if kl_train:
            axes[1].plot(kl_train, label="train KL(prior)")
        if kl_val:
            axes[1].plot(kl_val, label="val KL(prior)")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("KL"); axes[1].set_title("prior KL"); axes[1].legend()
    if log_scale:
        for a in axes:
            a.set_yscale("log")
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    return axes[0]


def archetypes(model, *, ax=None, save=None):
    """Cluster-ordered clone×phenotype heatmap + archetype centroids, ordered by the
    ``build_archetypes`` labels retained on the model."""
    import matplotlib.pyplot as plt

    labels = np.asarray(model.labels)
    prior = np.asarray(model.clone_phenotype_prior)
    centers = np.asarray(model.centers)
    order = np.argsort(labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) if ax is None else (ax.figure, [ax, ax.figure.add_subplot(122)])
    im0 = axes[0].imshow(prior[order, :], aspect="auto", cmap="viridis")
    axes[0].set_title("clone phenotype prior (cluster-ordered)")
    axes[0].set_xlabel("phenotype"); axes[0].set_ylabel("clone (by cluster)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(centers, aspect="auto", cmap="viridis")
    axes[1].set_title("archetype centroids")
    axes[1].set_xlabel("phenotype"); axes[1].set_ylabel("archetype")
    fig.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    return axes[0]
