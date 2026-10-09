"""``tcri.diag`` training diagnostics for a fitted model: the training and validation curves
(:func:`loss`) and the archetype heatmap (:func:`archetypes`)."""
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


def _last_call_start(model, n):
    """Index at which the last ``train()`` call's epochs start in a series of ``n`` epochs.

    ``training_record_`` counts epochs within the last call, whose epochs are the last
    ``epochs_run`` of the history: scvi appends a later call's epochs to an earlier call's.
    ``0`` when the record does not say how many epochs the call ran.
    """
    record = getattr(model, "training_record_", None) or {}
    if record.get("epochs_run") is None:
        return 0
    return max(n - int(record["epochs_run"]), 0)


def _after_ramp(model, n):
    """Index of the first epoch after the KL ramp in a series of ``n`` epochs, or ``None``.

    ``None`` when there is no record, the ramp did not complete, or no epoch follows it.
    """
    record = getattr(model, "training_record_", None) or {}
    if not record.get("ramp_completed"):
        return None
    start = _last_call_start(model, n) + int(record.get("ramp_completes_at_epoch") or 0)
    return start if start < n else None


def _fit_y(ax, values, log):
    """Set the y range of ``ax`` to span ``values`` with a 5% margin, in log space if ``log``."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)] if log else v[np.isfinite(v)]
    if not v.size:
        return
    lo, hi = (np.log10(v.min()), np.log10(v.max())) if log else (v.min(), v.max())
    pad = 0.05 * (hi - lo) if hi > lo else (0.05 if log else 0.05 * abs(hi) or 0.5)
    lo, hi = lo - pad, hi + pad
    ax.set_ylim((10 ** lo, 10 ** hi) if log else (lo, hi))


def loss(model, *, log_scale=False, ax=None, save=None):
    """Plot the training loss, the validation objective and the prior KL from
    ``model.history_``, each on its own panel.

    The training loss is ``elbo_train``: a step's negative ELBO, summed over the cells the data
    plate stands for, at that step's annealed KL weight. The validation objective is
    ``objective_validation_percell``, the per-cell objective early stopping selects on, evaluated
    at ``max_kl_weight``. They are different quantities on different scales, so each has its own
    axis. The epoch the last ``train()`` call selected, from ``model.training_record_``, is
    marked on the validation panel, counted across every call the history holds. Returns the
    training panel's axes.

    Once the KL ramp has completed, each panel's y range spans the epochs after it, so a small
    change late in the fit is readable; the earlier epochs, which can sit far above, run off the
    top. Without a completed ramp, the range spans every epoch.
    """
    import matplotlib.pyplot as plt

    hist = getattr(model, "history_", {}) or {}
    elbo_train = _series(hist, "elbo_train")
    objective_val = _series(hist, "objective_validation_percell")
    kl_train = _series(hist, "kl_divergence_with_prior_train_epoch")
    kl_val = _series(hist, "kl_divergence_with_prior_val")

    if ax is None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    else:
        fig = ax.figure
        axes = (ax.figure.axes[:3] if len(ax.figure.axes) >= 3
                else [ax, fig.add_subplot(312), fig.add_subplot(313)])

    axes[0].plot(elbo_train, label="train")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("-ELBO"); axes[0].set_title("Training loss")
    if objective_val:
        axes[1].plot(objective_val, color="C1", label="validation")
    selected = (getattr(model, "training_record_", None) or {}).get("selected_epoch")
    if selected is not None:
        selected = _last_call_start(model, len(objective_val or elbo_train)) + int(selected)
        axes[1].axvline(selected, color="0.3", linestyle="--", linewidth=1,
                        label=f"selected epoch ({selected})")
        axes[1].legend()
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("objective per cell")
    axes[1].set_title("Validation objective (per cell)")
    if kl_train or kl_val:
        if kl_train:
            axes[2].plot(kl_train, label="train KL(prior)")
        if kl_val:
            axes[2].plot(kl_val, label="val KL(prior)")
        axes[2].set_xlabel("epoch"); axes[2].set_ylabel("KL"); axes[2].set_title("prior KL")
        axes[2].legend()
    if log_scale:
        for a in axes:
            a.set_yscale("log")
    for a, panel in ((axes[0], [elbo_train]), (axes[1], [objective_val]),
                     (axes[2], [kl_train, kl_val])):
        tails = [series[start:] for series in panel
                 if series and (start := _after_ramp(model, len(series))) is not None]
        if tails:
            _fit_y(a, np.concatenate(tails), log_scale)
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
