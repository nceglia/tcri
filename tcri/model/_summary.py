"""The model's panel: printed when ``train()`` ends, and shown when the model is displayed.

The panel is a list of sections, :data:`SECTIONS`. Each is a small function of the model that
returns one ``(label, value)`` row, or ``None`` to be left out, so a section can be added,
removed or reordered without touching the others. Every value is read from the model and its
``training_record_``, so the panel printed at the end of a fit and the one shown later agree.
"""
from __future__ import annotations

from rich.text import Text

from .. import _output
from .._state import keys as K

#: How a fit ended, by ``training_record_["stop_reason"]``, and whether that is a flagged state.
_ENDED = {
    "patience": ("stopped early (patience)", False),
    "max_epochs": ("reached max_epochs", True),
    "non-finite objective": ("stopped: non-finite objective", True),
    "other limit": ("ended by another limit", True),
    "error": ("ended by an error", True),
}


def _record(model) -> dict:
    return getattr(model, "training_record_", None) or {}


def _training(model):
    rec = _record(model)
    if not rec:
        return "Training", Text("not trained")
    args = getattr(model, "train_kwargs_", None) or {}
    stopping = bool(args.get("early_stopping", True))
    ended, flagged = _ENDED.get(rec.get("stop_reason"), ("", False))
    if rec.get("stop_reason") == "max_epochs" and not stopping:
        ended, flagged = "ran max_epochs (early stopping off)", False
    first = Text(f"{int(rec['epochs_run']):,} epochs")
    if ended:
        first.append(" · ")
        first.append(ended, style="bold" if flagged else "")
    best, score = rec.get("selected_epoch"), rec.get("selected_score")
    second = (f"best epoch {int(best):,} · objective {_output.cell(score)}"
              if best is not None and score is not None else "no checkpoint selected")
    if stopping and args.get("early_stopping_patience") is not None:
        second += f" · patience {int(args['early_stopping_patience']):,}"
    seed = rec.get("call_seed", rec.get("seed"))
    if seed is not None:
        second += f" · seed {int(seed)}"
    return "Training", Text.assemble(first, "\n", second)


def _kl_ramp(model):
    rec = _record(model)
    if not rec:
        return None
    n = int(rec["n_epochs_kl_warmup"])
    if n == 0:
        return "KL ramp", Text("off")
    if rec.get("ramp_completed"):
        at = rec.get("ramp_completes_at_epoch")
        where = f" at epoch {int(at):,}" if at else " before this call"
        return "KL ramp", Text(f"{n:,} epochs · complete{where}")
    taken = float(rec.get("warmup_epochs_taken", 0.0))
    return "KL ramp", Text(f"{taken:.3g} of {n:,} epochs · did not complete", style="bold")


def _data(model):
    registry = model.adata_manager.registry
    obs = model.adata.obs
    lines = [f"{model.adata.n_obs:,} cells · "
             f"{obs[registry['clonotype_col']].nunique():,} clones · "
             f"{obs[registry['phenotype_col']].nunique():,} phenotypes"]
    covariate = registry["covariate_col"]
    levels = list(obs[covariate].astype("category").cat.categories)
    shown = ", ".join(map(str, levels[:6])) + (", …" if len(levels) > 6 else "")
    second = f"{covariate}: {shown}"
    replicate = registry.get(K.Config.REPLICATE)
    if replicate:
        second += f" · {replicate}: {obs[replicate].nunique():,} replicates"
    lines.append(second)
    return "Data", Text("\n".join(lines))


def _device(model):
    device = _record(model).get("device")
    return None if device is None else ("Device", Text(str(device)))


def _time(model):
    seconds = _record(model).get("duration_s")
    return None if seconds is None else ("Time", Text(_output.duration(seconds)))


def _nulls(model):
    """One line per reference fitted against this model on its AnnData: the fits, the epochs
    they ran, how many reached ``max_epochs`` or left the KL ramp incomplete, and the time, in
    all and per fit (from each null's stored training record)."""
    adata = model.adata
    references: dict = {}
    for fit in K.fits(adata):
        settings = adata.uns.get(K.fit_key(K.FIT_SETTINGS, fit)) or {}
        if str(settings.get("parent", "")) != str(model.name) or "kind" not in settings:
            continue
        record = dict((adata.uns.get(K.fit_key(K.TRAINING, fit)) or {}).get("record") or {})
        # reaching max_epochs is a condition only for a fit that had early stopping on
        record["_stopping"] = bool((settings.get("train") or {}).get("early_stopping", True))
        references.setdefault(str(settings.get("reference", settings["kind"])), []).append(record)
    if not references:
        return None
    lines = Text()
    for reference, records in references.items():
        if len(lines):
            lines.append("\n")
        epochs = [int(r.get("epochs_run", 0)) for r in records]
        span = (f"{min(epochs):,}" if min(epochs) == max(epochs)
                else f"{min(epochs):,}–{max(epochs):,}")
        lines.append(f"{reference}: {len(records)} fit{'s' if len(records) != 1 else ''} · "
                     f"{span} epochs")
        capped = sum(r["_stopping"] and str(r.get("stop_reason")) == "max_epochs" for r in records)
        ramp = sum(not bool(r.get("ramp_completed", True)) for r in records)
        for count, label in ((capped, "reached max_epochs"), (ramp, "ramp incomplete")):
            if count:
                lines.append(" · ")
                lines.append(f"{count} {label}", style="bold")
        seconds = [float(r["duration_s"]) for r in records if r.get("duration_s") is not None]
        if seconds:
            lines.append(f" · {_output.duration(sum(seconds))} "
                         f"({_output.duration(sum(seconds) / len(seconds))} per fit)")
    return "Nulls", lines


def _levels(values, limit=6) -> str:
    levels = [str(v) for v in values]
    return ", ".join(levels[:limit]) + (", …" if len(levels) > limit else "")


def setup_panel(adata, *, layer, clonotype_key, phenotype_key, covariate_key, batch_key,
                replicate):
    """The panel ``setup_anndata(print=True)`` prints: what was registered, with counts."""
    obs = adata.obs
    clonotypes = obs[clonotype_key].astype(str)
    pools = int(clonotypes.str.startswith("pooled@").groupby(clonotypes).first().sum())
    phenotypes = obs[phenotype_key].astype("category").cat.categories
    covariates = obs[covariate_key].astype("category").cat.categories
    rows = [
        ("Data", Text(f"{adata.n_obs:,} cells · {adata.n_vars:,} genes · counts in "
                      f"{'layer ' + repr(layer) if layer is not None else 'X'}")),
        ("Clonotype", Text(f"{clonotype_key} · {clonotypes.nunique():,} clones"
                           + (f", {pools:,} of them pools" if pools else ""))),
        ("Phenotype", Text(f"{phenotype_key} · {len(phenotypes):,}: {_levels(phenotypes)}")),
        ("Covariate", Text(f"{covariate_key} · {len(covariates):,}: {_levels(covariates)}")),
        ("Batch", Text(f"{batch_key} · {obs[batch_key].nunique():,} batches")),
        ("Replicate", Text(f"{replicate} · {obs[replicate].nunique():,} replicates")
         if replicate else Text("not registered: metrics need groupby=")),
    ]
    return _output.panel("setup_anndata", rows)


#: The panel's sections, in order. Add, remove or reorder here.
SECTIONS = [_training, _kl_ramp, _data, _device, _time, _nulls]


def model_panel(model):
    """The model's panel, as a Rich renderable."""
    rows = [section(model) for section in SECTIONS]
    return _output.panel(f"TCRIModel '{model.name}'", [r for r in rows if r is not None])
