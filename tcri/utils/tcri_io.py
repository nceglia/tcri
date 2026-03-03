
# tcri_io.py
# Utilities to save and load a TCRI run (model + AnnData) safely.
# - Avoids the non-picklable `tcri_manager` in adata.uns.
# - Persists the Pyro param store so posterior quantities (e.g., p_ct) can be recovered.
#
# Usage
# -----
# from tcri_io import save_tcri_session, load_tcri_session
#
# # After training
# save_tcri_session(model, adata, "runs/my_experiment")
#
# # Later (new session)
# model, adata = load_tcri_session("runs/my_experiment")  # returns a ready-to-use model + adata
#
# Notes
# -----
# * We rely on scvi-tools' BaseModelClass.save/load to serialize the module and registry.
# * We separately save/load Pyro's param store (for variational/posterior params).
# * We store setup metadata (column names, categories) to ensure category order is restored.
# * We write a sanitized .h5ad without `tcri_manager` so it’s portable.
#
# If you ever need to write adata again after loading, use `write_adata_safely(adata, path)`
# to avoid the same pickle issue.
#
import json
import os
import warnings
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import pyro

# Optional imports (only for types / nicer errors). The TCRIModel class lives in your codebase.
try:
    from anndata import AnnData
except Exception as e:
    raise RuntimeError("AnnData is required. Install `anndata` / `scanpy`.") from e

# ---- Filenames / layout -----------------------------------------------------
AD_FILE = "adata.h5ad"
SETUP_FILE = "setup.json"
PYRO_FILE = "pyro_params.pt"
META_FILE = "meta.json"

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _to_jsonable(x: Any) -> Any:
    \"\"\"Best-effort conversion to JSON-serializable (for init params/meta).\"\"\"
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(y) for y in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    # Torch tensors
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except Exception:
        pass
    # Fallback to string repr
    return str(x)

# ----------------------- AnnData safe writing --------------------------------
def _pop_nonserializables(adata: "AnnData") -> Dict[str, Any]:
    \"\"\"Remove obviously non-serializable objects from adata.uns (returns sidecar).

    Currently we know `adata.uns['tcri_manager']` cannot be pickled; drop it.
    If you'd like to protect other custom objects, add them here.
    \"\"\"
    sidecar = {}
    if "tcri_manager" in adata.uns:
        sidecar["tcri_manager"] = "dropped (AnnDataManager is not serializable)"
        adata.uns.pop("tcri_manager")
    return sidecar

def write_adata_safely(adata: "AnnData", path: str, *, compression: str = "gzip") -> None:
    \"\"\"Write an AnnData to .h5ad ensuring non-serializables are removed first.\"\"\"
    # Work on a shallow copy of .uns to avoid modifying the live object unintentionally
    # (we'll restore later if we removed something).
    removed = {}
    try:
        removed = _pop_nonserializables(adata)
        adata.write_h5ad(path, compression=compression)
    finally:
        # no-op, we don't restore the manager by default because it is tied to a Python session.
        # If you want to keep it around in-memory, adjust here.
        pass

# ------------------------ Setup metadata helpers -----------------------------
def _collect_setup_from_adata_or_model(adata: "AnnData", model: Any) -> Dict[str, Any]:
    \"\"\"Gather minimal setup info needed to re-run setup_anndata and restore category order.\"\"\"
    setup: Dict[str, Any] = {}
    # Preferred: metadata previously stored by your preprocessing code
    meta = adata.uns.get("tcri_metadata", {})
    if meta:
        setup.update({
            "phenotype_col": meta.get("phenotype_col"),
            "clone_col": meta.get("clone_col"),
            "covariate_col": meta.get("covariate_col"),
            "batch_col": meta.get("batch_col"),
        })
    # Category order (if available)
    for key in ("phenotype", "clonotype", "covariate"):
        cats_key = f"tcri_{key}_categories"
        if cats_key in adata.uns:
            setup[cats_key] = list(map(str, adata.uns[cats_key]))

    # Layer name (default to X)
    setup["layer"] = adata.uns.get("tcri_layer", "X")

    # As a fallback, try to pull from the model's adata_manager registry if present
    # (not strictly necessary post-training, but useful if metadata wasn't stored).
    try:
        reg = getattr(model, "adata_manager", None)
        if reg is not None and hasattr(reg, "registry"):
            r = reg.registry
            setup.setdefault("phenotype_col", r.get("phenotype_col"))
            setup.setdefault("clone_col", r.get("clonotype_col"))
            setup.setdefault("covariate_col", r.get("covariate_col"))
            setup.setdefault("batch_col", r.get("batch_col"))
            # If the layer was registered as a LayerField, it might appear under r["X"]["layer"] in some versions.
            if isinstance(r.get("X"), dict) and "layer" in r["X"]:
                setup["layer"] = r["X"]["layer"] or "X"
    except Exception:
        pass

    return setup

def _restore_category_order(adata: "AnnData", setup: Dict[str, Any]) -> None:
    \"\"\"Ensure that obs categorical columns use the same category order as during training.\"\"\"
    mapping = [
        ("phenotype_col", "tcri_phenotype_categories"),
        ("clone_col", "tcri_clonotype_categories"),
        ("covariate_col", "tcri_covariate_categories"),
    ]
    for col_key, cats_key in mapping:
        col = setup.get(col_key)
        cats = setup.get(cats_key)
        if not col or not cats or col not in adata.obs:
            continue
        # Cast to string categories to be robust across saved types
        adata.obs[col] = pd.Categorical(adata.obs[col].astype(str), categories=[str(c) for c in cats], ordered=True)

# --------------------------- Save / Load -------------------------------------
def save_tcri_session(
    model: Any,
    adata: "AnnData",
    out_dir: str,
    *,
    save_adata: bool = True,
    compression: str = "gzip",
) -> Dict[str, Any]:
    \"\"\"Save a trained TCRI model and its AnnData to `out_dir`.

    What gets written:
      - Model (via scvi BaseModelClass.save): weights & registry
      - Pyro param store:  {out_dir}/pyro_params.pt
      - AnnData (sanitized): {out_dir}/adata.h5ad
      - Setup metadata: {out_dir}/setup.json
      - Meta/version info: {out_dir}/meta.json

    Returns a dict of paths written.
    \"\"\"
    _ensure_dir(out_dir)

    # 1) Persist the model via scvi-tools BaseModelClass.save
    #    (Do not embed adata inside the saved model; we write it separately.)
    paths: Dict[str, Any] = {}
    if hasattr(model, "save"):
        model.save(out_dir, overwrite=True, save_anndata=False)
        paths["model_dir"] = out_dir
    else:
        raise RuntimeError("Expected `model.save` (scvi BaseModelClass) to exist on TCRIModel.")

    # 2) Persist Pyro param store (needed for posterior quantities like p_ct)
    try:
        pyro.get_param_store().save(os.path.join(out_dir, PYRO_FILE))
        paths["pyro"] = os.path.join(out_dir, PYRO_FILE)
    except Exception as e:
        warnings.warn(f"Could not save Pyro param store: {e}")

    # 3) Collect setup info and write it
    setup = _collect_setup_from_adata_or_model(adata, model)
    with open(os.path.join(out_dir, SETUP_FILE), "w") as f:
        json.dump(setup, f, indent=2)
    paths["setup"] = os.path.join(out_dir, SETUP_FILE)

    # 4) Write AnnData (sanitized)
    if save_adata:
        write_adata_safely(adata, os.path.join(out_dir, AD_FILE), compression=compression)
        paths["adata"] = os.path.join(out_dir, AD_FILE)

    # 5) Meta / versions
    meta = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "var_names_hash": str(pd.util.hash_pandas_object(pd.Index(adata.var_names)).sum()),
        "versions": {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "anndata": getattr(sc, "__version__", "unknown"),
            "scanpy": getattr(sc, "__version__", "unknown"),
            "torch": getattr(torch, "__version__", "unknown"),
            "pyro": getattr(pyro, "__version__", "unknown"),
        },
    }
    try:
        import scvi
        meta["versions"]["scvi-tools"] = getattr(scvi, "__version__", "unknown")
    except Exception:
        meta["versions"]["scvi-tools"] = "unknown"

    with open(os.path.join(out_dir, META_FILE), "w") as f:
        json.dump(meta, f, indent=2)
    paths["meta"] = os.path.join(out_dir, META_FILE)

    return paths

def load_tcri_session(
    run_dir: str,
    *,
    adata_path: Optional[str] = None,
    map_location: Optional[str] = None,
) -> Tuple[Any, "AnnData"]:
    \"\"\"Load a saved TCRI model + AnnData from `run_dir` and return (model, adata).

    Steps:
      1) Read adata (sanitized) from {run_dir}/adata.h5ad (or `adata_path` if provided)
      2) Restore category order and re-run TCRIModel.setup_anndata(adata, ...)
      3) Load the model via TCRIModel.load(run_dir, adata=adata)
      4) Restore Pyro param store from {run_dir}/pyro_params.pt (if available)

    Returns
    -------
    model : TCRIModel
    adata : AnnData
    \"\"\"
    from _model import TCRIModel  # your class

    # 1) Load adata
    ad_file = adata_path or os.path.join(run_dir, AD_FILE)
    if not os.path.exists(ad_file):
        raise FileNotFoundError(f"Could not find adata file at: {ad_file}")
    adata = sc.read_h5ad(ad_file)

    # 2) Setup metadata
    setup: Dict[str, Any] = {}
    setup_file = os.path.join(run_dir, SETUP_FILE)
    if os.path.exists(setup_file):
        with open(setup_file, "r") as f:
            setup = json.load(f)
    else:
        # Fallback: attempt to build it from adata only
        warnings.warn("setup.json not found; attempting to infer from adata.uns['tcri_metadata'].")
        setup = _collect_setup_from_adata_or_model(adata, model=None)

    # Restore category order if we have it
    _restore_category_order(adata, setup)

    # Re-run setup_anndata to (re)create a fresh manager inside this session
    # (Avoids storing a non-serializable manager inside the .h5ad)
    layer = setup.get("layer", "X")
    TCRIModel.setup_anndata(
        adata,
        layer=layer,
        clonotype_key=setup.get("clone_col", "unique_clone_id"),
        phenotype_key=setup.get("phenotype_col", "phenotype_col"),
        covariate_key=setup.get("covariate_col", "timepoint"),
        batch_key=setup.get("batch_col", "patient"),
    )

    # 3) Load the model via scvi BaseModelClass.load (does not trigger training)
    if hasattr(TCRIModel, "load"):
        model = TCRIModel.load(run_dir, adata=adata)
    else:
        raise RuntimeError("Expected `TCRIModel.load` to exist (scvi BaseModelClass).")

    # 4) Restore Pyro param store so posterior quantities (e.g., get_p_ct) use learned params
    pyro_file = os.path.join(run_dir, PYRO_FILE)
    if os.path.exists(pyro_file):
        try:
            pyro.clear_param_store()
            # Attempt to load with map_location if provided (Pyro >= 1.8 supports kwargs passthrough)
            if map_location is not None:
                try:
                    pyro.get_param_store().load(pyro_file, map_location=map_location)
                except TypeError:
                    # Older Pyro: no map_location argument
                    pyro.get_param_store().load(pyro_file)
                    if map_location != "cpu":
                        # Best-effort move to device
                        device = torch.device(map_location)
                        for k, v in list(pyro.get_param_store().items()):
                            pyro.get_param_store()[k] = v.to(device)
            else:
                pyro.get_param_store().load(pyro_file)
        except Exception as e:
            warnings.warn(f"Could not load Pyro param store: {e}")

    return model, adata
