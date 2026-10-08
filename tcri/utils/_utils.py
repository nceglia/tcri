from __future__ import print_function, division
from .._state import keys as K
from .._state.storage import _tcri_version
import os
import sys
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
from scipy.stats import fisher_exact
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors


from typing import Optional, Tuple, Dict, Any
import json as _json
import os as _os
import warnings as _warnings

import numpy as _np
import pandas as _pd
import scanpy as _sc
import anndata as _ad
import torch as _torch
import pyro as _pyro

import numpy as np


# `stars`, `auc_and_label_permutation` and `bootstrap_auc` live in tcri/_stats/_core.py.


#: Star-imported by ``tcri/utils/__init__.py``. Without this, `import *` would pull in every
#: module-level name the file happens to bind -- numpy, os, sys, matplotlib, typing
#: aliases, scipy functions, the private path constants -- and ``dir(tcri.ut)`` would
#: advertise them as API. ``governance/API_CONTRACT.md`` cannot catch that: its surface
#: check freezes tcri-DEFINED callables, so a re-exported third-party name is invisible to
#: it by construction.
__all__ = ["save_tcri_session", "load_tcri_session"]


def _resolve_TCRIModel():
    import importlib, importlib.util, os as _os
    # Try common import locations
    for name in ("tcri._model", "tcri.model", "_model"):
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "TCRIModel"):
                return mod.TCRIModel
        except Exception:
            pass
    # Try local sibling files (editable installs)
    here = _os.path.dirname(__file__)
    for rel in ("../_model.py", "../../_model.py"):
        fp = _os.path.normpath(_os.path.join(here, rel))
        if _os.path.exists(fp):
            spec = importlib.util.spec_from_file_location("tcri_model_local", fp)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "TCRIModel"):
                return mod.TCRIModel
    raise ModuleNotFoundError("Could not import TCRIModel.")

def _read_session_meta(run_dir: str) -> Dict[str, Any]:
    """``meta.json`` if it is there and readable, else an empty dict with a warning."""
    meta_file = _os.path.join(run_dir, META_FILE)
    if not _os.path.exists(meta_file):
        return {}
    try:
        with open(meta_file, "r") as f:
            return dict(_json.load(f))
    except Exception as e:
        _warnings.warn(f"Could not read {META_FILE}: {e}")
        return {}


def load_tcri_session(
    run_dir: str,
    *,
    adata_path: Optional[str] = None,
    map_location: Optional[str] = None,
    layer: Optional[str] = None,
):
    TCRIModel = _resolve_TCRIModel()

    # 0) What wrote this session. Checked BEFORE anything is loaded: a session from a newer tcri
    # may need files or keys this one knows nothing about, and reporting that after a partial load
    # is worse than not starting. Read once here; step 5 uses the same dict.
    meta = _read_session_meta(run_dir)
    found = int(meta.get("format_version", 0) or 0)
    if found > SESSION_FORMAT_VERSION:
        raise ValueError(
            f"the session at {run_dir!r} has format v{found}, written by tcri "
            f"{meta.get('versions', {}).get('tcri', 'unknown')}; this tcri "
            f"({_tcri_version()}) reads up to v{SESSION_FORMAT_VERSION}. Upgrade tcri to load it."
        )

    # 1) Load adata
    ad_file = adata_path or _os.path.join(run_dir, AD_FILE)
    if not _os.path.exists(ad_file):
        raise FileNotFoundError(f"Could not find adata file at: {ad_file}")
    adata = _sc.read_h5ad(ad_file)

    # 2) Setup metadata + categorical order
    setup = {}
    setup_file = _os.path.join(run_dir, SETUP_FILE)
    if _os.path.exists(setup_file):
        with open(setup_file, "r") as f:
            setup = _json.load(f)
    else:
        _warnings.warn("setup.json not found; attempting to infer from adata.uns['tcri_metadata'].")
        setup = _collect_setup_from_adata_or_model(adata, model=None)
    _restore_category_order(adata, setup)

    # Rebuild AnnData manager
    setup_layer = layer if layer is not None else setup.get("layer")
    if setup_layer == "X" and "X" not in adata.layers:
        setup_layer = None

    TCRIModel.setup_anndata(
        adata,
        layer=setup_layer,
        clonotype_key=setup.get("clone_col", "unique_clone_id"),
        phenotype_key=setup.get("phenotype_col", "phenotype_col"),
        covariate_key=setup.get("covariate_col", "timepoint"),
        batch_key=setup.get("batch_col", "patient"),
    )

    # 3) The Pyro store saved beside the model, read before anything is loaded: a corrupt file
    # fails the load with the process-global store untouched, rather than leaving a model whose
    # posteriors are not the fit.
    pyro_file = _os.path.join(run_dir, PYRO_FILE)
    state = None
    if _os.path.exists(pyro_file):
        try:
            # torch>=2.6 defaults torch.load(weights_only=True), which rejects the constraint
            # instances pyro stores. Self-produced artifacts only.
            state = _torch.load(pyro_file, map_location=map_location, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"could not read the Pyro store saved at {pyro_file}: {e}") from e

    # 4) Load the model. TCRIModule.on_load restores its store family -- its posteriors and its
    # nulls' parameters -- from the store saved in model.pt and leaves every other model's
    # entries as they are; the same family then comes from pyro_params.pt, onto map_location.
    model = TCRIModel.load(run_dir, adata=adata)
    if state is not None:
        model.module.restore_param_store(state, warn_on_replace=False,
                                         fitted=bool(getattr(model, "is_trained_", True)))

    # 5) The arguments the parent was actually fitted with. `tcri.null.*` replays them, and an
    # in-memory attribute does not survive a reload: without this a null built after a reload
    # would silently fall back to `train()`'s defaults and stop being the parent's model on
    # permuted labels. An older session has no record; `_train_kwargs` stays empty and
    # `tcri.null.*` raises rather than guessing.
    if meta.get("train_kwargs"):
        model._train_kwargs = dict(meta["train_kwargs"])

    return model, adata





import matplotlib.pyplot as plt






# === TCRI IO utilities ==========================================================
# Save / Load a trained TCRIModel together with a sanitized AnnData.
# Avoids serializing a non-picklable AnnDataManager (tcri_manager) by
# writing the .h5ad without it and reconstructing the manager on load.

# Filenames
AD_FILE = "adata.h5ad"
SETUP_FILE = "setup.json"
PYRO_FILE = "pyro_params.pt"
META_FILE = "meta.json"

#: The layout of a saved session: which files it has and which keys a load reads out of them.
#: Bump it when a load needs something an older session does not carry, so a session written by a
#: newer tcri is refused with a message rather than half-loaded. A session that records no
#: version at all is read without the check.
SESSION_FORMAT_VERSION = 1

def _ensure_dir(path: str) -> None:
    _os.makedirs(path, exist_ok=True)

def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(y) for y in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (_np.integer, _np.floating, _np.bool_)):
        return x.item()
    if isinstance(x, _np.ndarray):
        return x.tolist()
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    try:
        import torch as __torch
        if isinstance(x, __torch.Tensor):
            return x.detach().cpu().tolist()
    except Exception:
        pass
    return str(x)



def _collect_setup_from_adata_or_model(adata: "_ad.AnnData", model: Any) -> Dict[str, Any]:
    setup: Dict[str, Any] = {}
    meta = adata.uns.get(K.METADATA, {})
    if meta:
        setup.update({
            "phenotype_col": meta.get("phenotype_col"),
            "clone_col": meta.get("clone_col"),
            "covariate_col": meta.get("covariate_col"),
            "batch_col": meta.get("batch_col"),
        })
    for key in ("phenotype", "clonotype", "covariate"):
        cats_key = f"tcri_{key}_categories"
        if cats_key in adata.uns:
            setup[cats_key] = list(map(str, adata.uns[cats_key]))
    setup["layer"] = adata.uns.get("tcri_layer")
    try:
        reg = getattr(model, "adata_manager", None)
        if reg is not None and hasattr(reg, "registry"):
            r = reg.registry
            setup.setdefault("phenotype_col", r.get("phenotype_col"))
            setup.setdefault("clone_col", r.get("clonotype_col"))
            setup.setdefault("covariate_col", r.get("covariate_col"))
            setup.setdefault("batch_col", r.get("batch_col"))
            setup_args = r.get("setup_args", {})
            if isinstance(setup_args, dict) and "layer" in setup_args:
                setup["layer"] = setup_args["layer"]
            if isinstance(r.get("X"), dict) and "layer" in r["X"]:
                setup["layer"] = r["X"]["layer"]
    except Exception:
        pass
    return setup

def _restore_category_order(adata: "_ad.AnnData", setup: Dict[str, Any]) -> None:
    mapping = [
        ("phenotype_col", K.PHENOTYPE_CATEGORIES),
        ("clone_col", K.CLONOTYPE_CATEGORIES),
        ("covariate_col", K.COVARIATE_CATEGORIES),
    ]
    for col_key, cats_key in mapping:
        col = setup.get(col_key)
        cats = setup.get(cats_key)
        if not col or not cats or col not in adata.obs:
            continue
        adata.obs[col] = _pd.Categorical(
            adata.obs[col].astype(str),
            categories=[str(c) for c in cats],
            ordered=True,
        )

def save_tcri_session(
    model: Any,
    adata: "_ad.AnnData",
    out_dir: str,
    *,
    save_adata: bool = True,
    compression: str = "gzip",
) -> Dict[str, Any]:
    _ensure_dir(out_dir)
    paths: Dict[str, Any] = {}

    # 1) Save the scvi model (weights + registry). Do NOT embed anndata here.
    if hasattr(model, "save"):
        model.save(out_dir, overwrite=True, save_anndata=False)
        paths["model_dir"] = out_dir
    else:
        raise RuntimeError("Expected `model.save` (scvi BaseModelClass) to exist on TCRIModel.")

    # 2) Save Pyro param store
    try:
        _pyro.get_param_store().save(_os.path.join(out_dir, PYRO_FILE))
        paths["pyro"] = _os.path.join(out_dir, PYRO_FILE)
    except Exception as e:
        _warnings.warn(f"Could not save Pyro param store: {e}")

    # 3) Save setup metadata needed to rebuild the AnnData manager on load
    setup = _collect_setup_from_adata_or_model(adata, model)
    with open(_os.path.join(out_dir, SETUP_FILE), "w") as f:
        _json.dump(setup, f, indent=2)
    paths["setup"] = _os.path.join(out_dir, SETUP_FILE)

    # 4) Save the AnnData (plain h5ad; setup_anndata leaves no manager stash in uns)
    if save_adata:
        ad_path = _os.path.join(out_dir, AD_FILE)
        adata.uns.pop(K.LEGACY_MANAGER, None)  # defensive: never serialize a stray AnnDataManager
        adata.write_h5ad(ad_path, compression=compression)
        paths["adata"] = ad_path

    # 5) Meta / versions
    meta = {
        "format_version": SESSION_FORMAT_VERSION,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "var_names_hash": str(_pd.util.hash_pandas_object(_pd.Index(adata.var_names)).sum()),
        # What `train()` actually ran with, so `tcri.null.*` can replay it after a reload.
        # Empty for a model that was loaded and never trained, or a session that recorded none.
        "train_kwargs": dict(getattr(model, "_train_kwargs", {}) or {}),
        "name": str(getattr(model, "name", "")),
        "versions": {
            "tcri": _tcri_version(),
            "python": f"{_os.sys.version_info.major}.{_os.sys.version_info.minor}.{_os.sys.version_info.micro}",
            "anndata": getattr(_ad, "__version__", "unknown"),
            "scanpy": getattr(_sc, "__version__", "unknown"),
            "torch": getattr(_torch, "__version__", "unknown"),
            "pyro": getattr(_pyro, "__version__", "unknown"),
        },
    }

    with open(_os.path.join(out_dir, META_FILE), "w") as f:
        _json.dump(meta, f, indent=2)
    paths["meta"] = _os.path.join(out_dir, META_FILE)
    return paths
