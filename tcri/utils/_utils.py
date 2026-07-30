from __future__ import print_function, division
from .. import _keys as K
import os
import sys
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
from scipy.stats import fisher_exact#, binom_test
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

from contextlib import contextmanager

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


# stars / auc_and_label_permutation / bootstrap_auc → moved to tcri/_stats.py (PR1).


def _ensure_pyro_posterior_params(model, adata) -> None:
    import pyro, torch
    from pyro.distributions import constraints
    from torch import nn

    store = pyro.get_param_store()
    if "q_p_ct_raw" in store:
        return

    device = next(model.module.parameters()).device

    # infer ct_count
    if hasattr(model.module, "ct_count"):
        ct_count = int(model.module.ct_count)
    elif hasattr(model.module, "ct_to_cov"):
        ct_count = int(model.module.ct_to_cov.shape[0])
    elif hasattr(model.module, "ct_array"):
        ct_count = int(model.module.ct_array.max().item() + 1)
    else:
        raise RuntimeError("Could not infer ct_count from the model.")

    # infer P via classifier forward (most reliable)
    try:
        z = model.get_latent_representation(batch_size=8)
        with torch.no_grad():
            logits = model.module.classifier(torch.from_numpy(z[:1]).to(device))
        P = int(logits.shape[-1])
    except Exception:
        # fallback to adata phenotypes
        reg = getattr(model, "adata_manager", None)
        phen_col = reg.registry.get("phenotype_col") if (reg and hasattr(reg, "registry")) else None
        if phen_col and phen_col in adata.obs:
            P = int(adata.obs[phen_col].astype("category").cat.categories.size)
        else:
            raise RuntimeError("Could not infer P from classifier or adata.")

    _warnings.warn(
        "Pyro param store has no 'q_p_ct_raw'; re-initializing it to a uniform "
        "1/P simplex. Downstream posterior metrics (joint_distribution_posterior, "
        "phenotypic/clonotypic entropy, mutual information) will run on this "
        "uninformative prior instead of the trained posterior. This usually means "
        "the Pyro param store failed to load or was never saved; verify the model's "
        "Pyro params were persisted and restored (e.g. via save_tcri_session / "
        "load_tcri_session).",
        RuntimeWarning,
        stacklevel=2,
    )
    init = torch.full((ct_count, P), 1.0 / P, device=device)
    pyro.param("q_p_ct_raw", init, constraint=constraints.simplex)


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

from contextlib import contextmanager

@contextmanager
def _disable_scvi_onload_train():
    """
    Temporarily monkey-patch scvi's PyroBaseModuleClass.on_load to avoid
    the one-step warmup train that triggers EarlyStopping('elbo_validation').

    Compatible across scvi variants that pass different kwargs (e.g., pyro_param_store).
    """
    candidates = []
    try:
        from scvi.module.base import _base_module as _scvi_bm
        if hasattr(_scvi_bm, "PyroBaseModuleClass"):
            candidates.append(_scvi_bm)
    except Exception:
        pass
    # Some installations have the class re-exported elsewhere; try other modules if needed
    try:
        from scvi.module.base import _pyromodule as _scvi_pm  # may not exist in all versions
        if hasattr(_scvi_pm, "PyroBaseModuleClass"):
            candidates.append(_scvi_pm)
    except Exception:
        pass

    # If we couldn't import anything, just yield and hope load works.
    if not candidates:
        yield
        return

    import pyro as _pyro

    def _noop(self, *args, **kwargs):
        # Accept any args/kwargs (e.g., pyro_param_store) but do nothing.
        # Keep Pyro store clean; we'll load our own params after model.load().
        _pyro.clear_param_store()
        # If a store was provided, we intentionally ignore it here.

    origs = []
    try:
        for mod in candidates:
            cls = getattr(mod, "PyroBaseModuleClass", None)
            if cls is None:
                continue
            orig = getattr(cls, "on_load", None)
            if orig is not None:
                origs.append((cls, orig))
                setattr(cls, "on_load", _noop)
        yield
    finally:
        for cls, orig in origs:
            setattr(cls, "on_load", orig)


def _pyro_load(path, map_location=None):
    # torch>=2.6 defaults torch.load(weights_only=True), which rejects the
    # constraint instances pyro stores. Self-produced artifacts only.
    state = _torch.load(path, map_location=map_location, weights_only=False)
    _pyro.get_param_store().set_state(state)


def load_tcri_session(
    run_dir: str,
    *,
    adata_path: Optional[str] = None,
    map_location: Optional[str] = None,
    layer: Optional[str] = None,
):
    TCRIModel = _resolve_TCRIModel()

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

    # 3) Load model WITHOUT scvi's warmup train
    with _disable_scvi_onload_train():
        model = TCRIModel.load(run_dir, adata=adata)

    # 4) Restore Pyro param store
    pyro_file = _os.path.join(run_dir, PYRO_FILE)
    if _os.path.exists(pyro_file):
        try:
            _pyro.clear_param_store()
            if map_location is not None:
                try:
                    _pyro_load(pyro_file, map_location=map_location)
                except TypeError:
                    _pyro_load(pyro_file)
                    if map_location != "cpu":
                        device = _torch.device(map_location)
                        for k, v in list(_pyro.get_param_store().items()):
                            _pyro.get_param_store()[k] = v.to(device)
            else:
                _pyro_load(pyro_file)
        except Exception as e:
            _warnings.warn(f"Could not load Pyro param store: {e}")
    
    _ensure_pyro_posterior_params(model, adata)
    return model, adata





tcri_colors = [
    "#272822",  # Background
    "#AE81FF",  # Purple
    "#FD971F",  # Orange
    "#E6DB74",  # Yellow
    "#A6E22E",  # Green
    "#66D9EF",  # Blue
    "#75715E",  # Brown
    "#F92659",  # Pink
    "#D65F0E",  # Abricos
    "#F92672",  # Red
    "#1E1E1E",   # Black
    "#004d47",  # Darker Teal
    "#D291BC",  # Soft Pink
    "#3A506B",  # Dark Slate Blue
    "#5D8A5E",  # Sage Green
    "#A6A1E2",  # Dull Lavender
    "#E97451",  # Burnt Sienna
    "#6C8D67",  # Muted Lime Green
    "#832232",  # Dim Maroon
    "#669999",  # Desaturated Cyan
    "#C08497",  # Dusty Rose
    "#587B7F",  # Ocean Blue
    "#9A8C98",  # Muted Purple
    "#F28E7F",  # Salmon
    "#F3B61F",  # Goldenrod
    "#6A6E75",  # Iron Gray
    "#FFD8B1",  # Light Peach
    "#88AB75",  # Moss Green
    "#C38D94",  # Muted Rose
    "#6D6A75",  # Purple Gray
]

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

    # 4) Save the AnnData (plain h5ad; the manager stash is retired at setup_anndata)
    if save_adata:
        ad_path = _os.path.join(out_dir, AD_FILE)
        adata.uns.pop(K.LEGACY_MANAGER, None)  # defensive: never serialize a stray AnnDataManager
        adata.write_h5ad(ad_path, compression=compression)
        paths["adata"] = ad_path

    # 5) Meta / versions
    meta = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "var_names_hash": str(_pd.util.hash_pandas_object(_pd.Index(adata.var_names)).sum()),
        "versions": {
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
