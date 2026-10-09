"""Reconstruct a fitted null for inference, from the parent plus the AnnData.

A null is fully recoverable from the pair (parent session, AnnData) and nothing else: its
parameters live in the process-global store under its own namespace, which
``save_tcri_session`` of the PARENT already writes whole; its permutation, buffers and settings
travel in ``uns``. So the model object ``tcri.null.*`` returns is a convenience, never a
requirement, and there is no per-null session to save, find or keep in step.

This is what lets ``perturb.gene_importance`` take ``null_model="auto"`` like every metric: the
perturbation is a query on a MODEL rather than on a substrate, so it needs the null back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyro
import torch

from .._state import keys as K


def rebuild(model, adata, fit):
    """Return the fitted null named ``fit`` as a :class:`TCRIModel`.

    Built from the parent's ``init_params_`` with ``name`` and ``permutation`` replaced from the
    substrate, then given the buffers from ``uns``, the parameters from the store, and the
    training history and record its fit stored, for ``tcri.diag.loss``. It trains nothing and
    it fits nothing: if the store has no such namespace it raises, because a rebuild on fresh
    parameters would return a randomly initialized model that answers every question without
    complaint.
    """
    from ..model._model import TCRIModel, adopt, expect_params
    from ._nulls import _init_params

    fit = K.resolve_fit(adata, fit)
    if fit is None:
        raise ValueError("rebuild() needs a named fit; the main fit is the model you already have.")

    settings = adata.uns.get(K.fit_key(K.FIT_SETTINGS, fit))
    perm = adata.uns.get(K.fit_key(K.PERMUTATION, fit))
    if settings is None or perm is None:
        raise KeyError(
            f"no permutation record for fit {fit!r} in this AnnData. Only a fit built by "
            f"tcri.null.* can be rebuilt; a hand-written fit has to be passed as a model object."
        )
    # `.get(x) or y` is deliberately avoided on anything that has been through h5ad: a stored
    # list comes back as a numpy array and its truth value raises. These two are scalars today,
    # so the explicit form is about keeping one rule in this file rather than two.
    namespace = settings.get("namespace")
    namespace = fit if namespace is None else str(namespace)
    axis = settings.get("axis")
    axis = str(settings.get("kind")) if axis is None else str(axis)

    from ..model._model import _is_generated_name
    from ..model._module import _null_namespace

    # Exactly this namespace: a key_added sibling's keys (`<namespace>.b.…`) share the prefix.
    held = [k for k in pyro.get_param_store().keys() if _null_namespace(k) == namespace]
    parent = settings.get("parent")
    if not held and parent is not None and str(parent) != str(model.name):
        kind = settings.get("kind")
        keyed = str(settings.get("key_added", "") or "")
        main = str((adata.uns.get(K.FIT_SETTINGS) or {}).get("name", ""))
        if (not str(parent) or _is_generated_name(parent)) and _is_generated_name(model.name):
            remedy = (f"Refit it for this model with tcri.null.{kind}(model, adata"
                      + (f", key_added={keyed!r}" if keyed else "") + ")"
                      + ("" if main == str(model.name) else
                         ", after writing this model's main fit with model.to_anndata(adata)")
                      + ".")
        else:
            remedy = (f"Fit this model's own beside it with tcri.null.{kind}(model, adata, "
                      f"key_added=...).")
        raise RuntimeError(
            f"the store holds no parameters for {namespace!r}: fit {fit!r} on this object "
            f"belongs to parent {str(parent)!r}, not to this model ({model.name!r}). {remedy}"
        )
    restored = getattr(model.module, "_restored_nulls", None)
    if not held and restored is not None and namespace not in restored:
        raise RuntimeError(
            f"the store holds no parameters for {namespace!r}: this model was loaded from a "
            f"save that does not include that null, which was fitted after the save was made. "
            f"Its record in this AnnData is intact and metrics still read it; to rebuild the "
            f"model, refit it with tcri.null.{settings.get('kind')}(model, adata), or load a "
            f"save made after it."
        )
    if not held:
        raise RuntimeError(
            f"the store holds no parameters for {namespace!r}; load the session that fitted it "
            f"with tcri.utils.load_tcri_session, or refit with tcri.null.{settings.get('kind')}"
            f"(model, adata)."
        )

    # Register the parent's setup on THIS object first. Without it a rebuild on a derived
    # AnnData -- a copy, a slice, a zeroed matrix, which is the ordinary way
    # `perturb.gene_importance` is called -- fails in scvi's constructor with "setup with a
    # different model", naming a cause that has nothing to do with the request.
    adata = adopt(model, adata)

    with expect_params(namespace):
        null = TCRIModel(adata, name=namespace,
                         permutation=(axis, np.asarray(perm, dtype=np.int64)),
                         **_init_params(model))

    # Buffers first: BatchNorm running statistics are learned during the fit and are the one
    # piece of module state the constructor cannot re-derive from the permutation.
    saved = adata.uns.get(K.fit_key(K.BUFFERS, fit)) or {}
    live = dict(null.module.named_buffers())
    for name, value in saved.items():
        current = live.get(name)
        if current is None:
            continue
        value = torch.as_tensor(np.asarray(value))
        if tuple(value.shape) != tuple(current.shape):
            raise ValueError(
                f"buffer {name!r} of fit {fit!r} was saved with shape {tuple(value.shape)} but "
                f"this rebuild has {tuple(current.shape)}. The AnnData and the parent do not "
                f"describe the same fit."
            )
        current.copy_(value.to(dtype=current.dtype, device=current.device))

    # Then the parameters, from the store into the module's own tensors.
    pyro.module(null.module.pname("scvi"), null.module, update_module_params=True)
    # The parameters now sit on whatever device the store was loaded onto; the buffers copied above
    # are still on the constructor's. Moving the module puts both on one device.
    null.module.to(next(null.module.parameters()).device)
    null.module.eval()
    null.is_trained_ = True
    null._fit_name = fit

    # The curves and the record the fit left, so `tcri.diag.loss` reads a rebuilt null as it
    # reads the one `tcri.null.*` returned.
    training = adata.uns.get(K.fit_key(K.TRAINING, fit))
    if training is not None:
        history = training.get("history")
        null.history_ = {} if history is None else {
            str(name): pd.DataFrame(
                {str(name): np.asarray(series["value"], dtype=float)},
                index=pd.Index(np.asarray(series["epoch"], dtype=np.int64), name="epoch"))
            for name, series in history.items()
        }
        record = training.get("record")
        null.training_record_ = {} if record is None else {
            str(k): v.item() if isinstance(v, np.generic) else v for k, v in record.items()
        }
    return null
