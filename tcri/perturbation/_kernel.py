"""The one place ``tcri.perturb`` touches the fitted model.

Everything here is a deterministic function of the fitted parameters and an expression
matrix: the encoder's posterior mean, the head, and the model's own rule for combining the head
with the group prior, exactly as :meth:`TCRIModel.predict` computes it. Silencing a gene is
zeroing its column of ``x`` before the encoder. No encoder sample is ever drawn: the head reads
the posterior mean by contract, and a sample through it would measure the head's response to
noise it was never trained on, not to the gene.

The only stochastic element is the group prior ``p_ct``, which at ``n_samples > 0`` is drawn
from the guide's Dirichlet posterior the same way the metrics draw it (``_compute._joint``),
and enters only through the gate.
"""
from __future__ import annotations

import numpy as np
import torch
from scvi import REGISTRY_KEYS

from ..model._priors import encoder_posterior

__all__: list[str] = []  # private module

#: The floor ``predict()`` puts under the group prior before taking its log.
_EPS = 1e-8


def resolve_genes(adata, genes):
    """``genes`` -> ``(positions, names)`` in the caller's order, deduplicated.

    ``None`` is every gene of the object, in ``var_names`` order. Otherwise a name, a
    position, or a list of either. Unknown names raise ``KeyError`` naming them; a position
    outside ``[0, n_vars)`` raises ``IndexError``.
    """
    var_names = adata.var_names
    if genes is None:
        return list(range(adata.n_vars)), [str(v) for v in var_names]
    if not var_names.is_unique:
        raise ValueError(
            "adata.var_names are not unique, so a gene name does not identify a column; "
            "call adata.var_names_make_unique() first."
        )
    if isinstance(genes, (str, int, np.integer)):
        genes = [genes]
    positions, seen, missing = [], set(), []
    for g in genes:
        if isinstance(g, (int, np.integer)) and not isinstance(g, (bool, np.bool_)):
            j = int(g)
            if j < 0 or j >= adata.n_vars:
                raise IndexError(f"gene position {j} is out of range for {adata.n_vars} genes")
        else:
            name = str(g)
            if name not in var_names:
                missing.append(name)
                continue
            j = int(var_names.get_loc(name))
        if j not in seen:
            seen.add(j)
            positions.append(j)
    if missing:
        raise KeyError(
            f"{len(missing)} gene(s) not in adata.var_names: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    return positions, [str(var_names[j]) for j in positions]


def module_device(module):
    return next(module.parameters()).device


def prior_draws(module, *, n_samples, random_state):
    """The group prior as a ``[S', n_ct, P]`` float32 tensor on the module's device.

    ``S' = 1`` at ``n_samples = 0``: the posterior mean, which is what ``predict()`` uses.
    Otherwise ``n_samples`` draws from ``Dirichlet(λ'_m)``, the guide's own concentration
    (``get_conc_ct``), seeded through the same helper the metrics engine uses. The draw is
    made on the CPU in float64 -- MPS has no float64 -- and moved afterwards.
    """
    from .._compute._joint import _torch_seed  # lazy: the engine is a sibling layer

    if n_samples and int(n_samples) > 0:
        conc = module.get_conc_ct().detach().cpu().double()
        with _torch_seed(random_state):
            draws = torch.distributions.Dirichlet(conc).sample((int(n_samples),))
    else:
        draws = module.get_p_ct().detach().cpu().double().unsqueeze(0)
    return draws.to(device=module_device(module), dtype=torch.float32)


def head_logits(module, x, b, zero_cols=None):
    """``f_cls(μ_i)`` for a batch, with the columns in ``zero_cols`` silenced first."""
    if zero_cols is not None and len(zero_cols):
        x = x.clone()
        x[:, list(zero_cols)] = 0.0
    mu, _ = encoder_posterior(module.encoder, x, b)
    return module.classifier(mu)


def combine(cls_logits, log_prior, gate_prob, use_gate):
    """``softmax`` of the model's own rule -> ``[S', B, P]``.

    ``use_gate=True`` is the rule ``predict()`` applies: the gate when the model has one,
    the additive rule when ``gate_prob`` is ``None``. ``use_gate=False`` is the head alone,
    ``softmax(f_cls(μ_i))``, which does not depend on the prior at all.
    """
    if not use_gate:
        return torch.softmax(cls_logits, dim=-1).unsqueeze(0)
    ell = cls_logits.unsqueeze(0)
    if gate_prob is None:
        ell = ell + log_prior
    else:
        ell = float(gate_prob) * ell + (1.0 - float(gate_prob)) * log_prior
    return torch.softmax(ell, dim=-1)


def batches(model, adata, *, batch_size):
    """Yield ``(x, b, global_idx, n)`` per sequential minibatch, on the module's device.

    The loader is sequential, so batches follow ``obs`` row order -- the same property
    ``predict()`` relies on to align its output to ``obs_names``. ``global_idx`` is the cell's
    id in the training object, which is what indexes ``ct_array``; it is NOT the row position.
    """
    module = model.module
    device = module_device(module)
    loader = model._make_data_loader(adata=adata, batch_size=batch_size)
    for tensors in loader:
        x = tensors[REGISTRY_KEYS.X_KEY].to(device)
        b = tensors[REGISTRY_KEYS.BATCH_KEY].long().to(device)
        idx = tensors["indices"].long().view(-1).to(device)
        yield x, b, idx, int(x.shape[0])


def log_prior_for(draws, module, idx):
    """``log(p_ct^(s)[g(i)] + eps)`` for the cells ``idx`` -> ``[S', B, P]``."""
    ct = module.ct_array.to(draws.device)[idx]
    return torch.log(draws[:, ct, :] + _EPS)


def group_sums(probs, gid, n_groups):
    """Sum ``[S', B, P]`` probabilities into ``[S', n_groups, P]``, returned in float64 on the CPU.

    The within-batch sum is float32 on the device (a few thousand terms); the across-batch
    accumulation the caller does is float64. Float64 is never asked of the device.
    """
    out = torch.zeros(probs.shape[0], n_groups, probs.shape[-1],
                      dtype=probs.dtype, device=probs.device)
    out.index_add_(1, gid, probs)
    return out.cpu().double()
