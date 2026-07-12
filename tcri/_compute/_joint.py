"""Batched engine core — the clone×phenotype joint as a ``[S, n_clones, P]`` stack,
device-routed via :mod:`._xp` (§7.2). ``tools/joint_distribution`` is a thin
DataFrame wrapper over :func:`_joint_draws`.

Design invariants (§7.1/§7.8):
- Temperature-temper the base **once**: ``T==1`` is the exact identity (``base = p_ct``,
  no ``eps`` round-trip), else ``base = softmax(log(p_ct+1e-8)/T)``.
- Draws are made over **all ct rows at once** (a single seeded ``Dirichlet.sample((N,))``),
  then sliced per covariate — so the draw count == ``n_samples`` regardless of how many
  covariates/groups are requested (the draw-once invariant).
- ``n_samples=0`` is deterministic (base only; no Monte-Carlo, ``random_state`` ignored).
- ``use_logits=True`` folds per-cell logits with ``log(base)`` (gate-aware) exactly like
  ``predict()`` and scatter-adds per clone; ``use_logits=False`` returns the ct-level base.
- ``weighted`` scales each clone row by its **ct-keyed** cell count (the fix for the old
  clone-indexed ``Counter`` bug); ``weighted=False`` row-normalizes to a per-clone simplex.
"""
from __future__ import annotations

import contextlib

import numpy as np

from . import _xp

_EPS = 1e-8


@contextlib.contextmanager
def _torch_seed(random_state):
    """Seed the global torch RNG from ``random_state`` for the duration, then restore
    it (so the draw is reproducible without a permanent global side effect). ``None``
    leaves the RNG untouched (non-reproducible)."""
    import torch

    if random_state is None:
        yield
        return
    if isinstance(random_state, torch.Generator):
        seed = int(random_state.initial_seed())
    elif isinstance(random_state, (int, np.integer)):
        seed = int(random_state)
    elif hasattr(random_state, "integers"):  # numpy Generator
        seed = int(random_state.integers(0, 2**31 - 1))
    else:
        seed = int(np.random.default_rng(random_state).integers(0, 2**31 - 1))
    prev = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(prev)


def _joint_draws(
    p_ct,
    ct_to_cov,
    ct_to_c,
    ct_array,
    cov_array,
    *,
    local_scale,
    n_samples=0,
    temperature=1.0,
    use_logits=True,
    covariate_idx=None,
    logits=None,
    gate_prob=None,
    weighted=False,
    random_state=None,
    device=None,
):
    """Compute the joint stack per covariate.

    Returns ``(blocks, n_draws)`` where ``blocks`` is a list of
    ``(cov_idx, clone_idx[n_rows], J[S, n_rows, P])`` (host numpy, float64) — one block
    per covariate processed (all covariates when ``covariate_idx is None``) — and
    ``n_draws`` is the number of Dirichlet draws performed (``== n_samples``, or ``0``).
    """
    import torch

    dev = _xp.torch_device(device)
    p_ct_t = torch.as_tensor(np.asarray(p_ct), dtype=torch.float64, device=dev)  # [n_ct, P]
    ct_to_cov_t = torch.as_tensor(np.asarray(ct_to_cov), dtype=torch.long, device=dev)
    ct_to_c_t = torch.as_tensor(np.asarray(ct_to_c), dtype=torch.long, device=dev)
    ct_arr_t = torch.as_tensor(np.asarray(ct_array), dtype=torch.long, device=dev)
    cov_arr_t = torch.as_tensor(np.asarray(cov_array), dtype=torch.long, device=dev)
    n_ct, P = p_ct_t.shape

    # sanitize any non-finite p_ct rows to uniform (matches the legacy guard)
    bad = ~torch.isfinite(p_ct_t)
    if bad.any():
        p_ct_t = torch.where(bad, torch.full_like(p_ct_t, 1.0 / P), p_ct_t)

    # temper the base once (T==1 == exact identity, no eps round-trip)
    if float(temperature) == 1.0:
        base = p_ct_t
    else:
        base = torch.softmax(torch.log(p_ct_t + _EPS) / float(temperature), dim=-1)

    # draw over ALL ct rows once (shared-draw invariant), seeded
    if n_samples and int(n_samples) > 0:
        conc = torch.clamp(float(local_scale) * base, min=1e-3)
        with _torch_seed(random_state):
            bases = torch.distributions.Dirichlet(conc).sample((int(n_samples),))  # [N, n_ct, P]
        n_draws = int(n_samples)
        S = int(n_samples)
    else:
        bases = base.unsqueeze(0)  # [1, n_ct, P]
        n_draws = 0
        S = 1

    if use_logits:
        logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float64, device=dev)  # [n_obs, P]
        use_gate = gate_prob is not None and not (isinstance(gate_prob, float) and np.isnan(gate_prob))
        g = float(gate_prob) if use_gate else None

    covs = ([int(covariate_idx)] if covariate_idx is not None
            else sorted(int(c) for c in torch.unique(ct_to_cov_t).tolist()))

    blocks = []
    for m in covs:
        ct_rows = torch.nonzero(ct_to_cov_t == m, as_tuple=True)[0]  # ct indices at cov m
        clones_m = ct_to_c_t[ct_rows]                                # clone id per ct row
        # ct-keyed cell counts at this covariate (the weighting fix)
        counts_per_ct = torch.bincount(
            ct_arr_t[cov_arr_t == m], minlength=n_ct
        ).to(torch.float64)[ct_rows]  # [n_ct_m]

        if not use_logits:
            J = bases[:, ct_rows, :].clone()  # [S, n_ct_m, P] — each ct row is a clone
            if weighted:
                J = J * counts_per_ct[None, :, None]
        else:
            cells_m = torch.nonzero(cov_arr_t == m, as_tuple=True)[0]
            ct_of_cell = ct_arr_t[cells_m]                # global ct per cell
            lut = torch.full((n_ct,), -1, dtype=torch.long, device=dev)
            lut[ct_rows] = torch.arange(ct_rows.shape[0], device=dev)
            local_ct = lut[ct_of_cell]                    # [n_cells_m] local clone-row index
            b_cell = bases[:, ct_of_cell, :]              # [S, n_cells_m, P]
            log_b = torch.log(b_cell + _EPS)
            ell = logits_t[cells_m].unsqueeze(0)          # [1, n_cells_m, P]
            combine = (g * ell + (1.0 - g) * log_b) if use_gate else (ell + log_b)
            p_cell = torch.softmax(combine / float(temperature), dim=-1)  # [S, n_cells_m, P]
            J = torch.zeros((S, ct_rows.shape[0], P), dtype=torch.float64, device=dev)
            J.index_add_(1, local_ct, p_cell)             # sum P over cells per clone (row sum == count)
            if not weighted:
                J = J / J.sum(-1, keepdim=True).clamp_min(_EPS)

        blocks.append((m, _xp.asnumpy(clones_m), _xp.asnumpy(J)))

    return blocks, n_draws
