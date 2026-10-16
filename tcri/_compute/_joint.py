"""Batched engine core — the clone×phenotype joint as a ``[S, n_clones, P]`` stack,
device-routed via :mod:`._xp`. ``tools/joint_distribution`` is a thin
DataFrame wrapper over :func:`_joint_draws`.

Design invariants:
- Temperature-temper the base **once**: ``T==1`` is the exact identity (``base = p_ct``,
  no ``eps`` round-trip), else ``base = softmax(log(p_ct+1e-8)/T)``.
- Draws are made over **all ct rows at once** (a single seeded ``Dirichlet.sample((N,))``),
  then sliced per covariate — so the draw count == ``n_samples`` regardless of how many
  covariates/groups are requested (the draw-once invariant).
- ``n_samples=0`` is deterministic (base only; no Monte-Carlo, ``random_state`` ignored).
- ``use_logits=True`` folds per-cell logits with ``log(base)`` (gate-aware) exactly like
  ``predict()`` and scatter-adds per clone; ``use_logits=False`` returns the ct-level base.
- ``weighted`` scales each clone row by its **ct-keyed** cell count; ``weighted=False``
  row-normalizes to a per-clone simplex.
"""
from __future__ import annotations

import contextlib

import numpy as np

from . import _xp

_EPS = 1e-8


def _seed_int(random_state):
    """The integer seed ``random_state`` stands for, or ``None`` when it is ``None``.

    An int is itself; a torch generator gives its initial seed; a numpy generator gives one
    draw from itself, so it advances; anything else seeds a numpy generator that gives one.
    """
    import torch

    if random_state is None:
        return None
    if isinstance(random_state, torch.Generator):
        return int(random_state.initial_seed())
    if isinstance(random_state, (int, np.integer)):
        return int(random_state)
    if hasattr(random_state, "integers"):  # numpy Generator
        return int(random_state.integers(0, 2**31 - 1))
    return int(np.random.default_rng(random_state).integers(0, 2**31 - 1))


@contextlib.contextmanager
def _torch_seed(random_state):
    """Seed the global torch RNG from ``random_state`` for the duration, then restore
    it (so the draw is reproducible without a permanent global side effect). ``None``
    leaves the RNG untouched (non-reproducible)."""
    import torch

    seed = _seed_int(random_state)
    if seed is None:
        yield
        return
    prev = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(prev)


def _gate(gate_prob):
    """The classifier/prior gate as a float, or ``None`` for the additive combine. A stored NaN
    is the additive sentinel ``to_anndata`` writes, since h5ad cannot hold ``None``."""
    if gate_prob is None or (isinstance(gate_prob, float) and np.isnan(gate_prob)):
        return None
    return float(gate_prob)


def _posterior_bases(p_ct, *, local_scale, conc_ct=None, n_samples=0, temperature=1.0,
                     random_state=None, dev=None):
    """Every ct row's base, tempered once, and its seeded posterior draws.

    Returns ``(bases, n_draws)``: ``bases`` is a float64 torch tensor ``[S, n_ct, P]`` on
    ``dev``, the ``n_samples`` Dirichlet draws when ``n_samples > 0`` and the tempered mean as
    ``S = 1`` otherwise. ``n_draws`` is the number of draws made (``0`` for the mean).

    This is the one place a posterior draw of ``p_ct`` is made. :func:`_joint_draws` and
    ``diag.joint_distribution_ppc`` both call it, so the same ``random_state`` gives both the
    same draws; a second copy of this block would let them drift apart silently.
    """
    import torch

    p_ct_t = torch.as_tensor(np.asarray(p_ct), dtype=torch.float64, device=dev)  # [n_ct, P]
    P = p_ct_t.shape[1]

    # sanitize any non-finite p_ct rows to uniform, so one bad row cannot poison the draw
    bad = ~torch.isfinite(p_ct_t)
    if bad.any():
        p_ct_t = torch.where(bad, torch.full_like(p_ct_t, 1.0 / P), p_ct_t)

    # temper the base once (T==1 == exact identity, no eps round-trip)
    if float(temperature) == 1.0:
        base = p_ct_t
    else:
        base = torch.softmax(torch.log(p_ct_t + _EPS) / float(temperature), dim=-1)

    if not (n_samples and int(n_samples) > 0):
        return base.unsqueeze(0), 0  # [1, n_ct, P]

    # draw over ALL ct rows once (shared-draw invariant), seeded
    if conc_ct is not None:
        # Draw from the GUIDE's posterior concentration, not a reconstruction of it: the
        # `local_scale * base` fallback below is a fixed pseudo-count, identical for every
        # group regardless of how many cells supported it, so it cannot widen where the
        # data are thin.
        #
        # Only the MAGNITUDE is taken from the guide. The direction stays `base`, which is
        # the analysis-time tempered mean -- at temperature == 1 `base` is the guide's own
        # normalized direction, so the draw is the guide's own Dirichlet; at T != 1 the
        # deliberate analysis-time temper still applies to the mean while the posterior
        # WIDTH continues to come from the fit.
        conc_ct_t = torch.as_tensor(np.asarray(conc_ct), dtype=torch.float64, device=dev)
        if conc_ct_t.shape != p_ct_t.shape:
            raise ValueError(
                f"conc_ct has shape {tuple(conc_ct_t.shape)} but p_ct has "
                f"{tuple(p_ct_t.shape)}; they index the same ct rows and must match"
            )
        mag = conc_ct_t.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        conc = torch.clamp(mag * base, min=1e-3)
    else:
        conc = torch.clamp(float(local_scale) * base, min=1e-3)
    with _torch_seed(random_state):
        bases = torch.distributions.Dirichlet(conc).sample((int(n_samples),))  # [N, n_ct, P]
    return bases, int(n_samples)


def _cell_probs(bases, ct_of_cell, ell, gate, temperature=1.0):
    """Per-cell phenotype probabilities at each draw, ``[S, n_cells, P]``.

    ``softmax((g * ell + (1 - g) * log b) / T)``, or ``softmax((ell + log b) / T)`` when
    ``gate`` is ``None``, where ``b`` is the draw of the cell's own ct row and ``ell`` its
    classifier logits (``[1, n_cells, P]``). At ``T == 1`` this is the per-cell distribution
    ``predict()`` returns and the label likelihood's ``softmax(ℓ_i)``.

    The base is already tempered and the combine is divided by ``T`` here; at ``T == 1`` both
    are the identity. ``T != 1`` is a deliberate analysis-time temper, and the two ``T``'s do
    not cancel.
    """
    import torch

    b_cell = bases[:, ct_of_cell, :]                      # [S, n_cells, P]
    log_b = torch.log(b_cell + _EPS)
    combine = (gate * ell + (1.0 - gate) * log_b) if gate is not None else (ell + log_b)
    return torch.softmax(combine / float(temperature), dim=-1)


def _joint_draws(
    p_ct,
    ct_to_cov,
    ct_to_c,
    ct_array,
    cov_array,
    *,
    local_scale,
    conc_ct=None,
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
    ct_to_cov_t = torch.as_tensor(np.asarray(ct_to_cov), dtype=torch.long, device=dev)
    ct_to_c_t = torch.as_tensor(np.asarray(ct_to_c), dtype=torch.long, device=dev)
    ct_arr_t = torch.as_tensor(np.asarray(ct_array), dtype=torch.long, device=dev)
    cov_arr_t = torch.as_tensor(np.asarray(cov_array), dtype=torch.long, device=dev)
    bases, n_draws = _posterior_bases(p_ct, local_scale=local_scale, conc_ct=conc_ct,
                                      n_samples=n_samples, temperature=temperature,
                                      random_state=random_state, dev=dev)
    S, n_ct, P = bases.shape

    if use_logits:
        logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float64, device=dev)  # [n_obs, P]
        g = _gate(gate_prob)

    covs = ([int(covariate_idx)] if covariate_idx is not None
            else sorted(int(c) for c in torch.unique(ct_to_cov_t).tolist()))

    blocks = []
    for m in covs:
        ct_rows = torch.nonzero(ct_to_cov_t == m, as_tuple=True)[0]  # ct indices at cov m
        clones_m = ct_to_c_t[ct_rows]                                # clone id per ct row
        # cell counts keyed by ct row at this covariate -- the weights `weighted=True` applies,
        # so a clone's weight is its cell count AT this covariate, not across all of them
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
            ell = logits_t[cells_m].unsqueeze(0)          # [1, n_cells_m, P]
            J = torch.zeros((S, ct_rows.shape[0], P), dtype=torch.float64, device=dev)
            # Chunk over draws. The per-cell chain below holds four live
            # [S, n_cells_m, P] float64 tensors at once, which at realistic draw
            # counts and cell counts exhausts memory. Draws are independent, so
            # chunking is bit-identical and caps each temporary at the size below.
            n_cells_m = cells_m.shape[0]
            per_draw = max(1, n_cells_m * P)
            chunk = max(1, min(S, int(8_000_000 // per_draw)))  # ~64 MB/temp in float64
            for s0 in range(0, S, chunk):
                s1 = min(s0 + chunk, S)
                p_cell = _cell_probs(bases[s0:s1], ct_of_cell, ell, g, temperature)
                Jc = torch.zeros((s1 - s0, ct_rows.shape[0], P), dtype=torch.float64, device=dev)
                Jc.index_add_(1, local_ct, p_cell)        # sum P over cells per clone (row sum == count)
                J[s0:s1] = Jc
            if not weighted:
                J = J / J.sum(-1, keepdim=True).clamp_min(_EPS)

        blocks.append((m, _xp.asnumpy(clones_m), _xp.asnumpy(J)))

    return blocks, n_draws
