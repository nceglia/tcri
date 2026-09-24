"""Private numeric + device seam for the engine.

- :mod:`._xp`       — device dispatch (torch-first; CPU / torch-CUDA), ``asnumpy`` boundary.
- :mod:`._joint`    — ``_joint_draws``: the batched ``[S, n_clones, P]`` engine core.
- :mod:`._distance` — kl / l1 / jsd kernels over phenotype distributions.
- :mod:`._tables`   — the metric-table plumbing every ``tools`` metric reduces through
  (``metric_table``, ``build_result``, ``build_stats``, ``collapse_to_replicates``).

This is the LOWER layer: ``tools``/``plotting``/``diagnostics`` import down into it and it must
not import back up. ``_tables`` needs one symbol from ``tools._joint`` and takes it lazily inside
the function, never at module scope.

Nothing here is public API; GPU libs are imported lazily inside functions.
"""
