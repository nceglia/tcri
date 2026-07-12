"""Private numeric + device seam for the engine (grafiti ``_compute`` parity).

- :mod:`._xp`     — device dispatch (torch-first; CPU / torch-CUDA), ``asnumpy`` boundary.
- :mod:`._joint`  — ``_joint_draws``: the batched ``[S, n_clones, P]`` engine core.
- :mod:`._reduce` — batched entropy / mutual-information reductions (**bits / log2** default).

Nothing here is public API; GPU libs are imported lazily inside functions.
"""
