"""Device seam for the batched engine — **torch-first** (grafiti `_compute/_xp`
parity, adapted because tcri's numeric core is torch: the Dirichlet draws and the
softmax are `torch`, and torch≥2.4 is already a hard dep so torch.cuda is zero new
deps). `cupy` may be added later as a second backend; for now the ladder is
CPU / torch-CUDA. Every accelerated function returns a host numpy array via
:func:`asnumpy` (§7.4 guardrail 4). GPU libs are imported lazily inside functions
(guardrail 1) — importing this module never touches CUDA.
"""
from __future__ import annotations

import warnings

import numpy as np


def resolve_device(device: str | None) -> str:
    """Resolve ``device`` to ``"cpu"`` or ``"cuda"`` (§7.1 ladder).

    ``None``/``"cpu"`` → cpu; ``"mps"`` → cpu (first pass: the Dirichlet core has no
    Metal backend); ``"auto"``/``"gpu"``/``"cuda"`` → cuda only if torch reports a
    device, else cpu (an explicit ``"cuda"`` warns on fallback; ``auto``/``gpu`` are
    silent).
    """
    if device in (None, "cpu"):
        return "cpu"
    if device == "mps":
        return "cpu"
    if device in ("cuda", "gpu", "auto"):
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return "cuda"
        except Exception:
            pass
        if device == "cuda":
            warnings.warn(
                "device='cuda' requested but torch CUDA is unavailable; using CPU.",
                stacklevel=2,
            )
        return "cpu"
    warnings.warn(f"device={device!r} not recognized; using CPU.", stacklevel=2)
    return "cpu"


def torch_device(device: str | None):
    """The :class:`torch.device` for the resolved device."""
    import torch

    return torch.device("cuda" if resolve_device(device) == "cuda" else "cpu")


def asnumpy(x) -> np.ndarray:
    """Bring an array (possibly a torch tensor, possibly on GPU) back to host numpy."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)
