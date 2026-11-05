"""Device seam for the batched engine — **torch-first**, because the numeric core is torch:
the Dirichlet draws and the softmax are `torch` calls. The ladder is CPU / torch-CUDA.

Every accelerated function returns a host numpy array via :func:`asnumpy`. GPU libs are
imported lazily inside functions — importing this module never touches CUDA.
"""
from __future__ import annotations

import contextlib

import numpy as np

from .. import _output


def resolve_device(device: str | None) -> str:
    """Resolve ``device`` to ``"cpu"`` or ``"cuda"``.

    ``None``/``"cpu"`` → cpu; ``"mps"`` → cpu, since the Dirichlet core has no Metal
    backend; ``"auto"``/``"gpu"``/``"cuda"`` → cuda only if torch reports a
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
            _output.warn("device='cuda' requested but torch CUDA is unavailable; using CPU.")
        return "cpu"
    _output.warn(f"device={device!r} not recognized; using CPU.")
    return "cpu"


def torch_device(device: str | None):
    """The :class:`torch.device` for the resolved device."""
    import torch

    return torch.device("cuda" if resolve_device(device) == "cuda" else "cpu")


def rng_device(device) -> tuple[str, int | None]:
    """The generator a draw on ``device`` uses, as ``(kind, index)``.

    ``("cuda", i)`` for a CUDA device (an unindexed ``"cuda"`` means the current one),
    ``("mps", None)`` for MPS, which has one generator, and ``("cpu", None)`` otherwise.
    """
    import torch

    d = torch.device(device)
    if d.type == "cuda":
        return "cuda", d.index if d.index is not None else torch.cuda.current_device()
    if d.type == "mps":
        return "mps", None
    return "cpu", None


@contextlib.contextmanager
def seeded_rng(seed: int, device="cpu"):
    """Seed the CPU generator and the generator of ``device`` with ``seed`` for the block, then
    restore both.

    ``device`` is where the block draws. Only those two generators are seeded, so every other
    device's generator is left as it was; ``torch.manual_seed`` reseeds every device, and a fork
    that restores only some of them leaves the rest reseeded after the block. Draws inside the
    block match draws made after ``torch.manual_seed(seed)`` on the same device.
    """
    import torch

    kind, index = rng_device(device)
    cpu_state = torch.get_rng_state()
    if kind == "cuda":
        device_state = torch.cuda.get_rng_state(index)
    elif kind == "mps":
        device_state = torch.mps.get_rng_state()
    try:
        torch.default_generator.manual_seed(int(seed))
        if kind == "cuda":
            with torch.cuda.device(index):
                torch.cuda.manual_seed(int(seed))
        elif kind == "mps":
            torch.mps.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if kind == "cuda":
            torch.cuda.set_rng_state(device_state, index)
        elif kind == "mps":
            torch.mps.set_rng_state(device_state)


def asnumpy(x) -> np.ndarray:
    """Bring an array (possibly a torch tensor, possibly on GPU) back to host numpy."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)
