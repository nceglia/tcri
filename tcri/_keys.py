"""Backwards-compatible alias for :mod:`tcri._state.keys`.

The canonical import is ``from tcri._state import keys as K``. This shim exists so the ~25
existing ``from .. import _keys as K`` sites keep working; there is exactly one definition of
every key, in ``_state/keys.py``.
"""
from __future__ import annotations

from ._state.keys import *          # noqa: F401,F403
from ._state.keys import Config, colors  # noqa: F401
