"""``warn``: a warning attributed to the caller's line.

Python names the frame ``stacklevel`` levels up. tcri's warnings are raised at different depths
(inside a null fit, a reference run, the ``tl_result`` decorator), so a fixed ``stacklevel``
names a line inside tcri as often as the caller's. ``skip_file_prefixes`` (Python 3.12+) skips
every frame in the tcri package instead, so the warning names the first line outside it: the
caller's cell or script line. A warning raised under a dependency's frames (inside scvi's model
load, say) names that dependency's frame.

The module a warning is attributed to becomes the caller's, so ``filterwarnings(module=...)``
does not match tcri's warnings; filter them by category or message.
"""
from __future__ import annotations

import os
import warnings

#: The tcri package directory, with a trailing separator so ``tcri_extra/`` does not match, in
#: each spelling a frame's file name can carry: as imported (which may contain ``..``), absolute,
#: and with symlinks resolved.
_PACKAGE = os.path.dirname(os.path.dirname(__file__))
_TCRI_DIRS = tuple(dict.fromkeys(
    path + os.sep for path in (_PACKAGE, os.path.abspath(_PACKAGE), os.path.realpath(_PACKAGE))))


def warn(message: str, category: type[Warning] = UserWarning) -> None:
    """Issue ``message`` as ``category``, attributed to the first frame outside tcri."""
    warnings.warn(message, category, stacklevel=2, skip_file_prefixes=_TCRI_DIRS)
