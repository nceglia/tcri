from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, version as _version

try:
    __version__ = _version("tcri")
except _PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

from . import tools as tl          # engine-backed metrics
from . import preprocessing as pp
from . import plotting as pl
from . import utils as ut
from . import model as ml
from . import diagnostics as diag
from . import perturbation as perturb  # in-silico perturbation of a fitted model
from . import null            # permutation references, fitted as ordinary fits
from . import datasets        # synthetic cohorts, incl. one with a known-MI oracle
from . import get             # accessors for cached tl results

# The unified engine, re-exported top-level for prominence.
from .tools import joint_distribution

import sys as _sys

_sys.modules.update({f'{__name__}.{m}': globals()[m]
                     for m in ['tl', 'pp', 'pl', 'ut', 'ml', 'diag', 'perturb']})

#: The top-level surface. Without this, ``dir(tcri)`` would also advertise ``sys`` and
#: ``PackageNotFoundError`` -- module-machinery names bound at module scope that a user could
#: reasonably mistake for API. The private aliases (``_sys``, ``_PackageNotFoundError``) keep
#: them out of ``dir`` regardless; this makes the intended surface explicit rather than
#: incidental.
__all__ = [
    "tl", "pp", "pl", "ut", "ml", "diag", "perturb", "null", "datasets", "get",
    "tools", "preprocessing", "plotting", "utils", "model", "diagnostics", "perturbation",
    "joint_distribution", "__version__",
]