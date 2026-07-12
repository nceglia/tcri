from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("tcri")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

from . import tools as tl          # PR6: tl repointed metrics -> tools (engine-backed metrics)
from . import preprocessing as pp
from . import plotting as pl
from . import utils as ut
from . import model as ml

# The unified engine, re-exported top-level for prominence.
from .tools import joint_distribution

import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['tl', 'pp', 'pl', 'ut', 'ml']})