"""``tcri.ut`` — utilities.

The AUROC pair is surfaced here on purpose: they were implemented, tested, and unreachable
except by importing the private ``tcri._stats``.
"""
from ._utils import *          # noqa: F401,F403  (bounded by _utils.__all__)
from ._utils import __all__ as _utils_all
from .._stats import auc_and_label_permutation, bootstrap_auc   # noqa: F401

__all__ = [*_utils_all, "auc_and_label_permutation", "bootstrap_auc"]
