"""``tcri.ut`` — utilities.

The AUROC pair is surfaced here on purpose: they were implemented, tested, and unreachable
except by importing the private ``tcri._stats``.
"""
from ..utils._utils import *          # noqa: F401,F403  (no __all__ yet — tracked separately)
from .._stats import auc_and_label_permutation, bootstrap_auc   # noqa: F401
