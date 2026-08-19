"""Statistics primitives and the between-group contrast — private, shared.

Split from what used to be a flat ``tcri/_stats.py`` plus ``tcri/tools/_compare.py``. The two
belonged together: ``_compare.compare_groups`` is built entirely out of ``_core``'s primitives
(``hdi``, ``mann_whitney``, ``prob_direction``, ``stars``) and had no other reason to live under
``tools/``, where it read as a metric rather than as the machinery a metric reaches for.

``_core`` holds the primitives and ``_compare`` imports from it, rather than ``_compare``
importing the package ``__init__`` it is itself part of — that works in Python but inverts the
dependency and breaks the moment anything else is added here.

Nothing in this package is public API. ``compare_groups`` in particular is deliberately NOT on
``tcri.tl``: it is reached only through a metric's ``splitby`` argument, never as a step the user
performs. ``tests/test_removal_ledger.py`` pins that distinction.
"""
from __future__ import annotations

from ._core import (auc_and_label_permutation, bootstrap_auc, hdi, mann_whitney,
                    prob_direction, stars)
from ._compare import compare_groups

__all__ = ["hdi", "mann_whitney", "prob_direction", "stars",
           "auc_and_label_permutation", "bootstrap_auc", "compare_groups"]
