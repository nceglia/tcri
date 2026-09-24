"""Statistics primitives and the between-group contrast — private, shared.

``_core`` holds the primitives and ``_compare`` builds the contrast out of them:
``compare_groups`` is written entirely in terms of ``hdi``, ``mann_whitney``,
``prob_direction`` and ``stars``, which is why the two live together here rather than under
``tools/``, where a contrast reads as a metric rather than as the machinery a metric reaches
for.

``_compare`` imports from ``_core`` directly, rather than importing the package ``__init__``
it is itself part of — that works in Python but inverts the dependency and breaks the moment
anything else is added here.

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
