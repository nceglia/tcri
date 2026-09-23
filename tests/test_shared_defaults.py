"""Guard against defaults drifting apart across layers.

A knob is declared in more than one place: ``TCRIModel.__init__`` constructs
``TCRIModule``, and ``TCRIModel.train`` constructs ``UnifiedTrainingPlan``, each passing
its own value down. When the two declarations disagree, the outer one silently wins and
the inner one becomes **dead** — reachable only by constructing the inner object directly,
which is exactly what a test fixture or a downstream user does.

Four knobs are declared in both layers and so are pinned by value below:
``reconstruction_loss_scale``, ``local_scale``, ``n_steps_kl_warmup`` and ``global_scale`` (α).

A test that a value *arrives* at its target cannot catch this, because it does arrive. The
defect is that the two declared defaults differ, so a caller reading one signature is misled
about what the package does.

Cheap and signature-only: no model is constructed.
"""
from __future__ import annotations

import inspect

import pytest

from tcri.model._model import TCRIModel
from tcri.model._module import TCRIModule
from tcri.model._training import UnifiedTrainingPlan

#: (outer callable, inner callable, human label). The outer constructs the inner and
#: forwards these arguments, so a disagreement means the inner default is unreachable.
PAIRS = [
    (TCRIModel.__init__, TCRIModule.__init__, "TCRIModel.__init__ -> TCRIModule.__init__"),
    (TCRIModel.train, UnifiedTrainingPlan.__init__, "TCRIModel.train -> UnifiedTrainingPlan"),
]

#: Names that are deliberately allowed to differ, with the reason. Keep this empty unless
#: there is a real argument for a divergent default — "the inner one is never used" is not
#: one, since that is precisely the trap.
ALLOWED_DIVERGENCE: dict[str, str] = {}


def _defaults(fn):
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


@pytest.mark.parametrize("outer,inner,label", PAIRS, ids=[p[2] for p in PAIRS])
def test_shared_defaults_agree(outer, inner, label):
    """A knob declared in both layers must declare the SAME default in both."""
    o, i = _defaults(outer), _defaults(inner)
    mismatched = {
        k: (o[k], i[k])
        for k in set(o) & set(i)
        if o[k] != i[k] and k not in ALLOWED_DIVERGENCE
    }
    assert not mismatched, (
        f"{label}: defaults disagree, so the inner value is dead code that only bites "
        f"someone constructing the inner object directly — "
        + "; ".join(f"{k}: outer={ov!r} inner={iv!r}" for k, (ov, iv) in sorted(mismatched.items()))
    )


def test_the_known_drifted_knobs_are_pinned():
    """The four knobs declared in both layers, pinned by value, so an edit to one layer
    cannot silently reintroduce a split."""
    model, module = _defaults(TCRIModel.__init__), _defaults(TCRIModule.__init__)
    train, plan = _defaults(TCRIModel.train), _defaults(UnifiedTrainingPlan.__init__)

    assert model["local_scale"] == module["local_scale"] == 3.0
    assert model["global_scale"] == module["global_scale"] == 5.0
    assert train["n_steps_kl_warmup"] == plan["n_steps_kl_warmup"] == 2000
    assert train["reconstruction_loss_scale"] == 1e-2
