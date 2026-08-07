"""Conformance test for the TRAINING contract (``tcri/model/_training_contract.py``).

The model and metrics contracts are checked against the manuscript. This one cannot be —
Note 1 says nothing about optimization beyond "SVI in Pyro, mini-batching, KL scaling, Adam".
So half of it is *derived* from eq 7 and half is *authored* here, and the split has to stay
visible: a derived invariant is not negotiable, an authored bound is changeable with a reason.

This file checks the manifest's own hygiene and the structural claims that can be read off the
code. The BEHAVIOURAL assertions live in ``tests/test_training_invariants.py`` — that
separation is deliberate, because the failure this contract exists to prevent was a green tick
on a wiring check.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import re
import textwrap

import pytest

from tcri.model import _training_contract as TC
from tcri.model._training import UnifiedTrainingPlan


def test_manifest_is_complete():
    """Every entry states something and says how it is enforced."""
    assert TC.DERIVED_INVARIANTS and TC.AUTHORED_BOUNDS
    for key, inv in TC.DERIVED_INVARIANTS.items():
        assert inv.get("statement"), f"{key} has no statement"
        assert inv.get("status"), f"{key} has no status"
        assert inv.get("enforced_by"), f"{key} does not say what enforces it"
    for key, text in TC.AUTHORED_BOUNDS.items():
        assert len(text) > 60, f"{key} is too thin to act on"


def test_derived_and_authored_stay_separate():
    """A derived invariant follows from eq 7; an authored bound is our choice. Collapsing the
    two would let a preference acquire the authority of the manuscript."""
    assert not (set(TC.DERIVED_INVARIANTS) & set(TC.AUTHORED_BOUNDS))


def test_a_holds_claim_names_a_test_that_exists():
    """An invariant may only claim 'holds' if the test it names is real.

    This replaces a hardcoded assertion that I3/I4 read OPEN. That version had to be edited by
    hand the moment either was resolved, which makes it a reminder rather than a check — and it
    could not stop an invariant being flipped to 'holds' while pointing at a test nobody wrote.

    A contract that overstates its own coverage is worse than one that admits a gap, so the
    rule is general: 'holds' requires a file that exists and, when a ``::test_name`` is given,
    a function by that name inside it. Any other status ('OPEN', 'SPECIFIED', 'partial') may
    name a pending test freely — that is what those statuses are for.
    """
    repo = pathlib.Path(__file__).resolve().parents[1]

    for key, inv in TC.DERIVED_INVARIANTS.items():
        status, enforced = inv["status"], inv["enforced_by"]
        if not status.startswith("holds"):
            continue
        assert not enforced.lstrip().startswith("pending"), (
            f"{key} claims 'holds' but enforced_by is still 'pending'. Either the test exists "
            f"— name it — or the status is not 'holds'."
        )
        for ref in re.findall(r"tests/[\w/]+\.py(?:::\w+)?", enforced):
            path, _, func = ref.partition("::")
            full = repo / path
            assert full.exists(), (
                f"{key} claims 'holds' and cites {path}, which does not exist. An invariant "
                f"cannot be enforced by a file nobody wrote."
            )
            if func:
                assert re.search(rf"^def {re.escape(func)}\b", full.read_text(), re.M), (
                    f"{key} claims 'holds' and cites {ref}, but {full.name} defines no "
                    f"function named {func}."
                )


def test_specified_invariants_are_not_claimed_to_hold():
    """'SPECIFIED' means the design is settled but unimplemented. It must not read as done.

    I3 and I4 are written out in full in the manifest — criterion, scope, and the two
    param-store/BatchNorm hazards — ahead of the code, per the repo rule that the contract is
    updated first. Until the implementation lands they must not claim to hold.
    """
    for key, inv in TC.DERIVED_INVARIANTS.items():
        if not inv["status"].startswith("SPECIFIED"):
            continue
        assert "does not implement it yet" in inv["status"], (
            f"{key} is SPECIFIED but does not say plainly that the code has not landed."
        )
        assert inv["enforced_by"].lstrip().startswith("pending"), (
            f"{key} is SPECIFIED, so enforced_by must be marked pending. If the test now "
            f"exists, move the status to 'holds' — test_a_holds_claim_names_a_test_that_exists "
            f"will then verify the reference."
        )


def test_no_reset_knob_on_train():
    """B1: restarting the KL ramp is the behaviour DE-4 removes, and adding a reset to
    ``train()`` would also be an API-contract change to ``_contract.pyi``."""
    from tcri.model._model import TCRIModel

    params = inspect.signature(TCRIModel.train).parameters
    assert "reset_schedule" not in params, (
        "a reset knob reappeared on train(); see AUTHORED_BOUNDS['B1_monotone_terminating_annealing']"
    )


def test_warmup_counter_is_owned_by_the_module_not_the_plan():
    """B1/I5: ``train()`` builds a fresh plan per call, so a plan-local counter restarts the
    ramp on every resumed fit. The counter must live on the module, which survives."""
    assert not hasattr(UnifiedTrainingPlan, "_my_global_step"), (
        "the plan owns a warmup counter again — a fresh plan per train() call means the ramp "
        "restarts, which is DE-4"
    )
    src = inspect.getsource(UnifiedTrainingPlan.training_step)
    assert "self.module._kl_warmup_step" in src, (
        "training_step no longer reads the module-owned warmup counter"
    )


def _body_source(fn) -> str:
    """Source with the docstring removed.

    Necessary here: validation_step's docstring EXPLAINS the removed
    ``super().training_step()`` call, so a naive substring search matches the explanation and
    the test fails on a correct implementation. Match the code, not the prose about it.
    """
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    node = tree.body[0]
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body = node.body[1:]
    return ast.unparse(node)


def test_validation_step_does_not_call_training_step():
    """I2, structurally. The behavioural proof is in test_training_invariants.py; this catches
    a reintroduction at review time rather than after a fit."""
    src = _body_source(UnifiedTrainingPlan.validation_step)
    assert "super().training_step" not in src, (
        "validation_step calls training_step again — that reaches SVI.step() and the Pyro "
        "optimizer, so validation would update the guide parameters every metric reads (DE-1)"
    )
    assert "evaluate_loss" in src, (
        "validation_step should evaluate via SVI.evaluate_loss, which uses the same wrapped "
        "model/guide with no param_capture, no optim() and no zero_grads"
    )
