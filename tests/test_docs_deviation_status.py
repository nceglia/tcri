"""The living tracker must not contradict the contract about a deviation's status (DE-16).

`REFACTOR_AGENDA.md` is the diary: it records what was true when each entry was written, and
entries are never rewritten. `METHODS_CONFORMANCE.md` and `SANCTIONED_DEVIATIONS` carry the
LIVE status. Those two roles only stay separable if a superseded diary line says so.

DE-16: `REFACTOR_AGENDA.md:207` read "**Deferred:** **[E]** ..." long after `19db68e` raised
`reconstruction_loss_scale` 1e-3 -> 1e-2 and the contract recorded it resolved. Someone reading
the tracker — which `CLAUDE.md` tells contributors to read *before starting* — was told a
closed item was still open backlog. That is how the same work gets started twice.

This asserts the invariant rather than the instance: any diary line asserting a deviation is
deferred must either agree with the contract or be marked superseded. It is deliberately not a
prose-matching test; it compares two files' claims about the same object.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENDA = ROOT / "docs" / "contract" / "REFACTOR_AGENDA.md"
CONFORMANCE = ROOT / "docs" / "contract" / "METHODS_CONFORMANCE.md"

#: A diary line may keep an out-of-date claim if it carries this marker.
SUPERSEDED = "[SUPERSEDED"


def _resolved_deviations() -> set[str]:
    """Deviation ids the conformance record marks fixed or resolved."""
    text = CONFORMANCE.read_text()
    out = set()
    for row in text.splitlines():
        if not row.startswith("|"):
            continue
        cells = [c.strip() for c in row.split("|")]
        if len(cells) < 5:
            continue
        dev_id, status = cells[1], cells[-2].lower()
        if re.fullmatch(r"[A-Z]\d?", dev_id) and ("fixed" in status or "resolved" in status):
            out.add(dev_id)
    return out


def test_agenda_does_not_call_a_resolved_deviation_deferred():
    resolved = _resolved_deviations()
    assert resolved, "parsed no resolved deviations from METHODS_CONFORMANCE.md — parser is stale"

    offenders = []
    for n, line in enumerate(AGENDA.read_text().splitlines(), start=1):
        if "deferred" not in line.lower() or SUPERSEDED in line:
            continue
        for dev in resolved:
            if f"**[{dev}]**" in line or f"[{dev}]" in line:
                offenders.append((n, dev, line.strip()[:120]))

    assert not offenders, (
        "REFACTOR_AGENDA.md calls a deviation deferred that the contract records as resolved. "
        "The diary is append-only history, so annotate the line "
        f"'{SUPERSEDED} <date> — see METHODS_CONFORMANCE.md]' rather than rewriting it:\n"
        + "\n".join(f"  :{n} [{d}] {t}" for n, d, t in offenders)
    )


def test_the_diary_rule_is_written_down():
    """The convention only holds if it is stated where contributors are told to look."""
    head = AGENDA.read_text()[:4000]
    assert "append-only" in head.lower(), (
        "REFACTOR_AGENDA.md's 'How to use this doc' must state that the diary and audit log are "
        "append-only history and that METHODS_CONFORMANCE.md / SANCTIONED_DEVIATIONS carry live "
        "status — otherwise the next contributor rewrites a diary entry or trusts a stale one."
    )
