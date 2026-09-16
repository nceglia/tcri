"""The pull request check in ``.github/scripts/check_pr.py``.

Only :func:`problems` is tested: it takes the pull request, its changed files and a file reader, so
every case is a plain dict and a list. ``main`` is the thin GitHub API wrapper around it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "check_pr.py"
if not _SCRIPT.is_file():  # the sdist ships tests/ but not .github/
    pytest.skip("the pull request check is not part of the source distribution",
                allow_module_level=True)
_spec = importlib.util.spec_from_file_location("check_pr", _SCRIPT)
check_pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_pr)


def _pr(number=10, body="Closes #5", labels=(), milestone="0.13.0", base="main"):
    return {
        "number": number,
        "body": body,
        "labels": [{"name": name} for name in labels],
        "milestone": None if milestone is None else {"title": milestone},
        "base": {"ref": base},
    }


def _problems(pr, notes=None, extra=()):
    """``notes`` maps a changed release-note path to its text."""
    notes = {"docs/release-notes/10.feat.md": "Adds a thing.\n"} if notes is None else notes
    files = [(path, "added") for path in notes] + list(extra)
    return check_pr.problems(pr, files, lambda path: notes[path])


def test_a_complete_pull_request_passes():
    assert _problems(_pr()) == []


def test_missing_link_milestone_and_note_are_all_reported():
    found = _problems(_pr(body="", milestone=None), notes={})
    assert len(found) == 3
    assert any("milestone" in p for p in found)
    assert any("Link the issue" in p for p in found)
    assert any("Add one release note" in p for p in found)


def test_exemption_labels_pass():
    pr = _pr(body="", milestone=None, labels=("no issue", "no milestone", "no release note"))
    assert _problems(pr, notes={}) == []


@pytest.mark.parametrize("body", ["Closes #5", "Part of #129", "part of #3", "This fixes #12.",
                                  "Resolved #1", "closed #99"])
def test_link_keywords(body):
    assert _problems(_pr(body=body)) == []


@pytest.mark.parametrize("body", ["Closes #<!-- the issue this delivers -->", "See #5", "#5", "Closes 5"])
def test_unfilled_or_non_closing_links_are_rejected(body):
    assert any("Link the issue" in p for p in _problems(_pr(body=body)))


def test_note_named_for_another_pull_request_is_reported():
    notes = {"docs/release-notes/10.feat.md": "Mine.\n", "docs/release-notes/9.fix.md": "Below.\n"}
    assert any("another pull request" in p for p in _problems(_pr(), notes=notes))


def test_backport_to_a_release_branch_may_carry_the_original_note():
    notes = {"docs/release-notes/7.fix.md": "Fixed.\n"}
    backport = _pr(number=20, body="Backport of #7\nCloses #6", base="0.13.x")
    assert _problems(backport, notes=notes) == []
    on_main = _pr(number=20, body="Backport of #7\nCloses #6", base="main")
    assert any("another pull request" in p for p in _problems(on_main, notes=notes))


def test_unknown_type_is_reported():
    notes = {"docs/release-notes/10.docs.md": "Docs.\n"}
    assert any("not one of" in p for p in _problems(_pr(), notes=notes))


def test_two_lines_are_reported():
    notes = {"docs/release-notes/10.feat.md": "One.\nTwo.\n"}
    assert any("one line" in p for p in _problems(_pr(), notes=notes))


def test_length_limit_is_two_hundred_characters():
    at_limit = {"docs/release-notes/10.feat.md": "x" * 200 + "\n"}
    over = {"docs/release-notes/10.feat.md": "x" * 201 + "\n"}
    assert _problems(_pr(), notes=at_limit) == []
    assert any("one line" in p for p in _problems(_pr(), notes=over))


def test_two_notes_for_one_pull_request_are_reported():
    notes = {"docs/release-notes/10.feat.md": "A.\n", "docs/release-notes/10.fix.md": "B.\n"}
    assert any("Add one release note" in p for p in _problems(_pr(), notes=notes))


def test_orphan_note_is_reported():
    notes = {"docs/release-notes/10.feat.md": "A.\n", "docs/release-notes/+stray.feat.md": "B.\n"}
    assert any("Unexpected files" in p for p in _problems(_pr(), notes=notes))


@pytest.mark.parametrize("page", ["docs/release-notes/index.md", "docs/release-notes/0.13.0.md",
                                  "docs/release-notes/0.13.0a1.md"])
def test_index_and_version_pages_are_allowed(page):
    pr = _pr(labels=("no release note",))
    assert _problems(pr, notes={}, extra=[(page, "modified")]) == []


def test_removed_notes_are_ignored():
    # the release pull request deletes every note and adds the version page
    pr = _pr(labels=("no release note", "no issue"))
    removed = [("docs/release-notes/3.feat.md", "removed"), ("docs/release-notes/0.13.0.md", "added")]
    assert _problems(pr, notes={}, extra=removed) == []
