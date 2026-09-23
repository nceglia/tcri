#!/usr/bin/env python3
"""Report what a pull request is missing: a linked issue, a milestone, a release note.

Run by ``.github/workflows/check-pr.yml``. :func:`problems` is pure and unit-tested in
``tests/test_check_pr.py``. :func:`main` fetches the pull request fresh from the API rather than
reading the event payload, because re-running a job replays the original payload and would miss a
label or milestone added since. Standard library only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

TYPES = ("breaking", "feat", "fix", "perf")
MAX_NOTE = 200

#: GitHub's closing keywords, plus "Part of" for the pull requests below the top of a stack.
LINK = re.compile(r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|part of)\s+#\d+\b")
NOTE = re.compile(r"docs/release-notes/(\d+)\.([a-z]+)\.md")
#: Files that belong in docs/release-notes/ without being a note: the index and version pages.
PAGE = re.compile(r"docs/release-notes/(?:index|\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)\.md")


def problems(pr: dict, files: list[tuple[str, str]], read_file) -> list[str]:
    """Everything the pull request is missing, as messages; empty when it passes.

    ``pr`` is the REST pull request object; ``files`` is ``(filename, status)`` for every changed
    file; ``read_file(path)`` returns a file's text as it would land.
    """
    labels = {label["name"] for label in pr.get("labels") or []}
    body = pr.get("body") or ""
    number = pr["number"]
    found = []

    if pr.get("milestone") is None and "no milestone" not in labels:
        found.append("Set the milestone of the release this ships in, or add the `no milestone` label.")

    if not LINK.search(body) and "no issue" not in labels:
        found.append("Link the issue this delivers: `Closes #N`, or `Part of #N` below the top of a "
                     "stack, or add the `no issue` label.")

    present = [name for name, status in files
               if status != "removed" and name.startswith("docs/release-notes/")]
    stray = [name for name in present if not NOTE.fullmatch(name) and not PAGE.fullmatch(name)]
    if stray:
        found.append(f"Unexpected files in docs/release-notes/: {', '.join(stray)}. A release note "
                     f"is named `<PR number>.<type>.md`.")

    notes = [(match, name) for name in present for match in [NOTE.fullmatch(name)] if match]
    foreign = [name for match, name in notes if int(match[1]) != number]
    if foreign:
        found.append(f"Release notes named for another pull request: {', '.join(foreign)}. If the "
                     f"pull request below in a stack just merged, rebase this branch onto main.")

    if "no release note" not in labels:
        mine = [(match, name) for match, name in notes if int(match[1]) == number]
        if len(mine) != 1:
            found.append(f"Add one release note, `docs/release-notes/{number}.<type>.md`, or the "
                         f"`no release note` label.")
        else:
            match, name = mine[0]
            if match[2] not in TYPES:
                found.append(f"Release-note type `{match[2]}` is not one of {', '.join(TYPES)}.")
            lines = [line for line in read_file(name).splitlines() if line.strip()]
            if len(lines) != 1 or len(lines[0].strip()) > MAX_NOTE:
                found.append(f"A release note is one line of at most {MAX_NOTE} characters: {name}.")
    return found


def _gh_api(*args: str) -> str:
    return subprocess.run(["gh", "api", *args], check=True, capture_output=True, text=True).stdout


def main() -> int:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    repo = os.environ["GITHUB_REPOSITORY"]
    number = event["pull_request"]["number"]
    pr = json.loads(_gh_api(f"repos/{repo}/pulls/{number}"))
    rows = _gh_api("--paginate", f"repos/{repo}/pulls/{number}/files",
                   "--jq", ".[] | [.filename, .status] | @tsv")
    files = [tuple(row.split("\t")) for row in rows.splitlines() if row]
    found = problems(pr, files, lambda path: Path(path).read_text(encoding="utf-8"))

    summary = ["## check-pr", ""] + ([f"- {p}" for p in found] or ["Link, milestone and release note present."])
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write("\n".join(summary) + "\n")
    for p in found:
        print(f"::error::{p}")
    if not found:
        print("Link, milestone and release note present.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
