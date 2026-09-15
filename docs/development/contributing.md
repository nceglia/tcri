# Contributing

How work on tcri is planned, split into pull requests and merged. The rules themselves are stated
once, in [`governance/RULES.md`](https://github.com/nceglia/tcri/blob/main/governance/RULES.md).

## Goals and sub-issues

A **goal** is an outcome that takes more than one pull request, such as better Scirpy
integration. It is a GitHub issue labeled `goal`. Its **subtargets** are sub-issues of that issue:
one reviewable outcome each, delivered by one pull request or a stack. An issue has one parent, so
a sub-issue belongs to exactly one goal.

Work outside a goal links an ordinary issue. Bugs and feature requests have their own issue forms.

## Opening a goal

Open an issue with the **Goal** form (Issues → New issue → Goal). It asks for the outcome, the
subtargets, the footprint (files, modules, public symbols and stored results the goal touches) and
what is out of scope. Then create one issue per subtarget and attach it as a sub-issue. The goal
issue shows progress as sub-issues close.

Before starting a goal, compare its footprint with the open `goal` issues. Where two goals touch
the same file or public symbol, agree which lands first and note it on both issues.

## Stacks

A subtarget that needs several pull requests is a stack: each pull request branches from the one
below it, using GitHub's stacked pull requests.

- Every pull request in the stack says `Part of #N`, where `#N` is the sub-issue.
- The pull request that completes the sub-issue says `Closes #N` when everything the sub-issue asks
  can be checked before merging. Otherwise close the sub-issue by hand once it has been checked.
- Merge the stack into `main` **one pull request at a time, from the bottom, with a merge commit**.
  Do not merge several at once: each pull request should be its own commit on `main`.
- After the pull request below merges, the next one is rebased onto `main`. If you update a stacked
  branch yourself, rebase it; do not merge `main` into it.

## Milestones

Milestones are named after releases (`0.13.0`). Every pull request sets the milestone of the release
it ships in.

## Labels

- `goal` marks a goal issue.
- `no milestone` marks a pull request that is not part of a release; `no issue` marks one that links
  no issue.
- Area labels are optional and say which part of the package an issue or pull request touches:
  `area: pp`, `area: tl`, `area: pl`, `area: model`, `area: infra` (packaging, CI, docs build). They
  make it easier to spot two goals working on the same part of the package.

## Keeping `main` releasable

Any commit on `main` could become a release, so unfinished public API never lands there. A new
function or method stays private until the last pull request of its subtarget: give it an
underscore name, or do not import it into its `tcri.*` namespace. Methods on `TCRIModel` stay
underscore-named, because the API conformance test sees every public method. The last pull request
makes it public, adds it to `__all__`, declares it in `governance/API_CONTRACT.md`, and greps for the
old name.

## Rules

Contracts, tests and branching rules:
[`governance/RULES.md`](https://github.com/nceglia/tcri/blob/main/governance/RULES.md).
