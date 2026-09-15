# Contributing

How work on tcri is planned, split into pull requests and merged. The rules themselves are stated
once, in [`governance/RULES.md`](https://github.com/nceglia/tcri/blob/main/governance/RULES.md).

## Tracking issues and sub-issues

Work that takes more than one pull request, such as better Scirpy integration, has a **tracking
issue**: a GitHub issue labeled `tracking` that states the outcome. Its pieces are **sub-issues** of
that issue: one reviewable outcome each, delivered by one pull request or a stack. An issue has one
parent, so a sub-issue belongs to exactly one tracking issue.

Work outside a tracking issue links an ordinary issue. Bugs and feature requests have their own
issue forms.

## Opening a tracking issue

Open an issue with the **Tracking issue** form (Issues → New issue → Tracking issue). It asks for the
outcome, the sub-issues, the footprint (files, modules, public symbols and stored results the work
touches) and what is out of scope. Then create one issue per piece and attach it as a sub-issue. The
tracking issue shows progress as sub-issues close.

Before starting, compare the footprint with the open `tracking` issues. Where two touch the same
file or public symbol, agree which lands first and note it on both issues.

## Stacks

A sub-issue that needs several pull requests is a stack: each pull request branches from the one
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

- `tracking` marks a tracking issue.
- `no milestone` marks a pull request that is not part of a release; `no issue` marks one that links
  no issue.
- Area labels are optional and say which part of the package an issue or pull request touches:
  `area: pp`, `area: tl`, `area: pl`, `area: model`, `area: infra` (packaging, CI, docs build). They
  make it easier to spot two tracking issues working on the same part of the package.

## Keeping `main` releasable

Any commit on `main` could become a release, so unfinished public API never lands there. A new
function or method stays private until the last pull request of its sub-issue: give it an
underscore name, or do not import it into its `tcri.*` namespace. Methods on `TCRIModel` stay
underscore-named, because the API conformance test sees every public method. The last pull request
makes it public, adds it to `__all__`, declares it in `governance/API_CONTRACT.md`, and greps for the
old name.

## Rules

Contracts, tests and branching rules:
[`governance/RULES.md`](https://github.com/nceglia/tcri/blob/main/governance/RULES.md).
