# tcri — contributor rules

Single-cell TCR+RNA information-theory metrics on scvi-tools / pyro / scanpy.

**The policy is stated once, in `governance/RULES.md`.** This file is the short form. Where
the two disagree, `RULES.md` wins.

## The four contracts

One file each, one test each. The contract files are the source of truth; the manuscript is
provenance and is not kept in the repository.

| | file | enforced by |
|---|---|---|
| **Model** | `governance/MODEL_CONTRACT.md` | `tests/test_model_contract_conformance.py` (traces `model()`/`guide()` against the block in the file) |
| **Training** | `governance/TRAINING_CONTRACT.md` | `tests/test_training_invariants.py` (behavioural) |
| **Metrics** | `governance/METRICS_CONTRACT.md` | `tests/test_metrics_contract_conformance.py` (reference recomputation, identities, pinned values) |
| **API** | `governance/API_CONTRACT.md` | `tests/test_contract_conformance.py` (the stub in the file: set and signature equality) |

Code and its contract change in the same PR. A conformance failure means the change was
intended, in which case the contract file changes deliberately, or it is a regression. Never
loosen a test to make it pass.

## Working rules

- **Removal is a test.** When asked to remove a public symbol, add it to
  `tests/test_removal_ledger.py` so the suite fails, delete until it passes, and report the
  test output.
- **After moving or renaming anything, grep the whole repo** with no filters and no directory
  scoping, then run the thing that consumes the path: build the wheel, build the docs.
- **If a change affects a figure, render it and look.**
- Run tests with the pinned venv: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q`.
- `dev/` is gitignored scratch. `example/` and `examples/` are outputs of the package, not
  inputs to it.
- PRs are compact summaries of what was done. Follow-up work goes to GitHub issues.

## Workflow

- Work that takes more than one PR has a tracking issue (label `tracking`); its pieces are
  sub-issues.
- Every PR links the issue it delivers (`Closes #N`, or `Part of #N` below the top of a stack)
  and sets the milestone it ships in, or carries `no issue` / `no milestone`.
- Every PR adds one release-note line at `docs/release-notes/<PR>.<type>.md`, or carries
  `no release note`.
- `check-pr` reports what a PR is missing; read its output after pushing.
- New functions and methods stay private until the last PR of their sub-issue.
- Full text: `governance/RULES.md`; how-to: `docs/development/contributing.md`.

## Branching

- **Branch from a fresh `main`** (`git checkout main && git pull`), or, in a stack, from the
  branch of the PR below. Stacks merge into `main` one PR at a time from the bottom.
- **After a PR merges, return to `main` and pull** before work that is not the next PR of the
  same stack.
- **Before pushing to a branch with an open PR**, `git fetch && git rev-list --count
  <branch>..origin/<base>` must be `0`, where `<base>` is the PR's base branch. If not,
  rebase, and read what landed first.
