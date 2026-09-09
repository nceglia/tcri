# tcri — contributor rules

Single-cell TCR+RNA information-theory metrics on scvi-tools / pyro / scanpy.

**The policy is stated once, in `governance/RULES.md`.** This file is the short form. Where
the two disagree, `RULES.md` wins.

## What is locked, and what is not

The core implements two published documents, archived and hashed in `governance/source/`:
Supplementary Note 1 (the model, its objective, the in-silico perturbation) and the metrics
document (the entropies, mutual information, KL). Their equation numbers collide, so a manifest
reference always names its document.

- **Published math is locked.** Code may move *toward* the manuscript freely. The definition it
  is checked against moves only on an explicit instruction from a maintainer. Never rewrite an
  equation to make a result look better; never loosen a manifest to make a failing conformance
  test pass. Surface the implication instead.
- **The public surface is agreed, not frozen.** Adding a function or a module is a stub line, an
  `__all__` entry, and a tag. Changing an existing signature or definition is a contract change.
- **Everything that is not published math is open.** Extensions and experimental work may have
  their equations designed and changed in the repo. The manuscript has nothing to say about them.

If a document is ambiguous, **ask**. Never infer a definition from what makes the code, a test,
or a benchmark come out right.

## The four contracts

| | freezes | manifest | prose | test |
|---|---|---|---|---|
| **API** | the public *interface* | `tests/contracts/api.pyi` | `governance/API_CONTRACT.md` | `tests/test_contract_conformance.py` |
| **Model** | the generative *mathematics* | `tests/contracts/model.py` | `governance/MODEL_CONTRACT.md` | `tests/test_model_contract_conformance.py` |
| **Metrics** | what the *metrics compute* | `tests/contracts/metrics.py` | `governance/METRICS_CONTRACT.md` | `tests/test_metrics_contract_conformance.py` |
| **Training** | how the model is *fit* | `tests/contracts/training.py` | `governance/TRAINING_CONTRACT.md` | `tests/test_training_contract_conformance.py` + `tests/test_training_invariants.py` |

`governance/METHODS_CONFORMANCE.md` is the equation-by-equation code map and deviation history.
`normalize_mode` deliberately departs from the metrics document's eq 6 (`"min"` by default, the
note's mean denominator via `"average"`), and the training contract's `AUTHORED_BOUNDS` are ours
because the note is silent on the schedule. Both are recorded in their manifests.

## Working rules

- **Removal is a test.** When asked to remove something, add it to `tests/test_removal_ledger.py`
  so the suite fails, delete until it passes, and report the test output. Do not report a
  removal as done on any other evidence.
- **After moving or renaming anything, grep the whole repo** with no `--include` filters and no
  directory scoping, then run the thing that consumes the path: build the wheel, build the docs.
  Extensionless files (`CODEOWNERS`), docs trees, and Sphinx `automodule` references are exactly
  what a filtered grep hides.
- **If a change affects a figure, render it and look.** Twice a green suite passed a plot that
  was visibly wrong.
- Run tests with the pinned venv: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q`.
- `dev/` is gitignored scratch. `example/` and `examples/` are outputs of the package, not
  inputs to it.

## Branching

- **Branch from a fresh `main`.** `git checkout main && git pull` first. Never branch off a
  branch with an open PR.
- **After a PR merges, return to `main` and pull** before the next piece of work.
- **Before pushing to a branch with an open PR**, `git fetch && git rev-list --count
  <branch>..origin/main` must be `0`. If not, rebase, and read what landed first.
- PRs describe what was done. Follow-up work goes to GitHub issues, not PR prose or contract
  documents.
