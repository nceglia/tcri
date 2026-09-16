# Rules

Four contracts, one file each, one test file each. Everything else is ordinary code.

| contract | file | what it pins | enforced by |
|---|---|---|---|
| Model | `governance/MODEL_CONTRACT.md` | the generative model, the variational family, the objective | `tests/test_model_contract_conformance.py` traces `model()`/`guide()` against the block in the file |
| Training | `governance/TRAINING_CONTRACT.md` | how the model is fit: invariants of the objective and the bounds we author | `tests/test_training_invariants.py`, behaviourally |
| Metrics | `governance/METRICS_CONTRACT.md` | what every `tl` number means, its defaults, and its value on a reference joint | `tests/test_metrics_contract_conformance.py`: reference recomputation, identities, pinned values |
| API | `governance/API_CONTRACT.md` | the public surface and every signature | `tests/test_contract_conformance.py` parses the stub in the file: set equality and signature equality |

The contract files are the source of truth. The manuscript and its supplementary note are
provenance, not a specification, and are not kept in the repository.

## Changing things

- Code and its contract change in the same PR. A conformance failure means either the change
  was intended, in which case the contract file changes deliberately in that PR, or it is a
  regression. It never means loosen the test until it passes.
- Adding a public function: declare it in the stub in `API_CONTRACT.md`. Changing a signature:
  the same.
- Changing the model: update the block and the prose in `MODEL_CONTRACT.md`; add a test that
  fails on the parent commit for the behaviour that changed.
- Changing what a metric computes: update `METRICS_CONTRACT.md` including the pinned values,
  and say why in the PR.

## Workflow

- Work that takes more than one PR has a tracking issue, labeled `tracking`; its pieces are
  sub-issues. An issue has one parent.
- Every PR links the issue it delivers (`Closes #N`, or `Part of #N` below the top of a stack) and
  sets the milestone of the release it ships in. Labels `no issue` and `no milestone` are the
  exemptions. The procedure is in the contributing guide in the docs.
- Every PR adds one release-note line at `docs/release-notes/<PR>.<type>.md` (`breaking`, `feat`,
  `fix` or `perf`), or carries the `no release note` label.
- The `check-pr` workflow reports a missing link, milestone or release note on every PR.
- Versions come from git tags; no file holds a version. Releases are published GitHub Releases;
  the procedure is the release page in the docs.
- The shape of a stored `tl` result is versioned by `@tl_result(version=N)`. Changing its fields
  bumps N, keeps a reader for results that must still load, and carries a `breaking` release
  note; `tests/test_result_schemas.py` enforces it. Results written by past releases must still
  load: `tests/test_result_archives.py` reads an archive kept for each one.
- A saved session's layout is versioned by `SESSION_FORMAT_VERSION`; a session written by a newer
  tcri is refused rather than half-loaded.
- `main` is releasable at every merge: a new function or method stays private until the last PR of
  its sub-issue, which makes it public, adds it to `__all__` and declares it in `API_CONTRACT.md`.

## Working rules

- **Removal is a test.** When asked to remove a public symbol, add it to
  `tests/test_removal_ledger.py` so the suite fails, delete until it passes, and report the
  output.
- **After moving or renaming anything, grep the whole repository** with no filters, then run
  what consumes the path: build the wheel, build the docs.
- **If a change affects a figure, render it and look.**
- **Branch from a fresh `main`**, or, in a stack, from the branch of the pull request below. Stacks
  are GitHub stacked pull requests; they merge into `main` one pull request at a time from the
  bottom, with a merge commit. Update a stacked branch by rebasing. Before pushing, confirm the
  branch is not behind its base; if it is, rebase.
- Run tests with the pinned venv: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q`.
- `dev/` is gitignored scratch. `example/` and `examples/` are outputs of the package.
- PRs describe what was done, compactly. Follow-up work goes to GitHub issues.
