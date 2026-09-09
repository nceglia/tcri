# Rules

The policy for this repository, stated once. The contract documents hold *content*
(definitions, sites, deviations); the manifests under `tests/contracts/` hold *data*; the
conformance tests *enforce*. None of them restate the policy. Where another file disagrees with
this one, this one wins and the other file is stale.

> **Status (2026-09).** The three tiers and the lock described here land as code in the PR
> after this one. Until then the mechanism is the existing manifests, and `.github/CODEOWNERS`
> still gates `tests/contracts/api.pyi`, so an addition to the surface needs an owner's
> approval. That gate moves to the lock when the lock exists.

## What the contracts are for

Two things, and nothing else:

1. **Consistency with the manuscript for the core.** The published mathematics, the model in
   Supplementary Note 1 and the metrics in the metrics document, is what the package computes.
   An equation is never rewritten to make a result look better.
2. **An agreed public surface.** Which functions exist and with what signatures is known,
   written down, and does not drift.

The contracts are not a brake on new work. Adding a function, a module, or a new analysis is
routine; the rules below make it cheap.

**Source of truth.** The manuscript is upstream of every contract. Where a contract disagrees
with the archived document in `governance/source/`, the contract is wrong. Where a document is
ambiguous, ask the authors; never infer a definition from what makes the code, a test, or a
benchmark come out right. The two documents number their equations independently and the
numbers collide, so a manifest reference always names its document.

## Three tiers

Every public callable carries exactly one tag.

| tier | tag | in the surface stub | definition locked | who changes the math | what binds it |
|---|---|---|---|---|---|
| **published** | `@published` | yes | yes | nobody, except through the lock | model, metrics and training conformance; surface and signature |
| **extension** | `@extension` | yes | no | anyone, in the repo, with a docstring and a release note | surface and signature |
| **experimental** | `@experimental` | no | no | anyone | layering only: core never imports it |

*Published* means the function implements something in a source document: the generative
model and its objective, the entropies and mutual information, the in-silico perturbation. Its
definition lives in a manifest entry and its behaviour is pinned by identity tests.

*Extension* means it is ours: the delta family, the contrast statistics, the plots, the
preprocessing. The manuscript has nothing to say about these, so their equations may be
designed and changed here.

*Experimental* means development. It warns on call, has no stub entry, and may change or vanish
without notice. It may use the published model freely; it cannot change it, because the model
and metrics conformance tests trace the live objects regardless of who calls them.

Tags are by name. A `@published` function is linked to its manifest entry by the function's
name, and the manifest entry carries whatever citation text it has. Nothing new cites an
equation number.

## The lock

`tests/contracts/frozen.lock` holds one hash per published name, over its manifest entry and
its stub signature. A test recomputes the hashes and compares.

The lock is the single place a human overrides the core. Editing an equation in code fails the
identity tests. Editing the manifest fails the lock. Editing the lock is a diff in a file with
no other purpose, owned by `.github/CODEOWNERS`, updated by one command that requires a reason.

## Changing things

| you want to | do | touches the lock |
|---|---|---|
| add a function | stub line, `__all__` entry, a tag | no |
| add a module or namespace | `__init__` with `__all__`, a stub block, keep to the layering | no |
| fix code that disagrees with the manifest | fix the code; the identity tests are the detector | no |
| fix a manifest entry that disagrees with the manuscript | change the entry, run the lock command with a reason quoting the manuscript | yes |
| the manuscript was revised | new hash in `SOURCES`, then as above | yes |
| change an extension's math | change it, docstring, release note; the signature stays | no |
| change a published signature | stub and lock | yes |
| remove anything | add it to the removal ledger so the suite fails, delete until it passes, report the output | no |
| graduate | experimental → extension is a stub line; extension → published is a manifest entry and a lock entry | at the last step |

Code may move toward the manifest freely. The manifest and the lock move only on an explicit
instruction from a maintainer.

## What each test enforces

| test | enforces |
|---|---|
| `test_contract_conformance.py` | the set of tagged, non-experimental public callables equals the stub, and every signature matches |
| `test_model_contract_conformance.py` | a live trace of `model()`/`guide()` matches the model manifest: every declared site, no undeclared site, the semantic invariants, every deviation documented |
| `test_metrics_contract_conformance.py` | literal transcriptions of the metric equations and the numeric identities; sources archived and hashed; an open question is not also a sanctioned extension |
| `test_training_contract_conformance.py`, `test_training_invariants.py` | the derived invariants hold; every authored bound has a behavioural test, not a wiring check |
| `test_layout.py` | layering, explicit `__all__`, manifests outside the package, `governance/` outside `docs/` |
| `test_removal_ledger.py` | a symbol recorded as removed is absent from the public namespace |

A failing conformance test means stop and decide: either the change was intended, in which case
the manifest changes first and deliberately, or it is a regression, in which case the code is
fixed. It never means loosen the manifest until it passes.

## For agents

- When a document is ambiguous, ask. Do not infer.
- Never edit a published definition, a manifest, or the lock unless told to in the conversation.
  Surface the implication instead.
- Removal is a test: a claim that something was removed is a green ledger test, not a sentence.
- After moving or renaming anything, grep the whole repository with no filters, then run the
  thing that consumes the path: build the wheel, build the docs.
- Branch from a fresh `main`; before pushing to a branch with an open PR, confirm it is not
  behind.
- `dev/` is gitignored scratch. `example/` and `examples/` are outputs of the package, not
  inputs to it.

## Retired rules

Recorded so they are not rediscovered as requirements.

- **The disposition rule** (`API_CONTRACT.md` §11, "kept or dropped by one test, is it core").
  It sorted the pre-refactor functions. It is history, not a bar on adding non-core work.
- **"A symbol not in the stub must not be public after the refactor."** The refactor is done.
  The surface still matches the stub; additions are routine.
- **"`tl.*` takes an AnnData. That is the whole rule."** It recorded the deletion of a union
  type. The rule is: `tl` reads the AnnData substrate and takes no model; `diag`, and any
  analysis namespace that needs the trained networks, take an AnnData and a model.
- **"Never read the example notebooks."** The examples are now locked scripts. They remain
  outputs, not inputs.
- **"Removal is a hard bar."** It meant deletion is mandatory and read as the opposite. Removal
  is a test.
- **`dev/REFACTOR_AGENDA.md` as required reading.** The refactor it tracked is complete; its
  removal ledger lives in `test_removal_ledger.py`.
