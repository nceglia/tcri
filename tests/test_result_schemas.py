"""The shape of every stored ``tl`` result, pinned.

``@tl_result(version=N)`` declares the schema version of a stored result, but nothing in the code
forces the number to move when the fields do: the layouts live in comments in
``tcri/_state/schemas.py``, and ``pl`` and ``tcri.get`` read the frames by column name. Without this
test a renamed or dropped column passes the suite while making every result stored by an earlier
version unreadable.

This test runs each tool twice on the ``cohort`` fixture — once with only its required arguments,
once with grouping, splitting and draws — records the slots, their column names and index levels,
and compares that against ``tests/snapshots/tl_schemas.json``. Changing the fields fails until the
version is bumped and the snapshot updated::

    pytest tests/test_result_schemas.py --update-schema-snapshot

Names that come from the FIXTURE rather than the schema (the group and split columns, the phenotype
labels) are recorded as placeholders, so the snapshot describes tcri and not the test data. Every
call passes ``inplace=False``, and the minimal call runs on a copy with no registered replicate, so
the shared session fixture is left as it was and "always present" means always.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import tcri
from tcri import get
from tcri._state import keys as K
from tcri._state.storage import _REGISTRY

SNAPSHOT = Path(__file__).parent / "snapshots" / "tl_schemas.json"

#: Fixture column names that would otherwise be recorded as if they were part of the schema.
_GROUPBY, _SPLITBY = "patient", "disease_status"


def _writer(name: str) -> str:
    """The public call that writes a result — never the private module it lives in."""
    return "tcri.perturb.gene_importance" if name == "gene_importance" else f"tcri.tl.{name}"


def _calls(adata):
    """``name -> (minimal kwargs, maximal kwargs)``, from the fixture's own levels."""
    cov = list(adata.obs["covariate"].cat.categories)
    scalar = {"covariate": cov[0]}
    delta = {"cov_from": cov[0], "cov_to": cov[1]}
    wide = {"groupby": _GROUPBY, "splitby": _SPLITBY, "n_samples": 3}
    genes = {"genes": list(adata.var_names[:3])}
    return {
        # joint_distribution takes neither a reference nor grouping
        "joint_distribution": ({}, {"n_samples": 3}),
        "mutual_information": ({**scalar, "null_model": None}, {**scalar, **wide}),
        "clonotypic_entropy": ({**scalar, "null_model": None}, {**scalar, **wide}),
        "phenotypic_entropy": ({**scalar, "null_model": None}, {**scalar, **wide}),
        "phenotypic_flux": ({**delta, "null_model": None}, {**delta, **wide}),
        "delta_clonotypic_entropy": ({**delta, "null_model": None}, {**delta, **wide}),
        "delta_phenotypic_entropy": ({**delta, "null_model": None}, {**delta, **wide}),
        "gene_importance": ({**genes, "null_model": None}, {**genes, **wide}),
    }


def _run(name, kwargs, model, adata):
    """Call one tool with ``inplace=False``, so the shared fixture is left as it was."""
    if name == "gene_importance":
        return tcri.perturb.gene_importance(model, adata, inplace=False, **kwargs)
    return getattr(tcri.tl, name)(adata, inplace=False, **kwargs)


def _observe(result, placeholders):
    """The slots of a result: type, column names and index levels, fixture names masked."""
    def names(labels):
        return sorted({placeholders.get(str(label), str(label)) for label in labels})

    return {
        slot: {
            "type": type(value).__name__,
            "columns": names(value.columns) if isinstance(value, pd.DataFrame) else [],
            "index": [str(level) for level in value.index.names] if isinstance(value, pd.DataFrame) else [],
        }
        for slot, value in result.items()
    }


def _placeholders(adata):
    phenotypes = adata.uns.get(K.PHENOTYPE_CATEGORIES, [])
    return {_GROUPBY: "<groupby>", _SPLITBY: "<splitby>",
            **{str(p): "<phenotype>" for p in phenotypes}}


@pytest.fixture(scope="module")
def observed(cohort):
    """What every tool stores, at its minimal and maximal call."""
    model, adata = cohort
    placeholders = _placeholders(adata)

    # No registered replicate, so `groupby=None` adds no group column: "always" means always.
    bare = adata.copy()
    bare.uns[K.METADATA] = {**bare.uns[K.METADATA], "replicate": None}

    seen = {}
    for name, (minimal, maximal) in _calls(adata).items():
        slots = {}
        for label, kwargs, on in (("minimal", minimal, bare), ("maximal", maximal, adata)):
            for slot, shape in _observe(_run(name, kwargs, model, on), placeholders).items():
                entry = slots.setdefault(slot, {"type": shape["type"],
                                                "columns": {}, "index": {}})
                entry["type"] = shape["type"] if entry["type"] == "NoneType" else entry["type"]
                entry["columns"][label] = shape["columns"]
                entry["index"][label] = shape["index"]
        seen[get._RESULTS[name]] = {"writer": _writer(name),
                                    "version": _REGISTRY[get._RESULTS[name]].tcri_schema_version,
                                    "slots": slots}
    return seen


def test_stored_result_schemas_match_the_snapshot(observed, request):
    if request.config.getoption("--update-schema-snapshot"):
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps({"tools": observed}, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"snapshot rewritten: {SNAPSHOT}")

    pinned = json.loads(SNAPSHOT.read_text())["tools"]
    update = "rerun with --update-schema-snapshot"

    for key in sorted(set(pinned) | set(observed)):
        assert key in pinned, f"{key} is a new stored result: {update}"
        assert key in observed, (
            f"{key} is pinned but nothing writes it; removing a stored result is a removal "
            f"(tests/test_removal_ledger.py)"
        )
        was, now = pinned[key], observed[key]
        assert now["version"] >= was["version"], (
            f"{key}: schema version went down, {was['version']} -> {now['version']}"
        )
        if now["slots"] != was["slots"] and now["version"] == was["version"]:
            pytest.fail(
                f"{key}: the stored fields changed but @tl_result(version={now['version']}) did "
                f"not. Bump the version, add a `breaking` release note, then {update}.\n  was: {was['slots']}\n  now: {now['slots']}"
            )
        assert now == was, f"{key}: the snapshot is out of date: {update}"
