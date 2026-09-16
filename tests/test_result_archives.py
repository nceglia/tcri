"""Results written by past releases, read back by the current tcri.

The schema snapshot pins what the tools store *now*; it cannot notice that a result written by a
released tcri stopped loading. This test keeps one small ``.h5ad`` per release under
``tests/data/archives/<version>/`` and reads every result in it through ``tcri.get``, so a schema
change that leaves old results unreadable fails here rather than in someone's saved analysis.

An archive is written as part of cutting a release (see the release page in the docs)::

    pytest tests/test_result_archives.py --write-archive 0.13.0

It holds only the eight main-fit result entries and the phenotype names — no ``X``, no ``obs`` —
because that is all reading them back needs.
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata
import pytest
from anndata import AnnData

import tcri
from tcri import get
from tcri._state import keys as K
from tcri._state.schemas import validate
from tcri._state.storage import _REGISTRY

from .test_result_schemas import SNAPSHOT, _calls, _observe, _placeholders, _run

ARCHIVES = Path(__file__).parent / "data" / "archives"

#: Measured at about 0.8 MB for the eight results; the cap leaves room without hiding growth.
MAX_ARCHIVE_BYTES = 1_500_000


def _archives():
    return sorted(p for p in ARCHIVES.glob("*/results.h5ad")) if ARCHIVES.exists() else []


def _writer_versions(adata) -> set:
    """The tcri versions recorded in the stored results of an archive."""
    return {blob.get("tcri_version") for blob in adata.uns.values()
            if isinstance(blob, dict) and "tcri_version" in blob}


def test_write_archive(cohort, request):
    """Write this release's archive. Skipped unless ``--write-archive VERSION`` is passed."""
    version = request.config.getoption("--write-archive")
    if not version:
        pytest.skip("only when cutting a release: --write-archive VERSION")

    model, fixture = cohort
    adata = fixture.copy()
    for name, (_minimal, maximal) in _calls(adata).items():
        _run(name, maximal, model, adata, inplace=True)

    stored = {key: adata.uns[key] for key in get._RESULTS.values() if key in adata.uns}
    assert set(stored) == set(get._RESULTS.values()), "some tool stored nothing"
    # the phenotype names, so the columns named after them can be recognised when reading back
    stored[K.PHENOTYPE_CATEGORIES] = adata.uns[K.PHENOTYPE_CATEGORIES]

    out = ARCHIVES / version
    out.mkdir(parents=True, exist_ok=False)
    AnnData(uns=stored).write_h5ad(out / "results.h5ad")       # uncompressed: many small datasets
    (out / "anndata.txt").write_text(f"{anndata.__version__}\n")

    size = (out / "results.h5ad").stat().st_size
    assert size < MAX_ARCHIVE_BYTES, f"archive is {size} bytes, over the {MAX_ARCHIVE_BYTES} cap"


#: One case per archive on disk, so a missing archive directory skips rather than errors.
_each_archive = pytest.mark.parametrize(
    "archive",
    _archives() or [pytest.param(None, marks=pytest.mark.skip(reason="no archives written yet"))],
    ids=lambda p: p.parent.name if p is not None else "none",
)


@_each_archive
def test_an_archive_was_written_by_the_release_it_is_filed_under(archive):
    """The whole point of an archive is which release wrote it.

    Writing one from a working copy whose installed metadata is stale — a leftover editable
    install, a stray ``egg-info`` — silently files one release's results under another's name,
    and every later comparison is then against the wrong release.
    """
    recorded = _writer_versions(anndata.read_h5ad(archive))

    assert recorded == {archive.parent.name}, (
        f"{archive.parent.name} archive records writer version(s) {sorted(recorded)}; write it "
        f"from a checkout of the release tag, with tcri installed from that checkout"
    )


@_each_archive
def test_archived_results_still_load(archive):
    """Every result in a released archive loads, keeps its schema, and matches the pinned columns."""
    adata = anndata.read_h5ad(archive)
    pinned = json.loads(SNAPSHOT.read_text())["tools"]
    placeholders = _placeholders(adata)
    release = archive.parent.name

    for name, key in get._RESULTS.items():
        if key not in adata.uns:
            continue
        result = tcri.get.result(adata, name)
        validate(_REGISTRY[key].tcri_schema, result, name=f"{release}:{name}")

        for slot, shape in _observe(result, placeholders).items():
            if shape["type"] != "DataFrame":
                continue
            columns = pinned[key]["slots"][slot]["columns"]
            assert set(columns["minimal"]) <= set(shape["columns"]) <= set(columns["maximal"]), (
                f"{release}:{name}.{slot} columns {shape['columns']} are outside the pinned schema; "
                f"a reader for the archived version is missing"
            )
