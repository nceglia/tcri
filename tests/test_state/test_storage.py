"""The ``@tl_result`` storage convention.

The load-bearing tests here are the ones that go through a **real** ``write_h5ad`` →
``read_h5ad``. Encode-only assertions would pass under a naive label-keyed scheme and then lose
data the first time someone saved a session — h5py treats ``/`` as a path separator, so a
clonotype id like ``"TRB/1"`` used as a dict key silently becomes nested groups.

Every test stores through the real decorator rather than calling ``_encode`` directly, so the
params capture and the ``inplace``/``key_added`` handling are exercised too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from tcri._state.storage import (
    _REGISTRY,
    _encode,
    decode_blob,
    load_result,
    load_result_params,
    tl_result,
    with_resolved_params,
)
import tcri
from tcri.get import _PROVENANCE

#: Labels that break a naive encoder. The "/" entries are the ones that matter — h5py would
#: split them into groups. Real TCR ids and cell-type names look like this.
_EXOTIC = ["TRB/1", "CD8/naive", "naïve B", "with space", "αβ T"]


@pytest.fixture
def adata():
    return AnnData(X=np.zeros((4, 2), dtype="float32"))


def _store(adata, key, payload, version=1, **call_kwargs):
    """Store ``payload`` through the REAL decorator, not through ``_encode``."""

    @tl_result(key=key, version=version)
    def _tool(adata, *, groupby=None, key_added=None, inplace=True):
        return payload

    return _tool(adata, **call_kwargs)


# ── payload / params separation ──────────────────────────────────────────────

def test_dataframe_result_keeps_payload_and_params_separate(adata):
    df = pd.DataFrame({"MI": [0.1, 0.2]}, index=pd.Index(["P0", "P1"], name="patient"))
    returned = _store(adata, "tcri_x", df, version=7, groupby="patient")

    pd.testing.assert_frame_equal(returned, df)
    got = load_result(adata, "tcri_x")
    pd.testing.assert_frame_equal(got, df)
    assert got.index.name == "patient"

    # exact equality proves adata / key_added / inplace are excluded
    assert load_result_params(adata, "tcri_x") == {"groupby": "patient"}
    assert adata.uns["tcri_x"]["version"] == 7


def test_data_param_names_the_store_target_and_drops_the_objects_before_it(adata):
    """A tool that takes the fitted model first stores into ``adata`` and records neither.

    Everything before the data argument is an object the tool operates on, not a setting.
    Without the exclusion the model itself would be provenance -- unserialisable, and the
    ``.h5ad`` write would fail on it.
    """
    class _Model:
        pass

    @tl_result(key="tcri_q", data_param="adata")
    def _query(model, adata, *, genes=None, key_added=None, inplace=True):
        return pd.DataFrame({"value": [1.0]})

    _query(_Model(), adata, genes=["a"])
    assert load_result_params(adata, "tcri_q") == {"genes": ["a"]}

    with pytest.raises(TypeError, match="no parameter"):
        tl_result(key="tcri_q", data_param="nope")(_query)


def test_params_capture_includes_untouched_defaults(adata):
    """Provenance that records only explicitly-passed arguments cannot answer "what was this
    run with" — which is the entire point of storing it."""

    @tl_result(key="tcri_y")
    def _tool(adata, *, covariate="cov_0", n_samples=0, weighted=False,
              key_added=None, inplace=True):
        return pd.DataFrame({"v": [1.0]})

    _tool(adata, covariate="pre")
    params = load_result_params(adata, "tcri_y")
    assert params == {"covariate": "pre", "n_samples": 0, "weighted": False}


def test_inplace_false_skips_the_write_but_still_returns(adata):
    df = pd.DataFrame({"v": [1.0]})
    returned = _store(adata, "tcri_z", df, inplace=False)
    pd.testing.assert_frame_equal(returned, df)
    assert "tcri_z" not in adata.uns


def test_key_added_overrides_the_canonical_key(adata):
    df = pd.DataFrame({"v": [1.0]})
    _store(adata, "tcri_canon", df, key_added="my_run")
    assert "my_run" in adata.uns and "tcri_canon" not in adata.uns
    pd.testing.assert_frame_equal(load_result(adata, "my_run"), df)


def test_with_resolved_params_records_the_effective_value(adata):
    """A body that resolves ``groupby=None`` to the registry's replicate column must record the
    column it actually used, not the placeholder the caller passed."""

    @tl_result(key="tcri_r")
    def _tool(adata, *, groupby=None, key_added=None, inplace=True):
        effective = groupby or "patient"
        return with_resolved_params({"table": pd.DataFrame({"v": [1.0]})}, groupby=effective)

    result = _tool(adata)
    assert "__tcri_resolved_params__" not in result, "the tag leaked into the returned object"
    assert load_result_params(adata, "tcri_r")["groupby"] == "patient"


# ── h5ad safety ──────────────────────────────────────────────────────────────

def test_encode_keys_dataframe_columns_positionally(adata):
    df = pd.DataFrame([[1.0, 2.0]], columns=_EXOTIC[:2])
    blob = _encode(df)
    assert set(blob["data"]) == {"0", "1"}
    assert all("/" not in k for k in blob["data"]), "a user label was used as a dict key"
    assert list(blob["columns"]) == _EXOTIC[:2], "labels must survive as VALUES"
    pd.testing.assert_frame_equal(decode_blob(blob), df)


def test_duplicate_column_labels_survive():
    df = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["dup", "dup", "other"])
    pd.testing.assert_frame_equal(decode_blob(_encode(df)), df)


def test_label_keyed_dict_is_tagged(adata):
    payload = {c: np.arange(3.0) for c in _EXOTIC[:2]}
    blob = _encode(payload)
    assert blob["__tcri_map__"] == 1
    assert list(blob["keys"]) == _EXOTIC[:2]
    assert set(blob["values"]) == {"0", "1"}


@pytest.mark.parametrize("payload_kind", ["dataframe", "flat_map", "nested_map"])
def test_h5ad_roundtrip(tmp_path, payload_kind):
    """The test that actually matters: write a real .h5ad and read it back."""
    adata = AnnData(X=np.zeros((4, 2), dtype="float32"))

    if payload_kind == "dataframe":
        payload = pd.DataFrame(
            np.arange(10.0).reshape(5, 2), columns=["pre", "post"],
            index=pd.Index(_EXOTIC, name="clonotype"),
        )
    elif payload_kind == "flat_map":
        payload = {c: np.arange(3.0) for c in _EXOTIC}
    else:
        payload = {g: {c: np.arange(2.0) for c in _EXOTIC} for g in ("R", "NR")}

    _store(adata, "tcri_rt", payload, version=3)
    path = tmp_path / "rt.h5ad"
    adata.write_h5ad(path)

    import anndata as ad
    back = load_result(ad.read_h5ad(path), "tcri_rt")

    if payload_kind == "dataframe":
        pd.testing.assert_frame_equal(back, payload)
    elif payload_kind == "flat_map":
        assert set(back) == set(payload)
        for c in payload:
            np.testing.assert_allclose(back[c], payload[c])
    else:
        # dict payloads carry provenance through load_result -- see
        # test_load_result_provenance_asymmetry. tcri.get.result() is what strips it.
        assert set(back) - set(_PROVENANCE) == {"R", "NR"}
        for g in payload:
            assert set(back[g]) == set(payload[g])


def test_missing_result_raises_with_a_useful_message(adata):
    with pytest.raises(KeyError, match="run the matching tcri.tl tool first"):
        load_result(adata, "tcri_never_run")


def test_schema_check_names_the_missing_keys(adata):
    from typing import TypedDict

    class Needs(TypedDict):
        table: object
        result: object

    @tl_result(key="tcri_s", schema=Needs)
    def _tool(adata, *, key_added=None, inplace=True):
        return {"table": pd.DataFrame({"v": [1.0]})}   # 'result' missing

    with pytest.raises(ValueError, match=r"missing required keys \['result'\]"):
        _tool(adata)


def test_load_result_provenance_asymmetry_is_deliberate(adata):
    """``load_result`` returns ``params``/``version`` for a DICT payload but not for a
    DataFrame one, because the ``__tcri_df__`` tag wins in ``_decode`` and the siblings are
    dropped on the floor.

    That asymmetry is easy to trip over, so it is pinned here rather than discovered.
    ``tcri.get.result()`` is the accessor that normalises it — it strips provenance so the
    return mirrors the tool's natural result in BOTH cases.
    """
    _store(adata, "tcri_df_case", pd.DataFrame({"v": [1.0]}))
    _store(adata, "tcri_dict_case", {"table": pd.DataFrame({"v": [1.0]})})

    df_back = load_result(adata, "tcri_df_case")
    assert isinstance(df_back, pd.DataFrame)          # no provenance to strip

    dict_back = load_result(adata, "tcri_dict_case")
    assert {"params", "version"} <= set(dict_back)     # provenance rides along

    # params are reachable identically in both cases
    assert isinstance(load_result_params(adata, "tcri_df_case"), dict)
    assert isinstance(load_result_params(adata, "tcri_dict_case"), dict)


def test_a_multiindex_result_survives_h5ad(tmp_path):
    """`index.to_numpy()` on a MultiIndex yields an object array of TUPLES, and h5py has no
    writer for a tuple.

    `joint_distribution(n_samples>0)` puts a (clonotype, sample_id) MultiIndex on its
    `table`, so an encoder that passes the index straight through leaves the whole cached
    result unwritable to .h5ad. It fails as `TypeError: Can't implicitly convert non-string
    objects to strings` named against the enclosing group -- a long way from the index that
    caused it -- and only when someone saves a session, which is why the write is exercised
    here rather than the encoding alone.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    from tcri._state.storage import decode_blob, tl_result

    idx = pd.MultiIndex.from_product([["clone/1", "clone 2"], [0, 1]],
                                     names=["clonotype", "sample_id"])
    frame = pd.DataFrame({"a": np.arange(4.0), "b": np.arange(4.0)[::-1]}, index=idx)

    @tl_result(key="tcri_multi_probe", version=1)
    def probe(adata, *, key_added=None, inplace=True):
        return {"table": frame}

    adata = ad.AnnData(X=np.zeros((2, 2), dtype="float32"))
    probe(adata)

    path = tmp_path / "multi.h5ad"
    adata.write_h5ad(path)                      # the step h5py refuses on a tuple index
    back = decode_blob(ad.read_h5ad(path).uns["tcri_multi_probe"])["table"]

    assert isinstance(back.index, pd.MultiIndex)
    assert list(back.index.names) == ["clonotype", "sample_id"]
    pd.testing.assert_frame_equal(back, frame, check_dtype=False)


# ── provenance and schema versions ───────────────────────────────────────────

def test_the_blob_records_the_tool_and_the_tcri_version(adata):
    """Which tool wrote a result, and which tcri, without asking the caller to remember."""
    _store(adata, "tcri_x", {"table": pd.DataFrame({"v": [1.0]})}, version=3)

    blob = adata.uns["tcri_x"]
    assert blob["tool"] == "tcri_x"
    assert blob["version"] == 3
    assert blob["tcri_version"] == tcri.__version__


def test_get_result_strips_the_provenance():
    """`tcri.get.result` returns what the tool returned, provenance removed."""
    adata = AnnData(X=np.zeros((2, 2), dtype="float32"))
    _store(adata, "tcri_x", {"table": pd.DataFrame({"v": [1.0]})})

    got = tcri.get.result(adata, "tcri_x")
    assert set(got) == {"table"}


def _registered(monkeypatch, key, version):
    """Put a fake tool in the registry, as if `tcri` declared `key` at `version`."""
    class _Tool:
        tcri_schema_version = version
    monkeypatch.setitem(_REGISTRY, key, _Tool)


def test_a_newer_schema_version_refuses_to_load(adata, monkeypatch):
    _store(adata, "tcri_x", {"table": pd.DataFrame({"v": [1.0]})}, version=4)
    _registered(monkeypatch, "tcri_x", 2)

    with pytest.raises(ValueError, match=r"schema v4.*reads up to v2"):
        load_result(adata, "tcri_x")


def test_an_older_schema_version_says_to_recompute(adata, monkeypatch):
    _store(adata, "tcri_x", {"table": pd.DataFrame({"v": [1.0]})}, version=1)
    _registered(monkeypatch, "tcri_x", 2)

    with pytest.raises(ValueError, match="no longer read"):
        load_result(adata, "tcri_x")


def test_a_result_written_before_the_tool_key_existed_still_loads(adata, monkeypatch):
    """A result stored before the blob recorded a `tool` carries `version` alone; `tcri.get`
    supplies the tool from the registry."""
    _store(adata, "tcri_x", {"table": pd.DataFrame({"v": [1.0]})}, version=1)
    del adata.uns["tcri_x"]["tool"]
    _registered(monkeypatch, "tcri_x", 1)

    assert set(load_result(adata, "tcri_x", tool="tcri_x")) == {"table", "params", "version",
                                                                "tcri_version"}
    assert "table" in load_result(adata, "tcri_x")   # no tool passed: decoded as it stands


def test_tools_defined_outside_tcri_are_not_registered(adata):
    """A tool defined in a test file cannot displace a real one in the registry."""
    _store(adata, "tcri_not_registered", {"table": pd.DataFrame({"v": [1.0]})})

    assert "tcri_not_registered" not in _REGISTRY


def test_every_tool_tcri_get_knows_is_registered():
    from tcri import get

    assert set(get._RESULTS.values()) <= set(_REGISTRY)


def test_the_registry_records_each_tools_schema_version():
    versions = {key: tool.tcri_schema_version for key, tool in _REGISTRY.items()}

    assert versions and all(isinstance(v, int) and v >= 1 for v in versions.values())
