"""Smoke and invariant tests for joint_distribution (n_samples=0)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

# Load implementation without importing tcri/__init__.py (heavy optional deps).
_ROOT = Path(__file__).resolve().parents[2]
_PRE = _ROOT / "tcri" / "preprocessing" / "_preprocessing.py"
_NAME = "tcri.preprocessing._preprocessing"
if _NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_NAME, _PRE)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _mod
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
joint_distribution = sys.modules[_NAME].joint_distribution


def test_joint_distribution_n_samples_zero_shape(mock_adata):
    df = joint_distribution(mock_adata, "T1", n_samples=0, temperature=1.0)
    assert list(df.columns) == mock_adata.uns["tcri_phenotype_categories"]
    # One row per clonotype in this covariate (20 clones × 2 timepoints → 20 rows for T1)
    assert len(df) == 20


def test_joint_distribution_rows_sum_to_one(mock_adata):
    df = joint_distribution(mock_adata, "T1", n_samples=0, temperature=1.0)
    sums = df.to_numpy().sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5, atol=1e-5)


def test_joint_distribution_respects_clone_filter(mock_adata):
    # Odd-index clones (e.g. clone_1) do not appear under covariate T1 for this fixture layout
    sub = joint_distribution(
        mock_adata, "T1", n_samples=0, clones=["clone_0", "clone_2"]
    )
    # mock_adata uses two ct rows per (covariate, clone); expect duplicates in the index
    assert set(sub.index) == {"clone_0", "clone_2"}
    assert len(sub) == 4
    np.testing.assert_allclose(sub.to_numpy().sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)
