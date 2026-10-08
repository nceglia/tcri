"""Private helpers for clonotype labels, shared by preprocessing and the model.

This module may import numpy, pandas and :mod:`tcri._state.keys`, and nothing else from tcri, so
that both ``preprocessing`` and ``model`` can import it without reaching across layers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _missing_clonotypes(values: pd.Series) -> pd.Series:
    """Boolean mask of the cells that have no clonotype.

    A clonotype is missing when it is NaN or None, an empty or whitespace-only string, or the
    literal string ``"nan"``. ``"nan"`` is what ``astype(str)`` makes of NaN, and an empty string
    is how many tables write a missing value, so both are read as missing rather than as a clone
    of that name. This is the one definition of a missing clonotype in tcri.

    Parameters
    ----------
    values
        A clonotype column, of any dtype. For a categorical column each category is checked once
        and cells are read through their codes; an unused category marks no cell.

    Returns
    -------
    pd.Series
        Boolean, on the same index as ``values``; ``True`` where the clonotype is missing.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        bad = np.flatnonzero(_missing_clonotypes(pd.Series(values.cat.categories)).to_numpy())
        codes = values.cat.codes.to_numpy()
        return pd.Series((codes < 0) | np.isin(codes, bad), index=values.index, name=values.name)
    missing = values.isna().to_numpy(dtype=bool)
    if pd.api.types.is_string_dtype(values.dtype):
        text = values.astype("string")
        blank = (text.str.strip().eq("") | text.eq("nan")).fillna(False)
        missing = missing | blank.to_numpy(dtype=bool)
    return pd.Series(missing, index=values.index, name=values.name)
