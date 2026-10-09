"""Clone-like column helpers: ``setup_anndata``'s ``clonotype_key="auto"`` and the column list in
``pp.from_mudata``'s not-found error."""
from __future__ import annotations

from collections.abc import Collection

import pandas as pd

__all__ = ["clone_like_candidates", "resolve_clonotype_key"]

#: The families ``clonotype_key="auto"`` tries, in order.
_FAMILY_ORDER = ("clone_id", "cc", "size")


def _canonical_col(name: str, *, source_prefix: str) -> str:
    prefix = f"{source_prefix}:"
    return name[len(prefix):] if name.startswith(prefix) else name


def _name_family(canon: str) -> str | None:
    """The family a column name marks on its own: ``clone_id``, ``cc`` for ``cc_*``, or None."""
    if canon == "clone_id":
        return "clone_id"
    if canon.startswith("cc_"):
        return "cc"
    return None


def _size_pair(canon: str, cols: Collection[str], *, source_prefix: str) -> list[str]:
    """The present columns that a ``<base>_size`` column counts: ``<base>``, then
    ``<source_prefix>:<base>``. Empty when ``canon`` is not a ``_size`` column or its base is
    absent.

    Callers check this before :func:`_name_family`. Scirpy writes each clonotype definition with
    a size column (``clone_id_size``, ``cc_<x>_size``), and that column stands for its base; read
    by name first, ``cc_<x>_size`` would count as a second ``cc_*`` definition.
    """
    if not canon.endswith("_size"):
        return []
    base = canon[: -len("_size")]
    return [c for c in (base, f"{source_prefix}:{base}") if c in cols]


def clone_like_candidates(frame: pd.DataFrame, *, source_prefix: str) -> list[str]:
    cols = [str(c) for c in frame.columns]
    out = set()
    for col in cols:
        canon = _canonical_col(col, source_prefix=source_prefix)
        paired = _size_pair(canon, cols, source_prefix=source_prefix)
        if paired:
            out.update(paired)
        elif _name_family(canon) is not None:
            out.add(col)
    return sorted(out)


def resolve_clonotype_key(obs: pd.DataFrame, *, source_prefix: str = "airr") -> str:
    """The clonotype column that ``clonotype_key="auto"`` picks from ``obs``.

    Clone-like columns fall into three families, tried in order: ``clone_id``, ``cc_*``
    definitions, and any column with a ``<name>_size`` partner. A ``<base>_size`` column whose
    ``<base>`` is present stands for ``<base>`` and takes its family, so one ``cc_*`` definition
    with its size column is one candidate. Names may carry the ``<source_prefix>:`` prefix; a
    definition present under both names is read from the first column.

    Raises
    ------
    ValueError
        When the first family with a candidate has more than one definition, or when no column
        is clone-like.
    """
    cols = [str(c) for c in obs.columns]
    colset = set(cols)
    families: dict[str, dict[str, str]] = {}
    for col in cols:
        canon = _canonical_col(col, source_prefix=source_prefix)
        paired = _size_pair(canon, colset, source_prefix=source_prefix)
        if paired:
            col, canon = paired[0], canon[: -len("_size")]
            family = _name_family(canon) or "size"
        else:
            family = _name_family(canon)
        if family is not None:
            families.setdefault(family, {}).setdefault(canon, col)

    for family in _FAMILY_ORDER:
        found = families.get(family)
        if not found:
            continue
        if len(found) > 1:
            raise ValueError(
                "clonotype_key='auto' is ambiguous: multiple candidate clonotype columns were found "
                f"for family {family!r}: {sorted(found)}. Pass clonotype_key=... explicitly."
            )
        return next(iter(found.values()))

    raise ValueError(
        "clonotype_key='auto' found no clonotype candidates. "
        "Expected a clone-like column such as 'clone_id', 'cc_*', or a column with a paired "
        "'<name>_size' column."
    )
