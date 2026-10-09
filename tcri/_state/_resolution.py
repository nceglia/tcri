"""Shared field-resolution helpers used by setup and adapters."""
from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import pandas as pd

__all__ = ["clone_like_candidates", "resolve_clonotype_source"]


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


def _iter_sources(
    sources: Mapping[str, pd.DataFrame] | Sequence[tuple[str, pd.DataFrame]],
) -> list[tuple[str, pd.DataFrame]]:
    if isinstance(sources, Mapping):
        return [(str(k), v) for k, v in sources.items()]
    return [(str(k), v) for k, v in sources]


def resolve_clonotype_source(
    sources: Mapping[str, pd.DataFrame] | Sequence[tuple[str, pd.DataFrame]],
    *,
    clonotype_key: str,
    source_prefix: str = "airr",
    family_order: Sequence[str] = ("clone_id", "cc", "size"),
    source_preference: Mapping[str, int] | None = None,
) -> tuple[str, str, str, list[tuple[str, str, str, str]]]:
    """Resolve a clonotype column from one or more obs-like frames.

    A ``<base>_size`` column whose ``<base>`` is present stands for ``<base>`` and takes its
    family (``clone_id`` or ``cc``, otherwise ``size``), so one ``cc_*`` definition with its size
    column is one candidate.

    Returns ``(source_name, resolved_key, family, candidates)`` where candidates are tuples
    ``(family, canonical, source_name, key)``.
    """
    source_items = _iter_sources(sources)
    source_rank = (
        {name: i for i, (name, _) in enumerate(source_items)}
        if source_preference is None else
        {str(k): int(v) for k, v in source_preference.items()}
    )

    seen = set()
    candidates: list[tuple[str, str, str, str]] = []
    for source_name, frame in source_items:
        cols = [str(c) for c in frame.columns]
        colset = set(cols)
        for col in cols:
            canon = _canonical_col(col, source_prefix=source_prefix)
            paired = _size_pair(canon, colset, source_prefix=source_prefix)
            if paired:
                col, canon = paired[0], canon[: -len("_size")]
                family = _name_family(canon) or "size"
            else:
                family = _name_family(canon)
            if family is None:
                continue
            marker = (source_name, col)
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append((family, canon, source_name, col))

    if clonotype_key != "auto":
        checks = [clonotype_key]
        if ":" not in clonotype_key:
            checks.append(f"{source_prefix}:{clonotype_key}")
        for source_name, frame in source_items:
            for key in checks:
                if key in frame.columns:
                    return source_name, key, "explicit", candidates

        hint = ", ".join(sorted({c for _, f in source_items for c in clone_like_candidates(f, source_prefix=source_prefix)})[:8])
        names = ", ".join(name for name, _ in source_items)
        raise KeyError(
            f"clonotype_key={clonotype_key!r} was not found. "
            f"Expected it in: {names}. "
            f"{'Clone-like columns include: ' + hint if hint else 'No clone-like columns found.'}"
        )

    for family in family_order:
        subset = [c for c in candidates if c[0] == family]
        if not subset:
            continue
        by_canon: dict[str, list[tuple[str, str]]] = {}
        for _family, canonical, source_name, key in subset:
            by_canon.setdefault(canonical, []).append((source_name, key))
        if len(by_canon) > 1:
            options = sorted(by_canon)
            raise ValueError(
                "clonotype_key='auto' is ambiguous: multiple candidate clonotype columns were found "
                f"for family {family!r}: {options}. Pass clonotype_key=... explicitly."
            )
        canonical = next(iter(by_canon))
        chosen = sorted(
            by_canon[canonical],
            key=lambda item: source_rank.get(item[0], len(source_rank)),
        )[0]
        return chosen[0], chosen[1], family, candidates

    raise ValueError(
        "clonotype_key='auto' found no clonotype candidates. "
        "Expected a clone-like column such as 'clone_id', 'cc_*', or a column with a paired "
        "'<name>_size' column."
    )
