"""Shared field-resolution helpers used by setup and adapters."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

__all__ = ["clone_like_candidates", "resolve_clonotype_source"]


def _canonical_col(name: str, *, source_prefix: str) -> str:
    prefix = f"{source_prefix}:"
    return name[len(prefix):] if name.startswith(prefix) else name


def clone_like_candidates(frame: pd.DataFrame, *, source_prefix: str) -> list[str]:
    cols = [str(c) for c in frame.columns]
    out = set()
    for col in cols:
        canon = _canonical_col(col, source_prefix=source_prefix)
        if canon == "clone_id" or canon.startswith("cc_"):
            out.add(col)
            continue
        if canon.endswith("_size"):
            base = canon[:-5]
            prefixed = f"{source_prefix}:{base}"
            if base in cols:
                out.add(base)
            if prefixed in cols:
                out.add(prefixed)
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
            family = None
            if canon == "clone_id":
                family = "clone_id"
            elif canon.startswith("cc_"):
                family = "cc"
            elif canon.endswith("_size"):
                base = canon[:-5]
                if base in colset:
                    family, col, canon = "size", base, base
                elif f"{source_prefix}:{base}" in colset:
                    family, col, canon = "size", f"{source_prefix}:{base}", base
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
