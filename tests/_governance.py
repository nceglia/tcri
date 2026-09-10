"""Read the machine-checked block out of a contract file.

Each contract under ``governance/`` is one Markdown file. The part a test enforces sits in a
single fenced block opened with ```` ```python contract ````; the rest of the file is the
definition in prose. This keeps one file per contract and lets the test read the same text a
person reads.
"""
from __future__ import annotations

import re
from pathlib import Path

GOVERNANCE = Path(__file__).resolve().parents[1] / "governance"
_FENCE = re.compile(r"```python contract\n(.*?)\n```", re.DOTALL)


def contract_block(name: str) -> str:
    """The source text of the one ```python contract``` block in ``governance/<name>``."""
    path = GOVERNANCE / name
    assert path.is_file(), f"missing contract file {path}"
    blocks = _FENCE.findall(path.read_text())
    assert len(blocks) == 1, (
        f"{name} must carry exactly one ```python contract``` block, found {len(blocks)}"
    )
    return blocks[0]


def contract_namespace(name: str) -> dict:
    """Execute the block and return its names (plain lists, dicts and scalars)."""
    ns: dict = {}
    exec(compile(contract_block(name), f"governance/{name}#contract", "exec"), ns)
    return {k: v for k, v in ns.items() if not k.startswith("__")}
