#!/usr/bin/env python3
"""
Update the version of the tcri package.

The single source of truth for the version is the ``version`` field in
``pyproject.toml``. ``tcri.__version__`` and the docs (``docs/conf.py``) both
read it back via ``importlib.metadata``, so this script only edits
``pyproject.toml``.

Usage:
    python update_version.py 0.2.0     # set an explicit version
    python update_version.py --major   # 0.1.0 -> 1.0.0
    python update_version.py --minor   # 0.1.0 -> 0.2.0
    python update_version.py --patch   # 0.1.0 -> 0.1.1
"""

import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).parent / "pyproject.toml"
# Match the [project] `version = "X.Y.Z"` line (anchored to start of line).
VERSION_RE = re.compile(r'(?m)^(version\s*=\s*")([^"]+)(")')


def get_current_version(text: str) -> str:
    match = VERSION_RE.search(text)
    if not match:
        raise ValueError('Could not find `version = "..."` in pyproject.toml')
    return match.group(2)


def parse_version(version: str) -> tuple:
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version {version!r}; expected X.Y.Z")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise ValueError(f"Invalid version {version!r}; all parts must be integers")


def bump_version(version: str, part: str) -> str:
    major, minor, patch = parse_version(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Invalid part {part!r}; must be major, minor, or patch")


def main() -> None:
    if not PYPROJECT.exists():
        sys.exit(f"Error: {PYPROJECT} not found")

    text = PYPROJECT.read_text()
    current = get_current_version(text)
    print(f"Current version: {current}")

    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]
    if arg.startswith("--"):
        new_version = bump_version(current, arg[2:])
    else:
        parse_version(arg)  # validate format
        new_version = arg

    if new_version == current:
        print(f"Version already {current}; nothing to do.")
        return

    new_text = VERSION_RE.sub(rf"\g<1>{new_version}\g<3>", text, count=1)
    if new_text == text:
        sys.exit("Error: failed to update version in pyproject.toml")

    PYPROJECT.write_text(new_text)
    print(f"Version updated: {current} -> {new_version}")
    print(
        "Note: tcri.__version__ and the docs read this via importlib.metadata; "
        "reinstall (pip install -e .) to refresh the installed metadata."
    )


if __name__ == "__main__":
    main()
