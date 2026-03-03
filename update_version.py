#!/usr/bin/env python3
"""
Script to update the version of the tcri package in setup.py.

Usage:
    python update_version.py 0.0.2
    python update_version.py --major  # Bump major version (e.g., 0.0.1 -> 1.0.0)
    python update_version.py --minor  # Bump minor version (e.g., 0.0.1 -> 0.1.0)
    python update_version.py --patch  # Bump patch version (e.g., 0.0.1 -> 0.0.2)
"""

import re
import sys
from pathlib import Path


def get_current_version(setup_file: Path) -> str:
    """Extract current version from setup.py."""
    content = setup_file.read_text()
    match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", content)
    if not match:
        raise ValueError("Could not find version in setup.py")
    return match.group(1)


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse version string into (major, minor, patch) tuple."""
    parts = version.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version}. Expected format: X.Y.Z")
    try:
        return tuple(map(int, parts))
    except ValueError:
        raise ValueError(f"Invalid version format: {version}. All parts must be integers.")


def bump_version(version: str, part: str) -> str:
    """Bump the specified part of the version."""
    major, minor, patch = parse_version(version)
    
    if part == 'major':
        major += 1
        minor = 0
        patch = 0
    elif part == 'minor':
        minor += 1
        patch = 0
    elif part == 'patch':
        patch += 1
    else:
        raise ValueError(f"Invalid version part: {part}. Must be 'major', 'minor', or 'patch'.")
    
    return f"{major}.{minor}.{patch}"


def update_version(setup_file: Path, new_version: str) -> None:
    """Update version in setup.py."""
    content = setup_file.read_text()
    
    # Validate new version format
    parse_version(new_version)
    
    # Replace version in setup.py
    new_content = re.sub(
        r"(version\s*=\s*['\"])([^'\"]+)(['\"])",
        rf"\g<1>{new_version}\g<3>",
        content
    )
    
    if new_content == content:
        raise ValueError("Failed to update version in setup.py")
    
    setup_file.write_text(new_content)


def main():
    setup_file = Path(__file__).parent / "setup.py"
    
    if not setup_file.exists():
        print(f"Error: {setup_file} not found", file=sys.stderr)
        sys.exit(1)
    
    current_version = get_current_version(setup_file)
    print(f"Current version: {current_version}")
    
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    arg = sys.argv[1]
    
    # Determine new version
    if arg.startswith('--'):
        part = arg[2:]
        if part not in ['major', 'minor', 'patch']:
            print(f"Error: Invalid argument '{arg}'", file=sys.stderr)
            print(__doc__)
            sys.exit(1)
        new_version = bump_version(current_version, part)
    else:
        new_version = arg
    
    # Update version
    try:
        update_version(setup_file, new_version)
        print(f"Version updated: {current_version} -> {new_version}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
