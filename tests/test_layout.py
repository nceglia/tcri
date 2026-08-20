"""The package layout, as a test.

Every rule here was violated at least once and fixed by hand. A layout that is only ever
restored by someone noticing will drift back the moment nobody is looking, so the shape is
pinned rather than described.

What each rule is defending against, concretely:

* **Contract manifests inside the package.** They are enforcement, not code; nothing in ``tcri``
  imports them. While they sat in ``tcri/tools/`` and ``tcri/model/`` every wheel shipped them,
  handing users a frozen declaration they could mistake for an API.
* **Loose private modules at the top level.** ``_console`` (dead), ``_keys`` (a shim outliving
  its migration), ``_stats``, ``_distance`` — the top level had become where anything shared
  landed by default.
* **Upward imports.** ``_compute`` and ``_stats`` are lower layers. A module-level import back up
  into ``tools`` makes the package import-order dependent; one such edge existed and resolved
  only by luck.
* **An unbounded top-level surface.** With no ``__all__``, ``sys`` and ``PackageNotFoundError``
  were advertised by ``dir(tcri)`` purely because they were imported at module scope.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

import tcri

PKG = Path(tcri.__file__).parent
ROOT = PKG.parent

#: The only modules allowed to sit loose at the top of the package. Everything else belongs in a
#: subpackage named for its role (``_state``, ``_stats``, ``_compute``) or in a view directory.
ALLOWED_TOP_LEVEL = {"__init__.py", "get.py"}

#: Lower layers, in dependency order. Nothing here may import UP into a view package at module
#: scope. (Lazy in-function imports are allowed and used deliberately, with a note at the site.)
LOWER_LAYERS = ["_compute", "_stats", "_state"]
VIEW_PACKAGES = ["tools", "plotting", "preprocessing", "diagnostics", "model", "utils", "datasets"]


def _module_level_imports(path: Path):
    """(module, lineno) for every import at module scope — NOT inside a function or class.

    The distinction is the whole point: a lazy import inside a function is how `_compute` is
    allowed to reach one symbol in `tools` without inverting the layering.
    """
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            yield ("." * (node.level or 0)) + (node.module or ""), node.lineno


def test_no_loose_top_level_modules():
    loose = {p.name for p in PKG.glob("*.py")} - ALLOWED_TOP_LEVEL
    assert not loose, (
        f"loose private modules at the top of tcri/: {sorted(loose)}. Put shared code in the "
        f"subpackage named for its role — _state (persistence), _stats (statistics), _compute "
        f"(numerics) — rather than at the top level, which is how _console, _keys, _stats and "
        f"_distance all accumulated there."
    )


def test_no_contract_manifests_inside_the_package():
    # __pycache__ is skipped deliberately: a stale .pyc from a module deleted in this very pass
    # is build residue, not a layout violation, and it would make the guard fail for a reason
    # that has nothing to do with what it is defending.
    offenders = [str(p.relative_to(ROOT)) for p in PKG.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts
                 and ("contract" in p.name.lower() or p.suffix == ".pyi")]
    assert not offenders, (
        f"contract/enforcement files inside the installed package: {offenders}. They belong in "
        f"tests/contracts/ — nothing in tcri imports them, and anything under tcri/ ships in the "
        f"wheel."
    )


@pytest.mark.parametrize("layer", LOWER_LAYERS)
def test_lower_layers_do_not_import_up_at_module_scope(layer):
    d = PKG / layer
    if not d.is_dir():
        pytest.skip(f"{layer} is not a package")
    bad = []
    for py in d.rglob("*.py"):
        for mod, lineno in _module_level_imports(py):
            target = mod.lstrip(".")
            if any(target == v or target.startswith(v + ".") for v in VIEW_PACKAGES):
                bad.append(f"{py.relative_to(ROOT)}:{lineno} -> {mod}")
    assert not bad, (
        f"{layer} imports UP into a view package at module scope: {bad}. Lower layers must not "
        f"depend on their consumers — it makes the package import-order dependent. If the symbol "
        f"is genuinely needed, import it lazily inside the function and say why at the site."
    )


def test_top_level_surface_is_declared():
    undeclared = [n for n in dir(tcri) if not n.startswith("_") and n not in tcri.__all__]
    assert not undeclared, f"public names on tcri not in __all__: {undeclared}"


def test_governance_lives_outside_the_docs_tree():
    """`governance/` and `docs/contracts/` were one letter apart as `docs/contract`."""
    assert (ROOT / "governance").is_dir(), "governance/ is missing"
    assert not (ROOT / "docs" / "contract").exists(), (
        "docs/contract/ is back. The governance corpus lives in governance/ at the repo root; "
        "docs/contracts/ is the published reader page and the two names differ by one letter."
    )
    for name in ["API_CONTRACT.md", "MODEL_CONTRACT.md", "METRICS_CONTRACT.md",
                 "TRAINING_CONTRACT.md"]:
        assert (ROOT / "governance" / name).is_file(), f"governance/{name} is missing"


def test_the_wheel_ships_only_the_package():
    """The check that actually consumes the layout.

    Every layout defect in this pass was invisible to the unit suite and visible in a built
    artifact: contract manifests shipping, dead modules shipping. `git ls-files` is a cheap
    proxy for it — nothing outside `tcri/` may be inside `tcri/`.
    """
    tracked = subprocess.run(["git", "ls-files", "tcri/"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    strays = [f for f in tracked if not (f.endswith(".py") or f.endswith("py.typed"))]
    assert not strays, f"non-Python files tracked inside the package: {strays}"


#: The public view namespaces. Each must expose exactly what it declares.
VIEW_NAMESPACES = ["tl", "pl", "pp", "ut", "ml", "diag", "datasets", "get"]

#: `from __future__ import annotations` binds a module-scope `_Feature` object literally named
#: `annotations`. It is a language artifact present in every module using the import, not
#: something the package exports, and there is no way to suppress it short of dropping the
#: import. Excluded by name rather than by a blanket "ignore anything non-callable", which
#: would also hide real leaks like the AD_FILE/META_FILE path constants.
_FUTURE_ARTIFACTS = {"annotations"}


@pytest.mark.parametrize("ns", VIEW_NAMESPACES)
def test_namespace_exposes_exactly_its_all(ns):
    """A view namespace must declare its surface, and the declaration must be complete.

    `__all__` alone is not enough: it governs `from x import *`, but `dir(x)` — which is what
    tab-completion and `help()` show, and what a reader treats as the API — still lists every
    module-scope binding. `tcri.ut` was the worst case, advertising 27 public names for a
    namespace with 4: numpy, os, sys, matplotlib, typing aliases, scipy functions and the
    private session-path constants, all pulled in by `from ._utils import *` over a module with
    no `__all__`.

    The contract's surface check could not see any of it. `_public_callables` freezes callables
    whose `__module__` starts with "tcri", so a re-exported third-party name is invisible to it
    by construction — correctly, since freezing scipy's signature is not tcri's business. That
    is exactly why this test is separate: it asks a different question, whether the namespace
    exposes anything it did not mean to.
    """
    mod = getattr(tcri, ns)
    declared = getattr(mod, "__all__", None)
    assert declared is not None, (
        f"tcri.{ns} has no __all__. Every public namespace declares its surface; without one, "
        f"`import *` re-exports whatever the module happened to bind at import time."
    )
    public = {n for n in dir(mod) if not n.startswith("_")} - _FUTURE_ARTIFACTS
    undeclared = sorted(public - set(declared))
    assert not undeclared, (
        f"tcri.{ns} exposes names not in its __all__: {undeclared}. Either add them to __all__ "
        f"(and to tests/contracts/api.pyi, since that is a contract change), or alias the import "
        f"private — `import numpy as _np`. Adding to __all__ without declaring it in the contract "
        f"will fail test_public_surface_equals_the_contract."
    )


@pytest.mark.parametrize("ns", VIEW_NAMESPACES)
def test_namespace_all_contains_only_tcri_objects(ns):
    """Declaring a name is not the same as owning it.

    The companion test above asks whether anything is exposed but undeclared. That alone is
    satisfiable the wrong way: adding ``"np"`` to ``__all__`` makes the leak *declared* and the
    test green. The contract's surface check does not catch it either — ``_public_callables``
    requires ``callable(attr)``, and a module is not callable — so a re-exported third-party
    module could sit in the public API with both guards passing. This was found by mutating
    exactly that and watching the suite stay green.

    So: every name a namespace declares must be something tcri defines, or a tcri submodule.
    """
    import types

    mod = getattr(tcri, ns)
    foreign = []
    for name in getattr(mod, "__all__", []):
        obj = getattr(mod, name, None)
        origin = getattr(obj, "__module__", None) or getattr(obj, "__name__", "")
        if isinstance(obj, types.ModuleType):
            if not getattr(obj, "__name__", "").startswith("tcri"):
                foreign.append(f"{name} (module {obj.__name__})")
        elif origin and not str(origin).startswith("tcri"):
            foreign.append(f"{name} (from {origin})")
    assert not foreign, (
        f"tcri.{ns}.__all__ declares things tcri does not define: {foreign}. Re-exporting a "
        f"third-party name makes it part of tcri's public API — its signature, its deprecations, "
        f"its breakage. Import it privately (`import numpy as _np`) instead."
    )
