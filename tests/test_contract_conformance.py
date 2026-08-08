"""Contract conformance — the interface guardrail for the refactor.

``tcri/_contract.pyi`` freezes the public surface. Two things are checked:

1. **Set equality** — the set of public callables the package actually exposes equals
   the set the contract declares. Neither direction may drift.
2. **Signature equality** — for every name in that set, the live signature matches the
   ``.pyi`` declaration (parameter names, kinds, and which carry defaults).

Check 1 replaced a hand-maintained ``IMPLEMENTED`` allowlist. An allowlist can only
police what somebody remembered to add to it, so anything never listed was invisible:
not its signature, not even its existence. The same hole appeared in grafiti and in
tcri. When set equality went in it immediately found two things the allowlist had
never seen — ``TCRIModel.boost_phenotype_prior`` (a public method that multiplied the
eq-1 archetype concentration by a constant in place, undeclared and uncalled) and the
whole ``tcri.datasets`` namespace.

Adding a public function is now a contract change by construction: the test fails
until it is declared. That is the intent.

AST logic ported from grafiti's ``test_contract_conformance.py``.
"""
import ast
import importlib
import inspect
from pathlib import Path

import pytest

import tcri

PYI = Path(tcri.__file__).parent / "_contract.pyi"

#: Contract namespace -> the live object whose public callables it declares.
#: ``TCRIModel`` is the class itself; the rest are the ``tcri.*`` accessor modules.
NAMESPACES: dict[str, str] = {
    "TCRIModel": "tcri.ml:TCRIModel",
    "pp": "tcri:pp",
    "tl": "tcri:tl",
    "pl": "tcri:pl",
    "diag": "tcri:diag",
    "ut": "tcri:ut",
    "datasets": "tcri:datasets",
}


def _resolve(spec: str):
    module, _, attr = spec.partition(":")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _public_callables(obj) -> set[str]:
    """Public callables that tcri itself defines on ``obj``.

    Excludes underscore names, anything inherited from a base class outside tcri (scvi's
    BaseModelClass contributes ~40 methods that are not ours to freeze), and re-exported
    third-party callables, which are identified by ``__module__``.
    """
    names = set()
    for name in dir(obj):
        if name.startswith("_"):
            continue
        attr = getattr(obj, name, None)
        if not callable(attr):
            continue
        if not getattr(attr, "__module__", "").startswith("tcri"):
            continue
        if inspect.isclass(attr):
            continue
        names.add(name)
    return names


def _live_surface() -> set[str]:
    surface = set()
    for ns, spec in NAMESPACES.items():
        for name in _public_callables(_resolve(spec)):
            surface.add(f"{ns}.{name}")
    return surface


def _params_from_ast(a: ast.arguments):
    """(name, kind, has_default) per parameter of a .pyi FunctionDef."""
    params = []
    positional = list(a.posonlyargs) + list(a.args)
    n_def = len(a.defaults)
    for i, arg in enumerate(positional):
        kind = "POSITIONAL_ONLY" if arg in a.posonlyargs else "POSITIONAL_OR_KEYWORD"
        params.append((arg.arg, kind, i >= len(positional) - n_def))
    if a.vararg:
        params.append((a.vararg.arg, "VAR_POSITIONAL", False))
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        params.append((arg.arg, "KEYWORD_ONLY", default is not None))
    if a.kwarg:
        params.append((a.kwarg.arg, "VAR_KEYWORD", False))
    return params


def _strip_receiver(params):
    """Drop a leading self/cls so contract methods compare to live signatures."""
    if params and params[0][0] in ("self", "cls"):
        return params[1:]
    return params


def _live_params(fn):
    return [
        (p.name, p.kind.name, p.default is not inspect.Parameter.empty)
        for p in inspect.signature(fn).parameters.values()
    ]


def _contract_signatures():
    """Parse the .pyi: {"Namespace.func": [(name, kind, has_default), ...]}.

    Namespace classes (tl/pp/pl/diag/ut) and the real TCRIModel class both hold
    their functions as method FunctionDefs; keys are ``ClassName.funcname``.
    """
    tree = ast.parse(PYI.read_text())
    sigs = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, ast.FunctionDef):
                    sigs[f"{node.name}.{m.name}"] = _strip_receiver(_params_from_ast(m.args))
        elif isinstance(node, ast.FunctionDef):
            sigs[node.name] = _params_from_ast(node.args)
    return sigs


CONTRACT = _contract_signatures()


def test_contract_pyi_parses():
    """The frozen contract is present and declares a non-trivial surface."""
    assert PYI.exists(), "tcri/_contract.pyi missing"
    assert len(CONTRACT) >= 20, f"contract looks truncated: {len(CONTRACT)} entries"
    # every declared key is Namespace-qualified (no accidental bare defs)
    assert all("." in k for k in CONTRACT), "unexpected un-namespaced contract entry"


def test_public_surface_equals_the_contract():
    """The package exposes exactly what the contract declares — no more, no less.

    This is the check that a hand-maintained allowlist structurally cannot make. A
    function absent from the allowlist was not merely unchecked; it was invisible, so
    "the contract passes" said nothing about it.
    """
    live, declared = _live_surface(), set(CONTRACT)

    undeclared = sorted(live - declared)
    assert not undeclared, (
        f"public but NOT in the contract: {undeclared}\n"
        f"Every public callable is part of the frozen surface. Declare it in "
        f"tcri/_contract.pyi (a contract change, so CODEOWNERS applies), make it private "
        f"with a leading underscore, or delete it."
    )
    missing = sorted(declared - live)
    assert not missing, (
        f"declared in the contract but NOT public: {missing}\n"
        f"The contract promises these. Either implement/re-export them, or remove the "
        f"declaration — a contract that names things the package does not have is a "
        f"promise nothing keeps."
    )


@pytest.mark.parametrize("key", sorted(CONTRACT))
def test_signature_matches_contract(key):
    """Every declared function's live signature matches the frozen contract."""
    ns, _, name = key.partition(".")
    obj = getattr(_resolve(NAMESPACES[ns]), name)
    live = _strip_receiver(_live_params(obj))
    assert live == CONTRACT[key], (
        f"signature drift for {key}:\n  contract={CONTRACT[key]}\n  live    ={live}"
    )


def test_contract_is_not_trivially_small():
    """Guards against a truncated or half-parsed .pyi silently making the set check pass."""
    assert len(CONTRACT) >= 25, f"contract looks truncated: {len(CONTRACT)} entries"


def test_import_smoke():
    """`import tcri` is green and the public namespaces resolve."""
    for ns in ("tl", "pp", "pl", "ml", "ut"):
        assert hasattr(tcri, ns), f"tcri.{ns} missing"
    # diag is added in Phase 8; assert once it lands.


# migrated canonical keys (PR1) — must come from _keys.K.*, never a literal.
# legacy keys (tcri_clone_key/…, X_tcri_phenotypes) are exempt until their removal phase.
_MIGRATED_KEYS = [
    "METADATA", "P_CT", "LOCAL_SCALE", "CT_TO_COV", "CT_TO_C", "CT_ARRAY",
    "COV_ARRAY", "COVARIATE_CATEGORIES", "CLONOTYPE_CATEGORIES",
    "PHENOTYPE_CATEGORIES", "X_LOGITS", "X_LOGPOSTERIOR", "X_PROBABILITIES",
    "X_TCRI", "PHENOTYPE", "CLONE_SIZE",
]


def test_no_canonical_key_literals():
    """Migrated canonical keys must be accessed via `K.*` **in code** (subscripts /
    `.get(...)`), never a string literal. Docstrings and display/log strings may show
    the readable key name — this AST check only inspects real key-access code."""
    from tcri import _keys as K

    pkg = Path(tcri.__file__).parent
    forbidden = {getattr(K, n) for n in _MIGRATED_KEYS}
    offenders = []
    for py in pkg.rglob("*.py"):
        if py.name == "_keys.py":
            continue
        for node in ast.walk(ast.parse(py.read_text())):
            key = None
            if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                key = node.slice.value
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "get" and node.args
                  and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)):
                key = node.args[0].value
            if key in forbidden:
                offenders.append(f"{py.relative_to(pkg).as_posix()}:{node.lineno}: {key!r}")
    assert not offenders, (
        "canonical keys used as code literals (use K.*):\n  " + "\n  ".join(offenders)
    )
