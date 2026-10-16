"""API-contract conformance.

The stub inside ``governance/API_CONTRACT.md`` freezes the public surface. Three things are checked:

1. **Set equality** — the set of public callables the package actually exposes equals
   the set the contract declares. Neither direction may drift.
2. **Signature equality** — for every name in that set, the live signature matches the
   ``.pyi`` declaration (parameter names, kinds, and which carry defaults).
3. **Liveness** — every parameter of every declared function is read by the function that
   declares it. A parameter nothing reads is a promise the signature makes and the code
   ignores: passing it changes nothing, and checks 1 and 2 stay green.

Check 1 is set equality rather than an allowlist on purpose. An allowlist can only
police what somebody remembered to add to it, so a callable that was never listed is
invisible to it: not its signature, not even its existence.

Adding a public function is therefore a contract change by construction: the test fails
until it is declared in ``governance/API_CONTRACT.md``. That is the intent.
"""
import ast
import importlib
import inspect
import textwrap
from pathlib import Path

import pytest

import tcri

from tests._governance import contract_block

#: The stub is the ```python contract``` block of the API contract file. Read from the repo, not
#: from the installed package, so an installed copy can never shadow the contract.
STUB = contract_block("API_CONTRACT.md")

#: Contract namespace -> the live object whose public callables it declares.
#: ``TCRIModel`` is the class itself; the rest are the ``tcri.*`` accessor modules.
NAMESPACES: dict[str, str] = {
    "TCRIModel": "tcri.ml:TCRIModel",
    "pp": "tcri:pp",
    "tl": "tcri:tl",
    "pl": "tcri:pl",
    "get": "tcri:get",
    "diag": "tcri:diag",
    "perturb": "tcri:perturb",
    "null": "tcri:null",
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
    tree = ast.parse(STUB)
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


def _is_dunder(key: str) -> bool:
    """``TCRIModel.__init__`` is declared for its signature only; dunders are not part of the
    public-callable set (``_public_callables`` skips underscore names)."""
    return key.rpartition(".")[2].startswith("__")


def test_contract_stub_parses():
    """The frozen contract is present and declares a non-trivial surface."""
    assert len(CONTRACT) >= 20, f"contract looks truncated: {len(CONTRACT)} entries"
    # every declared key is Namespace-qualified (no accidental bare defs)
    assert all("." in k for k in CONTRACT), "unexpected un-namespaced contract entry"


def test_public_surface_equals_the_contract():
    """The package exposes exactly what the contract declares — no more, no less.

    This is the check that a hand-maintained allowlist structurally cannot make. A
    function absent from the allowlist was not merely unchecked; it was invisible, so
    "the contract passes" said nothing about it.
    """
    live, declared = _live_surface(), {k for k in CONTRACT if not _is_dunder(k)}

    undeclared = sorted(live - declared)
    assert not undeclared, (
        f"public but NOT in the contract: {undeclared}\n"
        f"Every public callable is part of the frozen surface. Declare it in the stub in "
        f"governance/API_CONTRACT.md, make it private with a leading underscore, or delete it."
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


def _layers(obj):
    """The public callable and every function it wraps, outermost first."""
    fn = getattr(obj, "__func__", obj)
    chain = [fn]
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
        chain.append(fn)
    return chain


def _definition(fn) -> ast.FunctionDef:
    """The ``def`` that compiled ``fn``, parsed from its own code object.

    ``inspect.getsource`` on a wrapper follows ``__wrapped__`` to the function it wraps; the
    code object does not, so each layer is read as itself.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn.__code__)))
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))


def _declared(node: ast.FunctionDef) -> set[str]:
    a = node.args
    named = [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
    return {p.arg for p in named if p is not None}


def _read(node: ast.FunctionDef) -> set[str]:
    """Names the body of ``node`` loads, nested functions included.

    Only a load of the parameter's own name counts. A store or a ``del`` is not a read, a
    default value is not part of the body, and ``locals()`` or a forwarded ``**kwargs`` excuses
    nothing: ``**kwargs`` is a parameter like any other and must itself be read.
    """
    return {n.id for stmt in node.body for n in ast.walk(stmt)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _unread(obj) -> list[str]:
    """Published parameters of ``obj`` that a function declaring them never reads.

    A decorated function publishes parameters from more than one ``def``: ``tl_result``
    publishes the wrapped body's own followed by the three it declares on its wrapper. Each
    parameter is checked against every ``def`` in the ``__wrapped__`` chain that declares it,
    so a decorator that reads ``groupby`` for its own bookkeeping cannot cover for a body that
    ignores it, and the parameters the decorator owns are the decorator's to read.
    """
    layers = [_definition(fn) for fn in _layers(obj)]
    unread = []
    for name, _, _ in _strip_receiver(_live_params(obj)):
        owners = [node for node in layers if name in _declared(node)]
        if not owners or any(name not in _read(node) for node in owners):
            unread.append(name)
    return unread


@pytest.mark.parametrize("key", sorted(CONTRACT))
def test_every_declared_parameter_is_read(key):
    """No declared function accepts a parameter that the function declaring it never reads.

    "Read" is weaker than "changes behavior": this catches the parameter nothing mentions,
    not the one that is mentioned and has no effect. That stronger property needs a behavioral
    test per knob.
    """
    ns, _, name = key.partition(".")
    unread = _unread(getattr(_resolve(NAMESPACES[ns]), name))
    assert not unread, (
        f"{key} declares {unread} but never reads them: a caller who passes one gets the same "
        f"result as one who does not. Read it, or remove it from the signature and from the "
        f"contract in governance/API_CONTRACT.md."
    )


def test_the_liveness_check_sees_through_tl_result():
    """The check reads each ``def`` in the chain as itself. ``tl_result``'s wrapper reads
    ``splitby`` for the reference's statistics, which does not make it read for a body that
    ignores it; ``null_model``, ``key_added`` and ``inplace`` are declared and read by the
    wrapper."""
    from tcri._state.storage import tl_result

    @tl_result(key="tcri_liveness_probe", default_null="phenotype")
    def _probe(adata, *, groupby=None, splitby=None, fit=None):
        return {"n_obs": adata.n_obs, "groupby": groupby, "fit": fit}

    assert _unread(_probe) == ["splitby"]


def test_contract_is_not_trivially_small():
    """Guards against a truncated or half-parsed .pyi silently making the set check pass."""
    assert len(CONTRACT) >= 25, f"contract looks truncated: {len(CONTRACT)} entries"


def test_import_smoke():
    """`import tcri` is green and the public namespaces resolve."""
    for ns in ("tl", "pp", "pl", "ml", "ut"):
        assert hasattr(tcri, ns), f"tcri.{ns} missing"


# Canonical keys that must come from `_state.keys` K.*, never a string literal, so that
# renaming one is a single edit rather than a grep across the package.
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
    from tcri._state import keys as K

    pkg = Path(tcri.__file__).parent
    forbidden = {getattr(K, n) for n in _MIGRATED_KEYS}
    offenders = []
    for py in pkg.rglob("*.py"):
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
