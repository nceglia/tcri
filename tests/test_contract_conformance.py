"""Contract conformance — the interface guardrail for the refactor.

``tcri/_contract.pyi`` freezes the target public surface. Each *implemented*
function's live signature is checked against its ``.pyi`` declaration (parameter
names, kinds, and which carry defaults). Declared-but-absent functions are the
worklist. Onboard a newly implemented function by adding it to ``IMPLEMENTED``
as its PR lands.

AST logic ported from grafiti's ``test_contract_conformance.py``.
"""
import ast
import importlib
import inspect
from pathlib import Path

import pytest

import tcri

PYI = Path(tcri.__file__).parent / "_contract.pyi"

# contract key ("Namespace.func" / "TCRIModel.method") -> (module, dotted attr).
# Each PR onboards its landed functions here; the signature test then enforces
# live == contract. PR4 landed the model→AnnData surface.
IMPLEMENTED: dict[str, tuple[str, str]] = {
    "tl.joint_distribution": ("tcri.tools._joint", "joint_distribution"),
    "tl.clonotypic_entropy": ("tcri.tools._entropy", "clonotypic_entropy"),
    "tl.phenotypic_entropy": ("tcri.tools._entropy", "phenotypic_entropy"),
    "tl.mutual_information": ("tcri.tools._mutual_information", "mutual_information"),
    "tl.phenotypic_flux": ("tcri.tools._flux", "phenotypic_flux"),
    "tl.compare_groups": ("tcri.tools._compare", "compare_groups"),
    "TCRIModel.setup_anndata": ("tcri.model._model", "TCRIModel.setup_anndata"),
    "TCRIModel.train": ("tcri.model._model", "TCRIModel.train"),
    "TCRIModel.get_latent_representation": ("tcri.model._model", "TCRIModel.get_latent_representation"),
    "TCRIModel.predict": ("tcri.model._model", "TCRIModel.predict"),
    "TCRIModel.get_p_ct": ("tcri.model._model", "TCRIModel.get_p_ct"),
    "TCRIModel.to_anndata": ("tcri.model._model", "TCRIModel.to_anndata"),
}


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


@pytest.mark.parametrize("key", sorted(IMPLEMENTED))
def test_signature_matches_contract(key):
    """Each implemented function's live signature matches the frozen contract."""
    module, attr = IMPLEMENTED[key]
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    live = _strip_receiver(_live_params(obj))
    assert live == CONTRACT[key], (
        f"signature drift for {key}:\n  contract={CONTRACT[key]}\n  live    ={live}"
    )


def test_report_unimplemented(capsys):
    """Informational worklist: declared-but-not-yet-implemented (never fails)."""
    todo = sorted(set(CONTRACT) - set(IMPLEMENTED))
    with capsys.disabled():
        print(f"\n[contract] implemented {len(IMPLEMENTED)}/{len(CONTRACT)}; "
              f"remaining worklist ({len(todo)}):")
        for k in todo:
            print(f"    ☐ {k}")
    assert True


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
