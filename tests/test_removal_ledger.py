"""The removal ledger, as a test.

A symbol removed from the public surface must stay removed, and a hand-ticked list cannot
answer "is it *still* gone?" — the question that matters once the deleting branch is long
merged. So the ledger is pinned here: a deleted symbol reappearing on the public surface, via a
revert, a bad merge or a helpful re-export, fails this test instead of shipping.

Most entries assert **absence from the public namespace**: gone from the namespace, from
``__all__`` and from the imports. Two entries are deliberately narrower and are handled separately
below: `compare_groups`, which was removed from the public surface but *kept* as an internal
helper, and the delta pair, which was removed and then reinstated.

The ledger also pins removed **stored-result columns** and the word they were named after. A
column has no namespace to be absent from, so it is checked on the payloads every tool stores;
and a retired word survives longest in prose, so that one is a source-tree check over the
package, the contracts and the docs.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

import tcri

#: namespace -> symbols removed from the public surface. Add a row here whenever a public
#: symbol is deleted; see the removal rule in ``governance/RULES.md``.
REMOVED = {
    # dead or out of scope, and the registration helpers folded into to_anndata / the session
    "pp": [
        "get_latent_embedding", "group_small_clones", "register_probability_columns",
        "remove_meaningless_genes", "gene_entropy", "classify_phenotypes",
        "register_model", "register_phenotype_key", "register_clonotype_key",
        "_compute_logits_and_prior", "joint_distribution_posterior", "joint_distribution",
    ],
    # consolidated into the metric surface that replaced them
    "tl": [
        "clone_fraction", "mi_compare", "delta_entropy_table", "flux_table",
        "clonotypic_entropy_base", "clonality", "dkl",
    ],
    # non-core plots — dropped, not moved to examples
    "pl": [
        "polar_plot", "probability_distribution", "bayesian_mutual_information",
        "probability_ternary", "top_clone_umap", "clone_size_umap",
        "plot_phenotype_probabilities", "compare_phenotypes", "ridge_delta_entropy",
        "clonality", "tcri_boxplot", "set_color_palette", "plot_pheno_sankey",
        "resolve_palette", "centropy", "pentropy",
    ],
    # out of the package
    "ut": [
        "probabilities", "write_adata_safely", "_pop_nonserializables",
        "build_nested_tcri_pgm", "draw_tcri_pgm_nested",
    ],
    # model/utils cleanup
    "ml": ["plot_loss", "plot_archetypes"],
}

#: Module-level names that must NOT be on the top-level package. They are module machinery, not
#: API: a plain ``import sys`` at the top of ``__init__.py`` is advertised by ``dir(tcri)``
#: unless it is aliased private. This pins both the aliasing and ``__all__``.
LEAKED_FROM_INIT = ["sys", "PackageNotFoundError", "version", "annotations"]

#: Deleted modules. ``hasattr`` on a namespace cannot catch these — they were never attributes of
#: a namespace, they were importable module paths — so they need an import check.
DELETED_MODULES = ["tcri._console", "tcri._keys", "tcri.tools._common", "tcri.tools._compare",
                   "tcri._distance"]


@pytest.mark.parametrize(
    ("namespace", "symbol"),
    [(ns, s) for ns, syms in REMOVED.items() for s in syms],
    ids=[f"{ns}.{s}" for ns, syms in REMOVED.items() for s in syms],
)
def test_removed_symbols_stay_removed(namespace, symbol):
    mod = getattr(tcri, namespace)
    assert not hasattr(mod, symbol), (
        f"tcri.{namespace}.{symbol} is back. It is listed as removed in REMOVED above (the "
        f"removal rule in governance/RULES.md). If the reinstatement is deliberate, remove it "
        f"from REMOVED here — do not delete this assertion."
    )


def test_compare_groups_is_internal_not_public():
    """Removed from the public surface, deliberately *kept* as an internal helper.

    It is not dead code; it is no longer a step the user performs. So the assertion is narrower
    than for the rest — absent from `tl`, still importable internally.
    """
    assert not hasattr(tcri.tl, "compare_groups")
    from tcri._stats import compare_groups  # noqa: F401


def test_the_delta_pair_was_reinstated_and_must_exist():
    """The counter-case, pinned so nobody "fixes" a failure by re-adding these to REMOVED.

    `compare_groups` contrasts between groups on a tidy frame and cannot compute a metric at two
    covariate levels, so it never replaced the paired entropies. `delta_entropy_table` stayed
    gone (it is in REMOVED above); these two are the public form the capability came back as.
    """
    assert hasattr(tcri.tl, "delta_clonotypic_entropy")
    assert hasattr(tcri.tl, "delta_phenotypic_entropy")


@pytest.mark.parametrize("name", LEAKED_FROM_INIT)
def test_module_machinery_is_not_public(name):
    assert not hasattr(tcri, name), (
        f"tcri.{name} is exposed at the top level. It is module machinery, not API — alias it "
        f"private (e.g. `import sys as _sys`) rather than deleting this assertion."
    )
    assert name not in getattr(tcri, "__all__", []), f"{name} must not be in tcri.__all__"


def test_top_level_surface_is_bounded_by_all():
    """Every public name on the package must be declared. This is what stops the next
    ``import x`` at module scope from silently becoming part of the surface."""
    undeclared = [n for n in dir(tcri) if not n.startswith("_") and n not in tcri.__all__]
    assert not undeclared, f"undeclared public names on tcri: {undeclared}"


#: Columns removed from every stored ``tl`` payload. The adjusted value replaced them: one
#: reported quantity per metric, named for what it is.
REMOVED_COLUMNS = ["excess", "excess_from", "excess_to"]

#: The word the removed columns were named after; retired from the package, the contracts and
#: the docs with them. Release notes are exempt, since the note announcing the rename names it.
RETIRED_WORDS = ["excess"]

_SCANNED = ("tcri", "governance", "docs", "README.md")
_TEXT_SUFFIXES = {".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".yml"}
_SKIPPED_PARTS = {"__pycache__", "_build", "release-notes"}


def test_removed_columns_stay_removed(cohort):
    """No slot of any stored result carries a removed column, at the widest call each tool has.

    The maximal calls of the schema snapshot are reused so the check covers every column a
    reference adds: ``groupby``, ``splitby``, draws, and the metric's default null.
    """
    from tests.test_result_schemas import _calls, _run

    model, adata = cohort
    for name, (_minimal, maximal) in _calls(adata).items():
        payload = _run(name, maximal, model, adata)
        for slot, frame in payload.items():
            if not isinstance(frame, pd.DataFrame):
                continue
            bad = [c for c in frame.columns
                   if c in REMOVED_COLUMNS or str(c).startswith("excess")]
            assert not bad, f"{name}.{slot} carries removed columns {bad}"


def test_the_retired_words_are_gone():
    """A retired word appears in no text file of the package, the contracts or the docs.

    Text files only, read as UTF-8: the docs carry images, and a test run leaves bytecode under
    the package. Built docs and release notes are skipped.
    """
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile("|".join(re.escape(w) for w in RETIRED_WORDS), re.IGNORECASE)
    hits = []
    for base in _SCANNED:
        path = root / base
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for f in files:
            rel = f.relative_to(root)
            if f.suffix not in _TEXT_SUFFIXES or _SKIPPED_PARTS & set(rel.parts):
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{rel}:{i}: {line.strip()[:80]}")
    assert not hits, (
        f"{len(hits)} mention(s) of a retired word; the first ten:\n  " + "\n  ".join(hits[:10])
    )


@pytest.mark.parametrize("mod", DELETED_MODULES)
def test_deleted_modules_stay_deleted(mod):
    """Import-path removals, which the namespace checks above cannot see.

    ``tcri._keys`` in particular would resolve happily if someone re-added the shim, and every
    `hasattr` test here would still pass.
    """
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod)
