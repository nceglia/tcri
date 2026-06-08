# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'TCRi'
description = 'Information Theoretic Framework for Paired Single Cell Gene Expression and TCR Sequencing'
copyright = '2022-2025'
author = 'Nicholas Ceglia'

# The full version, including alpha/beta/rc tags.
# Single source of truth is pyproject.toml. Prefer installed metadata; fall back
# to parsing pyproject.toml directly so the version is correct even when the
# package is not installed in the docs environment (e.g. on ReadTheDocs, where
# the heavy runtime deps are mocked rather than installed).
def _get_release():
    try:
        from importlib.metadata import version as _pkg_version
        return _pkg_version("tcri")
    except Exception:
        pass
    try:
        import tomllib  # Python 3.11+
        _pp = os.path.join(os.path.dirname(__file__), os.pardir, "pyproject.toml")
        with open(_pp, "rb") as _f:
            return tomllib.load(_f)["project"]["version"]
    except Exception:
        return "0.0.0"


release = _get_release()
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'myst_parser',
]

# MyST: enable dollar-delimited math ($...$ and $$...$$).
myst_enable_extensions = ["dollarmath"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here.
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]

html_logo = '../framework.png'
html_favicon = '../framework.png'

html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension -------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
# Read constructor signatures from ``__init__`` directly. Without this, classes
# whose (mocked) base injects ``__new__(*args, **kwargs)`` — e.g. TCRIModel via
# scvi's BaseModelClass — would render as ``TCRIModel(*args, **kwargs)`` on the
# deps-mocked ReadTheDocs build.
autodoc_class_signature = 'separated'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
# Show clean object names (e.g. ``joint_distribution``) rather than the full
# private module path in signatures and headings.
add_module_names = False
# Generate stub pages for any autosummary directives.
autosummary_generate = True
# Mock the heavy / native scientific dependencies so the API reference can be
# built from source without installing the full ML stack (torch, scvi-tools,
# scanpy, ...). autodoc still reads the real signatures and docstrings of TCRi's
# own code; only third-party imports are stubbed. numpy and pandas are kept real
# (they're light and let intersphinx resolve their types).
autodoc_mock_imports = [
    "torch",
    "pyro",
    "scvi",
    "sklearn",
    "scanpy",
    "anndata",
    "scipy",
    "matplotlib",
    "seaborn",
    "mpltern",
    "umap",
    "tqdm",
    "daft",
    "gseapy",
]

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scanpy': ('https://scanpy.readthedocs.io/en/stable/', None),
}
