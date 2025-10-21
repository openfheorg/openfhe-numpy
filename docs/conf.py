# conf.py
import os

project = "OpenFHE-Numpy"
author = "Ahmad Al Badawi, Tran Ngo, Yuriy Polyakov, Dmitriy Suponitskiy"
copyright = f"2025, {author}"
version = release = "1.4.2.0"
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

# Autosummary: generate stubs from our hub page (below)
autosummary_generate = True
# autosummary_imported_members = True  # enable if you re-export heavily in __init__.py

# Autodoc
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_preserve_defaults = True

# Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Source
source_suffix = {".rst": "restructuredtext", ".md": "myst"}
master_doc = "index"

# --- Conditional mocks (like openfhe-python) ---
autodoc_mock_imports = ["openfhe"]  # always safe to mock the heavy dep

NEED_MOCK = False
if os.environ.get("READTHEDOCS") == "True" or os.environ.get("CI") == "true":
    NEED_MOCK = True
else:
    try:
        import openfhe_numpy.openfhe_numpy  # noqa: F401
    except Exception:
        NEED_MOCK = True

if NEED_MOCK:
    autodoc_mock_imports.append("openfhe_numpy.openfhe_numpy")
