# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import types

# Add the project root to the Python path so Sphinx can find the modules
sys.path.insert(0, os.path.abspath('..'))

# Create lightweight module mocks for compiled extensions and external libs
# Use ModuleType to avoid MagicMock causing import-time TypeError
ext_mod = types.ModuleType('openfhe_numpy.openfhe_numpy')
ext_mod.__file__ = '<mock-openfhe_numpy>'
sys.modules['openfhe_numpy.openfhe_numpy'] = ext_mod

# Mock OpenFHE package if it's not available during doc build
sys.modules['openfhe'] = types.ModuleType('openfhe')

# Provide minimal stubs used by the Python code during import-time
class _EnumStub:
    ROW_MAJOR = 0
    COL_MAJOR = 1
    DIAG_MAJOR = 2

# Attach expected names to the mocked extension module
ext_mod.ArrayEncodingType = _EnumStub
ext_mod.ROW_MAJOR = _EnumStub.ROW_MAJOR
ext_mod.COL_MAJOR = _EnumStub.COL_MAJOR
ext_mod.DIAG_MAJOR = _EnumStub.DIAG_MAJOR

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OpenFHE-Numpy'
copyright = '2025, OpenFHE Team'
author = 'OpenFHE Team'
release = '0.0.1'
version = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # Auto-generate docs from docstrings
    'sphinx.ext.autosummary',       # Generate summary tables (API index)
    'sphinx.ext.napoleon',          # Support for Google/NumPy style docstrings
    'sphinx.ext.viewcode',          # Add source code links
    'sphinx.ext.intersphinx',       # Link to other projects' documentation
    # 'sphinx.ext.autosummary',       # Generate summary tables (disabled for now)
    'sphinx_autodoc_typehints',     # Better type hint support
    'myst_parser',                  # Parse Markdown files (*.md)
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Napoleon settings (for NumPy-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'
autodoc_preserve_defaults = True

# Mock imports for modules that can't be imported during documentation build
autodoc_mock_imports = [
    'openfhe_numpy.openfhe_numpy',
    'openfhe'
]

# Autosummary settings
autosummary_generate = False  # Disable automatic autosummary generation (reduce import-time errors)

# Intersphinx mapping (links to external docs)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Theme options for Read the Docs theme
html_theme_options = {
    'analytics_id': '',  # Provided by you
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': 'myst',
}

# The master toctree document
master_doc = 'index'

# Add version info
html_context = {
    'display_github': True,
    'github_user': 'openfheorg',
    'github_repo': 'openfhe-numpy',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}
