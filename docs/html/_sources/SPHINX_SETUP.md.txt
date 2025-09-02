# OpenFHE-Numpy Sphinx Documentation Setup

This document summarizes the Sphinx documentation setup for the OpenFHE-Numpy project, including environment, structure, build steps, and troubleshooting.

## Sphinx Environment

- **Sphinx**: 8.2.3
- **Theme**: sphinx-rtd-theme (3.0.2)
- **Extensions**: sphinx-autodoc-typehints (2.4.4), napoleon, viewcode, intersphinx, myst-parser (Markdown support)
- **NumPy**: For intersphinx linking

## Documentation Structure

    docs/
        conf.py            # Sphinx configuration
        index.rst          # Main documentation page (with toctree)
        installation.rst   # Installation instructions
        example.rst        # Example usage
        api.rst            # API page (with toctree for modules)
        tensor.rst         # Tensor API documentation
        constructor.rst    # Constructors API documentation
        matrix_arithmetic.rst # Matrix arithmetic API documentation
        enum.rst           # Enum API documentation
        SPHINX_SETUP.md    # This setup guide
        _build/html/       # Generated HTML output
        Makefile           # Build commands

## Organization & Navigation

- `index.rst` contains the main toctree, referencing all top-level docs: installation, example, api, SPHINX_SETUP.
- `api.rst` serves as the API reference landing page, with its own toctree for API modules (tensor, constructor, matrix_arithmetic, enum).
- All referenced `.rst` files should exist and be listed in the correct order for navigation.

## Configuration Features

- **Extensions**: autodoc, napoleon, viewcode, intersphinx, sphinx_autodoc_typehints, myst-parser
- **Theme**: Read the Docs theme for consistent styling
- **NumPy-style docstrings**: Supported via napoleon
- **Quick start**: Usage examples and code snippets
- **API reference**: Documents all modules and classes

## Building Documentation

```bash
cd docs
source ../venv/bin/activate
make clean
make html
```

## Viewing Documentation

Open `docs/_build/html/index.html` in a web browser, or run:
```bash
python -m http.server 8000 -d _build/html
```

## Rebuilding After Changes

```bash
make clean && make html
```

## Troubleshooting & Best Practices

- Ensure all referenced `.rst` files exist and are included in the correct toctree.
- If you see import errors for C++ extensions, verify the extension is built and installed in the active Python environment.
- For docstring formatting issues, use NumPy-style and check indentation.
- To resolve Sphinx warnings (e.g., missing `_static`), create the required directories or update `conf.py`.
- Always activate the correct virtual environment before building docs.

## Updating Structure

To add new API modules, create a new `.rst` file and add it to the `api.rst` toctree. Update `index.rst` if you want it to appear in the main navigation.
