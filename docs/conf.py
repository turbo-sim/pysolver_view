# API documentation uses the installed package (poetry install or pip install -e .).

# Define project metadata
project = "pysolver_view"
copyright = "2024, Roberto Agromayor"
author = "Roberto Agromayor"
release = "v0.6.9"

# Define extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "numpydoc",
]

# Document definitions in their canonical modules. Compatibility/lazy exports
# link to those definitions without documenting every imported object again.
autodoc_default_options = {"ignore-module-all": True}
numpydoc_show_class_members = False

# Exclude unnecessary files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# The generated package index links canonical modules and the compatibility API.

# Define theme
html_theme = "sphinx_book_theme"
# html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinx_rtd_theme'
