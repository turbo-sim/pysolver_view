import os
import sys

# Add package root dir to path
sys.path.insert(0, os.path.abspath(".."))

# Define project metadata
project = "pysolver_view"
copyright = "2024, Roberto Agromayor"
author = "Roberto Agromayor"
release = "v0.4.0"

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

# Exclude unnecessary files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
exclude_patterns.extend(["source/api/pysolver_view.rst"])

# Define theme
html_theme = "sphinx_book_theme"
# html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinx_rtd_theme'
