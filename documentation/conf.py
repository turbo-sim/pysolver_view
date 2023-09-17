import os
import sys
import subprocess
sys.path.insert(0, os.path.abspath('..'))  # Adjust this if needed

# Define project metadata
project = 'PySolverView'
# copyright = '2023, Roberto Agromayor'
author = 'Roberto Agromayor'
release = '1.0'

# Define extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'numpydoc',
]

# Exclude unnecessary files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
exclude_patterns.extend(['source/PySolverView.rst'])

# Define theme
# html_theme = 'pydata_sphinx_theme'
html_theme = 'sphinx_book_theme'
# html_theme = 'sphinx_rtd_theme'


def generate_docs():
    # sphinx-apidoc.exe -o source/ ../myPackage -e
    subprocess.run(["make.bat", "clean"])
    subprocess.run(["make.bat", "html"])


def generate_shortcut():

    # Check if the platform is Windows
    if sys.platform == "win32":
        import winshell

        # Get the current working directory
        current_dir = os.getcwd()

        # Define the source path of the file for which you want to create a shortcut
        source_path = os.path.join(current_dir, "_build/html/index.html")

        # Define the destination path for the shortcut
        destination_path = os.path.join(current_dir, "..", "documentation.html.lnk")

        # Create the shortcut
        shortcut = winshell.shortcut(source_path)
        shortcut.write(destination_path)



if __name__ == "__main__":
    generate_docs()
    generate_shortcut()



# # Configuration file for the Sphinx documentation builder.
# #
# # For the full list of built-in configuration values, see the documentation:
# # https://www.sphinx-doc.org/en/master/usage/configuration.html

# # -- Project information -----------------------------------------------------
# # https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

